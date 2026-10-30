#!/usr/bin/env python3
"""Resize MP4s to 256x448, sample 97 frames and extract matching first-frame PNGs.

Requires Python 3.8+, ffmpeg, and ffprobe; no pip packages. The encoder is
checked before processing: auto prefers libx264 and falls back to native mpeg4.
Sampling covers the whole clip, including its first and last frames. Short clips
repeat uniformly selected frames. Output FPS preserves the original duration.
Spatial resizing stretches directly to the requested size, without cropping.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading


DATA_ROOT = Path('/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats_new')
STOP = threading.Event()


def run(command, data=None):
    result = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-4000:].strip())
    return result.stdout


def probe(path, threads=1, ffprobe='ffprobe'):
    data = json.loads(run([
        ffprobe, '-v', 'error', '-threads', str(threads), '-select_streams', 'v:0', '-count_frames',
        '-show_entries',
        'stream=width,height,nb_read_frames,avg_frame_rate,duration,duration_ts,time_base,sample_aspect_ratio',
        '-of', 'json', str(path),
    ]))
    if not data.get('streams'):
        raise ValueError('No video stream: {}'.format(path))
    stream = data['streams'][0]
    stream['frames'] = int(stream.get('nb_read_frames', 0))
    if stream['frames'] < 1:
        raise ValueError('No decodable video frames: {}'.format(path))
    return stream


def duration(stream):
    # Integer timestamp duration avoids rounding e.g. 97/24 to 4.041667.
    for calculate in (
        lambda: int(stream['duration_ts']) * Fraction(stream['time_base']),
        lambda: Fraction(stream['frames']) / Fraction(stream['avg_frame_rate']),
        lambda: Fraction(stream['duration']),
    ):
        try:
            value = calculate()
            if value > 0:
                return value
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            pass
    raise ValueError('Cannot determine a positive source duration')


def validate(path, args):
    info = probe(path, args.threads, args.ffprobe)
    actual = (int(info['width']), int(info['height']), info['frames'])
    expected = (args.width, args.height, args.frames)
    if actual != expected or info.get('sample_aspect_ratio') != '1:1':
        raise ValueError('Output validation failed: {} (expected {}, SAR=1:1)'.format(actual, expected))


def temporary_path(parent, suffix):
    fd, name = tempfile.mkstemp(prefix='.resize-', suffix=suffix, dir=str(parent))
    os.close(fd)
    return Path(name)


def codec_options(args):
    if args.encoder == 'libx264':
        return ['-c:v', 'libx264', '-crf', str(args.crf), '-preset', args.preset]
    # MPEG-4 Part 2 requires a time-base denominator <= 65535. A fixed legal
    # time base also supports the fractional rates used to preserve duration.
    return ['-c:v', 'mpeg4', '-q:v', str(args.mpeg4_quality), '-enc_time_base', '1:60000']


def preflight(args):
    """Check the actual encoder and PNG output once, before starting workers."""
    listing = run([args.ffmpeg, '-hide_banner', '-encoders']).decode(errors='replace')
    available = set()
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[0]) == 6 and fields[0].startswith('V'):
            available.add(fields[1])
    requested = args.encoder
    candidates = ['libx264', 'mpeg4'] if requested == 'auto' else [requested]
    candidates = [name for name in candidates if name in available]
    if not candidates:
        raise ValueError(
            'FFmpeg {} has no usable requested encoder ({}). Run it with '
            '"-hide_banner -encoders", or select another FFmpeg build using --ffmpeg.'.format(args.ffmpeg, requested))
    errors = []
    with tempfile.TemporaryDirectory(prefix='cats-resize-preflight-') as directory:
        video, png = Path(directory) / 'check.mp4', Path(directory) / 'check.png'
        for candidate in candidates:
            args.encoder = candidate
            try:
                run([
                    args.ffmpeg, '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo',
                    '-pixel_format', 'rgb24', '-video_size', '{}x{}'.format(args.width, args.height),
                    '-framerate', '24', '-i', 'pipe:0', '-an',
                ] + codec_options(args) + [
                    '-pix_fmt', 'yuv420p', '-vf', 'setsar=1', '-filter_threads', str(args.threads),
                    '-threads', str(args.threads), '-frames:v', '1', str(video),
                ], data=bytes(args.width * args.height * 3))
                info = probe(video, args.threads, args.ffprobe)
                if (int(info['width']), int(info['height']), info['frames']) != (args.width, args.height, 1):
                    raise ValueError('Encoded preflight MP4 has unexpected dimensions or frame count')
                extract_first_frame(video, png, args)
                print('FFmpeg: {}\nEncoder: {}'.format(args.ffmpeg, candidate), flush=True)
                if requested == 'auto' and candidate == 'mpeg4':
                    print('Using native MPEG-4 Part 2 because libx264 is unavailable or failed its check '
                          '(quality={}; --crf/--preset apply only to libx264).'.format(args.mpeg4_quality), flush=True)
                return
            except Exception as error:
                errors.append('{}: {}'.format(candidate, error))
    raise ValueError('Encoder/PNG preflight failed before processing any source videos:\n' + '\n'.join(errors))


def encode(source, target, args):
    info = probe(source, args.threads, args.ffprobe)
    count = info['frames']
    fps = Fraction(args.frames, 1) / duration(info)
    # Exact round-half-up indices, including both endpoints; duplicates are kept.
    indices = [0] if args.frames == 1 else [
        (2 * i * (count - 1) + args.frames - 1) // (2 * (args.frames - 1))
        for i in range(args.frames)
    ]
    decoder = [
        args.ffmpeg, '-v', 'error', '-nostdin', '-xerror', '-threads', str(args.threads),
        '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
        '-vf', 'scale={}:{}:flags=lanczos,setsar=1'.format(args.width, args.height),
        '-filter_threads', str(args.threads), '-vsync', '0', '-pix_fmt', 'rgb24',
        '-threads', str(args.threads), '-f', 'rawvideo', 'pipe:1',
    ]
    encoder = [
        args.ffmpeg, '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '{}x{}'.format(args.width, args.height),
        '-framerate', str(fps), '-i', 'pipe:0', '-an',
    ] + codec_options(args) + [
        '-pix_fmt', 'yuv420p',
        '-vf', 'setsar=1', '-filter_threads', str(args.threads),
        '-threads', str(args.threads), '-vsync', '0', '-frames:v', str(args.frames),
        '-movflags', '+faststart', str(target),
    ]
    with tempfile.TemporaryFile() as dec_errors, tempfile.TemporaryFile() as enc_errors:
        dec = enc = None
        try:
            dec = subprocess.Popen(decoder, stdout=subprocess.PIPE, stderr=dec_errors)
            enc = subprocess.Popen(encoder, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=enc_errors)
            frame_bytes = args.width * args.height * 3
            decoded = selected = 0
            while True:
                if STOP.is_set():
                    raise RuntimeError('Interrupted')
                frame = dec.stdout.read(frame_bytes)
                if not frame:
                    break
                if len(frame) != frame_bytes:
                    raise ValueError('Truncated decoded frame')
                while selected < args.frames and indices[selected] == decoded:
                    enc.stdin.write(frame)
                    selected += 1
                decoded += 1
            enc.stdin.close()
            dec_status, enc_status = dec.wait(), enc.wait()
            if dec_status or enc_status:
                raise RuntimeError('FFmpeg decode/encode failed')
            if decoded != count or selected != args.frames:
                raise ValueError('Frame count changed while decoding: {} vs {}; selected {}'.format(decoded, count, selected))
        except Exception as error:
            for process in (dec, enc):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
            messages = []
            for log in (dec_errors, enc_errors):
                log.seek(0)
                messages.append(log.read().decode(errors='replace')[-3000:].strip())
            raise RuntimeError('{} {}'.format(error, ' '.join(messages)).strip()) from error
        finally:
            for pipe in (getattr(dec, 'stdout', None), getattr(enc, 'stdin', None)):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass


def extract_first_frame(video, png, args):
    run([args.ffmpeg, '-v', 'error', '-nostdin', '-y', '-threads', str(args.threads),
         '-i', str(video), '-map', '0:v:0', '-frames:v', '1', '-update', '1',
         '-threads', str(args.threads), str(png)])
    with png.open('rb') as handle:
        header = handle.read(24)
    if (header[:8] != b'\x89PNG\r\n\x1a\n' or
            (int.from_bytes(header[16:20], 'big'), int.from_bytes(header[20:24], 'big')) != (args.width, args.height)):
        raise ValueError('First-frame PNG validation failed')


def process_one(source, args):
    if STOP.is_set():
        raise RuntimeError('Interrupted')
    relative = source.relative_to(args.input_root)
    target = args.output_root / relative
    image = (args.first_frame_root / relative).with_suffix('.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    image.parent.mkdir(parents=True, exist_ok=True)
    video_tmp = image_tmp = None
    try:
        skipped = target.exists() and not args.overwrite
        if skipped:
            try:
                validate(target, args)
            except Exception as error:
                raise ValueError('Existing MP4 is invalid; use --overwrite to rebuild: {}'.format(error)) from error
        else:
            video_tmp = temporary_path(target.parent, '.mp4')
            encode(source, video_tmp, args)
            validate(video_tmp, args)
        image_tmp = temporary_path(image.parent, '.png')
        extract_first_frame(target if skipped else video_tmp, image_tmp, args)
        if video_tmp is not None:
            os.replace(str(video_tmp), str(target))
        os.replace(str(image_tmp), str(image))
        return 'skipped_verified' if skipped else 'processed'
    finally:
        for path in (video_tmp, image_tmp):
            if path is not None and path.exists():
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-root', type=Path, default=DATA_ROOT / 'GT_videos')
    parser.add_argument('--output-root', type=Path, default=DATA_ROOT / 'GT_videos_256x448@97')
    parser.add_argument('--first-frame-root', type=Path, default=DATA_ROOT / 'first_frames_256x448')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--frames', type=int, default=97)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1, help='FFmpeg threads per worker')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='FFmpeg executable name or absolute path')
    parser.add_argument('--ffprobe', default='ffprobe', help='FFprobe executable name or absolute path')
    parser.add_argument('--encoder', default='auto', choices=['auto', 'libx264', 'mpeg4'],
                        help='auto prefers libx264, then native mpeg4; checks encoding before processing')
    parser.add_argument('--mpeg4-quality', type=int, default=2, help='Native mpeg4 quality (1-31, smaller is higher quality)')
    parser.add_argument('--crf', type=int, default=18, help='libx264 only: H.264 CRF (0-51)')
    parser.add_argument('--preset', default='medium', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'])
    parser.add_argument('--overwrite', action='store_true', help='Rebuild existing output MP4s')
    parser.add_argument('--dry-run', action='store_true', help='List configuration without writing anything')
    parser.add_argument('--limit', type=int, help='Process only the first N input files')
    args = parser.parse_args()
    if min(args.width, args.height, args.frames, args.workers, args.threads) < 1 or args.width % 2 or args.height % 2:
        parser.error('Dimensions must be positive even numbers; frames/workers/threads must be positive')
    if not 0 <= args.crf <= 51 or not 1 <= args.mpeg4_quality <= 31 or (args.limit is not None and args.limit < 1):
        parser.error('CRF must be 0-51, mpeg4-quality must be 1-31, and --limit must be positive')
    roots = [args.input_root.resolve(), args.output_root.resolve(), args.first_frame_root.resolve()]
    args.input_root, args.output_root, args.first_frame_root = roots
    for i, left in enumerate(roots):
        for right in roots[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                parser.error('Input, output and first-frame roots must be separate, non-nested directories')
    if not args.input_root.is_dir():
        parser.error('Input directory does not exist: {}'.format(args.input_root))
    sources = sorted(p for p in args.input_root.rglob('*') if p.is_file() and p.suffix.lower() == '.mp4')
    if args.limit:
        sources = sources[:args.limit]
    print('Input: {}\nVideo output: {}\nFirst frames: {}\nFiles: {}; target: {}x{}, {} frames'.format(
        *roots, len(sources), args.width, args.height, args.frames), flush=True)
    if not sources:
        parser.error('No MP4 files found')
    if args.dry_run:
        for source in sources[:10]:
            print(source.relative_to(args.input_root))
        return 0
    for executable in ('ffmpeg', 'ffprobe'):
        selected = shutil.which(getattr(args, executable))
        if not selected:
            parser.error('Missing {} executable: {}'.format(executable, getattr(args, executable)))
        setattr(args, executable, selected)
    try:
        preflight(args)
    except Exception as error:
        parser.error(str(error))
    counts = {'processed': 0, 'skipped_verified': 0, 'failed': 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        try:
            for source in sources:
                futures[executor.submit(process_one, source, args)] = source
            for finished, future in enumerate(as_completed(futures), 1):
                relative = futures[future].relative_to(args.input_root)
                try:
                    status = future.result()
                    detail = ''
                except Exception as error:
                    status, detail = 'failed', ': {}'.format(error)
                counts[status] += 1
                print('[{}/{}] {} {}{}'.format(finished, len(sources), status, relative, detail), flush=True)
        except KeyboardInterrupt:
            STOP.set()
            for future in futures:
                future.cancel()
            print('\nInterrupted; cancelling queued work and cleaning up active files.', file=sys.stderr, flush=True)
            return 130
    print(json.dumps(counts), flush=True)
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
