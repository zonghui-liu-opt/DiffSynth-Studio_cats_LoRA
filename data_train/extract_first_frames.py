#!/usr/bin/env python3
"""Extract the first decoded frame of each resized MP4 into a matching PNG.

Requires Python 3.8+ and ffmpeg on PATH; no pip packages or ffprobe needed.
Preserves action subdirectories and filenames, changing only .mp4 to .png.
Images retain the video's 256x448 size. Existing valid PNGs are skipped;
invalid PNGs are rebuilt. Use --overwrite to regenerate every first frame.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def run(command):
    """Wait in short intervals so Ctrl+C also stops active FFmpeg processes."""
    if STOP.is_set():
        raise RuntimeError('Interrupted')
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        while True:
            if STOP.is_set():
                raise RuntimeError('Interrupted')
            try:
                _, errors = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            raise RuntimeError(errors.decode(errors='replace')[-4000:].strip())
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def check_png_size(path):
    with path.open('rb') as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('Invalid or empty PNG: {}'.format(path))
    size = (int.from_bytes(header[16:20], 'big'), int.from_bytes(header[20:24], 'big'))
    if size != (256, 448):
        raise ValueError('Expected 256x448, got {}x{}: {}'.format(*size, path))


def existing_png_is_valid(path, threads):
    try:
        check_png_size(path)
        run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-threads', str(threads),
             '-i', str(path), '-map', '0:v:0', '-frames:v', '1',
             '-threads', str(threads), '-f', 'null', '-'])
        return True
    except (OSError, ValueError, RuntimeError):
        if STOP.is_set():
            raise
        return False


def extract_one(video, args):
    if STOP.is_set():
        raise RuntimeError('Interrupted')
    output = args.output_root / video.relative_to(args.input_root).with_suffix('.png')
    if output.exists() and not args.overwrite and existing_png_is_valid(output, args.threads):
        return 'skipped_verified'
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix='.first-frame-', suffix='.png', dir=str(output.parent))
    os.close(fd)
    temporary = Path(filename)
    try:
        # No seeking or scaling: this is the first frame of the final training MP4.
        run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-y',
             '-threads', str(args.threads), '-i', str(video), '-map', '0:v:0',
             '-an', '-sn', '-dn', '-frames:v', '1', '-pix_fmt', 'rgb24',
             '-threads', str(args.threads), '-update', '1', str(temporary)])
        check_png_size(temporary)
        if STOP.is_set():
            raise RuntimeError('Interrupted')
        os.replace(str(temporary), str(output))
        return 'extracted'
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-root', type=Path, default=DATA_ROOT / 'GT_videos_256x448@97')
    parser.add_argument('--output-root', type=Path, default=DATA_ROOT / 'first_frames_256x448')
    parser.add_argument('--workers', type=int, default=4, help='Concurrent videos (default: 4)')
    parser.add_argument('--threads', type=int, default=1, help='FFmpeg threads per worker (default: 1)')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate existing PNGs')
    parser.add_argument('--dry-run', action='store_true', help='Preview paths without writing files')
    parser.add_argument('--limit', type=int, help='Process only the first N videos')
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers, threads and limit must be positive')
    args.input_root, args.output_root = args.input_root.resolve(), args.output_root.resolve()
    if (args.input_root == args.output_root or args.input_root in args.output_root.parents or
            args.output_root in args.input_root.parents):
        parser.error('Input and output roots must be separate, non-nested directories')
    if not args.input_root.is_dir():
        parser.error('Input directory does not exist: {}'.format(args.input_root))
    videos = sorted(p for p in args.input_root.rglob('*') if p.is_file() and p.suffix.lower() == '.mp4')
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        parser.error('No MP4 files found')
    outputs = set()
    for video in videos:
        output = video.relative_to(args.input_root).with_suffix('.png')
        if output in outputs:
            parser.error('Multiple MP4s map to the same PNG: {}'.format(output))
        outputs.add(output)
    print('Input: {}\nOutput: {}\nVideos: {}'.format(args.input_root, args.output_root, len(videos)), flush=True)
    if args.dry_run:
        for video in videos[:10]:
            relative = video.relative_to(args.input_root)
            print('{} -> {}'.format(relative, args.output_root / relative.with_suffix('.png')))
        return 0
    if not shutil.which('ffmpeg'):
        parser.error('ffmpeg is missing from PATH')
    counts = {'extracted': 0, 'skipped_verified': 0, 'failed': 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        try:
            for video in videos:
                futures[executor.submit(extract_one, video, args)] = video
            for finished, future in enumerate(as_completed(futures), 1):
                detail = ''
                try:
                    status = future.result()
                except Exception as error:
                    status, detail = 'failed', ': {}'.format(error)
                counts[status] += 1
                relative = futures[future].relative_to(args.input_root)
                print('[{}/{}] {} {}{}'.format(finished, len(videos), status, relative, detail), flush=True)
        except KeyboardInterrupt:
            STOP.set()
            for future in futures:
                future.cancel()
            print('\nInterrupted; cancelling work and cleaning temporary PNGs.', file=sys.stderr, flush=True)
            return 130
    print(json.dumps(counts), flush=True)
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
