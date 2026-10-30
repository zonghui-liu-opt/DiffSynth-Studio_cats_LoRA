#!/usr/bin/env python3
"""递归 resize 所有动作子目录中的 MP4，保留目录结构和文件名。

输出：256x448、24 FPS、97 帧、无音轨 MP4。保持所提供单目录版本的
直接缩放方式，不裁剪。编码为高质量有损输出，各编码器的质量参数不等价。
依赖 Python 3.8+、FFmpeg 和 FFprobe，无需 pip 包。
"""

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DATA_ROOT = Path('/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats_new')
SRC_DIR = DATA_ROOT / 'GT_videos'
DST_DIR = DATA_ROOT / 'GT_videos_256x448@97'
TARGET_W, TARGET_H, FPS, EXPECTED_FRAMES = 256, 448, 24, 97
PRIORITY_ENCODERS = ('libx264', 'libopenh264', 'h264_nvenc', 'mpeg4')


def run(command, data=None):
    result = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        # FFmpeg stderr is bytes here; do not call .decode() on text=True output.
        raise RuntimeError(result.stderr.decode(errors='replace')[-3000:].strip())
    return result.stdout


def build_ffmpeg_cmd(encoder, src, dst, args):
    cmd = ['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y', '-threads', str(args.threads)]
    if src is None:
        # Independent test frame: encoder detection must not depend on a real clip.
        cmd += ['-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
                '{}x{}'.format(TARGET_W, TARGET_H), '-framerate', str(FPS), '-i', 'pipe:0']
    else:
        cmd += ['-i', str(src)]
    cmd += ['-map', '0:v:0', '-vf', 'scale={}:{}:flags=lanczos,setsar=1'.format(TARGET_W, TARGET_H),
            '-pix_fmt', 'yuv420p', '-r', str(FPS), '-vsync', 'cfr', '-an', '-sn', '-dn',
            '-filter_threads', str(args.threads), '-threads', str(args.threads)]
    if encoder == 'libx264':
        cmd += ['-c:v', encoder, '-crf', str(args.crf), '-preset', args.preset, '-tune', 'film']
    elif encoder == 'libopenh264':
        # OpenH264 uses bitrate controls, not x264's CRF/preset/tune options.
        cmd += ['-c:v', encoder, '-b:v', args.openh264_bitrate, '-allow_skip_frames', '0']
    elif encoder == 'h264_nvenc':
        cmd += ['-c:v', encoder, '-rc', 'vbr', '-cq', str(args.crf), '-b:v', '0', '-preset', 'p5', '-tune', 'hq']
    else:
        cmd += ['-c:v', 'mpeg4', '-qscale:v', '2']
    if src is None:
        cmd += ['-frames:v', '1']
    return cmd + ['-movflags', '+faststart', str(dst)]


def probe(video, threads):
    result = json.loads(run([
        'ffprobe', '-v', 'error', '-threads', str(threads), '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate,sample_aspect_ratio',
        '-of', 'json', str(video),
    ]))
    if not result.get('streams'):
        raise ValueError('没有可解码的视频流: {}'.format(video))
    return result['streams'][0]


def validate(video, threads, expected_frames=EXPECTED_FRAMES):
    info = probe(video, threads)
    actual = (int(info['width']), int(info['height']), int(info['nb_read_frames']))
    expected = (TARGET_W, TARGET_H, expected_frames)
    if (actual != expected or Fraction(info['avg_frame_rate']) != FPS or
            info.get('sample_aspect_ratio') != '1:1'):
        raise ValueError('输出校验失败: 尺寸/帧数={}, 期望={}; FPS={}, 期望={}'.format(
            actual, expected, info['avg_frame_rate'], FPS))


def get_available_encoder(args):
    listing = run(['ffmpeg', '-hide_banner', '-encoders']).decode(errors='replace')
    available = set()
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[0]) == 6 and fields[0].startswith('V'):
            available.add(fields[1])
    candidates = PRIORITY_ENCODERS if args.encoder == 'auto' else (args.encoder,)
    failures = []
    with tempfile.TemporaryDirectory(prefix='cats-encoder-check-') as directory:
        test_video = Path(directory) / 'check.mp4'
        for encoder in candidates:
            if encoder not in available:
                continue
            try:
                run(build_ffmpeg_cmd(encoder, None, test_video, args), data=bytes(TARGET_W * TARGET_H * 3))
                validate(test_video, args.threads, expected_frames=1)
                print('[INFO] 编码器实际试编码通过: {}'.format(encoder), flush=True)
                return encoder
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                failures.append('{}: {}'.format(encoder, error))
                print('[WARN] {} 无法使用，尝试下一编码器。'.format(encoder), flush=True)
    raise RuntimeError('没有可用的视频编码器。\n' + '\n'.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', '--input-root', dest='src', type=Path, default=SRC_DIR)
    parser.add_argument('--dst', '--output-root', dest='dst', type=Path, default=DST_DIR)
    parser.add_argument('--encoder', choices=('auto',) + PRIORITY_ENCODERS, default='auto')
    parser.add_argument('--crf', type=int, default=10, help='libx264 CRF / NVENC CQ，默认 10')
    parser.add_argument('--preset', default='slow', help='libx264 preset，默认 slow')
    parser.add_argument('--openh264-bitrate', default='8M', help='OpenH264 目标码率，默认 8M')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true', help='重新生成已存在的输出')
    parser.add_argument('--dry-run', action='store_true', help='仅预览路径，不写文件')
    parser.add_argument('--limit', type=int, help='仅处理前 N 个视频，用于试跑')
    args = parser.parse_args()
    if args.threads < 1 or not 0 <= args.crf <= 51 or (args.limit is not None and args.limit < 1):
        parser.error('threads/limit 必须为正整数，crf 必须在 0-51 范围内')
    args.src, args.dst = args.src.resolve(), args.dst.resolve()
    if args.src == args.dst or args.src in args.dst.parents or args.dst in args.src.parents:
        parser.error('输入和输出目录必须独立，不能相同或互相嵌套')
    if not args.src.is_dir():
        parser.error('源目录不存在: {}'.format(args.src))
    # Recursively include every action directory and .mp4/.MP4 files.
    videos = sorted(p for p in args.src.rglob('*') if p.is_file() and p.suffix.lower() == '.mp4')
    if args.limit:
        videos = videos[:args.limit]
    print('[INFO] 源目录: {}\n[INFO] 输出目录: {}\n[INFO] 共 {} 个 MP4'.format(args.src, args.dst, len(videos)), flush=True)
    if not videos:
        parser.error('所有子目录中均未找到 MP4 文件')
    if args.dry_run:
        for video in videos[:10]:
            print('{} -> {}'.format(video, args.dst / video.relative_to(args.src)))
        return 0
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            parser.error('PATH 中找不到 {}'.format(executable))
    try:
        encoder = get_available_encoder(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    counts = {'processed': 0, 'skipped_verified': 0, 'failed': 0}
    failed = []
    for index, src in enumerate(videos, 1):
        relative = src.relative_to(args.src)
        dst = args.dst / relative
        temporary = None
        print('[{}/{}] {}'.format(index, len(videos), relative), end=' ... ', flush=True)
        try:
            if dst.exists() and not args.overwrite:
                # Reject an invalid existing output instead of silently replacing it.
                try:
                    validate(dst, args.threads)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    raise ValueError('已有输出不合格，添加 --overwrite 重建: {}'.format(error)) from error
                counts['skipped_verified'] += 1
                print('SKIP (已校验 97 帧)', flush=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, filename = tempfile.mkstemp(prefix='.resize-', suffix='.mp4', dir=str(dst.parent))
            os.close(fd)
            temporary = Path(filename)
            run(build_ffmpeg_cmd(encoder, src, temporary, args))
            validate(temporary, args.threads)
            os.replace(str(temporary), str(dst))
            counts['processed'] += 1
            print('OK (256x448, 24 FPS, 97 frames)', flush=True)
        except KeyboardInterrupt:
            print('\n[INFO] 已中断，清理当前临时文件。', flush=True)
            return 130
        except (OSError, ValueError, RuntimeError, KeyError, ZeroDivisionError) as error:
            counts['failed'] += 1
            failed.append(str(relative))
            print('FAILED\n  {}'.format(error), flush=True)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    print(json.dumps(counts), flush=True)
    if failed:
        print('[ERROR] 失败文件: {}'.format(json.dumps(failed, ensure_ascii=False)), flush=True)
        return 2
    print('[INFO] 全部完成: {}'.format(args.dst), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
