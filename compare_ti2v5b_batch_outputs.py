import argparse
import hashlib
import itertools
import json
from pathlib import Path

import imageio.v2 as iio_v2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HEADER_HEIGHT = 52
BACKGROUND_COLOR = (15, 23, 42)
LORA_FLAG_COLOR = (37, 99, 235)
MERGED_FLAG_COLOR = (124, 58, 237)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LoRA-runtime and merged-model batch MP4 outputs."
    )
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        help=(
            "Directory for annotated side-by-side videos. Defaults to "
            "<merged-dir parent>/comparison_videos."
        ),
    )
    parser.add_argument("--lora-flag", default="RUNTIME LORA")
    parser.add_argument("--merged-flag", default="MERGED MODEL")
    parser.add_argument(
        "--skip-comparison-videos",
        action="store_true",
        help="Only calculate metrics; do not create annotated comparison MP4 files.",
    )
    parser.add_argument(
        "--allow-different",
        action="store_true",
        help="Return exit code 0 even when decoded frames differ.",
    )
    return parser.parse_args()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compare_decoded_frames(lora_path, merged_path):
    frame_count = 0
    differing_frames = 0
    differing_values = 0
    total_values = 0
    absolute_error_sum = 0
    max_absolute_error = 0
    sentinel = object()

    lora_frames = iio.imiter(lora_path, plugin="FFMPEG")
    merged_frames = iio.imiter(merged_path, plugin="FFMPEG")
    for frame_id, (lora_frame, merged_frame) in enumerate(
        itertools.zip_longest(lora_frames, merged_frames, fillvalue=sentinel)
    ):
        if lora_frame is sentinel or merged_frame is sentinel:
            return {
                "decoded_equal": False,
                "error": f"frame count differs starting at frame {frame_id}",
                "frame_count": frame_count,
            }
        if lora_frame.shape != merged_frame.shape:
            return {
                "decoded_equal": False,
                "error": (
                    f"frame {frame_id} shape differs: "
                    f"{lora_frame.shape} vs {merged_frame.shape}"
                ),
                "frame_count": frame_count,
            }

        difference = np.abs(lora_frame.astype(np.int16) - merged_frame.astype(np.int16))
        frame_count += 1
        total_values += difference.size
        differing_values += int(np.count_nonzero(difference))
        absolute_error_sum += int(difference.sum(dtype=np.int64))
        frame_max = int(difference.max(initial=0))
        max_absolute_error = max(max_absolute_error, frame_max)
        if frame_max:
            differing_frames += 1

    return {
        "decoded_equal": differing_values == 0,
        "frame_count": frame_count,
        "differing_frames": differing_frames,
        "differing_values": differing_values,
        "mean_absolute_error": absolute_error_sum / total_values if total_values else 0.0,
        "max_absolute_error": max_absolute_error,
    }


def compare_video(lora_path, merged_path):
    lora_sha256 = sha256_file(lora_path)
    merged_sha256 = sha256_file(merged_path)
    result = {
        "filename": lora_path.name,
        "lora_sha256": lora_sha256,
        "merged_sha256": merged_sha256,
        "file_bytes_equal": lora_sha256 == merged_sha256,
    }
    if result["file_bytes_equal"]:
        result["decoded_equal"] = True
        result["decoded_comparison_skipped"] = True
    else:
        result.update(compare_decoded_frames(lora_path, merged_path))
    return result


def video_fps(path, default=24.0):
    try:
        metadata = iio.immeta(path, plugin="FFMPEG")
        return float(metadata.get("fps", default))
    except (OSError, ValueError, TypeError):
        return default


def normalize_rgb_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
        raise ValueError(f"Unsupported decoded frame shape: {frame.shape}")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return frame.astype(np.uint8, copy=False)


def letterbox_frame(frame, panel_height, panel_width):
    frame = normalize_rgb_frame(frame)
    height, width = frame.shape[:2]
    scale = min(panel_width / width, panel_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    image = Image.fromarray(frame)
    if (resized_width, resized_height) != (width, height):
        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    top = (panel_height - resized_height) // 2
    left = (panel_width - resized_width) // 2
    panel[top : top + resized_height, left : left + resized_width] = np.asarray(image)
    return panel


def annotation_font(size=22):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_centered_badge(draw, center_x, y, text, color, font, padding_x=10, padding_y=4):
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    left = round(center_x - text_width / 2 - padding_x)
    right = round(center_x + text_width / 2 + padding_x)
    bottom = y + text_height + padding_y * 2
    draw.rounded_rectangle((left, y, right, bottom), radius=6, fill=color)
    draw.text(
        (center_x - text_width / 2, y + padding_y - box[1]),
        text,
        fill="white",
        font=font,
    )


def compose_comparison_frame(
    lora_frame,
    merged_frame,
    lora_flag,
    merged_flag,
    panel_height,
    panel_width,
):
    lora_missing = lora_frame is None
    merged_missing = merged_frame is None
    if lora_missing:
        lora_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    else:
        lora_panel = letterbox_frame(lora_frame, panel_height, panel_width)
    if merged_missing:
        merged_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    else:
        merged_panel = letterbox_frame(merged_frame, panel_height, panel_width)

    canvas = np.empty(
        (panel_height + HEADER_HEIGHT, panel_width * 2, 3), dtype=np.uint8
    )
    canvas[:HEADER_HEIGHT] = BACKGROUND_COLOR
    canvas[HEADER_HEIGHT:, :panel_width] = lora_panel
    canvas[HEADER_HEIGHT:, panel_width:] = merged_panel
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    flag_font = annotation_font(22)
    draw_centered_badge(
        draw, panel_width / 2, 10, lora_flag, LORA_FLAG_COLOR, flag_font
    )
    draw_centered_badge(
        draw,
        panel_width * 1.5,
        10,
        merged_flag,
        MERGED_FLAG_COLOR,
        flag_font,
    )
    draw.line(
        (panel_width, HEADER_HEIGHT, panel_width, panel_height + HEADER_HEIGHT),
        fill="white",
        width=2,
    )
    return np.asarray(image)


def generate_comparison_video(
    lora_path,
    merged_path,
    output_path,
    lora_flag,
    merged_flag,
):
    sentinel = object()
    lora_frames = iter(iio.imiter(lora_path, plugin="FFMPEG"))
    merged_frames = iter(iio.imiter(merged_path, plugin="FFMPEG"))
    first_lora = next(lora_frames, sentinel)
    first_merged = next(merged_frames, sentinel)
    if first_lora is sentinel and first_merged is sentinel:
        raise ValueError(f"Both videos have no decoded frames: {lora_path.name}")

    available_first_frames = [
        normalize_rgb_frame(frame)
        for frame in (first_lora, first_merged)
        if frame is not sentinel
    ]
    panel_height = max(frame.shape[0] for frame in available_first_frames)
    panel_width = max(frame.shape[1] for frame in available_first_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = video_fps(lora_path)

    frame_pairs = itertools.chain(
        [(first_lora, first_merged)],
        itertools.zip_longest(lora_frames, merged_frames, fillvalue=sentinel),
    )
    with iio_v2.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2,
    ) as writer:
        for lora_frame, merged_frame in frame_pairs:
            annotated_frame = compose_comparison_frame(
                None if lora_frame is sentinel else lora_frame,
                None if merged_frame is sentinel else merged_frame,
                lora_flag,
                merged_flag,
                panel_height,
                panel_width,
            )
            writer.append_data(annotated_frame)
    return output_path


def main():
    args = parse_args()
    lora_dir = args.lora_dir.expanduser().resolve()
    merged_dir = args.merged_dir.expanduser().resolve()
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"Missing LoRA output directory: {lora_dir}")
    if not merged_dir.is_dir():
        raise FileNotFoundError(f"Missing merged output directory: {merged_dir}")

    lora_files = {path.name: path for path in sorted(lora_dir.glob("*.mp4"))}
    merged_files = {path.name: path for path in sorted(merged_dir.glob("*.mp4"))}
    if not lora_files:
        raise ValueError(f"No MP4 files found in {lora_dir}")
    if lora_files.keys() != merged_files.keys():
        missing_in_merged = sorted(lora_files.keys() - merged_files.keys())
        missing_in_lora = sorted(merged_files.keys() - lora_files.keys())
        raise ValueError(
            f"Output filenames differ: missing_in_merged={missing_in_merged}, "
            f"missing_in_lora={missing_in_lora}"
        )

    comparison_dir = None
    if not args.skip_comparison_videos:
        comparison_dir = (
            args.comparison_dir.expanduser().resolve()
            if args.comparison_dir
            else merged_dir.parent / "comparison_videos"
        )
        if comparison_dir in {lora_dir, merged_dir}:
            raise ValueError("comparison_dir must differ from lora_dir and merged_dir")
        comparison_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    for name in lora_files:
        result = compare_video(lora_files[name], merged_files[name])
        if comparison_dir is not None:
            output_path = comparison_dir / f"{Path(name).stem}_comparison.mp4"
            generate_comparison_video(
                lora_files[name],
                merged_files[name],
                output_path,
                args.lora_flag,
                args.merged_flag,
            )
            result["comparison_video"] = str(output_path)
        videos.append(result)

    report = {
        "lora_dir": str(lora_dir),
        "merged_dir": str(merged_dir),
        "comparison_dir": str(comparison_dir) if comparison_dir else None,
        "lora_flag": args.lora_flag,
        "merged_flag": args.merged_flag,
        "all_file_bytes_equal": all(video["file_bytes_equal"] for video in videos),
        "all_decoded_frames_equal": all(video["decoded_equal"] for video in videos),
        "videos": videos,
    }
    report_text = json.dumps(report, indent=2, ensure_ascii=False)
    print(report_text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text + "\n", encoding="utf-8")
    if not report["all_decoded_frames_equal"] and not args.allow_different:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
