import argparse
import hashlib
import itertools
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LoRA-runtime and merged-model batch MP4 outputs."
    )
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
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

    videos = [compare_video(lora_files[name], merged_files[name]) for name in lora_files]
    report = {
        "lora_dir": str(lora_dir),
        "merged_dir": str(merged_dir),
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
