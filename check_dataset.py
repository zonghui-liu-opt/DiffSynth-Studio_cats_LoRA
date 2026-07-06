import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio

from metrics_utils import tokens_per_sample


REQUIRED_COLUMNS = ("video", "prompt", "input_image")
FIXED_COLUMNS = REQUIRED_COLUMNS + ("height", "width", "bucket")


def detect_delimiter(metadata_path):
    sample = Path(metadata_path).read_text(encoding="utf-8")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return "\t" if "\t" in first_line else ","


def read_metadata(metadata_path, delimiter):
    with Path(metadata_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"metadata missing required columns: {', '.join(missing)}")
    return rows


def is_missing(value):
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def parse_resolution_value(value, name, row_id):
    if is_missing(value):
        raise ValueError(f"missing_{name}")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{name}:{value}") from exc
    if parsed <= 0 or parsed % 32 != 0:
        raise ValueError(f"{name}_not_positive_multiple_of_32:{parsed}")
    return parsed


def resolution_bucket(height, width):
    return f"{height}x{width}"


def resolve_target_resolution(row, video_stats, fallback_height, fallback_width, row_id):
    has_metadata_resolution = not is_missing(row.get("height")) and not is_missing(row.get("width"))
    if has_metadata_resolution:
        height = parse_resolution_value(row.get("height"), "height", row_id)
        width = parse_resolution_value(row.get("width"), "width", row_id)
        return height, width
    if video_stats is not None:
        height = parse_resolution_value(video_stats["height"], "height", row_id)
        width = parse_resolution_value(video_stats["width"], "width", row_id)
        return height, width
    if fallback_height is not None and fallback_width is not None:
        height = parse_resolution_value(fallback_height, "height", row_id)
        width = parse_resolution_value(fallback_width, "width", row_id)
        return height, width
    raise ValueError("missing_resolution")


def write_fixed_metadata(metadata_path, rows):
    fixed_path = Path(metadata_path).with_name("metadata_fixed.csv")
    with fixed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIXED_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in FIXED_COLUMNS})
    return fixed_path


def inspect_video_imageio(video_path):
    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 0.0) or 0.0)
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = sum(1 for _ in reader)
            reader.close()
            reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
        return {"width": width, "height": height, "frames": frame_count, "fps": fps}
    finally:
        reader.close()


def inspect_video_cv2(video_path):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("cv2 is not available") from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cv2 failed to open {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return {"width": width, "height": height, "frames": frames, "fps": fps}
    finally:
        capture.release()


def inspect_video_ffprobe(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den or 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream.get("nb_frames") or 0),
        "fps": fps,
    }


def inspect_video(video_path):
    errors = []
    for inspector in (inspect_video_imageio, inspect_video_cv2, inspect_video_ffprobe):
        try:
            return inspector(video_path)
        except Exception as exc:
            errors.append(f"{inspector.__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def validate_dataset(dataset_root, metadata_path, height=None, width=None, num_frames=None):
    if num_frames is None:
        raise ValueError("num_frames is required")
    dataset_root = Path(dataset_root)
    metadata_path = Path(metadata_path)
    delimiter = detect_delimiter(metadata_path)
    rows = read_metadata(metadata_path, delimiter)

    bad_rows = []
    stats = []
    target_stats = []
    fixed_rows = []
    for row_id, row in enumerate(rows):
        reasons = []
        fixed_row = {column: row[column] for column in REQUIRED_COLUMNS}
        video_path = dataset_root / row["video"]
        image_path = dataset_root / row["input_image"]
        video_stats = None
        if not video_path.exists():
            reasons.append("missing_video")
        else:
            try:
                video_stats = inspect_video(video_path)
                stats.append(video_stats)
                if video_stats["frames"] < num_frames:
                    reasons.append("insufficient_frames")
            except Exception as exc:
                reasons.append(f"video_read_error:{exc}")
        if not image_path.exists():
            reasons.append("missing_input_image")
        try:
            target_height, target_width = resolve_target_resolution(row, video_stats, height, width, row_id)
            bucket = resolution_bucket(target_height, target_width)
            fixed_row.update({"height": target_height, "width": target_width, "bucket": bucket})
            target_stats.append({"row": row_id, "height": target_height, "width": target_width, "bucket": bucket})
        except ValueError as exc:
            reasons.append(f"resolution_error:{exc}")
        fixed_rows.append(fixed_row)
        if reasons:
            bad_rows.append({"row": row_id, "video": row["video"], "reasons": reasons})

    fixed_path = write_fixed_metadata(metadata_path, fixed_rows)
    resolution_counts = Counter((item["height"], item["width"]) for item in target_stats)
    source_resolution_counts = Counter((item["height"], item["width"]) for item in stats)
    frame_counts = Counter(item["frames"] for item in stats)
    fps_counts = Counter(round(item["fps"], 3) for item in stats)
    bucket_counts = Counter(item["bucket"] for item in target_stats)
    bad_row_ids = {row["row"] for row in bad_rows}
    tokens_per_bucket = {}
    tokens_total = 0
    for item in target_stats:
        if item["row"] in bad_row_ids:
            continue
        token_count = tokens_per_sample(num_frames=num_frames, height=item["height"], width=item["width"])
        tokens_per_bucket[item["bucket"]] = token_count
        tokens_total += token_count
    unique_token_counts = sorted(set(tokens_per_bucket.values()))
    token_count = unique_token_counts[0] if len(unique_token_counts) == 1 else None
    return {
        "delimiter": "tab" if delimiter == "\t" else "comma",
        "fixed_path": str(fixed_path),
        "total_rows": len(rows),
        "good_samples": len(rows) - len(bad_rows),
        "bad_samples": len(bad_rows),
        "bad_rows": bad_rows,
        "resolution_counts": dict(resolution_counts),
        "source_resolution_counts": dict(source_resolution_counts),
        "bucket_counts": dict(bucket_counts),
        "frame_counts": dict(frame_counts),
        "fps_counts": dict(fps_counts),
        "recommended_height": None,
        "recommended_width": None,
        "tokens_per_video": token_count,
        "tokens_per_bucket": tokens_per_bucket,
        "tokens_total": tokens_total,
    }


def print_summary(summary):
    reason_counts = Counter(reason for row in summary["bad_rows"] for reason in row["reasons"])
    print(f"delimiter: {summary['delimiter']}")
    print(f"metadata_fixed_path: {summary['fixed_path']}")
    print(f"total_rows: {summary['total_rows']}")
    print(f"good_samples: {summary['good_samples']}")
    print(f"bad_samples: {summary['bad_samples']}")
    for reason, count in sorted(reason_counts.items()):
        print(f"{reason}: {count}")
    print(f"resolution_counts: {summary['resolution_counts']}")
    print(f"source_resolution_counts: {summary['source_resolution_counts']}")
    print(f"bucket_counts: {summary['bucket_counts']}")
    print(f"frame_counts: {summary['frame_counts']}")
    print(f"fps_counts: {summary['fps_counts']}")
    print("recommended_height: metadata")
    print("recommended_width: metadata")
    print(f"tokens_per_video: {summary['tokens_per_video'] if summary['tokens_per_video'] is not None else 'mixed'}")
    print(f"tokens_per_bucket: {summary['tokens_per_bucket']}")
    print(f"tokens_total: {summary['tokens_total']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Check Wan TI2V dataset metadata and media files.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--height", type=int, default=None, help="Deprecated fallback height. Prefer per-row metadata height.")
    parser.add_argument("--width", type=int, default=None, help="Deprecated fallback width. Prefer per-row metadata width.")
    parser.add_argument("--num_frames", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = validate_dataset(args.dataset_root, args.metadata_path, args.height, args.width, args.num_frames)
    print_summary(summary)


if __name__ == "__main__":
    main()
