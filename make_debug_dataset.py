import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def validate_args(args):
    if args.height % 32 != 0:
        raise ValueError("--height must be a multiple of 32")
    if args.width % 32 != 0:
        raise ValueError("--width must be a multiple of 32")
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num_frames must satisfy (F - 1) % 4 == 0")
    if args.num_videos < 1:
        raise ValueError("--num_videos must be positive")


def make_frames(index, height, width, num_frames):
    frames = []
    base = np.zeros((height, width, 3), dtype=np.uint8)
    y_grid = np.linspace(0, 80, height, dtype=np.uint8)[:, None]
    x_grid = np.linspace(0, 80, width, dtype=np.uint8)[None, :]
    base[..., 0] = (x_grid + index * 17) % 255
    base[..., 1] = (y_grid + index * 31) % 255
    base[..., 2] = (index * 47) % 255
    block_h = max(16, height // 4)
    block_w = max(16, width // 4)
    for frame_id in range(num_frames):
        frame = base.copy()
        x = (frame_id * 7 + index * 11) % max(1, width - block_w)
        y = (frame_id * 5 + index * 13) % max(1, height - block_h)
        color = np.array(
            [
                (60 + index * 29 + frame_id * 5) % 255,
                (180 + index * 19 + frame_id * 3) % 255,
                (100 + index * 23 + frame_id * 9) % 255,
            ],
            dtype=np.uint8,
        )
        frame[y:y + block_h, x:x + block_w] = color
        frames.append(frame)
    return frames


def write_video(path, frames, fps=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def write_first_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path, quality=95)


def create_dataset(output_dir, num_videos=8, height=96, width=160, num_frames=13, with_bad_samples=False):
    output_dir = Path(output_dir)
    video_dir = output_dir / f"videos_{height}x{width}"
    image_dir = output_dir / f"images_{height}x{width}"
    rows = []
    for index in range(num_videos):
        stem = f"{index:04d}_debug"
        frames = make_frames(index, height, width, num_frames)
        video_rel = f"{video_dir.name}/{stem}.mp4"
        image_rel = f"{image_dir.name}/{stem}.jpg"
        write_video(output_dir / video_rel, frames)
        write_first_frame(output_dir / image_rel, frames[0])
        rows.append(
            {
                "video": video_rel,
                "prompt": f"debug moving color block sample {index}",
                "input_image": image_rel,
            }
        )

    if with_bad_samples:
        short_frames = make_frames(10_000, height, width, max(1, num_frames - 8))
        short_video_rel = f"{video_dir.name}/bad_short.mp4"
        short_image_rel = f"{image_dir.name}/bad_short.jpg"
        write_video(output_dir / short_video_rel, short_frames)
        write_first_frame(output_dir / short_image_rel, short_frames[0])
        rows.append(
            {
                "video": short_video_rel,
                "prompt": "bad sample with insufficient frames",
                "input_image": short_image_rel,
            }
        )

        missing_frames = make_frames(20_000, height, width, num_frames)
        missing_video_rel = f"{video_dir.name}/bad_missing_image.mp4"
        write_video(output_dir / missing_video_rel, missing_frames)
        rows.append(
            {
                "video": missing_video_rel,
                "prompt": "bad sample with missing input image",
                "input_image": f"{image_dir.name}/does_not_exist.jpg",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "prompt", "input_image"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a synthetic Wan TI2V debug dataset.")
    parser.add_argument("--num_videos", type=int, default=8)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--output_dir", type=Path, default=Path("debug_data"))
    parser.add_argument("--with_bad_samples", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)
    metadata_path = create_dataset(
        args.output_dir,
        num_videos=args.num_videos,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        with_bad_samples=args.with_bad_samples,
    )
    print(f"metadata_path: {metadata_path}")


if __name__ == "__main__":
    main()
