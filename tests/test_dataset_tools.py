import subprocess
import sys

import pandas as pd


def test_debug_dataset_and_checker_report_bad_samples(tmp_path):
    dataset_root = tmp_path / "debug_data"

    subprocess.run(
        [
            sys.executable,
            "make_debug_dataset.py",
            "--num_videos",
            "4",
            "--height",
            "96",
            "--width",
            "160",
            "--num_frames",
            "13",
            "--output_dir",
            str(dataset_root),
            "--with_bad_samples",
        ],
        check=True,
    )

    metadata_path = dataset_root / "metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(256)
    assert sample.splitlines()[0].split("\t") == ["video", "prompt", "input_image"]

    rows = pd.read_csv(metadata_path, sep="\t")
    assert list(rows.columns) == ["video", "prompt", "input_image"]
    assert (dataset_root / "videos_96x160").is_dir()
    assert (dataset_root / "images_96x160").is_dir()

    result = subprocess.run(
        [
            sys.executable,
            "check_dataset.py",
            "--dataset_root",
            str(dataset_root),
            "--metadata_path",
            str(metadata_path),
            "--height",
            "96",
            "--width",
            "160",
            "--num_frames",
            "13",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "bad_samples: 2" in result.stdout
    assert "insufficient_frames" in result.stdout
    assert "missing_input_image" in result.stdout
    assert "tokens_per_video: 60" in result.stdout

    fixed_path = dataset_root / "metadata_fixed.csv"
    fixed_rows = pd.read_csv(fixed_path)
    assert list(fixed_rows.columns) == ["video", "prompt", "input_image"]
    assert len(fixed_rows) == len(rows)
