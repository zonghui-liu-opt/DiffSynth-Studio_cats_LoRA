import subprocess
import sys


def test_unified_dataset_loads_video_prompt_and_input_image(tmp_path):
    from diffsynth.core import UnifiedDataset

    dataset_root = tmp_path / "debug_data"
    subprocess.run(
        [
            sys.executable,
            "make_debug_dataset.py",
            "--num_videos",
            "3",
            "--height",
            "96",
            "--width",
            "160",
            "--num_frames",
            "13",
            "--output_dir",
            str(dataset_root),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "check_dataset.py",
            "--dataset_root",
            str(dataset_root),
            "--metadata_path",
            str(dataset_root / "metadata.csv"),
            "--height",
            "96",
            "--width",
            "160",
            "--num_frames",
            "13",
        ],
        check=True,
    )

    dataset = UnifiedDataset(
        base_path=str(dataset_root),
        metadata_path=str(dataset_root / "metadata_fixed.csv"),
        data_file_keys="video,input_image".split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=str(dataset_root),
            height=96,
            width=160,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=13,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )

    for idx in range(2):
        sample = dataset[idx]
        assert isinstance(sample["prompt"], str)
        assert len(sample["video"]) == 13
        assert sample["video"][0].size == (160, 96)
        assert len(sample["input_image"]) == 1
        assert sample["input_image"][0].size == (160, 96)
