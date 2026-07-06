import csv

from PIL import Image

from check_dataset import validate_dataset
from diffsynth.core import UnifiedDataset
from diffsynth.core.data import bucket_sampler


def write_image(path, size=(40, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(127, 127, 127)).save(path)


def write_metadata(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video", "prompt", "input_image", "height", "width"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_check_dataset_uses_metadata_resolution_buckets(tmp_path, monkeypatch):
    rows = [
        {"video": "videos/landscape.mp4", "prompt": "landscape", "input_image": "images/landscape.jpg", "height": 96, "width": 160},
        {"video": "videos/portrait.mp4", "prompt": "portrait", "input_image": "images/portrait.jpg", "height": 160, "width": 96},
        {"video": "videos/square.mp4", "prompt": "square", "input_image": "images/square.jpg", "height": 128, "width": 128},
    ]
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path, rows)
    for row in rows:
        (tmp_path / row["video"]).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / row["video"]).touch()
        write_image(tmp_path / row["input_image"])

    monkeypatch.setattr(
        "check_dataset.inspect_video",
        lambda path: {"width": 999, "height": 777, "frames": 9, "fps": 24.0},
    )

    summary = validate_dataset(tmp_path, metadata_path, num_frames=9)

    assert summary["resolution_counts"] == {(96, 160): 1, (160, 96): 1, (128, 128): 1}
    assert summary["bucket_counts"] == {"96x160": 1, "160x96": 1, "128x128": 1}
    with (tmp_path / "metadata_fixed.csv").open(newline="", encoding="utf-8") as handle:
        fixed_rows = list(csv.DictReader(handle))
    assert [(row["height"], row["width"], row["bucket"]) for row in fixed_rows] == [
        ("96", "160", "96x160"),
        ("160", "96", "160x96"),
        ("128", "128", "128x128"),
    ]


def test_unified_dataset_resizes_each_row_to_metadata_resolution(tmp_path):
    rows = [
        {"video": "media/landscape.jpg", "prompt": "landscape", "input_image": "media/landscape.jpg", "height": 96, "width": 160},
        {"video": "media/square.jpg", "prompt": "square", "input_image": "media/square.jpg", "height": 128, "width": 128},
    ]
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path, rows)
    for row in rows:
        write_image(tmp_path / row["video"], size=(40, 60))

    dataset = UnifiedDataset(
        base_path=str(tmp_path),
        metadata_path=str(metadata_path),
        data_file_keys=("video", "input_image"),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=str(tmp_path),
            num_frames=9,
            resize_mode="metadata",
        ),
    )

    assert dataset[0]["video"][0].size == (160, 96)
    assert dataset[0]["input_image"][0].size == (160, 96)
    assert dataset[1]["video"][0].size == (128, 128)
    assert dataset[1]["input_image"][0].size == (128, 128)


def test_resolution_bucket_sampler_falls_back_to_height_width(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(
        metadata_path,
        [
            {"video": "a.jpg", "prompt": "a", "input_image": "a.jpg", "height": 96, "width": 160},
            {"video": "b.jpg", "prompt": "b", "input_image": "b.jpg", "height": 128, "width": 128},
        ],
    )
    dataset = UnifiedDataset(base_path=str(tmp_path), metadata_path=str(metadata_path))

    sampler_cls = getattr(bucket_sampler, "ResolutionBucketSampler", None)
    assert sampler_cls is not None
    sampler = sampler_cls(dataset, shuffle=False)

    assert list(sampler.bucket_to_indices) == ["96x160", "128x128"]
