def test_read_metadata_accepts_tab_separated_prompt_and_input_image(tmp_path):
    from infer_cats_ti2v5b_lora_batch import read_metadata

    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "input_image\tprompt\nimages/cat_a.jpg\tcat A prompt\nimages/cat_b.jpg\tcat B prompt\n",
        encoding="utf-8",
    )

    rows = read_metadata(metadata_path)

    assert rows == [
        {"input_image": "images/cat_a.jpg", "prompt": "cat A prompt"},
        {"input_image": "images/cat_b.jpg", "prompt": "cat B prompt"},
    ]


def test_read_metadata_rejects_missing_required_columns(tmp_path):
    from infer_cats_ti2v5b_lora_batch import read_metadata

    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("image,prompt\nimages/cat.jpg,cat prompt\n", encoding="utf-8")

    try:
        read_metadata(metadata_path)
    except ValueError as exc:
        assert "input_image" in str(exc)
    else:
        raise AssertionError("Expected missing input_image column to raise ValueError")


def test_row_paths_and_size_follow_training_metadata_layout(tmp_path):
    from infer_cats_ti2v5b_lora_batch import input_image_path, output_video_path, row_size

    row = {"input_image": "images_480x832/cat.jpg", "height": "832", "width": "480"}

    assert input_image_path(tmp_path, row) == tmp_path / "images_480x832" / "cat.jpg"
    assert output_video_path(tmp_path / "pred_videos", 3, row) == tmp_path / "pred_videos" / "0003_cat.mp4"
    assert row_size(row) == (832, 480)
