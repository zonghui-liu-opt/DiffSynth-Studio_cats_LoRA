import numpy as np

from compare_ti2v5b_batch_outputs import HEADER_HEIGHT, compose_comparison_frame


def test_compose_comparison_frame_adds_header_and_two_panels():
    lora_frame = np.zeros((32, 48, 3), dtype=np.uint8)
    merged_frame = lora_frame.copy()

    frame = compose_comparison_frame(
        lora_frame,
        merged_frame,
        lora_flag="RUNTIME LORA",
        merged_flag="MERGED MODEL",
        panel_height=32,
        panel_width=48,
    )

    assert frame.shape == (32 + HEADER_HEIGHT, 96, 3)
    assert frame.dtype == np.uint8


def test_compose_comparison_frame_supports_different_frames():
    lora_frame = np.zeros((32, 48, 3), dtype=np.uint8)
    merged_frame = np.full((32, 48, 3), 255, dtype=np.uint8)

    frame = compose_comparison_frame(
        lora_frame,
        merged_frame,
        lora_flag="LORA",
        merged_flag="MERGED",
        panel_height=32,
        panel_width=48,
    )

    assert frame.shape == (32 + HEADER_HEIGHT, 96, 3)
