#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence


EXPECTED_LORA_PARAMS = 10_936_320
CLIP_LENGTH = 33
TOTAL_LOOP_FRAMES = 72
SUPPORTED_RESOLUTIONS = (512, 640, 672, 704)
PROMPT = "a realistic indoor room tour"
FORBIDDEN_INPUT_FIELDS = {
    "vace_reference_image",
    "native_support",
    "native_hole",
    "backside",
    "uncertain",
    "normal",
    "normal_confidence",
    "depth",
    "clean_depth",
    "clean_normal",
    "clean_semantic",
    "clean_instance",
}
REQUIRED_FIELDS = {
    "sample_id",
    "split",
    "start_index",
    "frame_indices",
    "video",
    "vace_video",
    "prompt",
    "num_frames",
    "condition_frame_stems",
    "clean_frame_stems",
}


def validate_frame_count(num_frames: int) -> int:
    num_frames = int(num_frames)
    if num_frames < 1 or num_frames % 4 != 1:
        raise ValueError(f"Wan temporal length must be positive 4n+1, got {num_frames}")
    return num_frames


def validate_spatial_resolution(width: int, height: int) -> tuple[int, int]:
    width, height = int(width), int(height)
    if width != height or width not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"P3.9 supported resolutions are square {SUPPORTED_RESOLUTIONS}, got {width}x{height}"
        )
    return width, height


def cyclic_indices(start: int, length: int = CLIP_LENGTH, total: int = TOTAL_LOOP_FRAMES) -> list[int]:
    if total < 1:
        raise ValueError("total frame count must be positive")
    if start < 0 or start >= total:
        raise ValueError(f"start must be in [0, {total}), got {start}")
    validate_frame_count(length)
    return [(int(start) + offset) % int(total) for offset in range(int(length))]


def build_record(
    *,
    sample_id: str,
    start_index: int,
    frame_indices: Sequence[int],
    frame_stems: Sequence[str],
    clean_video: str | Path,
    condition_video: str | Path,
    split: str,
    width: int = 512,
    height: int = 512,
    fps: int = 12,
) -> dict:
    if len(frame_indices) != CLIP_LENGTH or len(frame_stems) != CLIP_LENGTH:
        raise ValueError("P3.8 records require exactly 33 frames")
    width, height = validate_spatial_resolution(width, height)
    return {
        "sample_id": str(sample_id),
        "split": str(split),
        "start_index": int(start_index),
        "frame_indices": [int(index) for index in frame_indices],
        "condition_frame_stems": [str(stem) for stem in frame_stems],
        "clean_frame_stems": [str(stem) for stem in frame_stems],
        "video": str(Path(clean_video)),
        "vace_video": str(Path(condition_video)),
        "prompt": PROMPT,
        "num_frames": CLIP_LENGTH,
        "width": int(width),
        "height": int(height),
        "training_assumed_fps": int(fps),
        "fps_status": "synthetic_operational_not_ground_truth",
    }


def validate_records(records: Iterable[Mapping]) -> int:
    rows = list(records)
    if not rows:
        raise ValueError("manifest is empty")
    for row_id, row in enumerate(rows):
        forbidden = FORBIDDEN_INPUT_FIELDS.intersection(row)
        if forbidden:
            raise ValueError(f"row {row_id} contains forbidden fields: {sorted(forbidden)}")
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            raise ValueError(f"row {row_id} missing fields: {sorted(missing)}")
        if int(row["num_frames"]) != CLIP_LENGTH:
            raise ValueError(f"row {row_id} must contain exactly 33 frames")
        validate_spatial_resolution(int(row["width"]), int(row["height"]))
        indices = [int(index) for index in row["frame_indices"]]
        expected = cyclic_indices(int(row["start_index"]), CLIP_LENGTH, TOTAL_LOOP_FRAMES)
        if indices != expected:
            raise ValueError(f"row {row_id} violates cyclic 72-frame ordering")
        if list(row["condition_frame_stems"]) != list(row["clean_frame_stems"]):
            raise ValueError(f"row {row_id} condition/clean stems differ")
        if len(row["condition_frame_stems"]) != CLIP_LENGTH:
            raise ValueError(f"row {row_id} stem count is not 33")
        clean = Path(row["video"])
        condition = Path(row["vace_video"])
        if clean.expanduser().resolve() == condition.expanduser().resolve():
            raise ValueError(f"row {row_id} reuses one path for condition and target")
        for path in (clean, condition):
            if not path.is_file():
                raise FileNotFoundError(path)
    return len(rows)


def training_contract() -> dict:
    return {
        "branch": "l33",
        "data_file_keys": ("video", "vace_video"),
        "extra_inputs": "vace_video",
        "trainable_params": EXPECTED_LORA_PARAMS,
        "reference_adapter_params": 0,
    }
