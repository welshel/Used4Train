#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from p38_contract import CLIP_LENGTH, PROMPT, build_record, cyclic_indices, validate_frame_count


VALIDATION_STARTS = (0, 17, 31, 50)
FORBIDDEN_S0_KEYS = {
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
}
ALLOWED_S0_KEYS = {
    "sample_id",
    "split",
    "start_index",
    "frame_indices",
    "condition_frame_stems",
    "clean_frame_stems",
    "video",
    "vace_video",
    "prompt",
    "num_frames",
    "width",
    "height",
    "training_assumed_fps",
    "fps_status",
}


def load_pair_manifest(path: Path | str) -> list[dict]:
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    required = {"frame_index", "frame_stem", "condition_rgb", "clean_rgb"}
    for row_id, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"row {row_id} missing required keys: {sorted(missing)}")
        row["frame_index"] = int(row["frame_index"])
    rows.sort(key=lambda row: row["frame_index"])
    indices = [row["frame_index"] for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("pair manifest requires contiguous unique frame_index values starting at 0")
    for row in rows:
        for key in ("condition_rgb", "clean_rgb"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"missing {key}: {row[key]}")
    return rows


def deterministic_resize_pair(
    condition: Image.Image,
    clean: Image.Image,
    size: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    if condition.size != clean.size:
        raise ValueError(f"pair source sizes differ: {condition.size} vs {clean.size}")
    return (
        condition.convert("RGB").resize(size, Image.Resampling.LANCZOS),
        clean.convert("RGB").resize(size, Image.Resampling.LANCZOS),
    )


def split_starts(total: int, validation_starts: Sequence[int] = VALIDATION_STARTS) -> tuple[list[int], list[int]]:
    validation = [int(start) for start in validation_starts]
    if len(set(validation)) != len(validation) or any(start < 0 or start >= total for start in validation):
        raise ValueError("validation starts must be unique and within the closed loop")
    validation_set = set(validation)
    training = [start for start in range(total) if start not in validation_set]
    return training, validation


def build_clip_records(
    pair_rows: Sequence[dict],
    starts: Iterable[int],
    clip_root: Path | str,
    *,
    split: str = "train",
    clip_length: int = CLIP_LENGTH,
    prompt: str = PROMPT,
    size: tuple[int, int] = (512, 512),
    assumed_fps: int = 12,
) -> list[dict]:
    validate_frame_count(clip_length)
    clip_root = Path(clip_root)
    records = []
    for start in starts:
        indices = cyclic_indices(int(start), clip_length, len(pair_rows))
        stems = [str(pair_rows[index]["frame_stem"]) for index in indices]
        sample_id = f"L33_start{int(start):02d}_{clip_length}f"
        record = build_record(
            sample_id=sample_id,
            start_index=int(start),
            frame_indices=indices,
            frame_stems=stems,
            clean_video=clip_root / "clean" / f"{sample_id}.mp4",
            condition_video=clip_root / "condition" / f"{sample_id}.mp4",
            split=split,
            width=int(size[0]),
            height=int(size[1]),
            fps=int(assumed_fps),
        )
        record["prompt"] = prompt
        unexpected = set(record).difference(ALLOWED_S0_KEYS)
        if unexpected or FORBIDDEN_S0_KEYS.intersection(record):
            raise AssertionError(f"S0 manifest contains forbidden fields: {sorted(unexpected)}")
        if record["video"] == record["vace_video"]:
            raise AssertionError("condition and clean target paths must be different")
        records.append(record)
    return records


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def materialize_clip(pair_rows: Sequence[dict], record: dict) -> None:
    condition_frames: list[np.ndarray] = []
    clean_frames: list[np.ndarray] = []
    size = (int(record["width"]), int(record["height"]))
    for index, expected_stem in zip(record["frame_indices"], record["condition_frame_stems"]):
        row = pair_rows[index]
        if str(row["frame_stem"]) != expected_stem:
            raise AssertionError("frame stem changed after record construction")
        with Image.open(row["condition_rgb"]) as condition, Image.open(row["clean_rgb"]) as clean:
            condition_out, clean_out = deterministic_resize_pair(condition, clean, size)
            condition_frames.append(np.asarray(condition_out))
            clean_frames.append(np.asarray(clean_out))
    _write_video(Path(record["vace_video"]), condition_frames, record["training_assumed_fps"])
    _write_video(Path(record["video"]), clean_frames, record["training_assumed_fps"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact cyclic P3.8 high-resolution condition/clean clips")
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=CLIP_LENGTH)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--assumed-fps", type=int, default=12)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_rows = load_pair_manifest(args.pair_manifest)
    training_starts, validation_starts = split_starts(len(pair_rows))
    size = (args.width, args.height)
    train_root = args.output_dir / "train_clips"
    val_root = args.output_dir / "validation_clips"
    train_records = build_clip_records(
        pair_rows,
        training_starts,
        train_root,
        split="train",
        clip_length=args.clip_length,
        prompt=args.prompt,
        size=size,
        assumed_fps=args.assumed_fps,
    )
    validation_records = build_clip_records(
        pair_rows,
        validation_starts,
        val_root,
        split="validation",
        clip_length=args.clip_length,
        prompt=args.prompt,
        size=size,
        assumed_fps=args.assumed_fps,
    )
    write_jsonl(args.output_dir / "training_manifest.jsonl", train_records)
    write_jsonl(args.output_dir / "validation_manifest.jsonl", validation_records)
    if not args.manifest_only:
        for record in train_records + validation_records:
            materialize_clip(pair_rows, record)
    summary = {
        "pair_frames": len(pair_rows),
        "clip_length": args.clip_length,
        "resolution": [args.width, args.height],
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "validation_starts": validation_starts,
        "training_assumed_fps": args.assumed_fps,
        "clean_dataset_fps": "UNRESOLVED",
        "s0_condition_fields": ["vace_video"],
        "target_fields": ["video"],
        "auxiliary_condition_fields": [],
        "resize_filter": "PIL.Image.Resampling.LANCZOS",
        "source_policy": "resize_each_clip_frame_directly_from_pair_manifest_master_rgb",
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
