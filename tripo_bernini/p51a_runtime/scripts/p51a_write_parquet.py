#!/usr/bin/env python3
"""Serialize the audited P51A JSONL rows as official Bernini-R parquet input."""

import argparse
import json
from pathlib import Path

import pandas as pd


def validate_row(row: dict) -> None:
    required = {
        "scene_id", "sample_id", "start_index", "clip_length", "resolution", "fps",
        "input_frame_indices", "target_frame_indices", "inputs", "videos",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"{row.get('sample_id', '<unknown>')}: missing {missing}")
    if row["clip_length"] != 33:
        raise ValueError(f"{row['sample_id']}: clip_length must be 33")
    if len(row["input_frame_indices"]) != 33 or row["input_frame_indices"] != row["target_frame_indices"]:
        raise ValueError(f"{row['sample_id']}: input/target cyclic pairing is invalid")
    messages = json.loads(row["inputs"])
    signatures = [(message["type"], int(message.get("has_loss", message["type"].endswith("_gen")))) for message in messages]
    if signatures != [("text", 0), ("video", 0), ("video_gen", 1)]:
        raise ValueError(f"{row['sample_id']}: invalid source/target message routing: {signatures}")
    if len(row["videos"]) != 2:
        raise ValueError(f"{row['sample_id']}: expected adjusted source and clean target videos")
    for video in row["videos"]:
        if not Path(video["video_path"]).is_file():
            raise FileNotFoundError(video["video_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    if len(rows) != 72:
        raise ValueError(f"expected 72 cyclic starts, got {len(rows)}")
    if sorted(row["start_index"] for row in rows) != list(range(72)):
        raise ValueError("starts must be exactly 0..71")
    for row in rows:
        validate_row(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing parquet: {args.output}")
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
