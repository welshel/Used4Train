#!/usr/bin/env python3
"""Project a full condition/clean pair manifest to P39's strict condition-only schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.pair_manifest.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if len(rows) != 72 or [int(r["frame_index"]) for r in rows] != list(range(72)):
        raise ValueError("pair manifest must contain 72 contiguous frames")
    result = []
    for i, row in enumerate(rows):
        if not Path(row["condition_rgb"]).is_file():
            raise FileNotFoundError(row["condition_rgb"])
        result.append({"frame_index": i, "frame_stem": str(row["frame_stem"]), "condition_rgb": row["condition_rgb"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in result) + "\n")
    print(json.dumps({"output": str(args.output), "frame_count": 72}, indent=2))


if __name__ == "__main__":
    main()
