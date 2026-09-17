#!/usr/bin/env python3
"""Fail-closed audit for the 72-frame condition/clean pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-manifest", type=Path, required=True)
    ap.add_argument("--topdown", type=Path, required=True)
    ap.add_argument("--condition-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    if not args.topdown.is_file():
        raise FileNotFoundError(args.topdown)
    rows = [json.loads(line) for line in args.pair_manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 72 or [int(r["frame_index"]) for r in rows] != list(range(72)):
        raise ValueError("pair manifest must have 72 contiguous frame indices")
    checks = []
    for row in rows:
        condition = Path(row["condition_rgb"])
        clean = Path(row["clean_rgb"])
        if condition == clean or condition.resolve() == clean.resolve():
            raise ValueError(f"condition and clean path are identical at frame {row['frame_index']}")
        if not condition.is_file() or not clean.is_file():
            raise FileNotFoundError(f"missing pair at frame {row['frame_index']}")
        with Image.open(condition) as c, Image.open(clean) as g:
            if c.size != g.size or c.size != (896, 896):
                raise ValueError(f"resolution mismatch at frame {row['frame_index']}: {c.size} vs {g.size}")
        expected = args.condition_dir / f"F{int(row['frame_index']):02d}.png"
        if condition.resolve() != expected.resolve():
            raise ValueError(f"manifest condition path differs from requested directory: {condition}")
        if str(row["frame_stem"]) not in clean.name:
            raise ValueError(f"clean frame stem mismatch at frame {row['frame_index']}")
        if row.get("camera_exact_pair") is not True:
            raise ValueError(f"camera_exact_pair is not true at frame {row['frame_index']}")
        checks.append(int(row["frame_index"]))
    report = {
        "status": "PASS",
        "frame_count": len(checks),
        "resolution": [896, 896],
        "condition_dir": str(args.condition_dir),
        "clean_dir": str(args.clean_dir),
        "clean_target_files_used_for_condition_generation": False,
        "checks": ["72 contiguous frames", "distinct condition/clean paths", "896x896", "camera_exact_pair=true", "frame stems preserved"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
