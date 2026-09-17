#!/usr/bin/env python3
"""Read-only fail-closed audit for the organized P39 pipeline."""
from __future__ import annotations

import argparse
import json
import py_compile
from pathlib import Path

from PIL import Image


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.manifest.read_text())
    required = [
        "topdown", "topdown_camera", "clean_trajectory", "pair_manifest",
        "infinisplat_checkpoint", "condition_rgb_dir", "dataset_builder", "trainer",
    ]
    for key in required:
        path = Path(cfg[key])
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")
    with Image.open(cfg["topdown"]) as topdown:
        if topdown.size != (1000, 1000):
            raise ValueError(f"unexpected topdown resolution: {topdown.size}")
    pair = rows(Path(cfg["pair_manifest"]))
    if len(pair) != 72 or [int(r["frame_index"]) for r in pair] != list(range(72)):
        raise ValueError("pair manifest is not exactly 72 contiguous rows")
    for row in pair:
        if Path(row["condition_rgb"]).resolve() == Path(row["clean_rgb"]).resolve():
            raise ValueError("condition and clean RGB are the same file")
        if row.get("camera_exact_pair") is not True:
            raise ValueError("camera_exact_pair is not true")
        if not Path(row["condition_rgb"]).is_file() or not Path(row["clean_rgb"]).is_file():
            raise FileNotFoundError(row)
        if Path(row["condition_rgb"]).name != f"F{int(row['frame_index']):02d}.png":
            raise ValueError(f"condition frame naming mismatch at {row['frame_index']}")
        if Path(row["clean_rgb"]).stem != str(row["frame_stem"]):
            raise ValueError(f"clean frame stem mismatch at {row['frame_index']}")
        if not Path(row["clean_camera"]).is_file():
            raise FileNotFoundError(row["clean_camera"])
        with Image.open(row["condition_rgb"]) as c, Image.open(row["clean_rgb"]) as g:
            if c.size != g.size or c.size != (896, 896):
                raise ValueError(f"resolution mismatch at {row['frame_index']}")
    dataset_checks = {}
    for name, root in cfg["datasets"].items():
        root = Path(root)
        train = rows(root / "training_manifest.jsonl")
        val = rows(root / "validation_manifest.jsonl")
        if len(train) != 68 or len(val) != 4:
            raise ValueError(f"{name} expected 68/4 records, got {len(train)}/{len(val)}")
        dataset_checks[name] = {"training": len(train), "validation": len(val)}
    code_files = [Path(x) for x in cfg["code_files"]]
    for path in code_files:
        py_compile.compile(str(path), doraise=True)
    report = {
        "status": "PASS",
        "pair_frames": len(pair),
        "dataset_checks": dataset_checks,
        "compiled_files": len(code_files),
        "clean_rgb_used_for_condition_generation": False,
        "notes": [
            "condition generation consumes topdown RGB plus camera payload only",
            "clean RGB is a target/evaluation input to pair construction and training",
            "no camera fitting, optical flow, or homography is performed",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
