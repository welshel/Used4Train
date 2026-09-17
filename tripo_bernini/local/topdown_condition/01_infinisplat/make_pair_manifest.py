#!/usr/bin/env python3
"""Create the explicit condition/RGB pair manifest from the fixed 72 frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean_rgb_paths(clean_dir: Path) -> list[Path]:
    excluded = ("_depth", "_normal", "_semantic", "_instance", "_mask")
    # RGB files are the bare millisecond timestamp; *_bbox, *_lines, masks,
    # depth and semantic products are auxiliary QA assets.
    paths = [p for p in clean_dir.glob("*.png") if p.stem.isdigit() and not p.stem.endswith(excluded)]
    return sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--camera-payload", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--condition-only-output", type=Path)
    args = ap.parse_args()
    condition = sorted(args.condition_dir.glob("F*.png"), key=lambda p: int(p.stem[1:]))
    clean = clean_rgb_paths(args.clean_dir)
    payload = json.loads(args.camera_payload.read_text())
    poses = payload.get("poses", [])
    if len(condition) != 72 or len(clean) != 72 or len(poses) != 72:
        raise ValueError(f"expected 72 condition, clean, and camera frames; got {len(condition)}, {len(clean)}, {len(poses)}")
    rows = []
    for i, (condition_path, clean_path, pose) in enumerate(zip(condition, clean, poses)):
        if int(pose["frame_index"]) != i:
            raise ValueError("camera payload ordering is not 0..71")
        if str(pose["frame_stem"]) != clean_path.stem:
            raise ValueError(f"frame stem mismatch at {i}: {pose['frame_stem']} vs {clean_path.stem}")
        camera_path = Path(pose["clean_camera_file"])
        if not camera_path.is_file():
            raise FileNotFoundError(camera_path)
        rows.append({
            "frame_index": i,
            "frame_stem": clean_path.stem,
            "condition_rgb": str(condition_path),
            "clean_rgb": str(clean_path),
            "clean_camera": str(camera_path),
            "camera_exact_pair": bool(pose.get("camera_exact_pair", False)),
        })
    if not all(row["camera_exact_pair"] for row in rows):
        raise ValueError("camera payload does not certify exact pairing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    if args.condition_only_output:
        condition_rows = [
            {"frame_index": r["frame_index"], "frame_stem": r["frame_stem"], "condition_rgb": r["condition_rgb"]}
            for r in rows
        ]
        args.condition_only_output.parent.mkdir(parents=True, exist_ok=True)
        args.condition_only_output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in condition_rows) + "\n"
        )
    print(json.dumps({"output": str(args.output), "frame_count": 72}, indent=2))


if __name__ == "__main__":
    main()
