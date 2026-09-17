#!/usr/bin/env python3
"""Build the P32 source-canonical camera payload from clean camera metadata.

The clean cameras define the trajectory, but clean RGB/depth/normal/semantic
files are not read.  Only camera JSON and the explicit one-degree-of-freedom
scale from the audited P32 scale contract are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def matrix_from_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    c2w = np.asarray(payload["c2w"], dtype=np.float64)
    intrinsic = np.asarray(payload["intrinsic"], dtype=np.float64)
    if c2w.shape != (4, 4) or intrinsic.shape not in ((3, 3), (4, 4)):
        raise ValueError(f"invalid camera shapes in {path}")
    return c2w, intrinsic[:3, :3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-manifest", type=Path, required=True)
    ap.add_argument("--source-camera", type=Path, required=True)
    ap.add_argument("--scale-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.pair_manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 72 or [int(r["frame_index"]) for r in rows] != list(range(72)):
        raise ValueError("pair manifest must contain frame_index 0..71 exactly")
    source_c2w, _ = matrix_from_camera(args.source_camera)
    scale_payload = json.loads(args.scale_audit.read_text())
    lam = float(scale_payload["final_lambda"])
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f"invalid final_lambda: {lam}")
    source_inv = np.linalg.inv(source_c2w)
    poses = []
    for row in rows:
        camera_path = Path(row["clean_camera"])
        clean_c2w, K = matrix_from_camera(camera_path)
        relative = source_inv @ clean_c2w
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = relative[:3, :3]
        c2w[:3, 3] = lam * relative[:3, 3]
        width = int(round(2 * K[0, 2]))
        height = int(round(2 * K[1, 2]))
        if (width, height) != (896, 896):
            raise ValueError(f"unexpected clean camera resolution for {camera_path}: {width}x{height}")
        poses.append(
            {
                "frame_index": int(row["frame_index"]),
                "frame_stem": str(row["frame_stem"]),
                "clean_camera_file": str(camera_path),
                "c2w_infinisplat": c2w.tolist(),
                "K": K.tolist(),
                "width": width,
                "height": height,
                "camera_exact_pair": bool(row.get("camera_exact_pair", False)),
            }
        )
    if not all(p["camera_exact_pair"] for p in poses):
        raise ValueError("pair manifest contains a non-exact camera pair")
    result = {
        "schema": "P32_CLEAN_POSE_DIRECT_SOURCE_CANONICAL_V1",
        "source_camera": str(args.source_camera),
        "source_camera_sha256": sha256(args.source_camera),
        "source_c2w_world": source_c2w.tolist(),
        "lambda": lam,
        "scale_audit": str(args.scale_audit),
        "clean_camera_convention": "OpenCV C2W, ssl_z_up world",
        "renderer_camera_convention": "OpenCV C2W in InfiniSplat source-canonical world",
        "frame_count": 72,
        "poses": poses,
        "clean_target_files_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "frames": 72, "lambda": lam}, indent=2))


if __name__ == "__main__":
    main()
