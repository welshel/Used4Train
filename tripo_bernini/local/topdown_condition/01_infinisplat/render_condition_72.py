#!/usr/bin/env python3
"""Render the official InfiniSplat Gaussian artifact on the fixed 72-frame path."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infinisplat-repo", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--camera-payload", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    if not args.artifact.is_file() or not args.camera_payload.is_file():
        raise FileNotFoundError("artifact and camera payload are required")
    sys.path.insert(0, str(args.infinisplat_repo))
    import torch  # type: ignore
    from gsplat import rasterization  # type: ignore
    from src.demo.hf_runtime import load_gaussian_artifact  # type: ignore
    from src.demo.infer_single_image import filter_final_gaussian_floaters  # type: ignore

    payload = json.loads(args.camera_payload.read_text())
    poses = payload.get("poses", [])
    if len(poses) != 72 or [int(x["frame_index"]) for x in poses] != list(range(72)):
        raise ValueError("camera payload must contain exactly frames 0..71")
    artifact = load_gaussian_artifact(args.artifact)
    gaussians = filter_final_gaussian_floaters(artifact.gaussians).to(args.device)
    cameras = np.asarray([p["c2w_infinisplat"] for p in poses], dtype=np.float32)
    K = np.asarray(poses[0]["K"], dtype=np.float32)
    H, W = int(poses[0]["height"]), int(poses[0]["width"])
    if any(int(p["width"]) != W or int(p["height"]) != H for p in poses):
        raise ValueError("camera payload resolution changes over time")
    output_dir = args.output_dir
    frame_dir = output_dir / "rgb"
    frame_dir.mkdir(parents=True, exist_ok=True)
    extrinsics = torch.linalg.inv(torch.tensor(cameras, device=args.device)).unsqueeze(0)
    intrinsics = torch.tensor(K, dtype=torch.float32, device=args.device).unsqueeze(0).unsqueeze(0).repeat(1, 72, 1, 1)
    with torch.inference_mode():
        rgb, alpha, _ = rasterization(
            gaussians.mean_vectors.float(),
            gaussians.quaternions.float(),
            gaussians.singular_values.float(),
            gaussians.opacities.float(),
            gaussians.colors.float(),
            extrinsics,
            intrinsics,
            W,
            H,
            sh_degree=None,
            render_mode="RGB",
            packed=True,
            covars=gaussians.covariances.float(),
            eps2d=1e-8,
        )
    frames = (rgb[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    alpha_np = alpha[0, ..., 0].detach().cpu().numpy()
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"F{i:02d}.png")
    metadata = {
        "status": "OK",
        "frame_count": 72,
        "resolution": [W, H],
        "fps": args.fps,
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "camera_payload": str(args.camera_payload),
        "camera_payload_sha256": sha256(args.camera_payload),
        "clean_target_files_read": False,
        "coverage_mean": float((alpha_np > 1e-3).mean()),
        "coverage_median": float(np.median((alpha_np > 1e-3).mean(axis=(1, 2)))),
    }
    (output_dir / "render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.video:
        import imageio.v2 as imageio  # type: ignore

        args.video.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.video, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
