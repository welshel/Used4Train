#!/usr/bin/env python3
"""Run the official InfiniSplat RGB single-image inference.

This wrapper deliberately accepts one RGB topdown image only.  It never opens
the clean trajectory or any clean target asset.  The artifact and filtered PLY
are the only geometry inputs used by the downstream renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infinisplat-repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--topdown", type=Path, required=True)
    ap.add_argument("--topdown-camera", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--meta", type=Path, required=True)
    args = ap.parse_args()
    for path in (args.infinisplat_repo, args.checkpoint, args.topdown):
        if not path.exists():
            raise FileNotFoundError(path)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.ply.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.infinisplat_repo))
    # Use the official encoder/runtime functions, while explicitly preserving
    # the audited topdown camera intrinsics (including its principal point).
    from src.demo.hf_runtime import (  # type: ignore
        GaussianArtifact,
        export_filtered_gaussian_ply,
        save_gaussian_artifact,
    )
    from src.demo.infer_single_image import (  # type: ignore
        load_demo_config,
        load_demo_encoder,
        load_demo_image_bundle,
        run_single_image_inference,
    )
    import torch  # type: ignore

    topdown_camera = args.topdown_camera
    if not topdown_camera.is_file():
        raise FileNotFoundError(f"topdown camera JSON is required beside topdown image: {topdown_camera}")
    camera_payload = json.loads(topdown_camera.read_text())
    intrinsic = camera_payload.get("intrinsic")
    if intrinsic is None:
        raise KeyError(f"topdown camera has no intrinsic matrix: {topdown_camera}")
    matrix = [row[:3] for row in intrinsic[:3]]
    override = args.artifact.parent / "source_intrinsics_px.yaml"
    override.write_text("intrinsics_px:\n" + "".join(f"  - [{', '.join(f'{float(v):.12g}' for v in row)}]\n" for row in matrix))
    device = torch.device(args.device)
    cfg = load_demo_config("infinisplat_hypersim_rgb")
    encoder = load_demo_encoder(cfg=cfg, checkpoint_path=args.checkpoint, device=device)
    image_bundle = load_demo_image_bundle(image_path=args.topdown, intrinsics_override_path=override)
    output = run_single_image_inference(
        encoder=encoder,
        image=image_bundle.inference_image,
        intrinsics_px=image_bundle.inference_intrinsics.intrinsics_px,
        device=device,
    )
    _, height, width = image_bundle.inference_image.shape
    save_gaussian_artifact(
        GaussianArtifact(
            gaussians=output["gaussians"].to("cpu"),
            focal_length_px=image_bundle.inference_intrinsics.focal_length_px,
            image_shape=(height, width),
        ),
        args.artifact,
    )
    scene_ply = export_filtered_gaussian_ply(
        artifact_path=args.artifact,
        output_dir=args.ply.parent,
    )
    if scene_ply.resolve() != args.ply.resolve():
        scene_ply.replace(args.ply)
    metadata = {
        "status": "OK",
        "official_rgb_only": True,
        "topdown": str(args.topdown),
        "topdown_sha256": sha256(args.topdown),
        "topdown_camera": str(topdown_camera),
        "topdown_camera_sha256": sha256(topdown_camera),
        "intrinsics_override": str(override),
        "original_image_shape": list(image_bundle.original_image_shape),
        "inference_image_shape": [height, width],
        "source_intrinsics_px": matrix,
        "inference_intrinsics_px": image_bundle.inference_intrinsics.intrinsics_px.tolist(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "ply": str(args.ply),
        "ply_sha256": sha256(args.ply),
        "clean_target_files_read": False,
        "renderer_inputs": ["artifact/scene.ply", "explicit camera payload"],
    }
    args.meta.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
