#!/usr/bin/env python3
"""P51A fixed-start validator for transformer-only Bernini-R checkpoints.

The clean clip is intentionally used only after generation for metrics and a
three-column QA video.  The sole inference condition passed to Bernini is the
adjusted Tripo video.
"""

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bernini.pipeline import BerniniRendererPipeline
from bernini.weights import HIGH_NOISE_PREFIXES, load_transformer_state_dict


PROMPT = (
    "Translate this Tripo condition video into a realistic, temporally stable "
    "roomtour. Preserve scene layout and camera motion; remove synthetic "
    "artifacts, ghosting, translucency, and duplicate objects while retaining "
    "realistic indoor appearance."
)
FIXED_STARTS = (0, 17, 31, 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adjusted-clips", required=True)
    parser.add_argument("--clean-clips", required=True)
    parser.add_argument("--comparison-script", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check_checkpoint(checkpoint: Path):
    marker = checkpoint / "P51A_HF_EXPORT_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"checkpoint is not complete for validation: {marker}")
    info = json.loads(marker.read_text())
    if info.get("clean_used_as_inference_input") is not False:
        raise RuntimeError("checkpoint provenance does not certify clean-input exclusion")
    return info


def load_rgb_frames(video: Path, temp_root: Path):
    frame_dir = temp_root / video.stem
    frame_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vf", "scale=672:672", "-vsync", "0", str(frame_dir / "F%02d.png"),
        ],
        check=True,
    )
    paths = sorted(frame_dir.glob("F*.png"))
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in paths]


def ssim_global(a, b):
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float((2 * mu_a * mu_b + c1) * (2 * cov + c2) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def metrics(pred, target):
    if len(pred) != 33 or len(target) != 33:
        raise RuntimeError(f"expected 33 frames, got prediction={len(pred)}, target={len(target)}")
    mse = float(np.mean([(a - b) ** 2 for a, b in zip(pred, target)]))
    psnr = float(10 * math.log10(1.0 / max(mse, 1e-12)))
    ssim = float(np.mean([ssim_global(a, b) for a, b in zip(pred, target)]))
    pred_delta = [pred[i + 1] - pred[i] for i in range(32)]
    target_delta = [target[i + 1] - target[i] for i in range(32)]
    temporal_delta_error = float(np.mean([(a - b) ** 2 for a, b in zip(pred_delta, target_delta)]))
    flicker = float(np.mean([np.abs(a.mean((0, 1)) - b.mean((0, 1))).mean() for a, b in zip(pred_delta, target_delta)]))
    emse, essim = [], []
    for a, b in zip(pred, target):
        x, y = a.reshape(-1), b.reshape(-1)
        scale = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
        aa = np.clip(a * scale, 0.0, 1.0)
        emse.append(float(np.mean((aa - b) ** 2)))
        essim.append(ssim_global(aa, b))
    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim_global": ssim,
        "exposure_matched_mse": float(np.mean(emse)),
        "exposure_matched_ssim_global": float(np.mean(essim)),
        "temporal_delta_error": temporal_delta_error,
        "flicker_proxy": flicker,
        "frame_count": 33,
    }


def mean_metrics(rows):
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys if key != "frame_count"} | {"frame_count": 33}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    provenance = check_checkpoint(checkpoint)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    pipeline = BerniniRendererPipeline.from_pretrained(args.base, device=str(device), load_ckpt_weights=False)
    state, prefix = load_transformer_state_dict(str(checkpoint), HIGH_NOISE_PREFIXES)
    missing, unexpected = pipeline.model.diff_dec.transformer.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    pipeline.model.eval().to(device)

    per_start = {}
    pred_rows, condition_rows = [], []
    with tempfile.TemporaryDirectory(prefix="p51a_validate_", dir=output_dir) as tmp:
        temp_root = Path(tmp)
        for start in FIXED_STARTS:
            name = f"start{start:02d}"
            condition = Path(args.adjusted_clips) / f"cyclic_start{start:02d}.mkv"
            clean = Path(args.clean_clips) / f"cyclic_start{start:02d}.mkv"
            prediction = output_dir / f"{name}.mp4"
            if not condition.is_file() or not clean.is_file():
                raise FileNotFoundError(f"missing validation clip for {name}")
            # `video=condition` is the only visual inference argument.  Clean is
            # opened later below exclusively for metrics and comparison rendering.
            pipeline(
                PROMPT,
                video=str(condition),
                output_path=str(prediction),
                num_frames=33,
                max_image_size=672,
                num_inference_steps=args.num_inference_steps,
                guidance_mode="v2v",
                seed=args.seed,
                fps=16,
            )
            pipeline.model.to(device).eval()
            subprocess.run(
                [args.comparison_script, str(condition), str(prediction), str(clean), str(output_dir / f"comparison_{name}.mp4")],
                check=True,
            )
            condition_frames = load_rgb_frames(condition, temp_root / f"condition_{name}")
            prediction_frames = load_rgb_frames(prediction, temp_root / f"prediction_{name}")
            clean_frames = load_rgb_frames(clean, temp_root / f"clean_{name}")
            pred_metric = metrics(prediction_frames, clean_frames)
            condition_metric = metrics(condition_frames, clean_frames)
            pred_rows.append(pred_metric)
            condition_rows.append(condition_metric)
            per_start[name] = {
                "prediction_vs_clean": pred_metric,
                "condition_vs_clean": condition_metric,
                "condition_input": str(condition),
                "clean_target_for_qa_only": str(clean),
                "prediction": str(prediction),
            }

    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_provenance": provenance,
        "fixed_starts": list(FIXED_STARTS),
        "inference": {
            "base": args.base,
            "guidance_mode": "v2v",
            "num_frames": 33,
            "max_image_size": 672,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "clean_used_as_inference_input": False,
            "transformer_prefix": prefix,
        },
        "per_start": per_start,
        "mean_prediction_vs_clean": mean_metrics(pred_rows),
        "mean_condition_vs_clean": mean_metrics(condition_rows),
        "not_implemented": ["hf_edge_f1", "flow_warp_residual", "roi_metrics"],
    }
    metrics_path = output_dir / "metrics.json"
    temp_path = metrics_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, metrics_path)
    marker = output_dir / "P51A_VALIDATION_COMPLETE.json"
    marker_tmp = marker.with_suffix(".json.tmp")
    marker_tmp.write_text(json.dumps({"status": "complete", "metrics": str(metrics_path)}, indent=2) + "\n")
    os.replace(marker_tmp, marker)
    print(json.dumps(result["mean_prediction_vs_clean"], indent=2))


if __name__ == "__main__":
    main()
