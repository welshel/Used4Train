#!/usr/bin/env python3
"""P49A lossless-PNG VACE validation and target-only QA metrics."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "DiffSynth-Studio"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video

PROMPT = "a realistic indoor room tour"
NEGATIVE_PROMPT = "overexposed, blur, watermark, text, distorted geometry, flicker, low quality"
FRAME_COUNT = 72


def clip_indices(start: int, clip_len: int = 33):
    if not 0 <= int(start) < FRAME_COUNT:
        raise ValueError("start must be in [0, 71]")
    if int(clip_len) != 33:
        raise ValueError("P49A validation requires 33 frames")
    return [(int(start) + i) % FRAME_COUNT for i in range(int(clip_len))]


def load_rgb_frame(folder: Path, index: int, resolution: int = 672):
    path = folder / f"F{int(index):02d}.png"
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        return image


def global_ssim(a: np.ndarray, b: np.ndarray):
    x = a.astype(np.float64).reshape(-1, 3) / 255.0
    y = b.astype(np.float64).reshape(-1, 3) / 255.0
    mux, muy = x.mean(axis=0), y.mean(axis=0)
    vx, vy = x.var(axis=0), y.var(axis=0)
    cov = ((x - mux) * (y - muy)).mean(axis=0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2))
    return float(np.mean(score))


def gradients(image: np.ndarray):
    value = image.astype(np.float32) / 255.0
    gray = 0.299 * value[:, :, 0] + 0.587 * value[:, :, 1] + 0.114 * value[:, :, 2]
    dx = np.diff(gray, axis=1, append=gray[:, -1:])
    dy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.sqrt(dx * dx + dy * dy)


def exposure_match(output: np.ndarray, target: np.ndarray):
    source_mean = max(float(output.mean()), 1.0)
    matched = output.astype(np.float32) * (float(target.mean()) / source_mean)
    return np.clip(matched, 0, 255).astype(np.uint8)


def compute_frame_metrics(outputs, targets):
    if len(outputs) != len(targets) or len(outputs) == 0:
        raise ValueError("outputs and targets must be non-empty and aligned")
    output = np.stack(outputs).astype(np.float32)
    target = np.stack(targets).astype(np.float32)
    mse = float(np.mean((output - target) ** 2))
    psnr = float("inf") if mse == 0 else float(20 * math.log10(255.0) - 10 * math.log10(mse))
    ssim = float(np.mean([global_ssim(a, b) for a, b in zip(outputs, targets)]))
    matched = [exposure_match(a, b) for a, b in zip(outputs, targets)]
    matched_arr = np.stack(matched).astype(np.float32)
    exp_mse = float(np.mean((matched_arr - target) ** 2))
    exp_ssim = float(np.mean([global_ssim(a, b) for a, b in zip(matched, targets)]))
    grad_out = [gradients(a) for a in outputs]
    grad_tgt = [gradients(a) for a in targets]
    hf_mse = float(np.mean([(a - b) ** 2 for a, b in zip(grad_out, grad_tgt)]))
    grad_error = float(np.mean([np.abs(a - b).mean() for a, b in zip(grad_out, grad_tgt)]))
    f1s = []
    for a, b in zip(grad_out, grad_tgt):
        edge_a, edge_b = a > 0.08, b > 0.08
        tp = int(np.logical_and(edge_a, edge_b).sum())
        denom = int(edge_a.sum()) + int(edge_b.sum())
        f1s.append(1.0 if denom == 0 else 2.0 * tp / denom)
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "exposure_matched_mse": exp_mse,
        "exposure_matched_ssim": exp_ssim,
        "hf_mse": hf_mse,
        "hf_edge_f1": float(np.mean(f1s)),
        "gradient_error": grad_error,
    }


def compute_temporal_metrics(outputs, targets):
    if len(outputs) < 2:
        return {"temporal_delta": 0.0, "flicker": 0.0, "flow_warp_residual": 0.0}
    output_delta = np.array(
        [np.abs(outputs[i].astype(np.float32) - outputs[i - 1].astype(np.float32)).mean() / 255.0 for i in range(1, len(outputs))]
    )
    flow_residual = []
    try:
        import cv2
        for i in range(1, len(outputs)):
            source = targets[i - 1]
            destination = targets[i]
            source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            destination_gray = cv2.cvtColor(destination, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(source_gray, destination_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = source_gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
            map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
            warped = cv2.remap(outputs[i - 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            flow_residual.append(float(np.abs(warped.astype(np.float32) - outputs[i].astype(np.float32)).mean() / 255.0))
    except Exception:
        flow_residual = output_delta.tolist()
    return {
        "temporal_delta": float(output_delta.mean()),
        "flicker": float(output_delta.std()),
        "flow_warp_residual": float(np.mean(flow_residual)),
    }


def build_pipeline(model_dir: Path):
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(model_dir / "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(model_dir / "Wan2.1_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(model_dir / "google" / "umt5-xxl")),
        redirect_common_files=False,
    )


def run_validation(args):
    condition_dir = Path(args.condition_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    indices = clip_indices(args.start, args.num_frames)
    condition = [load_rgb_frame(condition_dir, index, args.resolution) for index in indices]
    target = [np.asarray(load_rgb_frame(target_dir, index, args.resolution), dtype=np.uint8) for index in indices]
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(Path(args.model_dir))
    if args.checkpoint is not None:
        pipe.load_lora(pipe.vace, args.checkpoint, alpha=1.0)
    generated = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        vace_video=condition,
        vace_reference_image=condition[0],
        vace_scale=1.0,
        height=args.resolution,
        width=args.resolution,
        num_frames=args.num_frames,
        num_inference_steps=args.inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
        output_type="quantized",
    )
    if len(generated) != args.num_frames:
        raise RuntimeError(f"expected {args.num_frames} output frames, received {len(generated)}")

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    arrays = []
    for frame, index in zip(generated, indices):
        frame = frame.convert("RGB")
        frame.save(frame_dir / f"F{index:02d}.png")
        arrays.append(np.asarray(frame, dtype=np.uint8))
    save_video(generated, str(output_dir / "validation.mp4"), fps=12, quality=8)

    metrics = compute_frame_metrics(arrays, target)
    temporal = compute_temporal_metrics(arrays, target)
    report = {
        "start": args.start,
        "frame_indices": indices,
        "checkpoint": args.checkpoint,
        "official_base_without_lora": args.checkpoint is None,
        "condition_source": str(condition_dir.resolve()),
        "target_source": str(target_dir.resolve()),
        "target_used_for_generation": False,
        "prompt": PROMPT,
        "seed": args.seed,
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "inference_steps": args.inference_steps,
        "metrics": metrics,
        "temporal": temporal,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--condition-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_validation(build_parser().parse_args())
