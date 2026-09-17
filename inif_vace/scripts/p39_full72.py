#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

from PIL import Image

from p38_contract import validate_spatial_resolution


TOTAL_FRAMES = 72
CONTEXT_FRAMES = 33
SEGMENT_FRAMES = 18
PRE_CONTEXT = 7
POST_CONTEXT = 8
ALLOWED_CONDITION_KEYS = {"frame_index", "frame_stem", "condition_rgb"}


def direct73_indices(total: int = TOTAL_FRAMES, start_index: int = 0) -> list[int]:
    if start_index < 0 or start_index >= total:
        raise ValueError(f"start_index must be in [0, {total})")
    ordered = [(start_index + offset) % total for offset in range(total)]
    return ordered + [start_index]


def canonicalize_rotated_frames(frames: list, start_index: int, total: int = TOTAL_FRAMES) -> list:
    if len(frames) != total:
        raise ValueError(f"expected {total} rotated frames, got {len(frames)}")
    canonical = [None] * total
    for offset, frame in enumerate(frames):
        canonical[(start_index + offset) % total] = frame
    return canonical


def overlap_windows(
    total: int = TOTAL_FRAMES,
    segment_frames: int = SEGMENT_FRAMES,
    pre_context: int = PRE_CONTEXT,
    post_context: int = POST_CONTEXT,
) -> list[dict]:
    if pre_context + segment_frames + post_context != CONTEXT_FRAMES:
        raise ValueError("overlap context must contain exactly 33 frames")
    if total % segment_frames != 0:
        raise ValueError("segment length must divide the closed loop")
    windows = []
    for segment_start in range(0, total, segment_frames):
        output_indices = [(segment_start + offset) % total for offset in range(segment_frames)]
        context_indices = [
            (segment_start - pre_context + offset) % total for offset in range(CONTEXT_FRAMES)
        ]
        windows.append(
            {
                "segment_start": segment_start,
                "context_indices": context_indices,
                "output_indices": output_indices,
                "central_slice": [pre_context, pre_context + segment_frames],
            }
        )
    return windows


def inference_contract() -> dict:
    return {
        "scene_input": "native_gaussian_rgb_condition",
        "camera": "encoded_in_native_gaussian_condition_frames",
        "text_prompt": "a realistic indoor room tour",
        "vace_reference_image": None,
        "clean_inputs": [],
        "target_leakage": False,
    }


def validate_condition_rows(rows: list[dict]) -> int:
    if len(rows) != TOTAL_FRAMES:
        raise ValueError(f"condition-only manifest must contain {TOTAL_FRAMES} rows")
    for expected_index, row in enumerate(rows):
        if set(row) != ALLOWED_CONDITION_KEYS:
            raise ValueError(
                f"condition-only manifest row {expected_index} must have exactly "
                f"{sorted(ALLOWED_CONDITION_KEYS)}"
            )
        if int(row["frame_index"]) != expected_index:
            raise ValueError("condition-only manifest indices must be contiguous 0..71")
    return len(rows)


def load_condition_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["frame_index"]))
    validate_condition_rows(rows)
    for row in rows:
        if not Path(row["condition_rgb"]).is_file():
            raise FileNotFoundError(row["condition_rgb"])
    return rows


def load_condition_frames(rows: list[dict], indices: list[int], size: tuple[int, int]) -> list[Image.Image]:
    frames = []
    for index in indices:
        with Image.open(rows[index]["condition_rgb"]) as image:
            frames.append(image.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a clean-leakage-free P3.9 full RoomTour")
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--condition-manifest", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("direct73", "overlap33"), required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--vace-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def validate_full72_resolution(width: int, height: int) -> tuple[int, int]:
    return validate_spatial_resolution(width, height)


def main() -> None:
    args = parse_args()
    validate_full72_resolution(args.width, args.height)
    if not args.lora.is_file():
        raise FileNotFoundError(args.lora)
    rows = load_condition_rows(args.condition_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(args.diffsynth_repo))
    os.chdir(args.diffsynth_repo)

    import torch
    from diffsynth import save_video
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    model_paths = [
        args.model_dir / "diffusion_pytorch_model.safetensors",
        args.model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        args.model_dir / "Wan2.1_VAE.pth",
    ]
    missing = [str(path) for path in model_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model files: {missing}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(path=str(path), offload_device="cpu") for path in model_paths],
    )
    pipe.load_lora(pipe.vace, str(args.lora), alpha=1)
    pipe.enable_vram_management()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    negative_prompt = (
        "overexposed, blurry, text, watermark, low quality, distorted geometry, "
        "camera drift, layout morph, furniture replacement, temporal flicker"
    )

    def generate(condition_frames: list[Image.Image], num_frames: int, seed: int):
        return pipe(
            prompt="a realistic indoor room tour",
            negative_prompt=negative_prompt,
            vace_video=condition_frames,
            vace_reference_image=None,
            height=args.height,
            width=args.width,
            num_frames=num_frames,
            seed=seed,
            tiled=True,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            vace_scale=args.vace_scale,
        )

    condition72 = load_condition_frames(rows, list(range(TOTAL_FRAMES)), (args.width, args.height))
    condition_path = args.output_dir / "P39_FULL72_CONDITION.mp4"
    save_video(condition72, str(condition_path), fps=args.fps, quality=5)
    generated_path: Path
    generated_frames = []
    windows_audit = []
    if args.method == "direct73":
        indices = direct73_indices(start_index=args.start_index)
        condition73 = load_condition_frames(rows, indices, (args.width, args.height))
        output73 = generate(condition73, 73, args.seed)
        if len(output73) != 73:
            raise ValueError(f"direct73 returned {len(output73)} frames")
        save_video(output73, str(args.output_dir / "P39_FULL73_DIRECT_DEBUG.mp4"), fps=args.fps, quality=5)
        generated_frames = canonicalize_rotated_frames(list(output73[:72]), args.start_index)
        generated_path = args.output_dir / "P39_FULL72_DIRECT_RAW.mp4"
        save_video(generated_frames, str(generated_path), fps=args.fps, quality=5)
        windows_audit.append(
            {
                "method": "direct73",
                "start_index": args.start_index,
                "indices": indices,
                "kept_local": [0, 71],
                "canonicalized": True,
                "model_boundary_transition": [
                    (args.start_index - 1) % TOTAL_FRAMES,
                    args.start_index,
                ],
            }
        )
    else:
        window_dir = args.output_dir / "windows"
        window_dir.mkdir(exist_ok=True)
        stitched = []
        for window_id, window in enumerate(overlap_windows()):
            context = load_condition_frames(
                rows, window["context_indices"], (args.width, args.height)
            )
            output = generate(context, CONTEXT_FRAMES, args.seed)
            if len(output) != CONTEXT_FRAMES:
                raise ValueError(f"window {window_id} returned {len(output)} frames")
            save_video(
                output,
                str(window_dir / f"window_{window_id}_start{window['segment_start']:02d}_33f.mp4"),
                fps=args.fps,
                quality=5,
            )
            begin, end = window["central_slice"]
            central = list(output[begin:end])
            stitched.extend(central)
            windows_audit.append({**window, "window_id": window_id, "seed": args.seed})
            torch.cuda.empty_cache()
        if len(stitched) != TOTAL_FRAMES:
            raise ValueError(f"overlap stitch returned {len(stitched)} frames")
        generated_frames = stitched
        generated_path = args.output_dir / "P39_FULL72_OVERLAP33_RAW.mp4"
        save_video(generated_frames, str(generated_path), fps=args.fps, quality=5)

    summary = {
        "status": "PASS",
        "method": args.method,
        "resolution": [args.width, args.height],
        "frames": len(generated_frames),
        "fps": args.fps,
        "fps_status": "operational_preview_not_ground_truth",
        "seed": args.seed,
        "start_index": args.start_index if args.method == "direct73" else None,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "vace_scale": args.vace_scale,
        "condition_manifest": str(args.condition_manifest),
        "lora": str(args.lora),
        "condition_video": str(condition_path),
        "output_video": str(generated_path),
        "windows": windows_audit,
        "inference_contract": inference_contract(),
        "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "peak_ram_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / f"P39_FULL72_{args.method.upper()}_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("P39_FULL72_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
