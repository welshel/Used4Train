#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

from p38_contract import validate_records


FIXED_VALIDATION_STARTS = (0, 17, 31, 50)
CANONICAL_START_ORDER = FIXED_VALIDATION_STARTS


def validation_seed(start: int, base_seed: int) -> int:
    return int(base_seed) + int(start)


def load_validation_records(path: Path | str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["start_index"]))
    starts = [int(row["start_index"]) for row in rows]
    expected = FIXED_VALIDATION_STARTS
    if starts != list(expected):
        raise ValueError(f"expected validation starts {expected}, got {starts}")
    validate_records(rows)
    return rows


def parse_starts(
    value: str | None,
    allowed_starts: set[int] | tuple[int, ...] = FIXED_VALIDATION_STARTS,
) -> set[int] | None:
    if value is None:
        return None
    starts = {int(item) for item in value.split(",") if item.strip()}
    unknown = starts.difference(allowed_starts)
    if unknown:
        raise ValueError(f"unsupported validation starts: {sorted(unknown)}")
    return starts


def validation_prefix(start: int) -> str:
    start = int(start)
    if start not in CANONICAL_START_ORDER:
        raise ValueError(f"unsupported validation start: {start}")
    return f"V{CANONICAL_START_ORDER.index(start)}_start{start:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed P3.8 high-resolution validation clips")
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--only-starts")
    parser.add_argument("--base-seed", type=int, default=1337)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--vace-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_validation_records(args.manifest)
    selected_starts = parse_starts(args.only_starts, set(FIXED_VALIDATION_STARTS))
    if selected_starts is not None:
        records = [row for row in records if int(row["start_index"]) in selected_starts]
    if not records:
        raise ValueError("no validation records selected")
    if not args.lora.is_file():
        raise FileNotFoundError(args.lora)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(args.diffsynth_repo))
    os.chdir(args.diffsynth_repo)

    import torch
    from diffsynth import VideoData, save_video
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    negative_prompt = (
        "overexposed, blurry, text, watermark, low quality, distorted geometry, "
        "camera drift, layout morph, furniture replacement, temporal flicker"
    )
    outputs = []
    start_time = time.time()
    for validation_id, record in enumerate(records):
        start = int(record["start_index"])
        width = int(record["width"])
        height = int(record["height"])
        num_frames = int(record["num_frames"])
        seed = validation_seed(start, args.base_seed)
        control = VideoData(record["vace_video"], height=height, width=width)
        control_frames = [control[index] for index in range(num_frames)]

        generated = pipe(
            prompt=record["prompt"],
            negative_prompt=negative_prompt,
            vace_video=control_frames,
            vace_reference_image=None,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            tiled=True,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            vace_scale=args.vace_scale,
        )
        prefix = validation_prefix(start)
        output_path = args.output_dir / f"{prefix}_{args.label}.mp4"
        save_video(generated, str(output_path), fps=args.fps, quality=5)
        condition_copy = args.output_dir / f"{prefix}_condition.mp4"
        clean_copy = args.output_dir / f"{prefix}_clean.mp4"
        if not condition_copy.exists():
            shutil.copy2(record["vace_video"], condition_copy)
        if not clean_copy.exists():
            shutil.copy2(record["video"], clean_copy)
        item = {
            "validation_id": CANONICAL_START_ORDER.index(start),
            "start_index": start,
            "seed": seed,
            "label": args.label,
            "branch": "l33",
            "reference_mode": "none",
            "reference_source": None,
            "reference_sha256": None,
            "lora": str(args.lora),
            "condition": str(condition_copy),
            "clean": str(clean_copy),
            "output": str(output_path),
        }
        outputs.append(item)
        print("P38_VALIDATION " + json.dumps(item), flush=True)
        torch.cuda.empty_cache()

    summary = {
        "label": args.label,
        "branch": "l33",
        "reference_mode": "none",
        "lora": str(args.lora),
        "base_seed": args.base_seed,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "vace_scale": args.vace_scale,
        "fps": args.fps,
        "clean_dataset_fps": "UNRESOLVED",
        "outputs": outputs,
        "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "peak_ram_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / f"validation_{args.label}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("P38_VALIDATION_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
