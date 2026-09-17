#!/usr/bin/env python3
"""Generate one auditable 33-frame P51A Full72 inference chunk.

The checkpoint is transformer-only and the condition video is the sole visual
input to Bernini.  A clean target is deliberately neither accepted nor opened.
"""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from bernini.pipeline import BerniniRendererPipeline
from bernini.weights import HIGH_NOISE_PREFIXES, load_transformer_state_dict


PROMPT = (
    "Translate this Tripo condition video into a realistic, temporally stable "
    "roomtour. Preserve scene layout and camera motion; remove synthetic "
    "artifacts, ghosting, translucency, and duplicate objects while retaining "
    "realistic indoor appearance."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition-video", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--route", choices=("adjusted", "raw"), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check_checkpoint(path: Path):
    marker = path / "P51A_HF_EXPORT_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"checkpoint is not complete: {marker}")
    payload = json.loads(marker.read_text())
    if payload.get("clean_used_as_inference_input") is not False:
        raise RuntimeError("checkpoint provenance does not certify clean-input exclusion")
    return payload


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    condition = Path(args.condition_video)
    output = Path(args.output_video)
    marker = output.with_suffix(".P51A_GENERATION_COMPLETE.json")
    if marker.is_file():
        if output.is_file():
            print(f"already complete: {output}")
            return
        raise RuntimeError(f"generation marker exists without video: {marker}")
    if not condition.is_file():
        raise FileNotFoundError(condition)
    provenance = check_checkpoint(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    # Preserve an output interrupted between encode and marker publication.
    for stale in (temporary, output):
        if stale.exists():
            archived = stale.with_name(f"{stale.name}.interrupted_{datetime.now().strftime('%Y%m%dT%H%M%S%z')}")
            shutil.move(str(stale), str(archived))

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    pipeline = BerniniRendererPipeline.from_pretrained(args.base, device=str(device), load_ckpt_weights=False)
    state, prefix = load_transformer_state_dict(str(checkpoint), HIGH_NOISE_PREFIXES)
    missing, unexpected = pipeline.model.diff_dec.transformer.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    pipeline.model.eval().to(device)
    pipeline(
        PROMPT,
        video=str(condition),
        output_path=str(temporary),
        num_frames=33,
        max_image_size=672,
        num_inference_steps=args.num_inference_steps,
        guidance_mode="v2v",
        seed=args.seed,
        fps=16,
    )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("Bernini pipeline returned without a non-empty temporary output video")
    os.replace(temporary, output)
    payload = {
        "status": "complete",
        "route": args.route,
        "start": args.start,
        "condition_video": str(condition),
        "output_video": str(output),
        "checkpoint": str(checkpoint),
        "checkpoint_provenance": provenance,
        "inference": {
            "num_frames": 33,
            "max_image_size": 672,
            "num_inference_steps": args.num_inference_steps,
            "guidance_mode": "v2v",
            "seed": args.seed,
            "clean_used_as_inference_input": False,
            "transformer_prefix": prefix,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_tmp = marker.with_suffix(".json.tmp")
    marker_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(marker_tmp, marker)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
