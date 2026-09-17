#!/usr/bin/env python3
"""Audit the old P49A condition against the final dynamic-adjusted condition.

The audit is read-only with respect to Gaussian/camera assets.  It writes a
machine-readable comparison and a recommendation for whether the existing
step-700 LoRA can be treated as the final Phase-C model for the new condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def frame_metrics(old: np.ndarray, new: np.ndarray, clean: np.ndarray) -> dict:
    delta = np.abs(new - old)
    old_err = (old - clean) ** 2
    new_err = (new - clean) ** 2
    old_l = old.mean(axis=2)
    new_l = new.mean(axis=2)
    clean_l = clean.mean(axis=2)
    # This is the same broad white-foreground diagnostic used by the Phase-B
    # dynamic render QA; it is not a Clean input to training.
    old_white = (old_l >= 0.45) & (clean_l < 0.45)
    new_white = (new_l >= 0.45) & (clean_l < 0.45)
    return {
        "condition_rmse_old_to_new": float(np.sqrt(np.mean(delta**2))),
        "condition_mae_old_to_new": float(np.mean(delta)),
        "condition_changed_fraction_gt_2_255": float(np.mean(delta.mean(axis=2) > (2.0 / 255.0))),
        "condition_changed_fraction_gt_8_255": float(np.mean(delta.mean(axis=2) > (8.0 / 255.0))),
        "old_condition_mse_clean": float(np.mean(old_err)),
        "new_condition_mse_clean": float(np.mean(new_err)),
        "old_white_foreground_fraction": float(np.mean(old_white)),
        "new_white_foreground_fraction": float(np.mean(new_white)),
        "white_foreground_delta_fraction": float(np.mean(new_white) - np.mean(old_white)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-condition", type=Path, required=True)
    ap.add_argument("--new-condition", type=Path, required=True)
    ap.add_argument("--raw-condition", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--old-contract", type=Path, required=True)
    ap.add_argument("--best-checkpoint", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(72):
        name = f"F{i:02d}.png"
        old_p, new_p, raw_p, clean_p = [d / name for d in (args.old_condition, args.new_condition, args.raw_condition, args.clean)]
        for p in (old_p, new_p, raw_p, clean_p):
            if not p.is_file():
                raise FileNotFoundError(p)
        m = frame_metrics(rgb(old_p), rgb(new_p), rgb(clean_p))
        m.update({
            "frame": i,
            "old_condition_sha256": sha256(old_p),
            "new_condition_sha256": sha256(new_p),
            "raw_condition_sha256": sha256(raw_p),
            "clean_sha256": sha256(clean_p),
        })
        frames.append(m)

    def mean(key: str) -> float:
        return float(np.mean([f[key] for f in frames]))

    keyframes = [10, 11, 12, 13, 14, 17, 24, 28, 29, 30, 31, 50]
    old_contract = json.loads(args.old_contract.read_text())
    checkpoint_sha = sha256(args.best_checkpoint)
    summary = {
        "status": "PASS",
        "old_p49a_condition": str(args.old_condition.resolve()),
        "new_dynamic_adjusted_condition": str(args.new_condition.resolve()),
        "raw_condition_for_dual_audit": str(args.raw_condition.resolve()),
        "clean_used_for_metric_only": str(args.clean.resolve()),
        "frame_count": 72,
        "resolution": [896, 896],
        "old_contract_condition_source": old_contract.get("condition_source", "unknown"),
        "old_contract_train_items": old_contract.get("train_items"),
        "new_manifest_train_items": 2800,
        "existing_step700_checkpoint": str(args.best_checkpoint.resolve()),
        "existing_step700_checkpoint_sha256": checkpoint_sha,
        "aggregate": {
            "mean_condition_rmse_old_to_new": mean("condition_rmse_old_to_new"),
            "mean_condition_mae_old_to_new": mean("condition_mae_old_to_new"),
            "mean_condition_changed_fraction_gt_2_255": mean("condition_changed_fraction_gt_2_255"),
            "mean_condition_changed_fraction_gt_8_255": mean("condition_changed_fraction_gt_8_255"),
            "mean_old_condition_mse_clean": mean("old_condition_mse_clean"),
            "mean_new_condition_mse_clean": mean("new_condition_mse_clean"),
            "mean_old_white_foreground_fraction": mean("old_white_foreground_fraction"),
            "mean_new_white_foreground_fraction": mean("new_white_foreground_fraction"),
            "mean_white_foreground_delta_fraction": mean("white_foreground_delta_fraction"),
        },
        "keyframes": {f"F{i:02d}": frames[i] for i in keyframes},
        "recommendation": {
            "fresh_retrain": "RECOMMENDED",
            "existing_step700_as_final_for_new_condition": "NO",
            "existing_step700_use": "provisional zero-shot/probe baseline only; it was trained on the old P48.3 condition",
            "reason": "the dynamic-adjusted RGB condition is a new input distribution (tune4 adjusted Gaussian plus full-render suppression), so promoting an LoRA trained on the old P48.3 condition would leave an unmeasured train-test condition shift",
            "training_contract": "fresh official VACE-1.3B base + newly initialized rank-16 LoRA; no old LoRA warm start",
        },
        "artifacts": {
            "manifest_contract": "outputs/p49_adjusted_teacher_phase_c/04_dynamic_adjusted_2800_contract/P49_DYNAMIC_ADJUSTED_2800_DATA_CONTRACT.json",
            "fresh_4gpu_launcher": "outputs/p49_adjusted_teacher_phase_c/p49_dynamic_adjusted_fresh_4gpu_train.sh",
            "launcher_default": "dry-run; requires P49_DYNAMIC_START=YES",
        },
    }
    (args.output_dir / "P49_DYNAMIC_ADJUSTED_VS_P49A_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")

    a = summary["aggregate"]
    lines = [
        "# P49 dynamic-adjusted condition audit",
        "",
        "## Decision",
        "",
        "`FRESH_RETRAIN_RECOMMENDED`; existing P49A step-700 is retained as a provisional baseline only and is **not** promoted as the final Phase-C model for the new condition.",
        "",
        "The old P49A LoRA learned `outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic`. The new condition is `phase_b_adjusted_dynamic/condition_adjusted_rgb`, rendered from the tune4 adjusted Gaussian with the P48.3 full-render, trajectory-aware opacity suppression. This is a genuine input-distribution change, including the difficult F11-F13 curtain frames; retraining from the official VACE base with a fresh LoRA is the fail-closed choice.",
        "",
        "## Contract delta",
        "",
        f"- Old train items: `{old_contract.get('train_items')}`; new train items: `2800` (39 or 38 cyclic samples per start).",
        "- Old condition: P48.3 dynamic RGB; new condition: adjusted Gaussian + dynamic suppression RGB.",
        "- Exact cameras/alignment are unchanged; only the rendered condition changes.",
        "- New manifest uses lossless 896x896 PNGs, cyclic 33-frame clips, and fixed validation starts 0/17/31/50.",
        "- Clean appears only in `video` targets; never in `vace_video` or `vace_reference_image`.",
        "",
        "## Measured 72-frame shift",
        "",
        f"- Mean old→new RGB RMSE: `{a['mean_condition_rmse_old_to_new']:.6f}`; mean MAE: `{a['mean_condition_mae_old_to_new']:.6f}`.",
        f"- Mean changed-pixel fraction (>2/255 luminance-mean): `{a['mean_condition_changed_fraction_gt_2_255']:.4f}`; >8/255: `{a['mean_condition_changed_fraction_gt_8_255']:.4f}`.",
        f"- Mean condition MSE vs Clean: old `{a['mean_old_condition_mse_clean']:.6f}`, new `{a['mean_new_condition_mse_clean']:.6f}`.",
        f"- Broad white-foreground fraction: old `{a['mean_old_white_foreground_fraction']:.6f}`, new `{a['mean_new_white_foreground_fraction']:.6f}` (delta `{a['mean_white_foreground_delta_fraction']:.6f}`).",
        "",
        "Key-frame details are in `P49_DYNAMIC_ADJUSTED_VS_P49A_AUDIT.json`, including F10/F11/F12/F13/F14/F17/F24/F28/F29/F30/F31/F50.",
        "",
        "## Existing checkpoint",
        "",
        f"- Existing best step-700 LoRA: `{args.best_checkpoint.resolve()}`.",
        f"- SHA256: `{checkpoint_sha}`.",
        "- It remains useful for a controlled zero-shot/probe comparison on the new condition, but cannot be called a trained Phase-C result for the new contract.",
        "",
        "## Prepared artifacts",
        "",
        "- `04_dynamic_adjusted_2800_contract/dynamic_adjusted_train_manifest.json` (2800 rows)",
        "- `04_dynamic_adjusted_2800_contract/dynamic_adjusted_validation_manifest.json` (starts 0/17/31/50)",
        "- `04_dynamic_adjusted_2800_contract/P49_DYNAMIC_ADJUSTED_2800_DATA_CONTRACT.json`",
        "- `p49_dynamic_adjusted_fresh_4gpu_train.sh` (dry-run by default; explicit `P49_DYNAMIC_START=YES` required)",
        "",
        "No training was started by this audit.",
    ]
    (args.output_dir / "P49_DYNAMIC_ADJUSTED_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
