#!/usr/bin/env python3
"""Select P51A's best validated checkpoint from fixed-start QA metrics."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/fs1/private/user/baitongyuan/projects/liuzh")
OUT = ROOT / "outputs/p51a_bernini_r13b_single_scene"
RUN = OUT / "training/run_700"
FINAL = OUT / "final"
STEPS = (100, 200, 300, 400, 500, 600, 700)


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def score(pred, condition):
    # All components compare the exact same four starts.  MSE and temporal
    # delta are relative improvements; SSIM is an absolute signed gain.
    mse_gain = (condition["mse"] - pred["mse"]) / max(condition["mse"], 1e-12)
    temporal_gain = (condition["temporal_delta_error"] - pred["temporal_delta_error"]) / max(condition["temporal_delta_error"], 1e-12)
    ssim_gain = pred["ssim_global"] - condition["ssim_global"]
    return 0.45 * mse_gain + 0.30 * temporal_gain + 0.25 * ssim_gain, {
        "relative_mse_improvement": mse_gain,
        "relative_temporal_delta_improvement": temporal_gain,
        "ssim_gain": ssim_gain,
    }


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    rows = []
    checkpoints = []
    for step in range(20, 701, 20):
        root = RUN / "checkpoints" / f"global_step_{step}"
        marker = root / "P51A_DCP_COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError(f"missing trusted DCP completion marker: {marker}")
        checkpoints.append({"step": step, "path": str(root), "dcp_completion": read_json(marker)})
    for step in STEPS:
        root = RUN / "checkpoints" / f"global_step_{step}"
        hf = root / "hf_ckpt"
        hf_marker = hf / "P51A_HF_EXPORT_COMPLETE.json"
        validation = OUT / "validation" / f"step{step:04d}"
        validation_marker = validation / "P51A_VALIDATION_COMPLETE.json"
        metrics_path = validation / "metrics.json"
        if not hf_marker.is_file() or not validation_marker.is_file() or not metrics_path.is_file():
            raise RuntimeError(f"missing complete validation artifacts at step {step}")
        metrics = read_json(metrics_path)
        pred = metrics["mean_prediction_vs_clean"]
        condition = metrics["mean_condition_vs_clean"]
        total, components = score(pred, condition)
        rows.append({
            "step": step,
            "score": total,
            "score_components": components,
            "prediction_vs_clean": pred,
            "condition_vs_clean": condition,
            "validation_metrics": str(metrics_path),
            "hf_checkpoint": str(hf),
        })
    best = max(rows, key=lambda row: (row["score"], row["prediction_vs_clean"]["psnr_db"], row["prediction_vs_clean"]["ssim_global"]))
    selection = {
        "status": "complete",
        "selection_method": "0.45_relative_mse + 0.30_relative_temporal_delta + 0.25_ssim_gain; fixed starts only",
        "validation_starts": [0, 17, 31, 50],
        "steps": rows,
        "best": best,
        "clean_used_as_inference_input": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(FINAL / "P51A_CHECKPOINT_SELECTION.json", selection)
    atomic_json(FINAL / "P51A_CHECKPOINT_SUMMARY.json", {
        "status": "complete",
        "fresh_run": True,
        "base_checkpoint": "ByteDance/Bernini-R-1.3B-Diffusers@ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce",
        "dcp_checkpoints": checkpoints,
        "validated_checkpoints": [{"step": row["step"], "hf_checkpoint": row["hf_checkpoint"], "score": row["score"]} for row in rows],
        "best_checkpoint": best,
    })
    atomic_json(FINAL / "P51A_VALIDATION_SUMMARY.json", {
        "status": "complete",
        "required_steps": list(STEPS),
        "step0": {"status": "not_run", "reason": "no frozen pre-finetune HF export was scheduled; step-1 smoke validates the inference chain"},
        "validation_starts": [0, 17, 31, 50],
        "metrics": rows,
        "clean_used_as_inference_input": False,
    })
    best_file = FINAL / "P51A_BEST_CHECKPOINT.txt"
    temporary = best_file.with_suffix(".txt.tmp")
    temporary.write_text(
        f"step={best['step']}\n"
        f"hf_checkpoint={best['hf_checkpoint']}\n"
        f"selection_score={best['score']:.10f}\n"
        f"selection_method={selection['selection_method']}\n"
    )
    os.replace(temporary, best_file)
    print(json.dumps(selection["best"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
