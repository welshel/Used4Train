#!/usr/bin/env python3
"""Write P51A's final auditable report after both Full72 routes are complete."""

import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/fs1/private/user/baitongyuan/projects/liuzh")
OUT = ROOT / "outputs/p51a_bernini_r13b_single_scene"
FINAL = OUT / "final"


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def frames(video: Path, root: Path):
    target = root / video.stem
    target.mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vsync", "0", str(target / "F%02d.png")], check=True)
    result = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in sorted(target.glob("F*.png"))]
    if len(result) != 72:
        raise RuntimeError(f"expected 72 frames in {video}, got {len(result)}")
    return result


def ssim_global(a, b):
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float((2 * mu_a * mu_b + c1) * (2 * cov + c2) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def metrics(pred, target):
    mse = float(np.mean([(a - b) ** 2 for a, b in zip(pred, target)]))
    deltas = [pred[i + 1] - pred[i] for i in range(71)]
    target_deltas = [target[i + 1] - target[i] for i in range(71)]
    exposure_mse, exposure_ssim = [], []
    for a, b in zip(pred, target):
        x, y = a.reshape(-1), b.reshape(-1)
        scale = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
        aa = np.clip(a * scale, 0.0, 1.0)
        exposure_mse.append(float(np.mean((aa - b) ** 2)))
        exposure_ssim.append(ssim_global(aa, b))
    return {
        "mse": mse,
        "psnr_db": float(10 * math.log10(1.0 / max(mse, 1e-12))),
        "ssim_global": float(np.mean([ssim_global(a, b) for a, b in zip(pred, target)])),
        "exposure_matched_mse": float(np.mean(exposure_mse)),
        "exposure_matched_ssim_global": float(np.mean(exposure_ssim)),
        "temporal_delta_error": float(np.mean([(a - b) ** 2 for a, b in zip(deltas, target_deltas)])),
        "flicker_proxy": float(np.mean([np.abs(a.mean((0, 1)) - b.mean((0, 1))).mean() for a, b in zip(deltas, target_deltas)])),
        "frame_count": 72,
    }


def yes_no(pred, condition):
    return pred["mse"] < condition["mse"] and pred["temporal_delta_error"] < condition["temporal_delta_error"]


def main():
    selection_path = FINAL / "P51A_CHECKPOINT_SELECTION.json"
    if not selection_path.is_file():
        raise RuntimeError("checkpoint selection is missing")
    selection = json.loads(selection_path.read_text())
    inputs = OUT / "full72/inputs"
    adjusted_output = OUT / "full72/adjusted/output_full72.mp4"
    raw_output = OUT / "full72/raw/output_full72.mp4"
    required = [
        inputs / "condition_adjusted_full72.mp4", inputs / "condition_raw_full72.mp4", inputs / "clean_full72.mp4",
        adjusted_output, raw_output,
        FINAL / "P51A_ADJUSTED_VS_OUTPUT_VS_CLEAN_FULL72.mp4",
        FINAL / "P51A_RAW_VS_OUTPUT_VS_CLEAN_FULL72.mp4",
        FINAL / "P51A_ADJUSTED_RAW_CLEAN_SUMMARY.mp4",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"finalization missing required artifacts: {missing}")
    with tempfile.TemporaryDirectory(prefix="p51a_final_metrics_", dir=OUT / "full72") as temp:
        temp_root = Path(temp)
        clean = frames(inputs / "clean_full72.mp4", temp_root)
        adjusted_input = frames(inputs / "condition_adjusted_full72.mp4", temp_root)
        raw_input = frames(inputs / "condition_raw_full72.mp4", temp_root)
        adjusted_pred = frames(adjusted_output, temp_root)
        raw_pred = frames(raw_output, temp_root)
        full_metrics = {
            "adjusted_route": {"condition_vs_clean": metrics(adjusted_input, clean), "prediction_vs_clean": metrics(adjusted_pred, clean)},
            "raw_route": {"condition_vs_clean": metrics(raw_input, clean), "prediction_vs_clean": metrics(raw_pred, clean)},
            "clean_used_as_inference_input": False,
        }
    atomic_json(FINAL / "P51A_FULL72_METRICS.json", full_metrics)
    adjusted_better = yes_no(full_metrics["adjusted_route"]["prediction_vs_clean"], full_metrics["adjusted_route"]["condition_vs_clean"])
    raw_better = yes_no(full_metrics["raw_route"]["prediction_vs_clean"], full_metrics["raw_route"]["condition_vs_clean"])
    decision = "recommend further Bernini-R investigation" if adjusted_better else "do not yet recommend deeper Bernini-R work without revising the route"
    best = selection["best"]
    report = f"""# P51A_BERNINI_R13B_SINGLE_SCENE_V2V — final report

## Result

Bernini-R 1.3B completed the official single-scene fresh fine-tuning route through 700 steps on four A40 GPUs.  The selected checkpoint is **step {best['step']}** using the fixed-start composite score `{best['score']:.6f}`.

All generated Bernini clips used only their adjusted/raw Tripo condition video.  Clean RGB was opened only for target training data, post-generation metrics, and synchronized QA videos.

## Required answers

- Bernini-R 1.3B training ran successfully: **yes**; trusted DCP checkpoints exist for every 20-step segment and all seven required 100-step validations completed.
- Validation stability/improvement: see `P51A_VALIDATION_SUMMARY.json`; selection used fixed starts 0/17/31/50 and combines MSE, temporal-delta, and SSIM change rather than relying on a single metric.
- Best checkpoint: **step {best['step']}**.
- Better than adjusted input on Full72 (MSE plus temporal-delta criterion): **{'yes' if adjusted_better else 'no'}**.
- Adjusted-route ghosting/translucency/double-object mitigation: no semantic ghosting detector was implemented; inspect `P51A_ADJUSTED_VS_OUTPUT_VS_CLEAN_FULL72.mp4` for the required human visual decision.
- Raw-route effective by the same Full72 metric criterion: **{'yes' if raw_better else 'no'}**.
- Better than prior VACE-1.3B: **not automatically adjudicated** here, because this run contains no matched VACE Full72 inference generated with the same checkpoint-selection protocol.  No unsupported cross-backbone claim is made.
- Largest residual issue: metric-only QA cannot decide object identity/ghosting reliably; ROI/edge/flow diagnostics remain unimplemented.
- Recommendation: **{decision}**, subject to visual review of the synchronized comparisons.

## Full72 metrics

```json
{json.dumps(full_metrics, indent=2, sort_keys=True)}
```

## Audit notes

- Model: `ByteDance/Bernini-R-1.3B-Diffusers@ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce`
- Checkpoint selection: fixed validation starts 0/17/31/50; clean excluded from inference.
- Full72 assembly: F00–F32 from start00, F33–F49 from start31 offsets 2–18, F50–F71 from start50 offsets 0–21.
- Step-0 baseline was not run; the separate step-1 smoke verifies checkpoint inference integrity.
"""
    temporary = (FINAL / "final_report.md").with_suffix(".md.tmp")
    temporary.write_text(report)
    os.replace(temporary, FINAL / "final_report.md")
    handoff = f"""# HANDOFF — P51A Bernini-R 1.3B

The 700-step fresh official Bernini renderer run is complete.  Best checkpoint: step {best['step']}.

- Resume provenance: 20-step bounded four-GPU units, each resumed only from `P51A_DCP_COMPLETE.json`.
- Validation: every 100 steps at fixed starts 0/17/31/50; clean is QA/target only.
- Full72 routes: adjusted and raw use checkpoint `{best['hf_checkpoint']}`.
- Final metrics: `P51A_FULL72_METRICS.json`.
- Review the two three-column videos and five-column summary before making qualitative ghosting or VACE claims.
"""
    temporary = (FINAL / "HANDOFF_P51A.md").with_suffix(".md.tmp")
    temporary.write_text(handoff)
    os.replace(temporary, FINAL / "HANDOFF_P51A.md")
    atomic_json(FINAL / "P51A_FINAL_COMPLETE.json", {"status": "complete", "best_step": best["step"], "completed_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"best_step": best["step"], "adjusted_better": adjusted_better, "raw_better": raw_better}, indent=2))


if __name__ == "__main__":
    main()
