#!/usr/bin/env python3
"""Aggregate P49A validation, build QA artifacts, and optionally promote Full72."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAME_COUNT = 72
STARTS = (0, 17, 31, 50)
ELIGIBLE_STEPS = (200, 350, 500, 700)
TIMELINE_STEPS = (0, 50, 100, 200, 350, 500, 700)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def promote_best_checkpoint(root: Path, step: int):
    source = root / "06_checkpoints" / f"step-{step}.safetensors"
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    destination = root / "06_checkpoints" / "best.safetensors"
    shutil.copy2(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    result = {
        "best_step": step,
        "source_checkpoint": str(source),
        "best_checkpoint": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": digest,
    }
    write_json(root / "06_checkpoints" / "P49A_BEST_CHECKPOINT.json", result)
    return result


def image_array(path: Path, size=672):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def gradients(image):
    x = image.astype(np.float32) / 255.0
    gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    dx = np.diff(gray, axis=1, append=gray[:, -1:])
    dy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.sqrt(dx * dx + dy * dy)


def ssim(a, b):
    x = a.astype(np.float64).reshape(-1, 3) / 255.0
    y = b.astype(np.float64).reshape(-1, 3) / 255.0
    mux, muy = x.mean(axis=0), y.mean(axis=0)
    vx, vy = x.var(axis=0), y.var(axis=0)
    cov = ((x - mux) * (y - muy)).mean(axis=0)
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))
    return float(np.mean(score))


def exposure_match(output, target):
    source_mean = max(float(output.mean()), 1.0)
    matched = output.astype(np.float32) * (float(target.mean()) / source_mean)
    return np.clip(matched, 0, 255).astype(np.uint8)


def pair_metrics(a, b):
    af, bf = a.astype(np.float32), b.astype(np.float32)
    mse = float(np.mean((af - bf) ** 2))
    matched = exposure_match(a, b)
    ga, gb = gradients(a), gradients(b)
    edge_a, edge_b = ga > 0.08, gb > 0.08
    tp = int(np.logical_and(edge_a, edge_b).sum())
    denom = int(edge_a.sum()) + int(edge_b.sum())
    return {
        "mse": mse,
        "psnr": float("inf") if mse == 0 else float(20 * math.log10(255.0) - 10 * math.log10(mse)),
        "ssim": ssim(a, b),
        "exposure_matched_mse": float(np.mean((matched.astype(np.float32) - bf) ** 2)),
        "exposure_matched_ssim": ssim(matched, b),
        "hf_mse": float(np.mean((ga - gb) ** 2)),
        "hf_edge_f1": 1.0 if denom == 0 else float(2 * tp / denom),
        "gradient_error": float(np.mean(np.abs(ga - gb))),
    }


def average_dict(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def temporal_metrics(frames, clean):
    deltas = [float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))) / 255.0) for i in range(1, len(frames))]
    clean_deltas = [float(np.mean(np.abs(clean[i].astype(np.float32) - clean[i - 1].astype(np.float32))) / 255.0) for i in range(1, len(clean))]
    try:
        import cv2

        residuals = []
        for i in range(1, len(frames)):
            g0 = cv2.cvtColor(clean[i - 1], cv2.COLOR_RGB2GRAY)
            g1 = cv2.cvtColor(clean[i], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = g0.shape
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            warped = cv2.remap(frames[i - 1], (x + flow[..., 0]).astype(np.float32), (y + flow[..., 1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            residuals.append(float(np.mean(np.abs(warped.astype(np.float32) - frames[i].astype(np.float32))) / 255.0))
    except Exception:
        residuals = deltas
    return {
        "temporal_delta": float(np.mean(deltas)),
        "flicker": float(np.std(deltas)),
        "flow_warp_residual": float(np.mean(residuals)),
        "clean_temporal_delta": float(np.mean(clean_deltas)),
        "temporal_delta_ratio_to_clean": float(np.mean(deltas) / max(np.mean(clean_deltas), 1e-8)),
    }


def validation_dir(root: Path, step: int, start: int):
    name = f"step{step}_start{start}" if step else f"step0_base_start{start}"
    return root / "07_start31_validation" / name


def collect_validations(root: Path):
    records = []
    for path in sorted((root / "07_start31_validation").glob("*/metrics.json")):
        data = read_json(path)
        name = path.parent.name
        match = re.fullmatch(r"step(\d+)(?:_base)?_start(\d+)", name)
        if not match:
            continue
        records.append({"name": name, "step": int(match.group(1)), "start": int(match.group(2)), **data})
    return records


def rank_best(records):
    summaries = {}
    for step in ELIGIBLE_STEPS:
        subset = [r for r in records if r["step"] == step and r["start"] in STARTS]
        if len(subset) != 4:
            continue
        summaries[step] = {
            "step": step,
            "starts": {str(r["start"]): {"metrics": r["metrics"], "temporal": r["temporal"]} for r in subset},
            "mean_metrics": average_dict([r["metrics"] for r in subset]),
            "mean_temporal": average_dict([r["temporal"] for r in subset]),
        }
    if len(summaries) != len(ELIGIBLE_STEPS):
        raise RuntimeError(f"Expected four-start results for {ELIGIBLE_STEPS}, received {sorted(summaries)}")
    criteria = [
        ("mean_metrics", "mse", False),
        ("mean_metrics", "ssim", True),
        ("mean_metrics", "exposure_matched_ssim", True),
        ("mean_metrics", "hf_edge_f1", True),
        ("mean_metrics", "gradient_error", False),
        ("mean_temporal", "flicker", False),
        ("mean_temporal", "flow_warp_residual", False),
    ]
    totals = {s: 0.0 for s in summaries}
    for group, metric, higher in criteria:
        ordered = sorted(summaries, key=lambda s: summaries[s][group][metric], reverse=higher)
        for rank, step in enumerate(ordered, 1):
            totals[step] += rank
    for step in summaries:
        summaries[step]["aggregate_rank_sum"] = totals[step]
        vals = [summaries[step]["starts"][str(s)]["metrics"]["ssim"] for s in STARTS]
        summaries[step]["four_start_ssim_range"] = float(max(vals) - min(vals))
    best = min(summaries, key=lambda s: (totals[s], -summaries[s]["mean_metrics"]["ssim"]))
    return best, summaries


def assemble_best_full72(root: Path, step: int):
    accum = [None] * FRAME_COUNT
    weights = np.zeros(FRAME_COUNT, dtype=np.float64)
    provenance = {str(i): [] for i in range(FRAME_COUNT)}
    for start in STARTS:
        folder = validation_dir(root, step, start) / "frames"
        for pos in range(33):
            idx = (start + pos) % FRAME_COUNT
            path = folder / f"F{idx:02d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            weight = math.sin(math.pi * (pos + 1) / 34.0) ** 2 + 1e-4
            arr = image_array(path).astype(np.float64)
            accum[idx] = arr * weight if accum[idx] is None else accum[idx] + arr * weight
            weights[idx] += weight
            provenance[str(idx)].append({"start": start, "clip_position": pos, "weight": weight})
    if np.any(weights == 0):
        raise RuntimeError("Four-start windows do not cover all 72 frames")
    frames = [np.rint(np.clip(accum[i] / weights[i], 0, 255)).astype(np.uint8) for i in range(FRAME_COUNT)]
    return frames, provenance


def font(size=18):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def labeled(arr, label, width=260):
    im = Image.fromarray(arr).resize((width, width), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, width + 28), "white")
    canvas.paste(im, (0, 28))
    ImageDraw.Draw(canvas).text((6, 5), label, fill="black", font=font(15))
    return canvas


def save_grid(rows, path: Path, title: str):
    cell_w = max(im.width for row in rows for im in row)
    cell_h = max(im.height for row in rows for im in row)
    title_h = 42
    out = Image.new("RGB", (cell_w * max(len(r) for r in rows), title_h + cell_h * len(rows)), (20, 20, 20))
    ImageDraw.Draw(out).text((10, 10), title, fill="white", font=font(20))
    for y, row in enumerate(rows):
        for x, im in enumerate(row):
            out.paste(im, (x * cell_w, title_h + y * cell_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def crop_norm(arr, roi):
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = roi
    return arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def roi_report(condition, generated, clean, roi, frames):
    rows = []
    for idx in frames:
        c, g, t = (crop_norm(source[idx], roi) for source in (condition, generated, clean))
        cm, gm = pair_metrics(c, t), pair_metrics(g, t)
        rows.append({"frame": idx, "condition_vs_clean": cm, "vace_vs_clean": gm, "mse_improvement_fraction": float((cm["mse"] - gm["mse"]) / max(cm["mse"], 1e-8))})
    return {"roi_normalized": roi, "frames": rows, "mean_mse_improvement_fraction": float(np.mean([r["mse_improvement_fraction"] for r in rows]))}


def object_qa(condition, generated, clean, root):
    definitions = {
        "Tabletop": {"roi": (0.18, 0.56, 0.86, 1.0), "frames": (29, 31, 35, 40)},
        "Plant": {"roi": (0.0, 0.43, 0.46, 1.0), "frames": (29, 31, 35, 40)},
        "Curtain": {"roi": (0.0, 0.12, 0.35, 0.94), "frames": (11, 12, 13, 28, 29, 30)},
        "Window": {"roi": (0.0, 0.17, 0.48, 0.93), "frames": (24, 28, 31, 35)},
        "Furniture": {"roi": (0.18, 0.55, 1.0, 1.0), "frames": (31, 35, 40, 50)},
    }
    reports = {}
    for name, spec in definitions.items():
        reports[name] = roi_report(condition, generated, clean, spec["roi"], spec["frames"])
        board_rows = []
        for idx in spec["frames"]:
            board_rows.append([
                labeled(crop_norm(condition[idx], spec["roi"]), f"F{idx:02d} Condition"),
                labeled(crop_norm(generated[idx], spec["roi"]), f"F{idx:02d} VACE"),
                labeled(crop_norm(clean[idx], spec["roi"]), f"F{idx:02d} Clean"),
            ])
        save_grid(board_rows, root / "09_object_qa" / f"P49A_{name.upper()}_BOARD.png", f"P49A {name} QA")
    write_json(root / "09_object_qa" / "P49A_OBJECT_METRICS.json", reports)
    return reports


def geometry_qa(condition, generated, clean, root):
    candidates = []
    for idx in (11, 12, 13, 24, 28, 29, 30, 31, 35, 40, 50):
        for gy in range(4):
            for gx in range(4):
                roi = (gx / 4, gy / 4, (gx + 1) / 4, (gy + 1) / 4)
                c, t = crop_norm(condition[idx], roi), crop_norm(clean[idx], roi)
                pm = pair_metrics(c, t)
                score = pm["gradient_error"] * (1.0 + (1.0 - pm["hf_edge_f1"]))
                candidates.append((score, idx, roi, pm))
    selected, used = [], set()
    for score, idx, roi, cm in sorted(candidates, reverse=True):
        if idx in used:
            continue
        used.add(idx)
        g, t = crop_norm(generated[idx], roi), crop_norm(clean[idx], roi)
        gm = pair_metrics(g, t)
        improvement = float((cm["mse"] - gm["mse"]) / max(cm["mse"], 1e-8))
        selected.append({"frame": idx, "roi_normalized": roi, "condition_vs_clean": cm, "vace_vs_clean": gm, "mse_improvement_fraction": improvement, "selection_score": score})
        if len(selected) == 6:
            break
    mean_imp = float(np.mean([x["mse_improvement_fraction"] for x in selected]))
    verdict = "YES" if mean_imp >= 0.15 and sum(x["mse_improvement_fraction"] > 0 for x in selected) >= 5 else ("BORDERLINE" if mean_imp > 0 else "NO")
    report = {"selection": "top condition-vs-clean edge-disagreement patch, at most one patch per frame", "patches": selected, "mean_mse_improvement_fraction": mean_imp, "vace_corrected": verdict}
    write_json(root / "10_geometry_mismatch" / "P49A_GEOMETRY_MISMATCH_QA.json", report)
    rows = []
    for item in selected:
        idx, roi = item["frame"], item["roi_normalized"]
        rows.append([
            labeled(crop_norm(condition[idx], roi), f"F{idx:02d} Condition"),
            labeled(crop_norm(generated[idx], roi), f"F{idx:02d} VACE"),
            labeled(crop_norm(clean[idx], roi), f"F{idx:02d} Clean"),
        ])
    save_grid(rows, root / "10_geometry_mismatch" / "P49A_GEOMETRY_MISMATCH_BOARD.png", f"P49A geometry mismatch: {verdict}")
    return report


def mild_residual_qa(condition, generated, clean, root):
    roi = (0.0, 0.0, 0.24, 1.0)
    report = roi_report(condition, generated, clean, roi, (28, 29, 30))
    improvements = [r["mse_improvement_fraction"] for r in report["frames"]]
    verdict = "YES" if min(improvements) > 0.10 else ("BORDERLINE" if np.mean(improvements) > 0 else "NO")
    report["vace_removed"] = verdict
    write_json(root / "10_geometry_mismatch" / "P49A_MILD_RESIDUAL_QA.json", report)
    rows = []
    for idx in (28, 29, 30):
        rows.append([
            labeled(crop_norm(condition[idx], roi), f"F{idx:02d} P48.3"),
            labeled(crop_norm(generated[idx], roi), f"F{idx:02d} VACE"),
            labeled(crop_norm(clean[idx], roi), f"F{idx:02d} Clean"),
        ])
    save_grid(rows, root / "10_geometry_mismatch" / "P49A_MILD_RESIDUAL_BOARD.png", f"P49A F28-F30 mild residual: {verdict}")
    return report


def timeline_board(root, condition, clean):
    frames = (31, 40, 55)
    rows = []
    for idx in frames:
        row = [labeled(condition[idx], f"F{idx:02d} Condition", 190)]
        for step in TIMELINE_STEPS:
            folder = validation_dir(root, step, 31) / "frames"
            row.append(labeled(image_array(folder / f"F{idx:02d}.png"), f"step {step}", 190))
        row.append(labeled(clean[idx], "Clean", 190))
        rows.append(row)
    save_grid(rows, root / "09_object_qa" / "P49A_LEARNING_TIMELINE_BOARD.png", "P49A learning timeline (start31)")


def encode_frames(frame_dir: Path, output: Path, fps=12):
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i", str(frame_dir / "frame_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(output)], check=True)


def make_triptych(frames_a, frames_b, frames_c, indices, frame_dir: Path, labels):
    frame_dir.mkdir(parents=True, exist_ok=True)
    for seq, idx in enumerate(indices):
        cells = [labeled(source[idx], f"{labels[col]} F{idx:02d}", 360) for col, source in enumerate((frames_a, frames_b, frames_c))]
        out = Image.new("RGB", (sum(c.width for c in cells), max(c.height for c in cells)), "black")
        x = 0
        for cell in cells:
            out.paste(cell, (x, 0)); x += cell.width
        out.save(frame_dir / f"frame_{seq:03d}.png")


def combine_training_log(root: Path):
    initial = root / "06_checkpoints" / "P49A_RUNTIME_LOG.csv"
    segment_paths = []
    legacy = root / "06_checkpoints" / "P49A_RESUME_RUNTIME_LOG.csv"
    if legacy.is_file():
        segment_paths.append((150, legacy))
    for path in (root / "06_checkpoints").glob("P49A_RESUME_RUNTIME_LOG_FROM_*.csv"):
        match = re.search(r"FROM_(\d+)\.csv$", path.name)
        if match:
            segment_paths.append((int(match.group(1)), path))

    rows_by_step = {}
    with initial.open() as f:
        for row in csv.DictReader(f):
            step = int(row["global_step"])
            if 1 <= step <= 150:
                rows_by_step[step] = {**row, "segment": "initial"}
    # Later continuation points override replayed steps from earlier segments.
    for start_step, path in sorted(segment_paths):
        with path.open() as f:
            for row in csv.DictReader(f):
                step = int(row["global_step"])
                if start_step < step <= 700:
                    rows_by_step[step] = {
                        **row,
                        "segment": f"resume_from_step{start_step}_optimizer_reinitialized",
                    }
    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    if [int(r["global_step"]) for r in rows] != list(range(1, 701)):
        raise RuntimeError("Combined training log is not exactly global steps 1-700")
    out = root / "05_training" / "P49A_TRAINING_LOG.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", "phase", "loss", "learning_rate", "optimizer_step_sec", "segment"])
        writer.writeheader(); writer.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--condition-dir", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--promote-full72", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    condition_dir, target_dir = Path(args.condition_dir), Path(args.target_dir)
    records = collect_validations(root)
    if len(records) != 19:
        raise RuntimeError(f"Expected 19 unique validation runs, received {len(records)}")
    best, summaries = rank_best(records)
    best_checkpoint = promote_best_checkpoint(root, best)
    condition = [image_array(condition_dir / f"F{i:02d}.png") for i in range(FRAME_COUNT)]
    clean = [image_array(target_dir / f"F{i:02d}.png") for i in range(FRAME_COUNT)]
    generated, provenance = assemble_best_full72(root, best)
    combined_metrics = average_dict([pair_metrics(g, t) for g, t in zip(generated, clean)])
    combined_temporal = temporal_metrics(generated, clean)
    condition_metrics = average_dict([pair_metrics(c, t) for c, t in zip(condition, clean)])

    training_rows = combine_training_log(root)
    validation_report = {"validation_run_count": len(records), "records": records, "best_four_start_step": best, "best_checkpoint": best_checkpoint, "selection_method": "sum of ranks across seven global/temporal criteria", "best_full72_assembled_metrics": combined_metrics, "best_full72_temporal": combined_temporal, "condition_vs_clean_full72": condition_metrics, "target_used_for_generation": False}
    write_json(root / "07_start31_validation" / "P49A_VALIDATION_METRICS.json", validation_report)
    four = {"best_step": best, "steps": summaries, "assembly": "four fixed cyclic 33-frame starts with squared-sine overlap weights", "provenance": provenance}
    write_json(root / "08_four_start" / "P49A_FOUR_START_SUMMARY.json", four)

    object_report = object_qa(condition, generated, clean, root)
    geometry_report = geometry_qa(condition, generated, clean, root)
    residual_report = mild_residual_qa(condition, generated, clean, root)
    timeline_board(root, condition, clean)

    start31_indices = [(31 + i) % FRAME_COUNT for i in range(33)]
    best31 = {idx: image_array(validation_dir(root, best, 31) / "frames" / f"F{idx:02d}.png") for idx in start31_indices}
    best31_frames = [None] * FRAME_COUNT
    for idx, arr in best31.items(): best31_frames[idx] = arr
    trip_dir = root / "12_comparisons" / "start31_triptych_frames"
    make_triptych(condition, best31_frames, clean, start31_indices, trip_dir, ("Tripo P48.3", f"VACE step{best}", "Clean"))
    encode_frames(trip_dir, root / "12_comparisons" / "P49A_TRIPO_VS_VACE_VS_CLEAN_START31.mp4")
    shutil.copy2(validation_dir(root, 0, 31) / "validation.mp4", root / "07_start31_validation" / "P49A_STEP0_BASE_TRIPO_START31.mp4")

    promotion = {
        "best_step": best,
        "four_start_ssim_range": summaries[best]["four_start_ssim_range"],
        "geometry_corrected": geometry_report["vace_corrected"],
        "mild_residual_removed": residual_report["vace_removed"],
        "object_mean_mse_improvement": {k: v["mean_mse_improvement_fraction"] for k, v in object_report.items()},
        "automatic_camera_layout_identity_proxy": "PASS" if combined_metrics["ssim"] > condition_metrics["ssim"] and summaries[best]["four_start_ssim_range"] <= 0.10 else "FAIL",
        "manual_visual_review_required": True,
    }
    write_json(root / "08_four_start" / "P49A_PROMOTION_GATE_PRELIMINARY.json", promotion)

    if args.promote_full72:
        frames_dir = root / "11_full72" / "best_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, arr in enumerate(generated):
            Image.fromarray(arr).save(frames_dir / f"frame_{i:03d}.png")
        encode_frames(frames_dir, root / "11_full72" / "P49A_BEST_FULL72.mp4")
        trip72 = root / "12_comparisons" / "full72_triptych_frames"
        make_triptych(condition, generated, clean, range(FRAME_COUNT), trip72, ("Tripo P48.3", f"VACE step{best}", "Clean"))
        encode_frames(trip72, root / "12_comparisons" / "P49A_TRIPO_VS_VACE_VS_CLEAN_FULL72.mp4")
    print(json.dumps({"best_step": best, "promotion": promotion, "training_rows": len(training_rows)}, indent=2))


if __name__ == "__main__":
    main()
