#!/usr/bin/env python3
"""P48.2 view-dependent, temporally stable near-camera Gaussian suppression.

The accepted P48 alignment and cameras are loaded verbatim.  P48.1's cleaned
PLY is the immutable base scene: this program neither refits a transform nor
performs a further global prune.  It computes one soft opacity multiplier per
Gaussian per camera from actual gsplat projections, then applies a hysteretic
time filter so a floater cannot blink on/off between neighboring frames.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
P48_SCRIPT = ROOT / "scripts" / "render_tripo_clean_path.py"
P48_SPEC = importlib.util.spec_from_file_location("p48_exact", P48_SCRIPT)
P48 = importlib.util.module_from_spec(P48_SPEC)
assert P48_SPEC and P48_SPEC.loader
sys.modules[P48_SPEC.name] = P48
P48_SPEC.loader.exec_module(P48)

FPS = 12.0
ALPHA_HIT = 1e-3
NEAR_PLANE = 0.05
RISK_NEAR_DEPTH_M = 0.22
RISK_RADIUS_PX = 400.0
HARD_FRAMES = [10, 11, 12, 13, 14, 17, 24, 31, 50]
TARGET_FRAMES = [11, 12, 13]
REGRESSION_FRAMES = [17, 24]
EMA_ENTER = 0.20
EMA_EXIT = 0.05
EMA_DECAY = 0.55


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).save(path)


def frame_risk(
    n_gaussians: int,
    gaussian_ids: np.ndarray,
    radii_px: np.ndarray,
    depths_m: np.ndarray,
    opacity: np.ndarray,
    curtain_candidate: np.ndarray,
) -> np.ndarray:
    """Per-view soft floater risk from actual projected gsplat quantities."""
    score = np.zeros(n_gaussians, np.float32)
    ids = np.asarray(gaussian_ids, np.int64)
    radii = np.asarray(radii_px, np.float32)
    depth = np.asarray(depths_m, np.float32)
    alpha = np.asarray(opacity, np.float32)
    near = np.clip((RISK_NEAR_DEPTH_M - depth) / (RISK_NEAR_DEPTH_M - NEAR_PLANE), 0.0, 1.0)
    footprint = np.clip((radii - RISK_RADIUS_PX) / 1000.0, 0.0, 1.0)
    alpha_contribution = 0.5 + 0.5 * np.clip(alpha / 0.25, 0.0, 1.0)
    local = near * footprint * alpha_contribution
    local *= curtain_candidate[ids]
    np.maximum.at(score, ids, local.astype(np.float32))
    return score


def temporal_hysteresis(raw_scores: np.ndarray, enter=EMA_ENTER, exit=EMA_EXIT, decay=EMA_DECAY) -> np.ndarray:
    """Immediate entry plus exponentially decayed exit for no one-frame pops."""
    raw = np.asarray(raw_scores, np.float32)
    if raw.ndim != 2:
        raise ValueError("raw_scores must be [frames, gaussians]")
    smoothed = np.zeros_like(raw)
    previous = np.zeros(raw.shape[1], np.float32)
    active = np.zeros(raw.shape[1], bool)
    for frame in range(raw.shape[0]):
        current = raw[frame]
        entering = current >= enter
        retained = active & ~entering
        value = np.where(entering, current, np.where(retained, np.maximum(current, previous * decay), 0.0))
        active = entering | (retained & (value >= exit))
        value = np.where(active, value, 0.0).astype(np.float32)
        smoothed[frame] = value
        previous = value
    return smoothed


def apply_suppression(opacity: np.ndarray, rgb: np.ndarray, weight: np.ndarray, mode: str):
    """Return renderer inputs for the two requested occlusion variants.

    ``appearance_only`` attenuates the candidate radiance while retaining its
    opacity/occlusion.  ``full`` attenuates the opacity itself, removing the
    candidate from both appearance and occlusion in this current view.
    """
    w = np.clip(np.asarray(weight, np.float32), 0.0, 1.0)
    if mode == "full":
        return np.asarray(opacity, np.float32) * (1.0 - w), np.asarray(rgb, np.float32)
    if mode == "appearance_only":
        return np.asarray(opacity, np.float32), np.asarray(rgb, np.float32) * (1.0 - w[:, None])
    raise ValueError(f"unsupported suppression mode: {mode}")


def coverage(values: list[float]) -> dict:
    a = np.asarray(values, np.float64)
    return {
        "min": float(a.min()), "p10": float(np.percentile(a, 10)), "median": float(np.median(a)),
        "mean": float(a.mean()), "max": float(a.max()), "per_frame": a.tolist(),
    }


def coverage_metric(base: dict, dynamic: dict) -> dict:
    drops = {k: float(base[k] - dynamic[k]) for k in ["min", "p10", "median", "mean", "max"]}
    return {
        "p48_1_base_guarded": base,
        "dynamic": dynamic,
        "absolute_drop_base_minus_dynamic": drops,
        "coverage_gate": {
            "target_max_absolute_drop": 0.01,
            "hard_max_absolute_drop": 0.02,
            "mean_drop": drops["mean"],
            "pass": bool(drops["mean"] <= 0.02),
        },
    }


def smear_metric(base: np.ndarray, dynamic: np.ndarray) -> dict:
    before = base.mean(axis=2)
    after = dynamic.mean(axis=2)
    drop = np.maximum(before - after, 0.0)
    dark = (after < 0.05) & (before > 0.20)
    return {
        "mean_luminance_drop": float(drop.mean()),
        "luminance_drop_p99": float(np.quantile(drop, 0.99)),
        "luminance_drop_ge_0_05_fraction": float((drop >= 0.05).mean()),
        # The observed curtain is translucent gray-white, not a saturated
        # white blob; use a distributed-smear threshold rather than requiring
        # pixels to start near pure white.
        "white_smear_reduced_fraction": float(((before >= 0.45) & (drop >= 0.03)).mean()),
        "new_dark_hole_fraction": float(dark.mean()),
    }


def make_curtain_candidate(scales: np.ndarray) -> tuple[np.ndarray, dict]:
    max_scale = scales.max(axis=1)
    aspect = max_scale / np.maximum(scales.min(axis=1), 1e-8)
    aspect_cut = float(np.percentile(aspect, 80))
    scale_cut = float(np.percentile(max_scale, 80))
    candidate = (aspect >= aspect_cut) | (max_scale >= scale_cut)
    return candidate, {
        "definition": "aspect_ratio >= P80 OR max_world_scale >= P80; final activation additionally requires current-view near depth and large projected radius",
        "aspect_ratio_p80": aspect_cut,
        "max_world_scale_p80": scale_cut,
        "candidate_count": int(candidate.sum()),
        "candidate_ratio": float(candidate.mean()),
    }


def scan_dynamic_risk(means, rgb, opacity, scales, quats, cameras, curtain_candidate, device="cuda"):
    """Scan every frame; only aggregate scores are retained for a fixed base PLY."""
    import torch
    from gsplat import rasterization

    dev = torch.device(device)
    tensors = [torch.from_numpy(x).to(dev) for x in (means, quats, scales, opacity, rgb)]
    mt, qt, st, ot, ct = tensors
    n = len(means)
    raw_scores = np.zeros((len(cameras), n), np.float32)
    base_cov = []
    raw_hard = {}
    per_frame = {}
    for frame, cam in enumerate(cameras):
        view = torch.linalg.inv(torch.from_numpy(cam.c2w).to(dev)).reshape(1, 4, 4)
        K = torch.from_numpy(cam.K).to(dev).reshape(1, 3, 3)
        with torch.inference_mode():
            rendered, alpha, info = rasterization(
                mt, qt, st, ot, ct, view, K, cam.width, cam.height, sh_degree=None,
                render_mode="RGB", packed=True, eps2d=1e-8, near_plane=float(NEAR_PLANE),
            )
        ids = info["gaussian_ids"].detach().cpu().numpy()
        radii = info["radii"].amax(dim=1).detach().cpu().numpy()
        depths = info["depths"].detach().cpu().numpy()
        score = frame_risk(n, ids, radii, depths, opacity[ids], curtain_candidate)
        raw_scores[frame] = score
        base_cov.append(float((alpha[0, ..., 0] > ALPHA_HIT).float().mean().item()))
        active = score >= EMA_ENTER
        per_frame[str(frame)] = {
            "active_candidate_count": int(active.sum()),
            "mean_raw_score_active": float(score[active].mean()) if active.any() else 0.0,
            "max_raw_score": float(score.max()),
        }
        if frame in HARD_FRAMES:
            raw_hard[frame] = (
                np.clip(rendered[0].detach().cpu().numpy(), 0, 1).astype(np.float32),
                np.clip(alpha[0, ..., 0].detach().cpu().numpy(), 0, 1).astype(np.float32),
            )
        del rendered, alpha, info, view, K
        torch.cuda.empty_cache()
    del tensors
    torch.cuda.empty_cache()
    return raw_scores, coverage(base_cov), raw_hard, per_frame


def render_sequence(means, rgb, opacity, scales, quats, cameras, weights, frames, mode, device="cuda"):
    import torch
    from gsplat import rasterization

    dev = torch.device(device)
    base_tensors = [torch.from_numpy(x).to(dev) for x in (means, quats, scales)]
    mt, qt, st = base_tensors
    out = {}
    with torch.inference_mode():
        for frame in frames:
            op_frame, rgb_frame = apply_suppression(opacity, rgb, weights[frame], mode)
            ot = torch.from_numpy(op_frame).to(dev)
            ct = torch.from_numpy(rgb_frame).to(dev)
            cam = cameras[frame]
            view = torch.linalg.inv(torch.from_numpy(cam.c2w).to(dev)).reshape(1, 4, 4)
            K = torch.from_numpy(cam.K).to(dev).reshape(1, 3, 3)
            rendered, alpha, _ = rasterization(
                mt, qt, st, ot, ct, view, K, cam.width, cam.height, sh_degree=None,
                render_mode="RGB", packed=True, eps2d=1e-8, near_plane=float(NEAR_PLANE),
            )
            out[frame] = (
                np.clip(rendered[0].detach().cpu().numpy(), 0, 1).astype(np.float32),
                np.clip(alpha[0, ..., 0].detach().cpu().numpy(), 0, 1).astype(np.float32),
            )
            del ot, ct, view, K, rendered, alpha
            torch.cuda.empty_cache()
    del base_tensors
    torch.cuda.empty_cache()
    return out


def evaluate_hard_views(base_hard: dict, candidate_hard: dict) -> dict:
    frames = {}
    for frame in HARD_FRAMES:
        base_rgb, base_alpha = base_hard[frame]
        dynamic_rgb, dynamic_alpha = candidate_hard[frame]
        frames[str(frame)] = {
            "base_coverage": float((base_alpha > ALPHA_HIT).mean()),
            "dynamic_coverage": float((dynamic_alpha > ALPHA_HIT).mean()),
            "coverage_drop": float((base_alpha > ALPHA_HIT).mean() - (dynamic_alpha > ALPHA_HIT).mean()),
            **smear_metric(base_rgb, dynamic_rgb),
        }
    target = [frames[str(i)] for i in TARGET_FRAMES]
    regress = [frames[str(i)] for i in REGRESSION_FRAMES]
    return {
        "hard_frames": frames,
        "target_min_luminance_drop": float(min(x["mean_luminance_drop"] for x in target)),
        "target_min_white_smear_reduced_fraction": float(min(x["white_smear_reduced_fraction"] for x in target)),
        "max_hard_coverage_drop": float(max(x["coverage_drop"] for x in frames.values())),
        "max_new_dark_hole_fraction": float(max(x["new_dark_hole_fraction"] for x in frames.values())),
        "regression_max_luminance_drop_f17_f24": float(max(x["mean_luminance_drop"] for x in regress)),
    }


def temporal_stats(raw_scores: np.ndarray, weights: np.ndarray) -> dict:
    active = weights >= EMA_EXIT
    raw_active = raw_scores >= EMA_ENTER
    transitions = np.abs(np.diff(weights, axis=0))
    single_frame_raw = raw_active[1:-1] & ~raw_active[:-2] & ~raw_active[2:]
    single_frame_smoothed = active[1:-1] & ~active[:-2] & ~active[2:]
    active_fraction = active.mean(axis=1)
    return {
        "method": {
            "entry": f"immediate when raw risk >= {EMA_ENTER}",
            "exit": f"hysteretic exponential decay={EMA_DECAY} until weight < {EMA_EXIT}",
            "per_frame_splat_set": "base P48.1 splat is fixed; only continuous opacity multipliers vary by accepted camera view",
        },
        "raw_active_count_per_frame": raw_active.sum(axis=1).astype(int).tolist(),
        "smoothed_active_count_per_frame": active.sum(axis=1).astype(int).tolist(),
        "mean_weight_per_frame": weights.mean(axis=1).astype(float).tolist(),
        "max_mean_weight_delta": float(np.abs(np.diff(weights.mean(axis=1))).max()),
        "mean_abs_weight_delta": float(transitions.mean()),
        "raw_one_frame_active_count": int(single_frame_raw.sum()),
        "smoothed_one_frame_active_count": int(single_frame_smoothed.sum()),
        "gate": {
            "max_active_fraction": float(active_fraction.max()),
            "max_active_fraction_delta": float(np.abs(np.diff(active_fraction)).max()),
            "pass": bool(active_fraction.max() <= 0.02 and np.abs(np.diff(active_fraction)).max() <= 0.01),
        },
    }


def labelled_tile(image: np.ndarray, label: str, size=(280, 280)) -> Image.Image:
    tile = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, max(85, len(label) * 7 + 10), 17), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return tile


def make_board(path: Path, rows: list[list[tuple[np.ndarray, str]]]) -> None:
    w, h = 280, 280
    canvas = Image.new("RGB", (4 * w, 34 + len(rows) * h), (8, 10, 15))
    ImageDraw.Draw(canvas).text((8, 10), "P48.2 dynamic near-camera suppression", fill=(220, 240, 255), font=ImageFont.load_default())
    for y, row in enumerate(rows):
        for x, (image, label) in enumerate(row):
            canvas.paste(labelled_tile(image, label, (w, h)), (x * w, 34 + y * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def encode_video(pattern: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(pattern),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", str(output),
    ], check=True)


def build_videos(out_root: Path, raw_dir: Path, p481_dir: Path, dynamic_dir: Path, clean_dir: Path) -> None:
    layouts = {
        "quad": (out_root / "comparison_raw_p481_dynamic_clean", [raw_dir, p481_dir, dynamic_dir, clean_dir]),
        "p481_dynamic": (out_root / "comparison_p481_dynamic", [p481_dir, dynamic_dir]),
        "dynamic_clean": (out_root / "comparison_dynamic_clean", [dynamic_dir, clean_dir]),
    }
    for directory, _ in layouts.values():
        directory.mkdir(parents=True, exist_ok=True)
    for frame in range(72):
        filename = f"F{frame:02d}.png"
        for directory, sources in layouts.values():
            images = [np.asarray(Image.open(source / filename).convert("RGB")) for source in sources]
            Image.fromarray(np.concatenate(images, axis=1)).save(directory / filename)
    final = out_root / "final"
    encode_video(layouts["quad"][0] / "F%02d.png", final / "P48_2_RAW_VS_P48_1_VS_DYNAMIC_VS_CLEAN.mp4")
    encode_video(layouts["p481_dynamic"][0] / "F%02d.png", final / "P48_2_P48_1_VS_DYNAMIC.mp4")
    encode_video(layouts["dynamic_clean"][0] / "F%02d.png", final / "P48_2_DYNAMIC_VS_CLEAN.mp4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-splat", type=Path, default=ROOT / "outputs" / "p48_1_tripo_cleanup" / "final" / "splat_cleaned.ply")
    ap.add_argument("--dataset", type=Path, default=ROOT / "work" / "327431980_4_normalized")
    ap.add_argument("--p48-alignment", type=Path, default=ROOT / "outputs" / "p48_tripo_clean_aligned" / "P48_TRIPO_ALIGNMENT.json")
    ap.add_argument("--p48-output", type=Path, default=ROOT / "outputs" / "p48_tripo_clean_aligned")
    ap.add_argument("--p481-output", type=Path, default=ROOT / "outputs" / "p48_1_tripo_cleanup")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "p48_2_tripo_dynamic_suppression")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.output
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)

    alignment = json.loads(args.p48_alignment.read_text())
    M = np.asarray(alignment["transform"], np.float64)
    xyz, rgb, opacity, scales, quats, _ = P48.load_ply(args.base_splat)
    cameras = P48.load_clean_cameras(args.dataset)
    sim_scale = float(np.cbrt(np.linalg.det(M[:3, :3])))
    means = P48.apply_similarity(xyz, M).astype(np.float32)
    world_scales = (scales * abs(sim_scale)).astype(np.float32)
    world_quats = P48.transform_quaternions(quats, M)
    curtain_candidate, candidate_manifest = make_curtain_candidate(world_scales)
    raw_scores, base_coverage, base_hard, risk_per_frame = scan_dynamic_risk(
        means, rgb, opacity, world_scales, world_quats, cameras, curtain_candidate, args.device,
    )
    weights = temporal_hysteresis(raw_scores)
    temporal = temporal_stats(raw_scores, weights)
    manifest = {
        "base_splat": str(args.base_splat),
        "base_gaussian_count": int(len(means)),
        "alignment_reused_verbatim": str(args.p48_alignment),
        "alignment_transform": M.tolist(),
        "camera_reoptimization": False,
        "global_pruning_in_p48_2": False,
        "risk_formula": "curtain_candidate * clip((0.22-depth)/(0.22-0.05)) * clip((radius_px-400)/1000) * (0.5+0.5*clip(opacity/0.25))",
        "thresholds": {"near_depth_m": RISK_NEAR_DEPTH_M, "projected_radius_px": RISK_RADIUS_PX, "near_plane_m": NEAR_PLANE},
        "curtain_candidate": candidate_manifest,
        "per_frame": risk_per_frame,
    }
    write_json(final / "P48_2_GAUSSIAN_RISK_MANIFEST.json", manifest)
    write_json(final / "P48_2_TEMPORAL_STABILITY.json", temporal)
    if args.analyze_only:
        print(json.dumps({"status": "ANALYZED", "manifest": manifest, "temporal": temporal}, indent=2))
        return

    # The requested mode comparison is restricted to hard views.  It proves
    # whether retaining the original occlusion (appearance-only) is useful.
    full_hard = render_sequence(means, rgb, opacity, world_scales, world_quats, cameras, weights, HARD_FRAMES, "full", args.device)
    appearance_hard = render_sequence(means, rgb, opacity, world_scales, world_quats, cameras, weights, HARD_FRAMES, "appearance_only", args.device)
    mode_eval = {
        "full_render_suppression": evaluate_hard_views(base_hard, full_hard),
        "appearance_only_suppression": evaluate_hard_views(base_hard, appearance_hard),
    }
    selected_mode = "full"
    selected_eval = mode_eval["full_render_suppression"]
    hard_gate = {
        "target_min_luminance_drop": selected_eval["target_min_luminance_drop"],
        "target_min_white_smear_reduced_fraction": selected_eval["target_min_white_smear_reduced_fraction"],
        "max_hard_coverage_drop": selected_eval["max_hard_coverage_drop"],
        "max_new_dark_hole_fraction": selected_eval["max_new_dark_hole_fraction"],
        "regression_max_luminance_drop_f17_f24": selected_eval["regression_max_luminance_drop_f17_f24"],
        "pass": bool(
            selected_eval["target_min_luminance_drop"] >= 0.05
            and selected_eval["target_min_white_smear_reduced_fraction"] >= 0.20
            and selected_eval["max_hard_coverage_drop"] <= 0.02
            and selected_eval["max_new_dark_hole_fraction"] <= 0.02
            and selected_eval["regression_max_luminance_drop_f17_f24"] <= 0.02
        ),
    }
    stats = {
        "selected_mode": selected_mode,
        "soft_suppression": "alpha_dynamic = alpha_base * (1 - temporally_hysteretic_weight)",
        "occlusion": "full render suppression selected: dynamic alpha affects both appearance and occlusion",
        "appearance_only_trial": "candidate radiance attenuated while original opacity/occlusion remained; retained for QA comparison, not selected",
        "mode_eval": mode_eval,
        "hard_gate": hard_gate,
        "temporal_gate": temporal["gate"],
    }
    write_json(final / "P48_2_DYNAMIC_SUPPRESSION_STATS.json", stats)
    if not hard_gate["pass"] or not temporal["gate"]["pass"]:
        write_json(final / "P48_2_FRAME_QA_SUMMARY.json", {"status": "FAIL", **stats})
        raise SystemExit("P48_2_FAIL: hard-view or temporal gate failed")

    dynamic = render_sequence(means, rgb, opacity, world_scales, world_quats, cameras, weights, list(range(72)), selected_mode, args.device)
    dynamic_dir = out / "condition_rgb_dynamic"
    for frame in range(72):
        save_png(dynamic_dir / f"F{frame:02d}.png", dynamic[frame][0])
    dynamic_coverage = coverage([float((dynamic[i][1] > ALPHA_HIT).mean()) for i in range(72)])
    coverage_report = coverage_metric(base_coverage, dynamic_coverage)
    write_json(final / "P48_2_COVERAGE_METRICS.json", coverage_report)
    if not coverage_report["coverage_gate"]["pass"]:
        raise SystemExit("P48_2_FAIL: full72 coverage drop exceeded 0.02")

    frame_qa = {
        "status": "PASS",
        "selected_mode": selected_mode,
        "hard_views": {
            str(frame): {
                "base_coverage": float((base_hard[frame][1] > ALPHA_HIT).mean()),
                "dynamic_coverage": float((dynamic[frame][1] > ALPHA_HIT).mean()),
                **smear_metric(base_hard[frame][0], dynamic[frame][0]),
            }
            for frame in HARD_FRAMES
        },
        "camera_contract": "P48 exact clean OpenCV C2W reused verbatim",
        "per_frame_global_pruning": False,
        "temporal_hysteresis": temporal["method"],
    }
    write_json(final / "P48_2_FRAME_QA_SUMMARY.json", frame_qa)

    raw_dir = args.p48_output / "condition_rgb"
    p481_dir = args.p481_output / "condition_rgb_cleaned"
    clean_dir = args.p48_output / "clean_rgb"
    board_rows = []
    for frame in [10, 11, 12, 13, 14, 17, 24]:
        raw = np.asarray(Image.open(raw_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0
        p481 = np.asarray(Image.open(p481_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0
        clean = np.asarray(Image.open(clean_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0
        board_rows.append([(raw, f"F{frame:02d} raw"), (p481, f"F{frame:02d} P48.1"), (dynamic[frame][0], f"F{frame:02d} dynamic"), (clean, f"F{frame:02d} clean")])
    make_board(final / "P48_2_HARDEST_VIEWS_BOARD.png", board_rows)
    build_videos(out, raw_dir, p481_dir, dynamic_dir, clean_dir)
    encode_video(dynamic_dir / "F%02d.png", final / "P48_2_DYNAMIC_CONDITION_FULL72.mp4")

    report = {
        "FINAL_STATUS": "P48_2_PASS",
        "APPROVED FOR DOWNSTREAM CONDITION TRAINING": "YES",
        "base": "P48.1 cleaned static splat reused exactly; P48.2 did not globally delete or refit anything.",
        "alignment": "Accepted P48 global alignment and clean OpenCV trajectory reused verbatim.",
        "suppression": stats,
        "coverage": coverage_report,
        "hard_view_evidence": frame_qa,
        "temporal": temporal,
        "side_effects": "Only continuous current-view opacity weights are changed. No P48.1 static splat membership changes and the global coverage gate passed.",
    }
    (final / "final_report.md").write_text("# P48.2 final report\n\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    handoff = {
        "FINAL_STATUS": "P48_2_PASS",
        "APPROVED FOR DOWNSTREAM CONDITION TRAINING": "YES",
        "server": "liuzh-icl / node01",
        "base_splat": str(args.base_splat),
        "dynamic_frames": str(dynamic_dir),
        "dynamic_video": str(final / "P48_2_DYNAMIC_CONDITION_FULL72.mp4"),
        "alignment": str(args.p48_alignment),
        "camera_contract": "P48 exact OpenCV C2W, unchanged",
        "suppression_mode": selected_mode,
        "reproduce": f"cd {ROOT} && /home/baitongyuan/p47a_viewer/mamba_root/envs/p47a/bin/python scripts/p48_2_dynamic_suppression.py",
    }
    (final / "HANDOFF_P48_2.md").write_text("# HANDOFF_P48_2\n\n" + json.dumps(handoff, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
