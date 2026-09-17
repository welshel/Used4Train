#!/usr/bin/env python3
"""P48.3 ellipsoid-, footprint-, and attribution-aware dynamic Gaussian cleanup.

This program uses the immutable P48.1 cleaned PLY and exact accepted P48
camera/alignment contract.  It adds only current-view continuous opacity
attenuation.  No camera, Sim(3), scene scale, or base Gaussian membership is
changed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P48 = load_module("p48_exact_p483", ROOT / "scripts" / "render_tripo_clean_path.py")
P482 = load_module("p482_shared_p483", ROOT / "scripts" / "p48_2_dynamic_suppression.py")

FPS = 12.0
ALPHA_HIT = 1e-3
NEAR_PLANE = 0.05
ELLIPSOID_K = 3.0
FRONT_EXTENT_CANDIDATE_M = 0.22
FOOTPRINT_PERCENTILES = (98.0, 99.7)
EMA_ALPHA = 0.65
# Original targets plus frames that required current-render residual attribution.
HARD_FRAMES = [10, 11, 12, 13, 14, 16, 17, 18, 21, 24, 28, 29]
TARGET_FRAMES = [11, 12, 13]
QA_COLUMNS = [
    "frame", "white_foreground_occlusion", "floater_present", "new_hole",
    "important_content_removed", "temporal_transition", "manual_status", "notes",
]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def quat_to_rot_wxyz(quats: np.ndarray) -> np.ndarray:
    q = np.asarray(quats, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = (q[:, i] for i in range(4))
    out = np.empty((len(q), 3, 3), np.float64)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def ellipsoid_front_depth(means_cam: np.ndarray, scales: np.ndarray, quats: np.ndarray, world_to_camera_R: np.ndarray, k=ELLIPSOID_K) -> np.ndarray:
    """Return ``mu_z - k*sigma_z`` for oriented 3-D Gaussian ellipsoids."""
    R_local_world = quat_to_rot_wxyz(quats)
    z_axis_in_local = np.einsum("j,njk->nk", np.asarray(world_to_camera_R, np.float64)[2], R_local_world)
    sigma_z = np.sqrt(np.sum((z_axis_in_local * np.asarray(scales, np.float64)) ** 2, axis=1))
    return (np.asarray(means_cam, np.float64)[:, 2] - float(k) * sigma_z).astype(np.float32)


def ellipsoid_dynamic_risk(front_depth: np.ndarray, projected_area_ratio: np.ndarray, opacity: np.ndarray, candidate: np.ndarray, footprint_lo: float, footprint_hi: float) -> np.ndarray:
    """Continuous per-view risk; near alone or large alone is insufficient."""
    front = np.asarray(front_depth, np.float32)
    area = np.asarray(projected_area_ratio, np.float32)
    alpha = np.asarray(opacity, np.float32)
    valid = np.asarray(candidate, bool)
    near_risk = np.clip((0.25 - front) / 0.25, 0.0, 1.0)
    log_area = np.log(np.maximum(area, 1e-12))
    footprint = np.clip((log_area - math.log(max(footprint_lo, 1e-12))) / max(math.log(max(footprint_hi, footprint_lo * 1.0001)) - math.log(max(footprint_lo, 1e-12)), 1e-8), 0.0, 1.0)
    # A projected alpha/area proxy is the runtime occlusion term.  The later
    # attribution pass replaces this proxy with actual raster contribution for
    # each visually flagged frame.
    occlusion_mass = alpha * np.sqrt(np.maximum(area, 0.0) / max(footprint_lo, 1e-12))
    occlusion = np.clip((occlusion_mass - 0.30) / 0.70, 0.0, 1.0)
    risk = near_risk * np.maximum(footprint, occlusion)
    return np.where(valid, risk, 0.0).astype(np.float32)


def continuous_ema(raw_scores: np.ndarray, alpha=EMA_ALPHA) -> np.ndarray:
    """Causal continuous EMA with immediate danger preservation and soft exit."""
    raw = np.asarray(raw_scores, np.float32)
    if raw.ndim != 2:
        raise ValueError("raw_scores must be [frames, candidates]")
    out = np.zeros_like(raw)
    previous = np.zeros(raw.shape[1], np.float32)
    for frame in range(raw.shape[0]):
        ema = float(alpha) * raw[frame] + (1.0 - float(alpha)) * previous
        # Never delay a current dangerous front extent, but let it decay
        # continuously after exit rather than through a binary hysteresis mask.
        previous = np.maximum(raw[frame], ema).astype(np.float32)
        out[frame] = previous
    return out


def restrict_to_candidates(weights: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(candidates, bool), np.asarray(weights, np.float32), 0.0).astype(np.float32)


def apply_attribution_boost(weights: np.ndarray, candidates: np.ndarray, verified_ids, gamma: float = 4.0) -> np.ndarray:
    """Continuously strengthen only attribution-verified trajectory candidates."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    out = restrict_to_candidates(weights, candidates)
    selected = np.zeros(len(candidates), bool)
    ids = np.asarray(list(verified_ids), np.int64)
    ids = ids[(ids >= 0) & (ids < len(selected))]
    if len(ids):
        selected[ids] = True
    selected &= np.asarray(candidates, bool)
    if out.ndim == 1:
        out[selected] = 1.0 - (1.0 - out[selected]) ** float(gamma)
    else:
        out[:, selected] = 1.0 - (1.0 - out[:, selected]) ** float(gamma)
    return restrict_to_candidates(out, candidates)

def apply_frame_envelope_boost(weights: np.ndarray, candidates: np.ndarray, strength: np.ndarray, gamma: float = 4.0) -> np.ndarray:
    """Use a continuous trajectory envelope without enabling non-candidates."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    envelope = np.clip(np.asarray(strength, np.float32), 0.0, 1.0)
    if np.asarray(weights).ndim != 2 or len(envelope) != np.asarray(weights).shape[0]:
        raise ValueError("strength must have one value per frame")
    exponent = 1.0 + (float(gamma) - 1.0) * envelope
    out = restrict_to_candidates(weights, candidates)
    selected = np.asarray(candidates, bool)
    out[:, selected] = 1.0 - (1.0 - out[:, selected]) ** exponent[:, None]
    return restrict_to_candidates(out, candidates)

def apply_post_suppression_attribution_floor(weights: np.ndarray, candidates: np.ndarray, frame_to_ids: dict, floor: float = 0.92) -> np.ndarray:
    """EMA-smooth residual occluder suppression found from dynamic raster attribution."""
    out = restrict_to_candidates(weights, candidates)
    raw = np.zeros_like(out)
    for frame, ids in frame_to_ids.items():
        if not 0 <= int(frame) < len(raw):
            continue
        chosen = np.asarray(list(ids), np.int64)
        chosen = chosen[(chosen >= 0) & (chosen < len(candidates))]
        chosen = chosen[np.asarray(candidates, bool)[chosen]]
        raw[int(frame), chosen] = float(floor)
    smoothed = continuous_ema(raw)
    return restrict_to_candidates(np.maximum(out, smoothed), candidates)
def projected_area(radii: np.ndarray, width: int, height: int) -> np.ndarray:
    r = np.asarray(radii, np.float32)
    return (np.pi * r[:, 0] * r[:, 1] / float(width * height)).astype(np.float32)


def camera_space(means: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(camera.c2w).astype(np.float32)
    homo = np.concatenate([means, np.ones((len(means), 1), np.float32)], axis=1)
    return (homo @ w2c.T)[:, :3].astype(np.float32), w2c[:3, :3]


def render_info(means, rgb, opacity, scales, quats, camera, device="cuda", colors=None):
    """Single gsplat render returning RGB/features, alpha, and projection info."""
    import torch
    from gsplat import rasterization

    dev = torch.device(device)
    feature = rgb if colors is None else colors
    mt, qt, st, ot, ct = [torch.from_numpy(x).to(dev) for x in (means, quats, scales, opacity, feature)]
    view = torch.linalg.inv(torch.from_numpy(camera.c2w).to(dev)).reshape(1, 4, 4)
    K = torch.from_numpy(camera.K).to(dev).reshape(1, 3, 3)
    with torch.inference_mode():
        rendered, alpha, info = rasterization(mt, qt, st, ot, ct, view, K, camera.width, camera.height, sh_degree=None, render_mode="RGB", packed=True, eps2d=1e-8, near_plane=float(NEAR_PLANE))
    image = rendered[0].detach().cpu().numpy().astype(np.float32)
    alpha_np = alpha[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    packed = {
        "gaussian_ids": info["gaussian_ids"].detach().cpu().numpy(),
        "radii": info["radii"].detach().cpu().numpy(),
        "depths": info["depths"].detach().cpu().numpy(),
    }
    del mt, qt, st, ot, ct, view, K, rendered, alpha, info
    torch.cuda.empty_cache()
    return np.clip(image, 0, 1), np.clip(alpha_np, 0, 1), packed


def footprint_distribution(means, rgb, opacity, scales, quats, cameras, device="cuda") -> dict:
    """First trajectory pass: empirical area distribution, no guessed threshold."""
    samples = []
    for frame, camera in enumerate(cameras):
        _, _, info = render_info(means, rgb, opacity, scales, quats, camera, device)
        area = projected_area(info["radii"], camera.width, camera.height)
        if len(area) > 8192:
            samples.append(area[:: max(1, len(area) // 8192)][:8192])
        else:
            samples.append(area)
    values = np.concatenate(samples)
    lo, hi = (float(np.percentile(values, q)) for q in FOOTPRINT_PERCENTILES)
    return {
        "definition": "pi * projected_major_radius_px * projected_minor_radius_px / (W*H), from actual gsplat projection radii",
        "sample_count": int(len(values)),
        "quantiles": {str(q): float(np.percentile(values, q)) for q in [50, 90, 95, 97, 98, 99, 99.5, 99.7, 99.9, 100]},
        "candidate_area_ratio_lo": lo,
        "risk_area_ratio_hi": hi,
    }


def build_candidate_set(means, rgb, opacity, scales, quats, cameras, footprint, device="cuda") -> tuple[np.ndarray, dict]:
    """Second pass: trajectory-only candidate set from ellipsoid front + area."""
    n = len(means)
    candidate = np.zeros(n, bool)
    hit_count = np.zeros(n, np.int16)
    min_front = np.full(n, np.inf, np.float32)
    max_area = np.zeros(n, np.float32)
    lo = float(footprint["candidate_area_ratio_lo"])
    for frame, camera in enumerate(cameras):
        _, _, info = render_info(means, rgb, opacity, scales, quats, camera, device)
        ids = info["gaussian_ids"]
        means_cam, w2c_R = camera_space(means[ids], camera)
        front = ellipsoid_front_depth(means_cam, scales[ids], quats[ids], w2c_R, ELLIPSOID_K)
        area = projected_area(info["radii"], camera.width, camera.height)
        hit = (front <= FRONT_EXTENT_CANDIDATE_M) & (area >= lo)
        candidate[ids[hit]] = True
        np.add.at(hit_count, ids[hit], 1)
        np.minimum.at(min_front, ids, front)
        np.maximum.at(max_area, ids, area)
    return candidate, {
        "base_gaussian_count": int(n),
        "candidate_count": int(candidate.sum()),
        "candidate_ratio": float(candidate.mean()),
        "rule": f"exists F00-F71 with z_front({ELLIPSOID_K}sigma) <= {FRONT_EXTENT_CANDIDATE_M}m AND projected_area_ratio >= global P{FOOTPRINT_PERCENTILES[0]}",
        "candidate_hit_count_quantiles": {str(q): float(np.percentile(hit_count[candidate], q)) for q in [0, 50, 90, 99, 100]} if candidate.any() else {},
        "candidate_min_front_depth_quantiles": {str(q): float(np.percentile(min_front[candidate], q)) for q in [0, 50, 90, 100]} if candidate.any() else {},
        "candidate_max_area_ratio_quantiles": {str(q): float(np.percentile(max_area[candidate], q)) for q in [0, 50, 90, 99, 100]} if candidate.any() else {},
    }


def scan_risk_sequence(means, rgb, opacity, scales, quats, cameras, candidates, footprint, device="cuda"):
    """Third pass: per-frame ellipsoid-aware risk and unmodified base renders."""
    n = len(means)
    raw = np.zeros((len(cameras), n), np.float32)
    base_coverage = []
    base_hard = {}
    per_frame = {}
    lo = float(footprint["candidate_area_ratio_lo"])
    hi = float(footprint["risk_area_ratio_hi"])
    for frame, camera in enumerate(cameras):
        image, alpha, info = render_info(means, rgb, opacity, scales, quats, camera, device)
        ids = info["gaussian_ids"]
        means_cam, w2c_R = camera_space(means[ids], camera)
        front = ellipsoid_front_depth(means_cam, scales[ids], quats[ids], w2c_R, ELLIPSOID_K)
        area = projected_area(info["radii"], camera.width, camera.height)
        local = ellipsoid_dynamic_risk(front, area, opacity[ids], candidates[ids], lo, hi)
        np.maximum.at(raw[frame], ids, local)
        base_coverage.append(float((alpha > ALPHA_HIT).mean()))
        active = local >= 0.05
        per_frame[str(frame)] = {
            "risked_candidate_count": int(active.sum()),
            "mean_risk": float(local[active].mean()) if active.any() else 0.0,
            "max_risk": float(local.max()) if len(local) else 0.0,
            "front_depth_min_m": float(front.min()) if len(front) else None,
            "area_ratio_max": float(area.max()) if len(area) else None,
        }
        if frame in HARD_FRAMES:
            base_hard[frame] = (image, alpha)
    return raw, P482.coverage(base_coverage), base_hard, per_frame


def render_dynamic(means, rgb, opacity, scales, quats, cameras, weights, frames, device="cuda"):
    return P482.render_sequence(means, rgb, opacity, scales, quats, cameras, weights, frames, "full", device)


def attribution_pass(means, rgb, opacity, scales, quats, cameras, raw_scores, base_hard, candidates, footprint, out_dir: Path, device="cuda") -> dict:
    """Actual raster feature attribution for the top frame-local candidates.

    Every candidate in a 32-channel batch receives a one-hot feature.  With
    all original opacities still present, feature rasterization records its
    alpha/transmittance-weighted contribution through the real renderer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for frame in HARD_FRAMES:
        ranked = np.flatnonzero((raw_scores[frame] > 0) & candidates)
        ranked = ranked[np.argsort(raw_scores[frame, ranked])[::-1][:32]]
        camera = cameras[frame]
        base_rgb = base_hard[frame][0]
        white_region = base_rgb.mean(axis=2) >= 0.45
        means_cam, w2c_R = camera_space(means[ranked], camera) if len(ranked) else (np.empty((0, 3), np.float32), np.eye(3))
        front = ellipsoid_front_depth(means_cam, scales[ranked], quats[ranked], w2c_R, ELLIPSOID_K) if len(ranked) else np.empty(0, np.float32)
        per_id = []
        top_value = np.zeros(white_region.shape, np.float32)
        top_id = np.full(white_region.shape, -1, np.int64)
        # 32 channels is the documented performant gsplat feature limit.
        for offset in range(0, len(ranked), 32):
            chunk = ranked[offset:offset + 32]
            features = np.zeros((len(means), len(chunk)), np.float32)
            features[chunk, np.arange(len(chunk))] = 1.0
            feature_image, _, _ = render_info(means, rgb, opacity, scales, quats, camera, device, colors=features)
            local_index = feature_image.argmax(axis=2)
            local_value = feature_image.max(axis=2)
            better = local_value > top_value
            top_value[better] = local_value[better]
            top_id[better] = chunk[local_index[better]]
            for col, gaussian_id in enumerate(chunk):
                contribution = feature_image[..., col]
                in_white = contribution[white_region]
                per_id.append({
                    "gaussian_id": int(gaussian_id),
                    "visible_pixel_count_alpha_gt_1e-3": int((contribution > 1e-3).sum()),
                    "actual_alpha_contribution_sum": float(contribution.sum()),
                    "actual_alpha_contribution_mean": float(contribution.mean()),
                    "actual_alpha_contribution_max": float(contribution.max()),
                    "white_region_contribution_sum": float(in_white.sum()),
                    "white_region_top1_pixel_count": int(((top_id == gaussian_id) & white_region).sum()),
                    "raw_dynamic_risk": float(raw_scores[frame, gaussian_id]),
                })
        per_id.sort(key=lambda x: (-x["white_region_contribution_sum"], -x["actual_alpha_contribution_sum"]))
        lookup = {int(g): i for i, g in enumerate(ranked)}
        for item in per_id:
            idx = lookup[item["gaussian_id"]]
            item["z_front_3sigma_m"] = float(front[idx])
        payload = {
            "frame": f"F{frame:02d}",
            "method": "32-channel one-hot gsplat feature render; non-candidate opacity is retained so each feature is alpha/transmittance weighted by the real scene.",
            "white_region_definition": "base P48.1 rendered luminance >= 0.45",
            "white_region_pixel_count": int(white_region.sum()),
            "ranked_candidate_count": int(len(ranked)),
            "top_culprits": per_id[:20],
        }
        write_json(out_dir / f"P48_3_FRAME_F{frame:02d}_CULPRIT_GAUSSIANS.json", payload)
        summary[str(frame)] = {
            "ranked_candidate_count": int(len(ranked)),
            "top_culprit_ids": [x["gaussian_id"] for x in per_id[:10]],
            "top_white_region_contribution": float(per_id[0]["white_region_contribution_sum"]) if per_id else 0.0,
        }
    write_json(out_dir / "P48_3_FRAME_ATTRIBUTION_SUMMARY.json", summary)
    return summary


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).save(path)


def labelled(image: np.ndarray, label: str, size=(224, 224)) -> Image.Image:
    tile = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, max(70, 7 * len(label) + 8), 17), fill=(0, 0, 0))
    draw.text((3, 3), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return tile


def make_hard_board(path: Path, raw_dir: Path, p481_dir: Path, p482_dir: Path, dynamic: dict, clean_dir: Path) -> None:
    w = h = 224
    canvas = Image.new("RGB", (5 * w, 34 + len(HARD_FRAMES) * h), (8, 10, 15))
    ImageDraw.Draw(canvas).text((8, 10), "P48.3 ellipsoid-aware dynamic cleanup", fill=(225, 240, 255), font=ImageFont.load_default())
    for row, frame in enumerate(HARD_FRAMES):
        images = [
            np.asarray(Image.open(raw_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0,
            np.asarray(Image.open(p481_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0,
            np.asarray(Image.open(p482_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0,
            dynamic[frame][0],
            np.asarray(Image.open(clean_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0,
        ]
        labels = [f"F{frame:02d} raw", f"F{frame:02d} P48.1", f"F{frame:02d} P48.2", f"F{frame:02d} P48.3", f"F{frame:02d} clean"]
        for col, (image, label) in enumerate(zip(images, labels)):
            canvas.paste(labelled(image, label, (w, h)), (col * w, 34 + row * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_review_sheets(out_dir: Path, dynamic_dir: Path, clean_dir: Path) -> list[Path]:
    """Create four inspection sheets with one 448px P48.3/Clean pair per row."""
    paths = []
    for start in [0, 18, 36, 54]:
        frames = list(range(start, start + 18))
        w, h = 448, 448
        canvas = Image.new("RGB", (2 * w, 34 + 18 * h), (8, 10, 15))
        ImageDraw.Draw(canvas).text((8, 10), f"P48.3 manual review F{start:02d}-F{start + 17:02d}: P48.3 | Clean", fill=(225, 240, 255), font=ImageFont.load_default())
        for row, frame in enumerate(frames):
            dynamic = np.asarray(Image.open(dynamic_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0
            clean = np.asarray(Image.open(clean_dir / f"F{frame:02d}.png").convert("RGB"), np.float32) / 255.0
            canvas.paste(labelled(dynamic, f"F{frame:02d} P48.3", (w, h)), (0, 34 + row * h))
            canvas.paste(labelled(clean, f"F{frame:02d} clean", (w, h)), (w, 34 + row * h))
        path = out_dir / f"P48_3_REVIEW_SHEET_{start:02d}_{start + 17:02d}.png"
        canvas.save(path)
        paths.append(path)
    return paths


def encode_video(pattern: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(pattern), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", str(output)], check=True)


def build_videos(out: Path, p481_dir: Path, p482_dir: Path, dynamic_dir: Path, clean_dir: Path) -> None:
    review = out / "06_full72" / "review_p481_p482_p483_clean"
    p482p483 = out / "06_full72" / "p482_p483"
    dynamic_clean = out / "06_full72" / "dynamic_clean"
    for directory in [review, p482p483, dynamic_clean]:
        directory.mkdir(parents=True, exist_ok=True)
    for frame in range(72):
        images = {}
        for key, directory in [("p481", p481_dir), ("p482", p482_dir), ("p483", dynamic_dir), ("clean", clean_dir)]:
            image = Image.open(directory / f"F{frame:02d}.png").convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 42, 17), fill=(0, 0, 0))
            draw.text((3, 3), f"F{frame:02d}", fill=(255, 255, 255), font=ImageFont.load_default())
            images[key] = np.asarray(image)
        Image.fromarray(np.concatenate([images["p481"], images["p482"], images["p483"], images["clean"]], axis=1)).save(review / f"F{frame:02d}.png")
        Image.fromarray(np.concatenate([images["p482"], images["p483"]], axis=1)).save(p482p483 / f"F{frame:02d}.png")
        Image.fromarray(np.concatenate([images["p483"], images["clean"]], axis=1)).save(dynamic_clean / f"F{frame:02d}.png")
    final = out / "final"
    encode_video(review / "F%02d.png", final / "P48_3_REVIEW_FULL72.mp4")
    encode_video(p482p483 / "F%02d.png", final / "P48_3_P48_2_VS_P48_3.mp4")
    encode_video(dynamic_clean / "F%02d.png", final / "P48_3_DYNAMIC_VS_CLEAN.mp4")


def make_timeline(path: Path, weights: np.ndarray, candidates: np.ndarray) -> None:
    top = np.argsort(weights.max(axis=0))[::-1][:6]
    plot = np.full((560, 1200, 3), 18, np.uint8)
    cv2.putText(plot, "P48.3 continuous suppression weights (top trajectory-risk candidates)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 235, 235), 1, cv2.LINE_AA)
    colors = [(70, 230, 70), (70, 170, 255), (220, 170, 60), (230, 80, 220), (60, 220, 220), (220, 220, 220)]
    for j, idx in enumerate(top):
        curve = weights[:, idx]
        points = np.array([[42 + int(t * 15.5), 490 - int(v * 380)] for t, v in enumerate(curve)], np.int32)
        cv2.polylines(plot, [points], False, colors[j], 2, cv2.LINE_AA)
        cv2.putText(plot, f"id {int(idx)}", (920, 65 + 24 * j), cv2.FONT_HERSHEY_SIMPLEX, .48, colors[j], 1, cv2.LINE_AA)
    cv2.putText(plot, "frame", (1120, 515), cv2.FONT_HERSHEY_SIMPLEX, .45, (200, 200, 200), 1, cv2.LINE_AA)
    Image.fromarray(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)).save(path)


def temporal_report(raw: np.ndarray, weights: np.ndarray, candidates: np.ndarray) -> dict:
    active = weights > 0.05
    delta = np.abs(np.diff(weights, axis=0))
    iou = []
    for f in range(71):
        a, b = active[f], active[f + 1]
        union = int((a | b).sum())
        iou.append(float((a & b).sum() / union) if union else 1.0)
    return {
        "method": f"continuous EMA alpha={EMA_ALPHA}; output=max(raw risk, EMA) so genuine danger is never binary-gated away",
        "raw_active_count_per_frame": (raw > 0.05).sum(axis=1).astype(int).tolist(),
        "smoothed_active_count_per_frame": active.sum(axis=1).astype(int).tolist(),
        "mean_suppression_per_frame": weights.mean(axis=1).astype(float).tolist(),
        "max_suppression_count": int(active.sum(axis=1).max()),
        "max_suppression_frame": int(active.sum(axis=1).argmax()),
        "risk_temporal_variation_mean_abs": float(delta.mean()),
        "suppression_mask_iou_adjacent_mean": float(np.mean(iou)),
        "suppression_mask_iou_adjacent_min": float(np.min(iou)),
        "gate": {
            "max_active_fraction": float(active.mean(axis=1).max()),
            "max_mean_weight_delta": float(np.abs(np.diff(weights.mean(axis=1))).max()),
            "pass": bool(active.mean(axis=1).max() <= 0.02 and np.abs(np.diff(weights.mean(axis=1))).max() <= 0.01),
        },
    }


def write_draft_manual_qa(path: Path, dynamic: dict, base_hard: dict) -> None:
    """Create an explicit review-required CSV; final labels are never default PASS."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QA_COLUMNS)
        writer.writeheader()
        for frame in range(72):
            writer.writerow({
                "frame": f"F{frame:02d}", "white_foreground_occlusion": "REVIEW",
                "floater_present": "REVIEW", "new_hole": "REVIEW",
                "important_content_removed": "REVIEW", "temporal_transition": "REVIEW",
                "manual_status": "REVIEW", "notes": "Awaiting individual visual comparison of P48.3 against Clean.",
            })


def validate_manual_qa_csv(path: Path) -> dict:
    """Validate explicit human F00-F71 annotations without fabricating a pass."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != QA_COLUMNS:
            raise ValueError(f"Manual QA columns must be exactly {QA_COLUMNS}")
        rows = list(reader)
    expected = [f"F{frame:02d}" for frame in range(72)]
    frames = [row.get("frame", "") for row in rows]
    errors = []
    if sorted(frames) != expected:
        errors.append("CSV must contain each of F00-F71 exactly once")
    allowed = {
        "white_foreground_occlusion": {"NONE", "MILD", "MODERATE", "SEVERE"},
        "floater_present": {"YES", "NO"},
        "new_hole": {"NONE", "MILD", "SEVERE"},
        "important_content_removed": {"YES", "NO"},
        "temporal_transition": {"PASS", "BORDERLINE", "FAIL"},
        "manual_status": {"PASS", "REVIEW", "FAIL"},
    }
    white = {key: 0 for key in ["NONE", "MILD", "MODERATE", "SEVERE"]}
    blocking = []
    for row in rows:
        frame = row.get("frame", "")
        for key, values in allowed.items():
            value = row.get(key, "")
            if value not in values:
                errors.append(f"{frame}: invalid {key}={value!r}")
        level = row.get("white_foreground_occlusion", "")
        if level in white:
            white[level] += 1
        if (
            level in {"MODERATE", "SEVERE"}
            or row.get("floater_present") == "YES"
            or row.get("new_hole") == "SEVERE"
            or row.get("important_content_removed") == "YES"
            or row.get("temporal_transition") != "PASS"
            or row.get("manual_status") != "PASS"
        ):
            blocking.append(frame)
    blocking = sorted(set(blocking))
    return {
        "frames_checked": len(rows),
        "white_foreground_occlusion": white,
        "blocking_frames": blocking,
        "validation_errors": errors,
        "eligible_for_finalization": not errors and not blocking and len(rows) == 72,
    }


def finalization_status(qa: dict, coverage: dict, temporal: dict) -> str:
    if not qa.get("eligible_for_finalization", False):
        return "P48_3_RUNTIME_FAIL"
    coverage_gate = coverage.get("gate", {})
    if float(coverage_gate.get("mean_drop", float("inf"))) > float(coverage_gate.get("hard", 0.02)):
        return "P48_3_FAIL_COVERAGE_REGRESSION"
    if not temporal.get("gate", {}).get("pass", False):
        return "P48_3_FAIL_TEMPORAL_POPPING"
    return "P48_3_PASS_MINOR_MILD_RESIDUAL" if qa["white_foreground_occlusion"].get("MILD", 0) else "P48_3_PASS"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat", type=Path, default=ROOT / "outputs" / "p48_1_tripo_cleanup" / "final" / "splat_cleaned.ply")
    ap.add_argument("--alignment", type=Path, default=ROOT / "outputs" / "p48_tripo_clean_aligned" / "P48_TRIPO_ALIGNMENT.json")
    ap.add_argument("--camera-metadata", type=Path, default=ROOT / "work" / "327431980_4_normalized")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "p48_3_tripo_dynamic_cleanup")
    ap.add_argument("--p48-output", type=Path, default=ROOT / "outputs" / "p48_tripo_clean_aligned")
    ap.add_argument("--p481-output", type=Path, default=ROOT / "outputs" / "p48_1_tripo_cleanup")
    ap.add_argument("--p482-output", type=Path, default=ROOT / "outputs" / "p48_2_tripo_dynamic_suppression")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--finalize-manual-qa", action="store_true")
    ap.add_argument("--manual-qa", type=Path)
    ap.add_argument("--attribution-boost-manifest", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.finalize_manual_qa:
        final = args.output / "final"
        qa_path = args.manual_qa or final / "P48_3_FRAME_BY_FRAME_MANUAL_QA.csv"
        qa = validate_manual_qa_csv(qa_path)
        coverage = json.loads((final / "P48_3_COVERAGE_METRICS.json").read_text())
        temporal = json.loads((final / "P48_3_TEMPORAL_STABILITY.json").read_text())
        status = finalization_status(qa, coverage, temporal)
        payload = {
            "FINAL_STATUS": status,
            "APPROVED_FOR_DOWNSTREAM_CONDITION_TRAINING": status.startswith("P48_3_PASS"),
            "base_splat": str(args.splat),
            "alignment_modified": False,
            "camera_modified": False,
            "method": "ellipsoid 3sigma + projected footprint + actual attribution; continuous EMA full-render opacity suppression",
            "manual_qa": qa,
            "coverage": coverage,
            "temporal": temporal,
        }
        write_json(final / "P48_3_FINALIZATION_AUDIT.json", payload)
        (final / "final_report.md").write_text("# P48.3 Final Report\n\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        (final / "HANDOFF_P48_3.md").write_text("# P48.3 Handoff\n\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n\nReproduce: `python scripts/p48_3_dynamic_gaussian_cleanup.py`\n\nFinalize: `python scripts/p48_3_dynamic_gaussian_cleanup.py --finalize-manual-qa`\n")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    out = args.output
    boost_manifest = args.attribution_boost_manifest or out / "03_risk_analysis" / "P48_3_ATTRIBUTION_VERIFIED_BOOST_IDS.json"
    boost_payload = json.loads(boost_manifest.read_text()) if boost_manifest.is_file() else {}
    boost_ids = boost_payload.get("boost_gaussian_ids", [])
    boost_gamma = float(boost_payload.get("continuous_boost_gamma", 1.0))
    frame_boost_gamma = float(boost_payload.get("frame_boost_gamma", boost_gamma))
    frame_boost_strength = np.asarray(boost_payload.get("frame_boost_strength", [0.0] * 72), np.float32)
    if frame_boost_strength.shape != (72,): raise ValueError("frame_boost_strength must contain exactly 72 values")
    supplemental_candidate_ids = boost_payload.get("supplemental_candidate_ids", [])
    residual_floor_by_frame = {int(frame): ids for frame, ids in boost_payload.get("post_suppression_residual_floor_by_frame", {}).items()}
    residual_floor = float(boost_payload.get("post_suppression_residual_floor", 0.0))
    folders = {name: out / name for name in ["00_handoff", "01_asset_audit", "02_gaussian_attribution", "03_risk_analysis", "04_dynamic_renderer", "05_iteration_qa", "06_full72", "07_manual_review", "08_metrics", "final"]}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    alignment = json.loads(args.alignment.read_text())
    M = np.asarray(alignment["transform"], np.float64)
    xyz, rgb, opacity, scales, quats, audit = P48.load_ply(args.splat)
    cameras = P48.load_clean_cameras(args.camera_metadata)
    sim_scale = float(np.cbrt(np.linalg.det(M[:3, :3])))
    means = P48.apply_similarity(xyz, M).astype(np.float32)
    world_scales = (scales * abs(sim_scale)).astype(np.float32)
    world_quats = P48.transform_quaternions(quats, M)
    asset = {"base_splat": str(args.splat), "base_gaussian_count": int(len(means)), "alignment": str(args.alignment), "alignment_transform": M.tolist(), "alignment_modified": False, "camera_modified": False, "global_pruning_in_p48_3": False, "near_plane_m": NEAR_PLANE, "ellipsoid_k_sigma": ELLIPSOID_K}
    write_json(folders["01_asset_audit"] / "P48_3_ASSET_AUDIT.json", asset)

    footprint = footprint_distribution(means, rgb, opacity, world_scales, world_quats, cameras, args.device)
    write_json(folders["03_risk_analysis"] / "P48_3_PROJECTED_FOOTPRINT_DISTRIBUTION.json", footprint)
    candidates, candidate_stats = build_candidate_set(means, rgb, opacity, world_scales, world_quats, cameras, footprint, args.device)
    supplemental = np.asarray(supplemental_candidate_ids, np.int64)
    supplemental = supplemental[(supplemental >= 0) & (supplemental < len(candidates))]
    candidates[supplemental] = True
    candidate_stats.update({"base_ellipsoid_candidate_count": candidate_stats["candidate_count"], "supplemental_residual_candidate_count": int(len(supplemental)), "candidate_count": int(candidates.sum()), "candidate_ratio": float(candidates.mean())})
    candidate_payload = {
        "base": asset, "footprint_distribution": footprint, "candidate_set": candidate_stats,
        "candidate_ids": np.flatnonzero(candidates).astype(int).tolist(),
        "attribution_verified_boost": {"manifest": str(boost_manifest), "id_count": len(boost_ids), "gamma": boost_gamma},
        "frame_envelope": {"nonzero_frames": np.flatnonzero(frame_boost_strength).astype(int).tolist(), "max_strength": float(frame_boost_strength.max())},
    }
    write_json(folders["03_risk_analysis"] / "P48_3_DYNAMIC_CANDIDATE_SET.json", candidate_payload)
    write_json(folders["final"] / "P48_3_CANDIDATE_GAUSSIANS.json", candidate_payload)

    raw, p481_coverage, base_hard, per_frame = scan_risk_sequence(means, rgb, opacity, world_scales, world_quats, cameras, candidates, footprint, args.device)
    weights = apply_attribution_boost(restrict_to_candidates(continuous_ema(raw), candidates[None, :]), candidates, boost_ids, boost_gamma)
    weights = apply_frame_envelope_boost(weights, candidates, frame_boost_strength, frame_boost_gamma)
    weights = apply_post_suppression_attribution_floor(weights, candidates, residual_floor_by_frame, residual_floor)
    temporal = temporal_report(raw, weights, candidates)
    risk_stats = {
        "formula": "risk=near_risk(z_front_3sigma)*max(projected_footprint_risk, projected_alpha_area_occlusion_proxy)",
        "per_frame": per_frame, "temporal": temporal,
        "attribution_verified_boost": {"manifest": str(boost_manifest), "id_count": len(boost_ids), "gamma": boost_gamma, "formula": "w=1-(1-w_ema)^gamma for verified candidates only"},
        "frame_envelope": {"nonzero_frames": np.flatnonzero(frame_boost_strength).astype(int).tolist(), "max_strength": float(frame_boost_strength.max()), "gamma": frame_boost_gamma, "formula": "w=1-(1-w)^((1+(gamma-1)*strength_t)) for candidates with nonzero view risk"},
        "post_suppression_residual_attribution": {"frames": sorted(residual_floor_by_frame), "floor": residual_floor, "supplemental_candidate_count": int(len(supplemental)), "formula": "w=max(w_dynamic, EMA(actual-residual-attribution floor))"},
    }
    write_json(folders["03_risk_analysis"] / "P48_3_DYNAMIC_RISK_STATS.json", risk_stats)
    write_json(folders["final"] / "P48_3_DYNAMIC_SUPPRESSION_STATS.json", risk_stats)
    write_json(folders["final"] / "P48_3_TEMPORAL_STABILITY.json", temporal)
    make_timeline(folders["final"] / "P48_3_SUPPRESSION_TIMELINE.png", weights, candidates)
    if args.analyze_only:
        print(json.dumps({"status": "ANALYZED", "candidate": candidate_stats, "footprint": footprint, "temporal": temporal}, indent=2))
        return

    dynamic_hard = render_dynamic(means, rgb, opacity, world_scales, world_quats, cameras, weights, HARD_FRAMES, args.device)
    dynamic = render_dynamic(means, rgb, opacity, world_scales, world_quats, cameras, weights, list(range(72)), args.device)
    dynamic_dir = folders["06_full72"] / "condition_rgb_dynamic"
    for frame in range(72):
        save_png(dynamic_dir / f"F{frame:02d}.png", dynamic[frame][0])
    dynamic_coverage = P482.coverage([float((dynamic[i][1] > ALPHA_HIT).mean()) for i in range(72)])
    p482_coverage = json.loads((args.p482_output / "final" / "P48_2_COVERAGE_METRICS.json").read_text())
    coverage = {
        "p48_1_guarded": p481_coverage,
        "p48_2": p482_coverage["dynamic"],
        "p48_3": dynamic_coverage,
        "drop_vs_p48_1": {k: float(p481_coverage[k] - dynamic_coverage[k]) for k in ["min", "p10", "median", "mean", "max"]},
    }
    coverage["gate"] = {"target": 0.01, "hard": 0.02, "mean_drop": coverage["drop_vs_p48_1"]["mean"], "pass": bool(coverage["drop_vs_p48_1"]["mean"] <= 0.02)}
    write_json(folders["08_metrics"] / "P48_3_COVERAGE_METRICS.json", coverage)
    write_json(folders["final"] / "P48_3_COVERAGE_METRICS.json", coverage)

    attribution = attribution_pass(means, rgb, opacity, world_scales, world_quats, cameras, raw, base_hard, candidates, footprint, folders["02_gaussian_attribution"], args.device)
    write_json(folders["final"] / "P48_3_FRAME_ATTRIBUTION_SUMMARY.json", attribution)
    raw_dir = args.p48_output / "condition_rgb"
    p481_dir = args.p481_output / "condition_rgb_cleaned"
    p482_dir = args.p482_output / "condition_rgb_dynamic"
    clean_dir = args.p48_output / "clean_rgb"
    make_hard_board(folders["final"] / "P48_3_HARDEST_VIEWS_BOARD.png", raw_dir, p481_dir, p482_dir, dynamic_hard, clean_dir)
    make_review_sheets(folders["final"], dynamic_dir, clean_dir)
    build_videos(out, p481_dir, p482_dir, dynamic_dir, clean_dir)
    encode_video(dynamic_dir / "F%02d.png", folders["final"] / "P48_3_DYNAMIC_CONDITION_FULL72.mp4")
    write_draft_manual_qa(folders["final"] / "P48_3_FRAME_BY_FRAME_MANUAL_QA.csv", dynamic_hard, base_hard)
    pending = {"FINAL_STATUS": "P48_3_RUNTIME_FAIL", "reason": "Full72 rendered; manual F00-F71 review must be finalized before a PASS may be issued.", "asset": asset, "coverage": coverage, "temporal": temporal, "attribution": attribution}
    (folders["final"] / "final_report.md").write_text("# P48.3 pending manual review\n\n" + json.dumps(pending, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "RENDERED_AWAITING_MANUAL_REVIEW", "output": str(out), "coverage": coverage, "temporal": temporal}, indent=2))


if __name__ == "__main__":
    main()
