#!/usr/bin/env python3
"""Phase A: conservative Clean-guided Gaussian geometry adjustment.

This is deliberately a *fixed-cardinality* teacher optimizer.  It starts from
the P48.1 cleaned PLY, keeps the P48 camera/alignment contract immutable, and
only nudges Gaussian means along camera rays using robust multi-view Clean
camera-Z measurements.  There is no prune, split, clone, or densification.

The optimizer is intentionally modest: points are moved only when projected
onto valid Clean depth, residuals are clipped, and displacement from the
P48.1 teacher is regularized.  Clean RGB/depth/edges are used for Phase-A
optimization and QA only; they are never emitted as an inference-time mask.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
P48 = None
FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
          "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
DTYPE = np.dtype([(x, "<f4") for x in FIELDS])
GATE = [0, 17, 31, 41, 50, 64, 71]
FRAME_COUNT = 72
ALPHA_HIT = 1e-3
DEPTH_EDGE_THRESHOLD = 0.03


def load_p48():
    global P48
    p = ROOT / "scripts" / "render_tripo_clean_path.py"
    spec = importlib.util.spec_from_file_location("phase_a_p48", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_a_p48"] = mod
    spec.loader.exec_module(mod)
    P48 = mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(v: Any):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def load_raw(path: Path):
    hb, n, header = P48._read_ply_header(path)
    names = [line.split()[2].decode() for line in header if line.startswith(b"property float ")]
    if names != FIELDS:
        raise ValueError(f"unexpected schema {names}")
    with path.open("rb") as f:
        f.seek(hb)
        raw = np.fromfile(f, dtype=DTYPE, count=n)
    if len(raw) != n:
        raise ValueError(f"truncated PLY: {len(raw)} != {n}")
    return raw.copy(), header, hb


def save_raw(path: Path, raw: np.ndarray, header: list[bytes]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.writelines(header)
        raw.tofile(f)


def load_depths(cameras):
    depth, valid, rgb, lines = {}, {}, {}, {}
    meta = []
    for i, cam in enumerate(cameras):
        cj = json.loads(Path(cam.json_path).read_text())
        dp = Path(cam.json_path).with_name(Path(cam.json_path).name.replace("_camera_para.json", "_depth.png"))
        if not dp.exists():
            raise FileNotFoundError(dp)
        raw = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        scale = float(cj.get("depth_scale", 0.0))
        if raw is None or raw.ndim != 2 or scale <= 0:
            raise ValueError(f"invalid metric depth contract: {dp}")
        z = raw.astype(np.float32) / scale
        m = (raw > 0) & np.isfinite(z) & (z > 0)
        im = np.asarray(Image.open(cam.rgb_path).convert("RGB"), np.uint8)
        # Clean line target is a derived QA/training asset, not an inference input.
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        edge_rgb = cv2.Canny(gray, 60, 150) > 0
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        edge_z = (np.sqrt(gx * gx + gy * gy) >= DEPTH_EDGE_THRESHOLD) & m
        lines[i] = edge_rgb | edge_z
        depth[i], valid[i], rgb[i] = z, m, im
        meta.append({"frame": i, "camera_json": str(cam.json_path), "depth": str(dp),
                     "depth_scale": scale, "valid_ratio": float(m.mean()), "sha256": sha256(dp)})
    return depth, valid, rgb, lines, meta


def project_world(points: np.ndarray, cam):
    R = cam.c2w[:3, :3].astype(np.float64).T
    t = cam.c2w[:3, 3].astype(np.float64)
    pc = (R @ (points.astype(np.float64) - t).T).T
    z = pc[:, 2]
    ok = z > 0.05
    u = cam.K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + cam.K[0, 2]
    v = cam.K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + cam.K[1, 2]
    return u, v, z, ok


def optimize_means(world0, scales, opacity, cameras, depth, valid, lines, frames, iters=2,
                   step=0.28, max_residual=0.65, max_disp=0.16):
    """Robust ray-depth correction; returns world means and detailed statistics."""
    world = world0.astype(np.float64).copy()
    n = len(world)
    total_updates = np.zeros(n, np.float64)
    frame_stats = []
    # Opacity is useful as a soft confidence but never hard-deletes a point.
    op = np.clip(opacity.astype(np.float64), 0.0, 1.0)
    # Use a conservative footprint confidence: very large splats are often floaters.
    smean = np.mean(np.asarray(scales, np.float64), axis=1)
    for iteration in range(iters):
        num = np.zeros_like(world)
        den = np.zeros(n, np.float64)
        pos_w = np.zeros(n, np.float64)
        neg_w = np.zeros(n, np.float64)
        obs_n = np.zeros(n, np.int32)
        iter_rows = []
        for fi in frames:
            cam = cameras[fi]
            u, v, z, ok = project_world(world, cam)
            xi = np.rint(u).astype(np.int32)
            yi = np.rint(v).astype(np.int32)
            ok &= (xi >= 1) & (xi < cam.width - 1) & (yi >= 1) & (yi < cam.height - 1)
            ids = np.flatnonzero(ok)
            if len(ids) == 0:
                continue
            zz = depth[fi][yi[ids], xi[ids]]
            vv = valid[fi][yi[ids], xi[ids]]
            rr = zz - z[ids]
            good = vv & np.isfinite(rr) & (np.abs(rr) <= max_residual)
            ids = ids[good]; rr = rr[good]
            if len(ids) == 0:
                continue
            # Robust confidence suppresses points far from the measured surface.
            # Use image gradients to retain a little extra weight at true edges.
            grad = lines[fi][yi[ids], xi[ids]].astype(np.float64)
            conf = op[ids] * (0.35 + 0.65 * np.exp(-np.abs(rr) / 0.20))
            conf *= (0.75 + 0.25 * grad)
            # Gaussian projected support (in source units) dampens tiny outliers.
            pixrad = smean[ids] * float(cam.K[0, 0]) / np.maximum(z[ids], 0.05)
            conf *= np.clip(0.35 + pixrad / 4.0, 0.35, 1.0)
            conf *= np.clip(np.abs(rr) / 0.01, 0.0, 1.0)
            # Foreground phantom geometry (predicted z substantially nearer
            # than Clean, rr>0) is the dominant failure mode.  Give those
            # residuals a cautious extra vote while down-weighting corrections
            # that would pull points even closer to the camera.
            conf *= np.where(rr > 0.05, 1.8, 0.35)
            # Camera forward is the ray direction in world space.  Moving along
            # it changes metric camera-Z without changing the projected pixel.
            fw = cam.c2w[:3, 2].astype(np.float64)
            delta = fw[None, :] * rr[:, None] * float(step)
            num[ids] += conf[:, None] * delta
            den[ids] += conf
            pos_w[ids] += conf * (rr > 0)
            neg_w[ids] += conf * (rr < 0)
            obs_n[ids] += 1
            iter_rows.append({"frame": fi, "points": int(len(ids)), "median_residual_m": float(np.median(rr)),
                              "p90_abs_residual_m": float(np.percentile(np.abs(rr), 90)),
                              "weighted_update_m": float(np.sum(conf * np.abs(rr)) / max(np.sum(conf), 1e-8))})
        update = num / np.maximum(den[:, None], 1e-8)
        # Only apply a correction when the observed camera-Z residual has a
        # consistent sign across views.  Conflicting signs are usually
        # occlusion/semantic mismatch rather than a reliable surface offset;
        # leaving those Gaussians untouched is safer than creating ghosts.
        consistency = np.maximum(pos_w, neg_w) / np.maximum(den, 1e-8)
        update[(obs_n < 2) | (consistency < 0.65)] = 0.0
        # Average multi-view correction, hard-clipped to a small teacher edit.
        mag = np.linalg.norm(update, axis=1)
        clip = np.minimum(1.0, max_disp / np.maximum(mag, 1e-8))
        update *= clip[:, None]
        update[den <= 1e-8] = 0.0
        world += update
        total_updates += np.linalg.norm(update, axis=1)
        frame_stats.append({"iteration": iteration, "frames": iter_rows,
                            "updated_gaussians": int(np.count_nonzero(den > 0)),
                            "consistent_gaussians": int(np.count_nonzero((obs_n >= 2) & (consistency >= .65))),
                            "median_displacement_m": float(np.median(np.linalg.norm(update, axis=1))),
                            "p95_displacement_m": float(np.percentile(np.linalg.norm(update, axis=1), 95)),
                            "max_displacement_m": float(np.max(np.linalg.norm(update, axis=1)))})
    return world.astype(np.float32), {"iterations": frame_stats,
                                     "total_displacement": total_updates.astype(np.float32)}


def render_one(world, rgb, opacity, scales, quats, cam, device="cuda"):
    import torch
    from gsplat import rasterization
    dev = torch.device(device)
    mt = torch.from_numpy(world.astype(np.float32)).to(dev)
    qt = torch.from_numpy(quats.astype(np.float32)).to(dev)
    st = torch.from_numpy(scales.astype(np.float32)).to(dev)
    ot = torch.from_numpy(opacity.astype(np.float32)).to(dev)
    ct = torch.from_numpy(rgb.astype(np.float32)).to(dev)
    view = torch.linalg.inv(torch.from_numpy(cam.c2w).to(dev)).reshape(1, 4, 4)
    K = torch.from_numpy(cam.K).to(dev).reshape(1, 3, 3)
    with torch.inference_mode():
        im, alpha, _ = rasterization(mt, qt, st, ot, ct, view, K, cam.width, cam.height,
                                     sh_degree=None, render_mode="RGB", packed=True, eps2d=1e-8)
        ed, _, _ = rasterization(mt, qt, st, ot, ct, view, K, cam.width, cam.height,
                                 sh_degree=None, render_mode="ED", packed=True, eps2d=1e-8, near_plane=0.05)
    out = np.clip(im[0].detach().cpu().numpy(), 0, 1).astype(np.float32)
    a = np.clip(alpha[0, ..., 0].detach().cpu().numpy(), 0, 1).astype(np.float32)
    d = ed[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    del mt, qt, st, ot, ct, view, K, im, alpha, ed
    torch.cuda.empty_cache()
    return out, a, d


def depth_edges(z, m):
    z = np.asarray(z, np.float32); m = np.asarray(m, bool)
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return (np.sqrt(gx * gx + gy * gy) >= DEPTH_EDGE_THRESHOLD) & m


def edge_distance(a, b):
    if not a.any() or not b.any():
        return None
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.concatenate([db[a], da[b]]).mean())


def depth_metric(pred, pv, target, tv, mask=None):
    both = pv & tv & np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    if mask is not None:
        both &= mask
    if not both.any():
        return {"count": 0, "relative_median": None, "relative_mean": None,
                "phantom_20cm_ratio": None, "phantom_50cm_ratio": None}
    delta = pred - target
    rel = np.abs(delta) / np.maximum(target, 1e-3)
    den = max(int(mask.sum()) if mask is not None else int(both.size), 1)
    return {"count": int(both.sum()), "relative_median": float(np.median(rel[both])),
            "relative_mean": float(np.mean(rel[both])),
            "abs_median_m": float(np.median(np.abs(delta[both]))),
            "phantom_20cm_ratio": float((both & (delta < -0.20) & (rel > .10)).sum() / den),
            "phantom_50cm_ratio": float((both & (delta < -0.50) & (rel > .20)).sum() / den),
            "recessed_20cm_ratio": float((both & (delta > .20) & (rel > .10)).sum() / den)}


def rgb_metric(pred, clean):
    p = np.asarray(pred, np.float32); c = np.asarray(clean, np.float32) / 255.0
    mse = float(np.mean((p - c) ** 2))
    return {"mse": mse, "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
            "mean_abs": float(np.mean(np.abs(p - c)))}


def save_png(path: Path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(a)
    if a.dtype != np.uint8:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    Image.fromarray(a).save(path)


def depth_vis(z, m):
    z = np.asarray(z, np.float32); m = np.asarray(m, bool)
    out = np.zeros((*z.shape, 3), np.uint8)
    if m.any():
        vv = z[m]; lo, hi = np.percentile(vv, [2, 98])
        q = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        out = np.repeat(np.uint8(q * 255)[..., None], 3, axis=2)
        out[~m] = 0
    return out


def make_board(path: Path, rows, title):
    if not rows:
        return
    w, h = 256, 256
    cols = max(len(r) for r in rows)
    out = Image.new("RGB", (w * cols, 30 + h * len(rows)), (20, 20, 25))
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title, fill="white", font=ImageFont.load_default())
    for y, row in enumerate(rows):
        for x, (im, label) in enumerate(row):
            if im.ndim == 2:
                im = np.repeat(im[..., None], 3, axis=2)
            if im.dtype != np.uint8:
                im = np.uint8(np.clip(im, 0, 1) * 255)
            tile = Image.fromarray(im).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
            out.paste(tile, (x * w, 30 + y * h))
            ImageDraw.Draw(out).text((x * w + 4, 34 + y * h), label, fill="yellow", font=ImageFont.load_default())
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def encode(pattern: Path, output: Path, fps=12):
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(pattern.parent.glob(pattern.name.replace("%02d", "*.png")))
    if not files:
        return None
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(pattern),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", str(output)]
    try:
        subprocess.run(cmd, check=True)
        return "ffmpeg-libx264"
    except Exception:
        arr = np.asarray(Image.open(files[0]).convert("RGB")); h, w = arr.shape[:2]
        wr = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in files:
            wr.write(cv2.cvtColor(np.asarray(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR))
        wr.release(); return "opencv-mp4v"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat", type=Path, default=ROOT / "outputs/p48_1_tripo_cleanup/final/splat_cleaned.ply")
    ap.add_argument("--dataset", type=Path, default=ROOT / "work/327431980_4_normalized")
    ap.add_argument("--alignment", type=Path, default=ROOT / "outputs/p48_tripo_clean_aligned/P48_TRIPO_ALIGNMENT.json")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/phase_a_clean_adjusted_gaussian")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--step", type=float, default=.28)
    ap.add_argument("--max-disp", type=float, default=.16)
    ap.add_argument("--opt-frames", type=str, default="all",
                    help="comma-separated Clean frame IDs used for correction; all uses 0..71")
    ap.add_argument("--render-full", action="store_true")
    args = ap.parse_args()
    load_p48()
    root = args.output
    for d in ["00_audit", "01_optimization", "02_render_raw", "03_render_adjusted", "04_clean_targets",
              "condition_adjusted_rgb", "condition_adjusted_depth", "condition_adjusted_lines", "condition_raw_rgb",
              "condition_raw_depth", "condition_raw_lines", "final"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    raw, header, hb = load_raw(args.splat)
    xyz0 = np.column_stack([raw[x] for x in ("x", "y", "z")]).astype(np.float32)
    # P48 renderer's decoded values / world conversion are immutable contracts.
    rgb = np.clip(.5 + P48.SH_C0 * np.column_stack([raw[x] for x in ("f_dc_0", "f_dc_1", "f_dc_2")]), 0, 1).astype(np.float32)
    opacity = (1.0 / (1.0 + np.exp(-np.nan_to_num(raw["opacity"], nan=-20, posinf=20, neginf=-20)))).astype(np.float32)
    scales = np.exp(np.column_stack([raw[x] for x in ("scale_0", "scale_1", "scale_2")])).astype(np.float32)
    quats = np.column_stack([raw[x] for x in ("rot_0", "rot_1", "rot_2", "rot_3")]).astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    cameras = P48.load_clean_cameras(args.dataset)
    if len(cameras) != FRAME_COUNT:
        raise RuntimeError("camera contract is not exactly 72 frames")
    alignment = json.loads(args.alignment.read_text())
    M = np.asarray(alignment["transform"], np.float64)
    if M.shape != (4, 4):
        raise RuntimeError("invalid immutable alignment")
    clean_depth, clean_valid, clean_rgb, clean_lines, depth_meta = load_depths(cameras)
    sim = float(np.cbrt(np.linalg.det(M[:3, :3])))
    world0 = P48.apply_similarity(xyz0, M).astype(np.float32)
    world_scales = scales * abs(sim)
    write_json(root / "00_audit/PHASE_A_INPUT_MANIFEST.json", {
        "splat": str(args.splat), "splat_sha256": sha256(args.splat), "gaussian_count": int(len(raw)),
        "property_schema": FIELDS, "dataset": str(args.dataset), "alignment": str(args.alignment),
        "alignment_sha256": sha256(args.alignment), "transform": M.tolist(), "camera_count": 72,
        "camera_contract": "P48 exact OpenCV C2W immutable", "clean_depth": depth_meta,
        "base": "P48.1 cleaned splat; no P48.3 dynamic suppression baked into PLY",
        "forbidden_operations": ["prune", "densify", "split", "clone", "camera_edit", "alignment_refit"]})
    opt_frames = list(range(FRAME_COUNT)) if args.opt_frames.lower() == "all" else [int(x) for x in args.opt_frames.split(",") if x.strip()]
    if not opt_frames or any(x < 0 or x >= FRAME_COUNT for x in opt_frames):
        raise ValueError("--opt-frames must contain valid frame IDs")
    world_adj, opt_stats = optimize_means(world0, world_scales, opacity, cameras, clean_depth, clean_valid,
                                          clean_lines, opt_frames, args.iters, args.step, .65, args.max_disp)
    # Convert back to source coordinates using exact inverse P48 transform.
    invM = np.linalg.inv(M)
    xyz_adj = P48.apply_similarity(world_adj, invM).astype(np.float32)
    out_raw = raw.copy()
    for j, name in enumerate(("x", "y", "z")):
        out_raw[name] = xyz_adj[:, j]
    # Verify all non-xyz payload bytes stay exactly unchanged.
    changed = {name: int(np.count_nonzero(out_raw[name] != raw[name])) for name in FIELDS}
    if any(changed[n] for n in FIELDS if n not in ("x", "y", "z")):
        raise RuntimeError(f"unexpected non-xyz edits {changed}")
    save_raw(root / "splat_adjusted.ply", out_raw, header)
    disp = np.linalg.norm(world_adj.astype(np.float64) - world0.astype(np.float64), axis=1)
    write_json(root / "01_optimization/PHASE_A_OPTIMIZATION_STATS.json", {
        "base_splat": str(args.splat), "adjusted_splat": str(root / "splat_adjusted.ply"),
        "gaussian_count_before": int(len(raw)), "gaussian_count_after": int(len(out_raw)),
        "changed_fields": changed, "iters": args.iters, "step": args.step, "max_disp_m": args.max_disp,
        "optimization_frames": opt_frames,
        "displacement_m": {"median": float(np.median(disp)), "mean": float(np.mean(disp)),
                           "p90": float(np.percentile(disp, 90)), "p95": float(np.percentile(disp, 95)),
                           "max": float(np.max(disp)), "moved_gt_1mm": int((disp > .001).sum()),
                           "moved_gt_5cm": int((disp > .05).sum())}, "iterations": opt_stats["iterations"]})
    # Render gate first; full72 is useful to downstream Phase B when requested.
    frames = list(range(FRAME_COUNT)) if args.render_full else GATE
    inv_out = np.linalg.inv(M)
    rows_metrics = []
    raw_outputs, adj_outputs = {}, {}
    for i in frames:
        raw_im, raw_a, raw_d = render_one(world0, rgb, opacity, world_scales, P48.transform_quaternions(quats, M), cameras[i], args.device)
        adj_im, adj_a, adj_d = render_one(world_adj, rgb, opacity, world_scales, P48.transform_quaternions(quats, M), cameras[i], args.device)
        raw_outputs[i] = (raw_im, raw_a, raw_d); adj_outputs[i] = (adj_im, adj_a, adj_d)
        np.save(root / "02_render_raw" / f"F{i:02d}_depth.npy", raw_d); np.save(root / "03_render_adjusted" / f"F{i:02d}_depth.npy", adj_d)
        save_png(root / "02_render_raw" / f"F{i:02d}.png", raw_im); save_png(root / "03_render_adjusted" / f"F{i:02d}.png", adj_im)
        save_png(root / "04_clean_targets" / f"F{i:02d}.png", clean_rgb[i])
        # Condition packs are created for every rendered frame when --render-full;
        # gate mode still leaves exact gate diagnostics without pretending 72 frames.
        for sub, im, dep, a in [("condition_raw_rgb", raw_im, raw_d, raw_a), ("condition_adjusted_rgb", adj_im, adj_d, adj_a)]:
            if args.render_full:
                save_png(root / sub / f"F{i:02d}.png", im)
        if args.render_full:
            for sub, dep, a, prefix in [("condition_raw_depth", raw_d, raw_a, "raw"), ("condition_adjusted_depth", adj_d, adj_a, "adjusted")]:
                np.save(root / sub / f"F{i:02d}.npy", dep)
                e = depth_edges(dep, (a > ALPHA_HIT) & np.isfinite(dep) & (dep > 0))
                save_png(root / sub.replace("depth", "lines") / f"F{i:02d}.png", e.astype(np.uint8))
        pv0 = (raw_a > ALPHA_HIT) & np.isfinite(raw_d) & (raw_d > 0); pv1 = (adj_a > ALPHA_HIT) & np.isfinite(adj_d) & (adj_d > 0)
        dm0 = depth_metric(raw_d, pv0, clean_depth[i], clean_valid[i]); dm1 = depth_metric(adj_d, pv1, clean_depth[i], clean_valid[i])
        em0 = edge_distance(depth_edges(raw_d, pv0), depth_edges(clean_depth[i], clean_valid[i]))
        em1 = edge_distance(depth_edges(adj_d, pv1), depth_edges(clean_depth[i], clean_valid[i]))
        rows_metrics.append({"frame": i, "raw_depth": dm0, "adjusted_depth": dm1, "raw_edge_distance_px": em0,
                             "adjusted_edge_distance_px": em1, "raw_rgb": rgb_metric(raw_im, clean_rgb[i]),
                             "adjusted_rgb": rgb_metric(adj_im, clean_rgb[i]), "raw_coverage": float(pv0.mean()),
                             "adjusted_coverage": float(pv1.mean())})
    # Summary and boards.
    def med(path):
        vals = [r["raw_depth"][path] for r in rows_metrics if r["raw_depth"].get(path) is not None]
        return float(np.median(vals)) if vals else None
    def med_adj(path):
        vals = [r["adjusted_depth"][path] for r in rows_metrics if r["adjusted_depth"].get(path) is not None]
        return float(np.median(vals)) if vals else None
    raw_rel, adj_rel = med("relative_median"), med_adj("relative_median")
    raw_ph, adj_ph = med("phantom_20cm_ratio"), med_adj("phantom_20cm_ratio")
    raw_edge = float(np.median([r["raw_edge_distance_px"] for r in rows_metrics if r["raw_edge_distance_px"] is not None])) if rows_metrics else None
    adj_edge = float(np.median([r["adjusted_edge_distance_px"] for r in rows_metrics if r["adjusted_edge_distance_px"] is not None])) if rows_metrics else None
    improvement = {"relative_median_fraction": None if raw_rel is None or adj_rel is None else float((raw_rel - adj_rel) / max(raw_rel, 1e-8)),
                   "phantom_fraction": None if raw_ph is None or adj_ph is None else float((raw_ph - adj_ph) / max(raw_ph, 1e-8)),
                   "edge_fraction": None if raw_edge is None or adj_edge is None else float((raw_edge - adj_edge) / max(raw_edge, 1e-8))}
    # A conservative gate: at least 2% depth or phantom improvement and no
    # >10% aggregate coverage loss.  This is fail-closed by design.
    raw_cov = float(np.mean([r["raw_coverage"] for r in rows_metrics])) if rows_metrics else 0
    adj_cov = float(np.mean([r["adjusted_coverage"] for r in rows_metrics])) if rows_metrics else 0
    coverage_drop = raw_cov - adj_cov
    clear = any(x is not None and x >= .02 for x in (improvement["relative_median_fraction"], improvement["phantom_fraction"], improvement["edge_fraction"]))
    status = "PASS" if clear and coverage_drop <= .02 else "FAIL"
    write_json(root / "final/PHASE_A_GATE_METRICS.json", {"status": status, "gate_frames": GATE, "rows": rows_metrics,
              "summary": {"raw_relative_median": raw_rel, "adjusted_relative_median": adj_rel, "raw_phantom_20cm": raw_ph,
                          "adjusted_phantom_20cm": adj_ph, "raw_edge_distance_px": raw_edge, "adjusted_edge_distance_px": adj_edge,
                          "raw_coverage": raw_cov, "adjusted_coverage": adj_cov, "coverage_drop": coverage_drop,
                          "improvement": improvement}, "criteria": {"clear_improvement_fraction": .02, "coverage_drop_max": .02}})
    # Boards require at least all gate frames, with raw/adjusted/clean RGB and depth.
    b_rgb, b_dep = [], []
    for i in GATE:
        if i not in raw_outputs:
            continue
        ri, ra, rd = raw_outputs[i]; ai, aa, ad = adj_outputs[i]
        b_rgb.append([(ri, f"F{i:02d} raw"), (ai, f"F{i:02d} adjusted"), (clean_rgb[i], f"F{i:02d} clean")])
        b_dep.append([(depth_vis(rd, (ra > ALPHA_HIT) & (rd > 0)), f"F{i:02d} raw Z"),
                      (depth_vis(ad, (aa > ALPHA_HIT) & (ad > 0)), f"F{i:02d} adjusted Z"),
                      (depth_vis(clean_depth[i], clean_valid[i]), f"F{i:02d} clean Z")])
    make_board(root / "final/PHASE_A_GATE_RGB_BOARD.png", b_rgb, "Phase A RGB: P48.1 raw vs adjusted vs Clean")
    make_board(root / "final/PHASE_A_GATE_DEPTH_BOARD.png", b_dep, "Phase A depth: raw vs adjusted vs Clean")
    # Gate-only synchronized video for the audit (frame labels are exact IDs).
    cmp = root / "final/phase_a_gate_frames"; cmp.mkdir(parents=True, exist_ok=True)
    for r in rows_metrics:
        i = r["frame"]
        ri = np.uint8(np.clip(raw_outputs[i][0], 0, 1) * 255); ai = np.uint8(np.clip(adj_outputs[i][0], 0, 1) * 255); ci = clean_rgb[i]
        Image.fromarray(np.concatenate([ri, ai, ci], axis=1)).save(cmp / f"F{i:02d}.png")
    encode(cmp / "F%02d.png", root / "final/PHASE_A_RAW_ADJUSTED_CLEAN_GATE.mp4", 6)
    report = {"FINAL_STATUS": "PHASE_A_" + status, "status": status, "base": str(args.splat),
              "adjusted": str(root / "splat_adjusted.ply"), "alignment_immutable": True, "camera_immutable": True,
              "gaussian_count": int(len(raw)), "no_prune_densify_split_clone": True, "clean_used_for": ["phase_a_optimization", "qa"],
              "gate_frames": GATE, "summary": {"raw_relative_median": raw_rel, "adjusted_relative_median": adj_rel,
              "raw_phantom_20cm": raw_ph, "adjusted_phantom_20cm": adj_ph, "raw_edge_distance_px": raw_edge,
              "adjusted_edge_distance_px": adj_edge, "coverage_drop": coverage_drop, "improvement": improvement},
              "next_phase_allowed": bool(status == "PASS"), "script": str(Path(__file__).resolve())}
    write_json(root / "final/PHASE_A_REPORT.json", report)
    (root / "final/final_report.md").write_text("# Phase A Clean-adjusted Gaussian teacher\n\n" + json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n")
    (root / "final/HANDOFF_PHASE_A.md").write_text("# HANDOFF Phase A\n\n" + json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(jsonable(report), indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
