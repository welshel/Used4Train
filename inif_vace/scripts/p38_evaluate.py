#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


STARTS = (0, 17, 31, 50)
DISPLAY_NAMES = {
    "condition": "Condition",
    "e0_800": "E0@800 (17f)",
    "l33_400": "L33@400",
    "l33_800": "L33@800 PRO6000",
    "clean": "Clean",
}
MODEL_LABELS = ("e0_800", "l33_400", "l33_800")
ALL_LABELS = ("condition", "e0_800", "l33_400", "l33_800", "clean")


@dataclass(frozen=True)
class ClipPaths:
    start: int
    condition: Path
    clean: Path
    e0_800: Path
    l33_400: Path
    l33_800: Path


def read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"no frames decoded: {path}")
    return np.stack(frames).astype(np.float32) / 255.0, fps


def rgb_to_gray(video: np.ndarray) -> np.ndarray:
    return np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video])


def sobel_magnitude(video: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(video)
    magnitudes = []
    for frame in gray:
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        magnitudes.append(np.sqrt(gx * gx + gy * gy))
    return np.stack(magnitudes)


def edge_stack(video: np.ndarray) -> np.ndarray:
    gray = (rgb_to_gray(video) * 255.0).astype(np.uint8)
    return np.stack([cv2.Canny(frame, 70, 140).astype(np.float32) / 255.0 for frame in gray])


def build_high_frequency_mask(clean: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """Select per-frame detail regions from clean and deployable condition only."""

    if clean.shape != condition.shape:
        raise ValueError("clean and condition must have matching shapes")
    clean_gradient = sobel_magnitude(clean)
    condition_gradient = sobel_magnitude(condition)
    discrepancy_gradient = sobel_magnitude(np.abs(clean - condition))
    score = clean_gradient + 0.35 * condition_gradient + 0.20 * discrepancy_gradient
    masks = []
    height, width = clean.shape[1:3]
    border_y = max(2, int(height * 0.03))
    border_x = max(2, int(width * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for frame_score in score:
        positive = frame_score[frame_score > 0]
        threshold = float(np.percentile(positive, 78)) if positive.size else 1.0
        mask = (frame_score >= threshold).astype(np.uint8)
        mask[:border_y] = 0
        mask[-border_y:] = 0
        mask[:, :border_x] = 0
        mask[:, -border_x:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        minimum_area = max(16, int(height * width * 0.01))
        if int(mask.sum()) < minimum_area:
            usable = frame_score.copy()
            usable[:border_y] = 0
            usable[-border_y:] = 0
            usable[:, :border_x] = 0
            usable[:, -border_x:] = 0
            indices = np.argpartition(usable.ravel(), -minimum_area)[-minimum_area:]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask.ravel()[indices] = 1
        masks.append(mask.astype(bool))
    return np.stack(masks)


def compute_high_frequency_metrics(
    output: np.ndarray, clean: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    if output.shape != clean.shape:
        raise ValueError(f"shape mismatch: {output.shape} vs {clean.shape}")
    if mask.shape != output.shape[:3]:
        raise ValueError(f"high-frequency mask shape mismatch: {mask.shape} vs {output.shape[:3]}")
    if not mask.any():
        raise ValueError("high-frequency mask is empty")
    pixel_error = np.mean((output - clean) ** 2, axis=3)
    output_gradient = sobel_magnitude(output)
    clean_gradient = sobel_magnitude(clean)
    output_edges = edge_stack(output).astype(bool)
    clean_edges = edge_stack(clean).astype(bool)
    selected_output_edges = output_edges & mask
    selected_clean_edges = clean_edges & mask
    true_positive = float(np.logical_and(selected_output_edges, selected_clean_edges).sum())
    predicted = float(selected_output_edges.sum())
    target = float(selected_clean_edges.sum())
    if predicted == 0.0 and target == 0.0:
        edge_f1 = 1.0
    else:
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / target if target else 0.0
        edge_f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {
        "hf_region_fraction": float(mask.mean()),
        "hf_mse": float(pixel_error[mask].mean()),
        "hf_gradient_error": float(np.abs(output_gradient - clean_gradient)[mask].mean()),
        "hf_edge_f1": float(edge_f1),
        "hf_gradient_magnitude": float(output_gradient[mask].mean()),
        "hf_clean_gradient_magnitude": float(clean_gradient[mask].mean()),
    }


def video_ssim(output: np.ndarray, clean: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    scores = []
    for x, y in zip(output, clean):
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        score = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sx + sy + c2) + 1e-12
        )
        scores.append(float(np.clip(score.mean(), -1.0, 1.0)))
    return float(np.mean(scores))


def exposure_match(output: np.ndarray, clean: np.ndarray) -> np.ndarray:
    matched = np.empty_like(output)
    for index, (source, target) in enumerate(zip(output, clean)):
        source_flat = source.reshape(-1, 3)
        target_flat = target.reshape(-1, 3)
        source_mean = source_flat.mean(axis=0)
        target_mean = target_flat.mean(axis=0)
        source_var = ((source_flat - source_mean) ** 2).mean(axis=0)
        covariance = ((source_flat - source_mean) * (target_flat - target_mean)).mean(axis=0)
        gain = np.clip(covariance / (source_var + 1e-8), 0.25, 4.0)
        bias = target_mean - gain * source_mean
        matched[index] = np.clip(source * gain + bias, 0.0, 1.0)
    return matched


def compute_clean_flows(clean: np.ndarray) -> list[np.ndarray]:
    gray = (rgb_to_gray(clean) * 255.0).astype(np.uint8)
    return [
        cv2.calcOpticalFlowFarneback(
            gray[index], gray[index + 1], None, 0.5, 4, 21, 4, 7, 1.5, 0
        )
        for index in range(len(gray) - 1)
    ]


def warp_forward_approximately(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x.astype(np.float32) - flow[..., 0]).astype(np.float32)
    map_y = (grid_y.astype(np.float32) - flow[..., 1]).astype(np.float32)
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def flow_warp_residual(video: np.ndarray, flows: list[np.ndarray]) -> float:
    residuals = []
    for index, flow in enumerate(flows):
        warped = warp_forward_approximately(video[index], flow)
        residuals.append(float(np.mean(np.abs(warped - video[index + 1]))))
    return float(np.mean(residuals)) if residuals else 0.0


def bounding_box(mask: np.ndarray, padding: int = 12) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    height, width = mask.shape
    if len(xs) == 0:
        return (0, 0, width, height)
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(width, int(xs.max()) + 1 + padding)
    y1 = min(height, int(ys.max()) + 1 + padding)
    return x0, y0, x1, y1


def build_salient_object_mask(
    clean: np.ndarray, condition: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if clean.shape != condition.shape:
        raise ValueError("clean and condition must have matching shapes")
    height, width = clean.shape[1:3]
    edge = sobel_magnitude(clean).mean(axis=0)
    temporal = rgb_to_gray(clean).std(axis=0)
    discrepancy = np.mean(np.abs(clean - condition), axis=(0, 3))
    edge /= float(np.percentile(edge, 99) + 1e-8)
    temporal /= float(np.percentile(temporal, 99) + 1e-8)
    discrepancy /= float(np.percentile(discrepancy, 99) + 1e-8)
    yy, xx = np.mgrid[0:height, 0:width]
    center = np.exp(-(((xx - width / 2) / (0.58 * width)) ** 2 + ((yy - height / 2) / (0.58 * height)) ** 2))
    score = (0.50 * edge + 0.30 * temporal + 0.20 * discrepancy) * (0.55 + 0.45 * center)
    border_y = max(2, int(height * 0.04))
    border_x = max(2, int(width * 0.04))
    score[:border_y] = 0
    score[-border_y:] = 0
    score[:, :border_x] = 0
    score[:, -border_x:] = 0
    positive = score[score > 0]
    threshold = float(np.percentile(positive, 82)) if positive.size else 1.0
    mask = (score >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        component_scores = []
        for label in range(1, count):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            component_scores.append((float(score[component].mean()) * math.sqrt(area), label))
        keep = {label for _, label in sorted(component_scores, reverse=True)[:3]}
        mask = np.isin(labels, list(keep)).astype(np.uint8)
    minimum_area = max(16, int(height * width * 0.01))
    if int(mask.sum()) < minimum_area:
        flat_indices = np.argpartition(score.ravel(), -minimum_area)[-minimum_area:]
        mask = np.zeros((height, width), np.uint8)
        mask.ravel()[flat_indices] = 1
        mask = cv2.dilate(mask, kernel, iterations=1)
    boolean_mask = mask.astype(bool)
    return boolean_mask, bounding_box(boolean_mask)


def object_temporal_excess(output: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any() or len(output) < 2:
        return 0.0
    output_motion = np.abs(np.diff(output, axis=0)).mean(axis=3)[:, mask].mean()
    clean_motion = np.abs(np.diff(clean, axis=0)).mean(axis=3)[:, mask].mean()
    return float(max(0.0, output_motion - clean_motion))


def object_temporal_delta_error(output: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any() or len(output) < 2:
        return 0.0
    delta_error = np.abs(np.diff(output, axis=0) - np.diff(clean, axis=0)).mean(axis=3)
    return float(delta_error[:, mask].mean())


def compute_metrics(
    output: np.ndarray,
    clean: np.ndarray,
    object_mask: np.ndarray | None = None,
    clean_flows: list[np.ndarray] | None = None,
) -> dict[str, float]:
    if output.shape != clean.shape:
        raise ValueError(f"shape mismatch: {output.shape} vs {clean.shape}")
    if object_mask is None:
        object_mask = np.ones(output.shape[1:3], dtype=bool)
    if clean_flows is None:
        clean_flows = compute_clean_flows(clean)
    mse = float(np.mean((output - clean) ** 2))
    output_edges, clean_edges = edge_stack(output), edge_stack(clean)
    temporal_delta_error = float(np.mean(np.abs(np.diff(output, axis=0) - np.diff(clean, axis=0))))
    edge_temporal_error = float(
        np.mean(np.abs(np.diff(output_edges, axis=0) - np.diff(clean_edges, axis=0)))
    )
    if output.shape[0] >= 3:
        output_second = output[2:] - 2 * output[1:-1] + output[:-2]
        clean_second = clean[2:] - 2 * clean[1:-1] + clean[:-2]
        flicker_proxy = float(np.mean(np.abs(output_second - clean_second)))
    else:
        flicker_proxy = 0.0
    output_warp = flow_warp_residual(output, clean_flows)
    clean_warp = flow_warp_residual(clean, clean_flows)
    laplacian_values = []
    edge_magnitudes = []
    for frame in output:
        gray = cv2.cvtColor((frame * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian_values.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        gx = cv2.Sobel(gray.astype(np.float32) / 255.0, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32) / 255.0, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitudes.append(float(np.sqrt(gx * gx + gy * gy).mean()))
    matched = exposure_match(output, clean)
    matched_mse = float(np.mean((matched - clean) ** 2))
    matched_delta = float(np.mean(np.abs(np.diff(matched, axis=0) - np.diff(clean, axis=0))))
    return {
        "mse": mse,
        "psnr": -10.0 * math.log10(mse) if mse > 0 else float("inf"),
        "ssim": video_ssim(output, clean),
        "exposure_matched_mse": matched_mse,
        "exposure_matched_psnr": -10.0 * math.log10(matched_mse) if matched_mse > 0 else float("inf"),
        "exposure_matched_ssim": video_ssim(matched, clean),
        "exposure_matched_temporal_delta_error": matched_delta,
        "temporal_delta_error": temporal_delta_error,
        "edge_temporal_error": edge_temporal_error,
        "flow_warp_residual": output_warp,
        "clean_flow_warp_residual": clean_warp,
        "flow_warp_excess": max(0.0, output_warp - clean_warp),
        "flicker_proxy": flicker_proxy,
        "motion_energy": float(np.mean(np.abs(np.diff(output, axis=0)))),
        "clean_motion_energy": float(np.mean(np.abs(np.diff(clean, axis=0)))),
        "laplacian_variance": float(np.mean(laplacian_values)),
        "edge_magnitude": float(np.mean(edge_magnitudes)),
        "object_temporal_delta_error": object_temporal_delta_error(output, clean, object_mask),
        "object_temporal_excess_variation": object_temporal_excess(output, clean, object_mask),
    }


def clip_input_scores(clean: np.ndarray, condition: np.ndarray, start: int) -> dict[str, float | int]:
    clean_edges = sobel_magnitude(clean)
    condition_edges = sobel_magnitude(condition)
    clean_gray = rgb_to_gray(clean)
    edge_density = float((clean_edges > np.percentile(clean_edges, 75)).mean())
    edge_strength = float(clean_edges.mean())
    edge_mismatch = float(np.mean(np.abs(clean_edges - condition_edges)))
    temporal_std = float(clean_gray.std(axis=0).mean())
    temporal_edge = float(np.abs(np.diff(clean_edges, axis=0)).mean())
    condition_gap = float(np.mean(np.abs(clean - condition)))
    edge_score = edge_strength + 0.75 * temporal_edge + 0.35 * edge_mismatch
    complex_score = temporal_std + 0.55 * temporal_edge + 0.35 * condition_gap + 0.15 * edge_density
    return {
        "start": start,
        "edge_density": edge_density,
        "edge_strength": edge_strength,
        "edge_mismatch": edge_mismatch,
        "temporal_std": temporal_std,
        "temporal_edge": temporal_edge,
        "condition_gap": condition_gap,
        "edge_score": edge_score,
        "complex_score": complex_score,
    }


def choose_validation_roles(scores: list[dict[str, float | int]]) -> dict[str, int]:
    edge = max(scores, key=lambda row: float(row["edge_score"]))
    complex_row = max(scores, key=lambda row: float(row["complex_score"]))
    return {"V_EDGE": int(edge["start"]), "V_COMPLEX": int(complex_row["start"])}


def resolve_paths(root: Path, start: int) -> ClipPaths:
    l33_400 = root / "04_validation/l33_400"
    e0 = root / "04_validation/e0_800_33f"
    l33_800 = root / "pro6000_resume/12_validation/l33_800"
    prefix = {0: "V0", 17: "V1", 31: "V2", 50: "V3"}[start]
    paths = ClipPaths(
        start=start,
        condition=l33_400 / f"{prefix}_start{start:02d}_condition.mp4",
        clean=l33_400 / f"{prefix}_start{start:02d}_clean.mp4",
        e0_800=e0 / f"{prefix}_start{start:02d}_e0_800_33f.mp4",
        l33_400=l33_400 / f"{prefix}_start{start:02d}_l33_400.mp4",
        l33_800=l33_800 / f"{prefix}_start{start:02d}_l33_800_pro6000.mp4",
    )
    for path in paths.__dict__.values():
        if isinstance(path, Path) and not path.is_file():
            raise FileNotFoundError(path)
    return paths


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def labeled_frame(frame: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    resized = cv2.resize(to_uint8(frame), (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 30, width, 3), 18, np.uint8)
    canvas[30:] = resized
    cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def crop_frame(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1]


def make_samepose_board(videos: dict[str, np.ndarray], start: int, output: Path) -> None:
    rows = []
    for frame_index in (0, 8, 16, 24, 32):
        cells = [
            labeled_frame(videos[label][frame_index], f"{DISPLAY_NAMES[label]} F{frame_index:02d}", (280, 280))
            for label in ALL_LABELS
        ]
        rows.append(np.concatenate(cells, axis=1))
    board = np.concatenate(rows, axis=0)
    header = np.full((54, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, f"P3.7 same-pose comparison | start {start:02d}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(output, np.concatenate([header, board], axis=0))


def make_object_board(
    videos: dict[str, np.ndarray],
    box: tuple[int, int, int, int],
    start: int,
    output: Path,
) -> None:
    rows = []
    for frame_index in (0, 5, 10, 16, 22, 27, 32):
        cells = [
            labeled_frame(crop_frame(videos[label][frame_index], box), f"{DISPLAY_NAMES[label]} F{frame_index:02d}", (260, 220))
            for label in ALL_LABELS
        ]
        rows.append(np.concatenate(cells, axis=1))
    board = np.concatenate(rows, axis=0)
    header = np.full((54, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, f"Object persistence crop | input-selected start {start:02d} | box={box}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(output, np.concatenate([header, board], axis=0))


def make_enter_leave_board(videos: dict[str, np.ndarray], start: int, output: Path) -> None:
    selected = (0, 4, 8, 12, 16, 20, 24, 28, 32)
    rows = []
    for label in ALL_LABELS:
        cells = [labeled_frame(videos[label][index], f"F{index:02d}", (180, 180)) for index in selected]
        row = np.concatenate(cells, axis=1)
        cv2.putText(row, DISPLAY_NAMES[label], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 230, 60), 2, cv2.LINE_AA)
        rows.append(row)
    board = np.concatenate(rows, axis=0)
    header = np.full((54, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, f"Enter/leave temporal sequence | start {start:02d}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(output, np.concatenate([header, board], axis=0))


def make_complex_track_board(
    videos: dict[str, np.ndarray],
    box: tuple[int, int, int, int],
    start: int,
    output: Path,
) -> None:
    selected = tuple(range(0, 33, 4))
    rows = []
    for label in ALL_LABELS:
        cells = [
            labeled_frame(crop_frame(videos[label][index], box), f"F{index:02d}", (180, 150))
            for index in selected
        ]
        row = np.concatenate(cells, axis=1)
        cv2.putText(row, DISPLAY_NAMES[label], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 230, 60), 2, cv2.LINE_AA)
        rows.append(row)
    board = np.concatenate(rows, axis=0)
    header = np.full((54, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, f"Complex-object track | input-selected start {start:02d}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(output, np.concatenate([header, board], axis=0))


def heatmap_rgb(values: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(values, 98) + 1e-8)
    normalized = np.clip(values / scale, 0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def make_temporal_error_board(videos: dict[str, np.ndarray], start: int, output: Path) -> None:
    clean_delta = np.diff(videos["clean"], axis=0)
    rows = []
    for label in MODEL_LABELS:
        delta_error = np.abs(np.diff(videos[label], axis=0) - clean_delta).mean(axis=3)
        mean_map = heatmap_rgb(delta_error.mean(axis=0))
        max_map = heatmap_rgb(np.percentile(delta_error, 90, axis=0))
        cells = [
            labeled_frame(mean_map.astype(np.float32) / 255.0, f"{DISPLAY_NAMES[label]} mean delta error", (420, 420)),
            labeled_frame(max_map.astype(np.float32) / 255.0, f"{DISPLAY_NAMES[label]} P90 delta error", (420, 420)),
        ]
        rows.append(np.concatenate(cells, axis=1))
    board = np.concatenate(rows, axis=0)
    header = np.full((54, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, f"Temporal error heatmaps | start {start:02d} | lower/darker is better", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(output, np.concatenate([header, board], axis=0))


def make_sharpness_board(
    videos: dict[str, np.ndarray],
    start: int,
    metrics_by_label: dict[str, dict[str, float]],
    output: Path,
) -> None:
    rows = []
    for frame_index in (4, 16, 28):
        cells = []
        for label in ALL_LABELS:
            metric = metrics_by_label.get(label, {})
            suffix = f"LapVar {metric.get('laplacian_variance', float('nan')):.1f}" if metric else "reference"
            cells.append(labeled_frame(videos[label][frame_index], f"{DISPLAY_NAMES[label]} F{frame_index:02d} | {suffix}", (300, 300)))
        rows.append(np.concatenate(cells, axis=1))
    board = np.concatenate(rows, axis=0)
    header = np.full((54, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, f"Sharpness comparison | start {start:02d}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(output, np.concatenate([header, board], axis=0))


def make_full_playback_board(videos: dict[str, np.ndarray], start: int, output: Path) -> None:
    panels = []
    blocks = (range(0, 11), range(11, 22), range(22, 33))
    for block in blocks:
        for label in ALL_LABELS:
            cells = [labeled_frame(videos[label][index], f"F{index:02d}", (118, 118)) for index in block]
            row = np.concatenate(cells, axis=1)
            cv2.putText(row, DISPLAY_NAMES[label], (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 230, 60), 2, cv2.LINE_AA)
            panels.append(row)
    board = np.concatenate(panels, axis=0)
    header = np.full((50, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, f"Full 33-frame playback sheet | start {start:02d}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(output, np.concatenate([header, board], axis=0))


def write_side_by_side(videos: dict[str, np.ndarray], start: int, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height, header_height = 384, 384, 38
    frame_width = cell_width * len(ALL_LABELS)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, cell_height + header_height))
    if not writer.isOpened():
        raise ValueError(f"cannot create video: {path}")
    for frame_index in range(33):
        canvas = np.full((cell_height + header_height, frame_width, 3), 20, np.uint8)
        for column, label in enumerate(ALL_LABELS):
            frame = cv2.resize(to_uint8(videos[label][frame_index]), (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            x0 = column * cell_width
            canvas[header_height:, x0 : x0 + cell_width] = frame
            cv2.putText(canvas, DISPLAY_NAMES[label], (x0 + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"start {start:02d} | frame {frame_index:02d}/32", (frame_width - 245, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 210, 255), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def aggregate(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    numeric_keys = [key for key in rows[0] if key not in {"start", "model"}]
    for label in MODEL_LABELS:
        selected = [row for row in rows if row["model"] == label]
        result[label] = {
            key: float(np.mean([float(row[key]) for row in selected])) for key in numeric_keys
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    final = args.root / "pro6000_resume/final"
    boards = final / "boards"
    comparisons = final / "comparison_videos"
    full_playback = final / "full_playback"
    for directory in (final, boards, comparisons, full_playback):
        directory.mkdir(parents=True, exist_ok=True)

    loaded: dict[int, dict[str, np.ndarray]] = {}
    fps_by_start: dict[int, float] = {}
    object_masks: dict[int, np.ndarray] = {}
    object_boxes: dict[int, tuple[int, int, int, int]] = {}
    input_scores = []
    contract = []
    for start in STARTS:
        paths = resolve_paths(args.root, start)
        videos = {}
        fps_values = {}
        for label in ALL_LABELS:
            video, fps = read_video(getattr(paths, label))
            videos[label] = video
            fps_values[label] = fps
            contract.append({
                "start": start,
                "label": label,
                "path": str(getattr(paths, label)),
                "frames": int(video.shape[0]),
                "height": int(video.shape[1]),
                "width": int(video.shape[2]),
                "fps": fps,
            })
            if video.shape != (33, 512, 512, 3) or abs(fps - 12.0) > 1e-6:
                raise ValueError(f"validation contract failed: {contract[-1]}")
        loaded[start] = videos
        fps_by_start[start] = fps_values["clean"]
        mask, box = build_salient_object_mask(videos["clean"], videos["condition"])
        object_masks[start] = mask
        object_boxes[start] = box
        input_scores.append(clip_input_scores(videos["clean"], videos["condition"], start))

    roles = choose_validation_roles(input_scores)
    (final / "VALIDATION_ROLE_SELECTION.json").write_text(json.dumps({
        "selection_policy": "clean_and_condition_only_no_generated_output_used",
        "roles": roles,
        "input_scores": input_scores,
        "object_boxes": {str(start): object_boxes[start] for start in STARTS},
    }, indent=2) + "\n")
    (final / "VIDEO_CONTRACT_AUDIT.json").write_text(json.dumps(contract, indent=2) + "\n")

    rows = []
    metrics_by_start: dict[int, dict[str, dict[str, float]]] = {}
    for start in STARTS:
        videos = loaded[start]
        flows = compute_clean_flows(videos["clean"])
        metrics_by_start[start] = {}
        for label in MODEL_LABELS:
            values = compute_metrics(videos[label], videos["clean"], object_masks[start], flows)
            metrics_by_start[start][label] = values
            rows.append({"start": start, "model": label, **values})
        for label in ("condition", "clean"):
            metrics_by_start[start][label] = compute_metrics(videos[label], videos["clean"], object_masks[start], flows)

    summary = aggregate(rows)
    (final / "P37_METRICS_SUMMARY.json").write_text(json.dumps({
        "aggregate": summary,
        "per_clip": rows,
        "roles": roles,
        "metric_notes": {
            "flow_warp_residual": "Output residual under optical flow estimated only from clean frames.",
            "flow_warp_excess": "max(0, output warp residual - clean self-warp residual).",
            "object_mask": "Selected only from clean+condition edge, motion and discrepancy signals.",
            "exposure_matched": "Per-frame per-channel affine fit to clean, used only for analysis.",
        },
    }, indent=2) + "\n")
    write_csv(final / "TEMPORAL_METRICS.csv", rows)
    write_csv(final / "EDGE_TEMPORAL_METRICS.csv", [
        {key: row[key] for key in ("start", "model", "edge_temporal_error", "temporal_delta_error", "flow_warp_residual", "flow_warp_excess", "flicker_proxy")}
        for row in rows
    ])
    write_csv(final / "SHARPNESS_METRICS.csv", [
        {key: row[key] for key in ("start", "model", "laplacian_variance", "edge_magnitude", "mse", "psnr", "ssim")}
        for row in rows
    ])
    write_csv(final / "OBJECT_PERSISTENCE_AUDIT.csv", [
        {
            "start": row["start"],
            "model": row["model"],
            "role": ",".join(role for role, selected in roles.items() if selected == row["start"]),
            "object_box": str(object_boxes[int(row["start"])]),
            "object_temporal_delta_error": row["object_temporal_delta_error"],
            "object_temporal_excess_variation": row["object_temporal_excess_variation"],
            "manual_identity_drift": "PENDING_VISUAL_QA",
            "manual_stretching": "PENDING_VISUAL_QA",
            "manual_persistence": "PENDING_VISUAL_QA",
        }
        for row in rows
    ])

    edge_start = roles["V_EDGE"]
    complex_start = roles["V_COMPLEX"]
    make_samepose_board(loaded[edge_start], edge_start, boards / "L17_VS_L33_SAMEPOSE_BOARD.png")
    make_object_board(loaded[complex_start], object_boxes[complex_start], complex_start, boards / "L17_VS_L33_OBJECT_PERSISTENCE_BOARD.png")
    make_enter_leave_board(loaded[complex_start], complex_start, boards / "ENTER_LEAVE_TEMPORAL_BOARD.png")
    make_complex_track_board(loaded[complex_start], object_boxes[complex_start], complex_start, boards / "COMPLEX_OBJECT_TRACK_BOARD.png")
    make_temporal_error_board(loaded[complex_start], complex_start, boards / "TEMPORAL_ERROR_BOARD.png")
    make_sharpness_board(loaded[edge_start], edge_start, metrics_by_start[edge_start], boards / "SHARPNESS_BOARD.png")
    for start in STARTS:
        make_full_playback_board(loaded[start], start, full_playback / f"FULL_PLAYBACK_START{start:02d}.png")
        write_side_by_side(loaded[start], start, comparisons / f"P37_START{start:02d}_CONDITION_E0_L33_400_L33_800_CLEAN.mp4", fps_by_start[start])

    def fmt(label: str, key: str) -> str:
        return f"{summary[label][key]:.6f}"

    comparison_md = f"""# P3.7 Exposure-Matched and Temporal Comparison

Validation contract: four fixed 33-frame clips, 512×512, 12 fps, identical starts and seeds.

V_EDGE: start `{edge_start:02d}`. V_COMPLEX: start `{complex_start:02d}`. Both roles were selected from clean+condition signals only; generated outputs were not used for selection.

| Model | Raw MSE ↓ | PSNR ↑ | SSIM ↑ | Exposure-matched MSE ↓ | Exposure-matched SSIM ↑ | Temporal delta ↓ | Edge temporal ↓ | Flow-warp excess ↓ | Object delta error ↓ | LapVar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0@800 (17f) | {fmt('e0_800','mse')} | {fmt('e0_800','psnr')} | {fmt('e0_800','ssim')} | {fmt('e0_800','exposure_matched_mse')} | {fmt('e0_800','exposure_matched_ssim')} | {fmt('e0_800','temporal_delta_error')} | {fmt('e0_800','edge_temporal_error')} | {fmt('e0_800','flow_warp_excess')} | {fmt('e0_800','object_temporal_delta_error')} | {fmt('e0_800','laplacian_variance')} |
| L33@400 | {fmt('l33_400','mse')} | {fmt('l33_400','psnr')} | {fmt('l33_400','ssim')} | {fmt('l33_400','exposure_matched_mse')} | {fmt('l33_400','exposure_matched_ssim')} | {fmt('l33_400','temporal_delta_error')} | {fmt('l33_400','edge_temporal_error')} | {fmt('l33_400','flow_warp_excess')} | {fmt('l33_400','object_temporal_delta_error')} | {fmt('l33_400','laplacian_variance')} |
| L33@800 PRO6000 | {fmt('l33_800','mse')} | {fmt('l33_800','psnr')} | {fmt('l33_800','ssim')} | {fmt('l33_800','exposure_matched_mse')} | {fmt('l33_800','exposure_matched_ssim')} | {fmt('l33_800','temporal_delta_error')} | {fmt('l33_800','edge_temporal_error')} | {fmt('l33_800','flow_warp_excess')} | {fmt('l33_800','object_temporal_delta_error')} | {fmt('l33_800','laplacian_variance')} |

Metric interpretation:

- Pixel/SSIM scores measure reconstruction agreement, not hallucination safety by themselves.
- Temporal-delta, edge-temporal, flow-warp excess and object-delta error are the primary persistence diagnostics.
- Exposure matching is analysis-only and never changes generated video.
"""
    (final / "EXPOSURE_MATCHED_COMPARISON.md").write_text(comparison_md)
    print(json.dumps({"status": "PASS", "roles": roles, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
