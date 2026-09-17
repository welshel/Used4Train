#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from p38_evaluate import (
    build_salient_object_mask,
    compute_clean_flows,
    compute_high_frequency_metrics,
    compute_metrics,
    labeled_frame,
    read_video,
    to_uint8,
    write_rgb,
)


STARTS = (0, 17, 31, 50)
PREFIX = {0: "V0", 17: "V1", 31: "V2", 50: "V3"}
DISPLAY = {
    "condition": "Condition",
    "baseline": "L33@800 (trained 512)",
    "candidate": "H640@100",
    "clean": "Clean",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_map(values: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(values, 99) + 1e-8)
    return np.clip(values / scale, 0.0, 1.0)


def scalar_gradient(values: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(values.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def rgb_gradient(values: np.ndarray) -> np.ndarray:
    channels = [scalar_gradient(values[..., channel]) for channel in range(values.shape[-1])]
    return np.mean(channels, axis=0)


def build_structural_mask(
    clean_video: np.ndarray,
    record: dict,
    pair_rows: list[dict],
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = clean_video.shape[1:3]
    masks = []
    counts = {"clean_gradient": 0, "lines": 0, "depth": 0, "normal": 0}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for local_index, frame_index in enumerate(record["frame_indices"]):
        row = pair_rows[int(frame_index)]
        gray = cv2.cvtColor((clean_video[local_index] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        score = normalize_map(scalar_gradient(gray.astype(np.float32) / 255.0))
        counts["clean_gradient"] += 1

        line_path = Path(row["clean_rgb"]).with_name(f"{row['frame_stem']}_lines.png")
        if line_path.is_file():
            with Image.open(line_path) as image:
                rgba = np.asarray(image.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS))
            line_rgb = rgba[..., :3].astype(np.float32) / 255.0
            line_alpha = rgba[..., 3].astype(np.float32) / 255.0
            line_signal = np.maximum(rgb_gradient(line_rgb), scalar_gradient(line_alpha))
            score += 0.70 * normalize_map(line_signal)
            counts["lines"] += 1

        depth_path = Path(row["clean_depth"])
        if depth_path.is_file():
            with Image.open(depth_path) as image:
                depth = np.asarray(image.resize((width, height), Image.Resampling.NEAREST)).astype(np.float32)
            valid = depth > 0
            if valid.any():
                lo, hi = np.percentile(depth[valid], (1, 99))
                depth = np.clip((depth - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)
                score += 0.65 * normalize_map(scalar_gradient(depth))
                counts["depth"] += 1

        normal_path = Path(row["clean_normal"])
        if normal_path.is_file():
            with Image.open(normal_path) as image:
                normal = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
            score += 0.65 * normalize_map(rgb_gradient(normal))
            counts["normal"] += 1

        border_y = max(2, int(height * 0.03))
        border_x = max(2, int(width * 0.03))
        score[:border_y] = 0
        score[-border_y:] = 0
        score[:, :border_x] = 0
        score[:, -border_x:] = 0
        positive = score[score > 0]
        threshold = float(np.percentile(positive, 78)) if positive.size else 1.0
        mask = (score >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        masks.append(mask.astype(bool))
    return np.stack(masks), counts


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    result = {}
    for model in ("baseline", "candidate"):
        selected = [row for row in rows if row["model"] == model]
        keys = [key for key in selected[0] if key not in ("start", "model")]
        result[model] = {key: float(np.mean([float(row[key]) for row in selected])) for key in keys}
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_video(videos: dict[str, np.ndarray], path: Path, fps: float = 12.0) -> None:
    width, height, header = 320, 320, 38
    labels = ("condition", "baseline", "candidate", "clean")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * len(labels), height + header)
    )
    if not writer.isOpened():
        raise ValueError(f"cannot create {path}")
    for frame_index in range(33):
        canvas = np.full((height + header, width * len(labels), 3), 18, np.uint8)
        for column, label in enumerate(labels):
            frame = cv2.resize(to_uint8(videos[label][frame_index]), (width, height), interpolation=cv2.INTER_AREA)
            x0 = column * width
            canvas[header:, x0 : x0 + width] = frame
            cv2.putText(canvas, DISPLAY[label], (x0 + 7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def make_keyframe_board(videos: dict[str, np.ndarray], path: Path, title: str) -> None:
    rows = []
    for frame_index in (0, 8, 16, 24, 32):
        rows.append(
            np.concatenate(
                [labeled_frame(videos[label][frame_index], f"{DISPLAY[label]} F{frame_index:02d}", (300, 300)) for label in DISPLAY],
                axis=1,
            )
        )
    board = np.concatenate(rows, axis=0)
    header = np.full((56, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, title, (16, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(path, np.concatenate([header, board], axis=0))


def make_track_board(
    videos: dict[str, np.ndarray], path: Path, title: str, box: tuple[int, int, int, int]
) -> None:
    x0, y0, x1, y1 = box
    blocks = []
    for first in (0, 11, 22):
        model_rows = []
        for label in ("baseline", "candidate", "clean"):
            cells = []
            for frame_index in range(first, min(first + 11, 33)):
                crop = videos[label][frame_index, y0:y1, x0:x1]
                cells.append(labeled_frame(crop, f"F{frame_index:02d}", (160, 110)))
            row = np.concatenate(cells, axis=1)
            cv2.putText(row, DISPLAY[label], (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 40), 1, cv2.LINE_AA)
            model_rows.append(row)
        blocks.append(np.concatenate(model_rows, axis=0))
    board = np.concatenate(blocks, axis=0)
    header = np.full((52, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.73, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(path, np.concatenate([header, board], axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="l33_800_infer640")
    parser.add_argument("--candidate-label", default="h640_900")
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boards = args.output_dir / "boards"
    videos_out = args.output_dir / "videos"
    boards.mkdir(exist_ok=True)
    videos_out.mkdir(exist_ok=True)

    records = {int(row["start_index"]): row for row in load_jsonl(args.validation_manifest)}
    pair_rows = load_jsonl(args.pair_manifest)
    rows = []
    hf_rows = []
    loaded = {}
    boxes = {}
    structural_counts = {}
    for start in STARTS:
        prefix = f"{PREFIX[start]}_start{start:02d}"
        paths = {
            "condition": args.candidate_dir / f"{prefix}_condition.mp4",
            "baseline": args.baseline_dir / f"{prefix}_{args.baseline_label}.mp4",
            "candidate": args.candidate_dir / f"{prefix}_{args.candidate_label}.mp4",
            "clean": args.candidate_dir / f"{prefix}_clean.mp4",
        }
        videos = {}
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            video, fps = read_video(path)
            if video.shape != (33, 640, 640, 3) or abs(fps - 12.0) > 1e-6:
                raise ValueError(f"video contract failed: {label} {path} {video.shape} {fps}")
            videos[label] = video
        loaded[start] = videos
        object_mask, box = build_salient_object_mask(videos["clean"], videos["condition"])
        boxes[start] = box
        structural_mask, counts = build_structural_mask(videos["clean"], records[start], pair_rows)
        structural_counts[start] = counts
        flows = compute_clean_flows(videos["clean"])
        for label in ("baseline", "candidate"):
            metrics = compute_metrics(videos[label], videos["clean"], object_mask, flows)
            hf_metrics = compute_high_frequency_metrics(videos[label], videos["clean"], structural_mask)
            rows.append({"start": start, "model": label, **metrics, **hf_metrics})
            hf_rows.append({"start": start, "model": label, **hf_metrics})
        write_comparison_video(videos, videos_out / f"P38_START{start:02d}_COMPARISON.mp4")

    summary = aggregate(rows)
    historical = json.loads(args.historical_metrics.read_text())["aggregate"]["l33_800"]
    delta = {key: summary["candidate"][key] - summary["baseline"][key] for key in summary["baseline"]}
    result = {
        "status": "PASS",
        "resolution": [640, 640],
        "historical_l33_800_512": historical,
        "aggregate": summary,
        "candidate_minus_baseline": delta,
        "per_clip": rows,
        "object_boxes": {str(key): value for key, value in boxes.items()},
        "structural_qa_sources": {
            "clean_gradient": "QA_ONLY",
            "clean_lines": "QA_ONLY",
            "clean_depth_edges": "QA_ONLY",
            "clean_normal_edges": "QA_ONLY",
            "counts_by_start": structural_counts,
        },
        "inference_target_leakage": False,
    }
    (args.output_dir / "P38_METRICS_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(args.output_dir / "P38_ALL_METRICS.csv", rows)
    write_csv(args.output_dir / "HIGH_FREQUENCY_REGION_METRICS.csv", hf_rows)

    start = 31
    make_keyframe_board(loaded[start], boards / "P38_SHARPNESS_BOARD.png", "P3.8 sharpness | start31")
    make_keyframe_board(loaded[start], boards / "P38_512_VS_HIGHRES_BOARD.png", "L33@800 trained512 vs H640@100 | inference640")
    make_track_board(loaded[start], boards / "P38_COMPLEX_OBJECT_BOARD.png", "P3.8 complex-object persistence | start31", boxes[start])
    make_track_board(loaded[start], boards / "P38_CURTAIN_PLANT_TABLE_BOARD.png", "P3.8 curtain / plant / table track | start31", (0, 220, 640, 640))
    make_track_board(loaded[start], boards / "P38_TEMPORAL_PERSISTENCE_BOARD.png", "P3.8 temporal persistence | start31", (0, 0, 640, 640))
    print(json.dumps({"status": "PASS", "aggregate": summary, "delta": delta}, indent=2))


if __name__ == "__main__":
    main()
