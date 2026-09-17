#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from p38_compare import build_structural_mask, load_jsonl
from p38_evaluate import (
    build_salient_object_mask,
    compute_high_frequency_metrics,
    compute_metrics,
    labeled_frame,
    read_video,
    to_uint8,
    write_rgb,
)


def transition_metrics(
    output: np.ndarray,
    clean: np.ndarray,
    previous_index: int,
    next_index: int,
) -> dict[str, float | int]:
    output_delta = output[next_index] - output[previous_index]
    clean_delta = clean[next_index] - clean[previous_index]
    output_jump = float(np.abs(output_delta).mean())
    clean_jump = float(np.abs(clean_delta).mean())
    return {
        "previous_index": int(previous_index),
        "next_index": int(next_index),
        "output_rgb_jump": output_jump,
        "clean_rgb_jump": clean_jump,
        "jump_excess": max(0.0, output_jump - clean_jump),
        "transition_delta_error": float(np.abs(output_delta - clean_delta).mean()),
    }


def classify_transition(metrics: dict[str, float | int]) -> str:
    error = float(metrics["transition_delta_error"])
    excess = float(metrics["jump_excess"])
    if error <= 0.020 and excess <= 0.010:
        return "PASS"
    if error <= 0.040 and excess <= 0.020:
        return "BORDERLINE"
    return "FAIL"


def summarize_transitions(
    output: np.ndarray,
    clean: np.ndarray,
    method: str,
    model_boundary: tuple[int, int] | None = None,
) -> dict:
    transitions = []
    normalized_method = method.upper()
    is_direct = normalized_method.startswith("DIRECT")

    if is_direct:
        if model_boundary is not None:
            previous_index, next_index = model_boundary
            item = transition_metrics(output, clean, previous_index, next_index)
            item.update(
                {
                    "name": f"{previous_index:02d}_to_{next_index:02d}_model_boundary",
                    "kind": "model_boundary",
                    "status": classify_transition(item),
                }
            )
            transitions.append(item)
        segment_status = "PASS"
    else:
        for previous_index, next_index, name in (
            (17, 18, "17_to_18"),
            (35, 36, "35_to_36"),
            (53, 54, "53_to_54"),
        ):
            item = transition_metrics(output, clean, previous_index, next_index)
            item.update(
                {
                    "name": name,
                    "kind": "segment_seam",
                    "status": classify_transition(item),
                }
            )
            transitions.append(item)
        seam_statuses = [item["status"] for item in transitions]
        segment_status = (
            "FAIL"
            if "FAIL" in seam_statuses
            else "BORDERLINE"
            if "BORDERLINE" in seam_statuses
            else "PASS"
        )

    loop = transition_metrics(output, clean, 71, 0)
    loop.update(
        {
            "name": "71_to_00_loop",
            "kind": "loop",
            "status": classify_transition(loop),
        }
    )
    transitions.append(loop)

    model_boundary_status = None
    if is_direct and model_boundary is not None:
        model_boundary_status = transitions[0]["status"]

    return {
        "transitions": transitions,
        "segment_seams": segment_status,
        "model_boundary": model_boundary_status,
        "model_boundary_transition": list(model_boundary) if model_boundary is not None else None,
        "loop_f71_to_f00": loop["status"],
    }


def write_video(path: Path, frames: list[np.ndarray], fps: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def build_clean_video(pair_rows: list[dict], path: Path, size: tuple[int, int]) -> None:
    frames = []
    for row in pair_rows:
        with Image.open(row["clean_rgb"]) as image:
            frames.append(np.asarray(image.convert("RGB").resize(size, Image.Resampling.LANCZOS)))
    write_video(path, frames)


def make_keyframes(videos: dict[str, np.ndarray], path: Path) -> None:
    rows = []
    names = {"condition": "Condition", "output": "P3.8 Best", "clean": "Clean QA"}
    for frame_index in (0, 8, 16, 24, 32, 40, 48, 56, 64, 71):
        cells = [
            labeled_frame(videos[label][frame_index], f"{names[label]} F{frame_index:02d}", (360, 270))
            for label in ("condition", "output", "clean")
        ]
        rows.append(np.concatenate(cells, axis=1))
    board = np.concatenate(rows, axis=0)
    header = np.full((58, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, "P3.8 Full72 keyframes | raw output", (18, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(path, np.concatenate([header, board], axis=0))


def make_loop_board(videos: dict[str, np.ndarray], path: Path) -> None:
    indices = list(range(66, 72)) + list(range(0, 6))
    names = {"condition": "Condition", "output": "P3.8 Best", "clean": "Clean QA"}
    rows = []
    for label in ("condition", "output", "clean"):
        cells = [labeled_frame(videos[label][index], f"F{index:02d}", (180, 150)) for index in indices]
        row = np.concatenate(cells, axis=1)
        cv2.putText(row, names[label], (62, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 40), 1, cv2.LINE_AA)
        rows.append(row)
    board = np.concatenate(rows, axis=0)
    header = np.full((56, board.shape[1], 3), 245, np.uint8)
    cv2.putText(header, "P3.8 loop seam | F66-F71 then F00-F05", (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (25, 25, 25), 2, cv2.LINE_AA)
    write_rgb(path, np.concatenate([header, board], axis=0))


def make_comparison_video(
    videos: dict[str, np.ndarray], path: Path, indices: list[int], fps: int = 12
) -> None:
    width, height, header = 400, 400, 40
    labels = ("condition", "output", "clean")
    names = {"condition": "Condition", "output": "P3.8 Best RAW", "clean": "Clean QA"}
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height + header)
    )
    if not writer.isOpened():
        raise ValueError(f"cannot create video: {path}")
    for index in indices:
        canvas = np.full((height + header, width * 3, 3), 18, np.uint8)
        for column, label in enumerate(labels):
            frame = cv2.resize(to_uint8(videos[label][index]), (width, height), interpolation=cv2.INTER_AREA)
            x0 = column * width
            canvas[header:, x0 : x0 + width] = frame
            cv2.putText(canvas, names[label], (x0 + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"F{index:02d}", (width * 3 - 60, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 210, 255), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--condition-video", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-boundary", type=int, nargs=2, metavar=("PREVIOUS", "NEXT"))
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = load_jsonl(args.pair_manifest)
    clean_path = args.output_dir / "P38_FULL72_CLEAN_QA.mp4"
    build_clean_video(pair_rows, clean_path, (640, 640))

    paths = {
        "condition": args.condition_video,
        "output": args.output_video,
        "clean": clean_path,
    }
    videos = {}
    for label, path in paths.items():
        video, fps = read_video(path)
        if video.shape != (72, 640, 640, 3) or abs(fps - args.fps) > 1e-6:
            raise ValueError(f"full72 contract failed: {label} {path} {video.shape} {fps}")
        videos[label] = video

    object_mask, object_box = build_salient_object_mask(videos["clean"], videos["condition"])
    record = {"frame_indices": list(range(72))}
    structural_mask, structural_counts = build_structural_mask(videos["clean"], record, pair_rows)
    metrics = compute_metrics(videos["output"], videos["clean"], object_mask)
    metrics.update(compute_high_frequency_metrics(videos["output"], videos["clean"], structural_mask))

    transition_summary = summarize_transitions(
        videos["output"],
        videos["clean"],
        method=args.method,
        model_boundary=tuple(args.model_boundary) if args.model_boundary else None,
    )
    transitions = transition_summary["transitions"]
    segment_status = transition_summary["segment_seams"]
    model_boundary_status = transition_summary["model_boundary"]
    loop_status = transition_summary["loop_f71_to_f00"]
    transition_statuses = [item["status"] for item in transitions]
    result = {
        "status": "PASS" if "FAIL" not in transition_statuses else "FAIL",
        "method": args.method,
        "resolution": [640, 640],
        "frames": 72,
        "fps": args.fps,
        "fps_status": "operational_preview_not_ground_truth",
        "metrics": metrics,
        "transitions": transitions,
        "segment_seams": segment_status,
        "model_boundary": model_boundary_status,
        "model_boundary_transition": transition_summary["model_boundary_transition"],
        "loop_f71_to_f00": loop_status,
        "object_box": object_box,
        "structural_qa_sources": structural_counts,
        "inference_target_leakage": False,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    (args.output_dir / "P38_FULL72_METRICS.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "P38_FULL72_TRANSITIONS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)

    make_keyframes(videos, args.output_dir / "P38_FULL72_KEYFRAMES.png")
    make_loop_board(videos, args.output_dir / "P38_LOOP_SEAM_BOARD.png")
    make_comparison_video(videos, args.output_dir / "P38_FULL72_COMPARISON.mp4", list(range(72)), args.fps)
    make_comparison_video(
        videos,
        args.output_dir / "P38_LOOP_SEAM_PREVIEW.mp4",
        list(range(66, 72)) + list(range(0, 6)),
        args.fps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
