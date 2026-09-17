#!/usr/bin/env python3
"""Build the Phase-C contract for the final dynamic-adjusted RGB condition.

This is deliberately a contract-only utility: it never starts training and it
never copies Clean pixels into a condition channel.  Each row contains an
exact cyclic Fxx-to-Fxx pairing, with the adjusted render in ``vace_video``
and the Clean RGB frame in ``video`` (target only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from PIL import Image

FRAME_COUNT = 72
CLIP_LEN = 33
TRAIN_ITEMS = 2800
VALIDATION_STARTS = (0, 17, 31, 50)
PROMPT = "a realistic indoor room tour"


def frame_paths(folder: Path) -> list[Path]:
    paths = [folder / f"F{i:02d}.png" for i in range(FRAME_COUNT)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} lossless frames: {missing[:4]}")
    for p in paths:
        with Image.open(p) as im:
            if im.mode != "RGB" or im.size != (896, 896):
                raise ValueError(f"{p}: expected RGB 896x896, got {im.mode} {im.size}")
    return paths


def cyclic(paths: list[Path], start: int) -> list[str]:
    return [str(paths[(start + j) % FRAME_COUNT]) for j in range(CLIP_LEN)]


def row(adjusted: list[Path], clean: list[Path], start: int, idx: int, split: str) -> dict:
    return {
        "id": f"{split}_{idx:04d}_start{start:02d}",
        "start": start,
        "clip_len": CLIP_LEN,
        "prompt": PROMPT,
        # Clean is target-only.  It is intentionally not listed in either
        # vace_video or vace_reference_image.
        "video": cyclic(clean, start),
        "vace_video": cyclic(adjusted, start),
        "vace_reference_image": [str(adjusted[start])],
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit_rows(rows: Iterable[dict], clean_root: Path) -> dict:
    rows = list(rows)
    violations = []
    starts = []
    for r in rows:
        starts.append(int(r["start"]))
        if len(r["video"]) != CLIP_LEN or len(r["vace_video"]) != CLIP_LEN:
            violations.append({"id": r["id"], "reason": "clip_len"})
        for j, (target, cond) in enumerate(zip(r["video"], r["vace_video"])):
            expected = (int(r["start"]) + j) % FRAME_COUNT
            if Path(target).name != f"F{expected:02d}.png" or Path(cond).name != f"F{expected:02d}.png":
                violations.append({"id": r["id"], "j": j, "reason": "Fxx mismatch"})
            # Resolve paths relative to the project root.  A condition path
            # containing the clean root is an explicit target-leakage error.
            if str(clean_root) in str(Path(cond).resolve()):
                violations.append({"id": r["id"], "j": j, "reason": "clean in condition"})
        ref = Path(r["vace_reference_image"][0])
        if ref.name != f"F{int(r['start']):02d}.png":
            violations.append({"id": r["id"], "reason": "reference mismatch"})
    return {
        "rows": len(rows),
        "start_histogram": {str(i): starts.count(i) for i in range(FRAME_COUNT)},
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjusted-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--train-items", type=int, default=TRAIN_ITEMS)
    args = ap.parse_args()
    if args.train_items != TRAIN_ITEMS:
        raise ValueError(f"this contract is fixed at exactly {TRAIN_ITEMS} train items")

    adjusted = frame_paths(args.adjusted_dir)
    raw = frame_paths(args.raw_dir)
    clean = frame_paths(args.clean_dir)
    if [p.name for p in adjusted] != [p.name for p in raw] or [p.name for p in adjusted] != [p.name for p in clean]:
        raise AssertionError("condition and target frame names are not identical F00-F71")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = [row(adjusted, clean, i % FRAME_COUNT, i, "train") for i in range(args.train_items)]
    val = [row(adjusted, clean, s, i, "val") for i, s in enumerate(VALIDATION_STARTS)]
    train_path = args.output_dir / "dynamic_adjusted_train_manifest.json"
    val_path = args.output_dir / "dynamic_adjusted_validation_manifest.json"
    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False) + "\n")
    val_path.write_text(json.dumps(val, indent=2, ensure_ascii=False) + "\n")

    train_audit = audit_rows(train, args.clean_dir.resolve())
    val_audit = audit_rows(val, args.clean_dir.resolve())
    if train_audit["status"] != "PASS" or val_audit["status"] != "PASS":
        raise RuntimeError("manifest audit failed; refusing to write a PASS contract")

    contract = {
        "status": "PASS",
        "contract_name": "P49 dynamic-adjusted Phase-C fresh retrain",
        "condition_variant": "tune4 adjusted Gaussian + P48.3 trajectory-aware full-render suppression",
        "frame_count": FRAME_COUNT,
        "clip_len": CLIP_LEN,
        "train_items": args.train_items,
        "validation_starts": list(VALIDATION_STARTS),
        "adjusted_condition": str(args.adjusted_dir.resolve()),
        "raw_condition_for_dual_audit": str(args.raw_dir.resolve()),
        "clean_target": str(args.clean_dir.resolve()),
        "pairing": "condition Fxx and target Fxx selected by identical cyclic indices",
        "lossless_png_input": True,
        "mp4_decoding_for_training": False,
        "inference_clean_input": False,
        "target_leakage": "Clean RGB appears only under video target; vace_video and reference contain adjusted Gaussian renders only",
        "condition_f00_sha256": sha256(adjusted[0]),
        "condition_keyframe_sha256": {f"F{i:02d}": sha256(adjusted[i]) for i in (10, 11, 12, 13, 14, 17, 24, 28, 29, 30)},
        "clean_f00_sha256": sha256(clean[0]),
        "raw_f00_sha256": sha256(raw[0]),
        "files": {
            "train": str(train_path.resolve()),
            "validation": str(val_path.resolve()),
            "train_audit": train_audit,
            "validation_audit": val_audit,
        },
    }
    (args.output_dir / "P49_DYNAMIC_ADJUSTED_2800_DATA_CONTRACT.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "NO_TARGET_LEAKAGE.md").write_text(
        "# P49 dynamic-adjusted target leakage audit\n\n"
        "- Clean RGB is used only as the diffusion target (`video`).\n"
        "- `vace_video` and `vace_reference_image` contain only the dynamic-adjusted Gaussian PNGs.\n"
        "- No Clean RGB/depth/lines/normals/masks are inference inputs.\n"
        "- Every clip is exact cyclic Fxx-to-Fxx, with fixed validation starts 0/17/31/50.\n"
        "- The 2800-item train manifest covers all 72 starts before repeating.\n"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
