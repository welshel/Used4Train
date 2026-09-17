#!/usr/bin/env python3
"""Build the P49A exact-pair, cyclic 33-frame RGB/VACE data contract."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image

FRAME_COUNT = 72
DEFAULT_CLIP_LEN = 33
VALIDATION_STARTS = (0, 17, 31, 50)
PROMPT = "a realistic indoor room tour"

def _frames(folder: Path) -> list[Path]:
    paths = [folder / f"F{i:02d}.png" for i in range(FRAME_COUNT)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing F00-F71 frames ({len(missing)}): {missing[:3]}")
    return paths

def _check_rgb(paths: list[Path], expected_size=(896, 896)) -> None:
    for p in paths:
        with Image.open(p) as im:
            if im.mode != "RGB":
                raise ValueError(f"{p} is {im.mode}; expected RGB")
            if im.size != expected_size:
                raise ValueError(f"{p} is {im.size}; expected {expected_size}")

def _clip(paths: list[Path], start: int, clip_len: int) -> list[str]:
    if not 0 <= int(start) < FRAME_COUNT:
        raise ValueError(f"start must be in [0,71], got {start}")
    if clip_len != 33:
        raise ValueError("P49A requires exactly 33 frames")
    return [str(paths[(int(start) + j) % FRAME_COUNT]) for j in range(clip_len)]

def build_clip_manifest(condition_dir, clean_dir, starts=range(FRAME_COUNT), clip_len=DEFAULT_CLIP_LEN):
    condition_dir, clean_dir = Path(condition_dir), Path(clean_dir)
    condition = _frames(condition_dir); clean = _frames(clean_dir)
    _check_rgb(condition); _check_rgb(clean)
    rows = []
    for start in starts:
        start = int(start)
        rows.append({
            "id": f"train_start{start:02d}", "start": start, "clip_len": clip_len,
            "prompt": PROMPT,
            "video": _clip(clean, start, clip_len),
            "vace_video": _clip(condition, start, clip_len),
            "vace_reference_image": [str(condition[start])],
        })
    return rows

def build_validation_manifest(condition_dir, clean_dir, starts=VALIDATION_STARTS, clip_len=DEFAULT_CLIP_LEN):
    rows = build_clip_manifest(condition_dir, clean_dir, starts=starts, clip_len=clip_len)
    for row in rows: row["id"] = f"val_start{row['start']:02d}"
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--train-items", type=int, default=700)
    args = ap.parse_args()
    if args.train_items < 72:
        raise ValueError("train-items must cover all 72 starts")
    base = build_clip_manifest(args.condition_dir, args.clean_dir, starts=[i % FRAME_COUNT for i in range(args.train_items)])
    # Keep exactly the requested number of optimizer samples and balanced cyclic starts.
    for i, row in enumerate(base): row["id"] = f"train_{i:04d}_start{row['start']:02d}"
    val = build_validation_manifest(args.condition_dir, args.clean_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_manifest.json").write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "validation_manifest.json").write_text(json.dumps(val, indent=2, ensure_ascii=False) + "\n")
    contract = {
        "condition_dir": str(args.condition_dir.resolve()), "target_dir": str(args.clean_dir.resolve()),
        "condition_source": "P48.3 lossless dynamic RGB PNGs", "target_source": "P48 exact clean RGB PNGs",
        "frame_count": FRAME_COUNT, "clip_len": DEFAULT_CLIP_LEN, "resolution_source": [896,896],
        "training_resolution": [672,672], "train_items": len(base), "validation_starts": list(VALIDATION_STARTS),
        "train_start_histogram": {str(i): sum(r["start"] == i for r in base) for i in range(FRAME_COUNT)},
        "frame_pairing": "target Fxx and condition Fxx are selected by the same cyclic index; no MP4 decode",
        "target_leakage": "Clean RGB is only video target; no Clean pixels/masks/depth are included in vace_video or reference inputs",
        "prompt": PROMPT,
        "files": {"train": str(args.output_dir / "train_manifest.json"), "validation": str(args.output_dir / "validation_manifest.json")},
    }
    (args.output_dir / "P49A_DATA_CONTRACT.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status":"PASS", "train_items":len(base), "validation_starts":list(VALIDATION_STARTS), "contract":str(args.output_dir / "P49A_DATA_CONTRACT.json")}, indent=2))

if __name__ == "__main__": main()
