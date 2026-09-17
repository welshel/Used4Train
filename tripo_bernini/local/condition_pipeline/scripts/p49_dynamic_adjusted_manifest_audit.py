#!/usr/bin/env python3
"""Independent file-level audit for the 2800-item dynamic-adjusted contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    args = ap.parse_args()
    out = args.manifest_dir
    train = json.loads((out / "dynamic_adjusted_train_manifest.json").read_text())
    val = json.loads((out / "dynamic_adjusted_validation_manifest.json").read_text())
    clean = str((args.project_root / "outputs/p48_tripo_clean_aligned/clean_rgb").resolve())
    cond = str((args.project_root / "outputs/phase_b_adjusted_dynamic/condition_adjusted_rgb").resolve())
    errors, missing, clean_in_condition = [], [], []
    starts = [int(r["start"]) for r in train]
    for rows in (train, val):
        for r in rows:
            start = int(r["start"])
            if len(r.get("video", [])) != 33 or len(r.get("vace_video", [])) != 33:
                errors.append({"id": r["id"], "reason": "clip_len"})
            for j, (target, condition) in enumerate(zip(r["video"], r["vace_video"])):
                expected = (start + j) % 72
                if Path(target).name != f"F{expected:02d}.png" or Path(condition).name != f"F{expected:02d}.png":
                    errors.append({"id": r["id"], "j": j, "reason": "Fxx mismatch"})
                tp, cp = (args.project_root / target).resolve(), (args.project_root / condition).resolve()
                if not tp.is_file() or not cp.is_file():
                    missing.append({"target": str(tp), "condition": str(cp)})
                if str(cp).startswith(clean):
                    clean_in_condition.append({"id": r["id"], "path": str(cp)})
            reference = (args.project_root / r["vace_reference_image"][0]).resolve()
            if not reference.is_file():
                missing.append({"reference": str(reference)})
            if str(reference).startswith(clean):
                clean_in_condition.append({"id": r["id"], "reference": str(reference)})

    inventory = []
    for i in range(72):
        p = args.project_root / "outputs/phase_b_adjusted_dynamic/condition_adjusted_rgb" / f"F{i:02d}.png"
        inventory.append({"frame": f"F{i:02d}", "exists": p.is_file(), "sha256": sha256(p) if p.is_file() else None})
    result = {
        "status": "PASS" if not (errors or missing or clean_in_condition) else "FAIL",
        "train_rows": len(train),
        "validation_rows": len(val),
        "train_clip_references": len(train) * 33,
        "validation_clip_references": len(val) * 33,
        "train_start_histogram": {str(i): starts.count(i) for i in range(72)},
        "validation_starts": [int(r["start"]) for r in val],
        "frame_pairing_errors": errors,
        "missing_files": missing,
        "clean_condition_leakage": clean_in_condition,
        "condition_root": cond,
        "clean_root": clean,
        "all_unique_condition_frames_exist": all(x["exists"] for x in inventory),
        "condition_frame_inventory": inventory,
        "target_entries_are_clean": all(str((args.project_root / p).resolve()).startswith(clean) for r in train + val for p in r["video"]),
        "condition_entries_are_dynamic_adjusted": all(str((args.project_root / p).resolve()).startswith(cond) for r in train + val for p in r["vace_video"]),
        "all_condition_entries_not_clean": all(not str((args.project_root / p).resolve()).startswith(clean) for r in train + val for p in r["vace_video"]),
        "reference_entries_not_clean": all(not str((args.project_root / r["vace_reference_image"][0]).resolve()).startswith(clean) for r in train + val),
    }
    path = out / "P49_DYNAMIC_ADJUSTED_MANIFEST_AUDIT.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "train_rows", "validation_rows", "frame_pairing_errors", "missing_files", "clean_condition_leakage", "all_unique_condition_frames_exist", "target_entries_are_clean", "condition_entries_are_dynamic_adjusted", "all_condition_entries_not_clean", "reference_entries_not_clean")}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
