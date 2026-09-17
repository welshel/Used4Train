#!/usr/bin/env python3
"""Build source-only Full72 inputs and three raw cyclic clips for P51A QA."""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/fs1/private/user/baitongyuan/projects/liuzh")
OUT = ROOT / "outputs/p51a_bernini_r13b_single_scene"
SOURCES = {
    "adjusted": ROOT / "outputs/phase_b_adjusted_dynamic/condition_adjusted_rgb",
    "raw": ROOT / "outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic",
    "clean": ROOT / "outputs/p48_tripo_clean_aligned/clean_rgb",
}
STARTS = (0, 31, 50)


def run_ffmpeg(command):
    subprocess.run(command, check=True)


def archive_if_exists(path: Path):
    if path.exists():
        archived = path.with_name(f"{path.name}.interrupted_{datetime.now().strftime('%Y%m%dT%H%M%S%z')}")
        shutil.move(str(path), str(archived))


def assert_frames(directory: Path):
    expected = [directory / f"F{i:02d}.png" for i in range(72)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source frames: {missing[:3]}")


def make_full_video(source: Path, output: Path):
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    archive_if_exists(temporary)
    archive_if_exists(output)
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "16", "-start_number", "0",
        "-i", str(source / "F%02d.png"), "-frames:v", "72", "-vf", "scale=672:672",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(temporary),
    ])
    os.replace(temporary, output)


def make_raw_clip(source: Path, start: int, output: Path, work: Path):
    listing = work / f"raw_cyclic_start{start:02d}.txt"
    listing.write_text("\n".join(f"file '{source / f'F{(start + offset) % 72:02d}.png'}'" for offset in range(33)) + "\n")
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    archive_if_exists(temporary)
    archive_if_exists(output)
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-r", "16",
        "-i", str(listing), "-vf", "scale=672:672", "-vsync", "0", "-an", "-c:v", "ffv1", "-pix_fmt", "rgb24", str(temporary),
    ])
    os.replace(temporary, output)


def main():
    inputs = OUT / "full72/inputs"
    marker = inputs / "P51A_INPUTS_COMPLETE.json"
    if marker.is_file():
        print(f"already complete: {marker}")
        return
    inputs.mkdir(parents=True, exist_ok=True)
    for source in SOURCES.values():
        assert_frames(source)
    make_full_video(SOURCES["adjusted"], inputs / "condition_adjusted_full72.mp4")
    make_full_video(SOURCES["raw"], inputs / "condition_raw_full72.mp4")
    make_full_video(SOURCES["clean"], inputs / "clean_full72.mp4")
    raw_work = inputs / "concat"
    raw_work.mkdir(exist_ok=True)
    for start in STARTS:
        make_raw_clip(SOURCES["raw"], start, inputs / f"raw_cyclic_start{start:02d}.mkv", raw_work)
    payload = {
        "status": "complete",
        "frame_count": 72,
        "frame_stems": [f"F{i:02d}" for i in range(72)],
        "raw_cyclic_starts": list(STARTS),
        "sources": {key: str(value) for key, value in SOURCES.items()},
        "clean_used_as_inference_input": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = marker.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, marker)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
