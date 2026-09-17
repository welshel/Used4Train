#!/usr/bin/env python3
"""Assemble the exact F00--F71 timeline from bounded Bernini 33-frame chunks."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# target_start, source_chunk_start, first source offset, number of frames
SEGMENTS = ((0, 0, 0, 33), (33, 31, 2, 17), (50, 50, 0, 22))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("adjusted", "raw"), required=True)
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--output-video", required=True)
    return parser.parse_args()


def run(command):
    subprocess.run(command, check=True)


def extract(video: Path, output_dir: Path):
    output_dir.mkdir()
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vsync", "0", str(output_dir / "frame%03d.png")])
    frames = sorted(output_dir.glob("frame*.png"))
    if len(frames) != 33:
        raise RuntimeError(f"expected 33 frames from {video}, got {len(frames)}")
    return frames


def main():
    args = parse_args()
    chunks = Path(args.chunks_dir)
    output = Path(args.output_video)
    marker = output.with_suffix(".P51A_ASSEMBLY_COMPLETE.json")
    if marker.is_file():
        if output.is_file():
            print(f"already complete: {output}")
            return
        raise RuntimeError(f"assembly marker exists without output: {marker}")
    required = {start: chunks / f"start{start:02d}.mp4" for _, start, _, _ in SEGMENTS}
    for start, video in required.items():
        completion = video.with_suffix(".P51A_GENERATION_COMPLETE.json")
        if not video.is_file() or not completion.is_file():
            raise FileNotFoundError(f"missing complete generation for start {start}: {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    for stale in (temporary, output):
        if stale.exists():
            archived = stale.with_name(f"{stale.name}.interrupted_{datetime.now().strftime('%Y%m%dT%H%M%S%z')}")
            shutil.move(str(stale), str(archived))
    with tempfile.TemporaryDirectory(prefix=f"p51a_assemble_{args.route}_", dir=output.parent) as temp:
        temp_root = Path(temp)
        decoded = {start: extract(video, temp_root / f"start{start:02d}") for start, video in required.items()}
        selected = temp_root / "selected"
        selected.mkdir()
        mapping = []
        for target_start, source_start, offset, count in SEGMENTS:
            for local in range(count):
                target = target_start + local
                source_offset = offset + local
                shutil.copy2(decoded[source_start][source_offset], selected / f"F{target:02d}.png")
                mapping.append({"target_frame": target, "source_chunk_start": source_start, "source_clip_offset": source_offset})
        if sorted(path.name for path in selected.glob("F*.png")) != [f"F{i:02d}.png" for i in range(72)]:
            raise RuntimeError("assembled Full72 frame mapping is incomplete")
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "16", "-start_number", "0",
            "-i", str(selected / "F%02d.png"), "-frames:v", "72", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(temporary),
        ])
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not create the assembled Full72 video")
    os.replace(temporary, output)
    payload = {
        "status": "complete",
        "route": args.route,
        "frame_count": 72,
        "frame_mapping": mapping,
        "clean_used_as_inference_input": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = marker.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, marker)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
