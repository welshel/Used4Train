"""Build P51A source manifests and lossless cyclic-clip concat lists."""
import argparse
import json
from pathlib import Path

from p51a_data_contract import (
    CLEAN_DIR,
    CLIP_LENGTH,
    CONDITION_DIR,
    PROJECT_ROOT,
    build_training_rows,
    cyclic_indices,
)

OUTPUT_ROOT = PROJECT_ROOT / 'outputs/p51a_bernini_r13b_single_scene'
FPS = 16
PROMPT = (
    'Translate this Tripo condition video into a realistic, temporally stable roomtour. '
    'Preserve scene layout and camera motion; remove synthetic artifacts, ghosting, translucency, '
    'and duplicate objects while retaining realistic indoor appearance.'
)


def concat_file_lines(frame_dir: Path, start_index: int):
    """Create FFmpeg concat entries for one exact ordered cyclic clip."""
    return [f"file '{frame_dir / f'F{frame_index:02d}.png'}'" for frame_index in cyclic_indices(start_index)]


def build_parquet_row(start_index: int, source_video: str, target_video: str):
    """Create one official renderer-v2v row: source video has no loss, clean target does."""
    frame_indices = cyclic_indices(start_index)
    messages = [
        {'type': 'text', 'text': PROMPT, 'has_loss': 0},
        {'type': 'video', 'has_loss': 0},
        {'type': 'video_gen', 'has_loss': 1},
    ]
    return {
        'scene_id': 'P51A_SINGLE_SCENE',
        'sample_id': f'cyclic_start{start_index:02d}',
        'start_index': start_index,
        'clip_length': CLIP_LENGTH,
        'resolution': [672, 672],
        'fps': FPS,
        'input_frame_indices': frame_indices,
        'target_frame_indices': list(frame_indices),
        'inputs': json.dumps(messages, separators=(',', ':')),
        'videos': [
            {'video_path': source_video, 'duration': CLIP_LENGTH / FPS, 'crop_method': 'left'},
            {'video_path': target_video, 'duration': CLIP_LENGTH / FPS, 'crop_method': 'left'},
        ],
    }


def write_concat_lists(output_root: Path):
    concat_root = output_root / 'clips' / '_concat'
    for start_index in range(72):
        for role, frame_dir in (('adjusted', CONDITION_DIR), ('clean', CLEAN_DIR)):
            path = concat_root / role / f'cyclic_start{start_index:02d}.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('\n'.join(concat_file_lines(frame_dir, start_index)) + '\n')


def write_source_manifest(output_root: Path):
    path = output_root / 'manifests' / 'training_source.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for source_row in build_training_rows():
            start_index = source_row['start_index']
            row = build_parquet_row(
                start_index,
                str(output_root / 'clips' / 'adjusted' / f'cyclic_start{start_index:02d}.mkv'),
                str(output_root / 'clips' / 'clean' / f'cyclic_start{start_index:02d}.mkv'),
            )
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-concat', action='store_true')
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    if not args.write_concat and not args.write_manifest:
        parser.error('choose --write-concat and/or --write-manifest')
    if args.write_concat:
        write_concat_lists(OUTPUT_ROOT)
    if args.write_manifest:
        print(write_source_manifest(OUTPUT_ROOT))


if __name__ == '__main__':
    main()
