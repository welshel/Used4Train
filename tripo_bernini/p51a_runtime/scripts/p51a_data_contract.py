"""Auditable data contract for P51A paired video-to-video training."""
from pathlib import Path

PROJECT_ROOT = Path('/fs1/private/user/baitongyuan/projects/liuzh')
CONDITION_DIR = PROJECT_ROOT / 'outputs/phase_b_adjusted_dynamic/condition_adjusted_rgb'
CLEAN_DIR = PROJECT_ROOT / 'outputs/p48_tripo_clean_aligned/clean_rgb'
RAW_DIR = PROJECT_ROOT / 'outputs/p48_3_tripo_dynamic_cleanup/06_full72/condition_rgb_dynamic'
FRAME_STEMS = [f'F{i:02d}' for i in range(72)]
CLIP_LENGTH = 33
VALIDATION_STARTS = [0, 17, 31, 50]


def cyclic_indices(start_index: int, clip_length: int = CLIP_LENGTH, total_frames: int = 72):
    """Return an ordered cyclic temporal clip without altering source frames."""
    if not 0 <= start_index < total_frames:
        raise ValueError(f'start_index must be in [0, {total_frames - 1}], got {start_index}')
    if clip_length <= 0:
        raise ValueError('clip_length must be positive')
    return [(start_index + offset) % total_frames for offset in range(clip_length)]


def _frame_paths(directory: Path, frame_indices):
    return [str(directory / f'F{index:02d}.png') for index in frame_indices]


def build_training_rows():
    """Build 72 RGB-to-RGB cyclic clips with exact input/target frame alignment."""
    rows = []
    for start_index in range(72):
        frame_indices = cyclic_indices(start_index)
        rows.append(
            {
                'scene_id': 'P51A_SINGLE_SCENE',
                'sample_id': f'cyclic_start{start_index:02d}',
                'start_index': start_index,
                'clip_length': CLIP_LENGTH,
                'resolution': [672, 672],
                'input_frame_indices': frame_indices,
                'target_frame_indices': list(frame_indices),
                'input_frame_paths': _frame_paths(CONDITION_DIR, frame_indices),
                'target_frame_paths': _frame_paths(CLEAN_DIR, frame_indices),
                'source_video_role': 'adjusted_condition',
                'target_video_role': 'clean_target',
            }
        )
    return rows


def build_inference_plan(route: str):
    """Return source-only inference metadata; clean remains excluded by construction."""
    condition_dir = {'adjusted': CONDITION_DIR, 'raw': RAW_DIR}.get(route)
    if condition_dir is None:
        raise ValueError("route must be 'adjusted' or 'raw'")
    return {
        'route': route,
        'condition_dir': str(condition_dir),
        'output_frame_indices': list(range(72)),
        'clip_length': CLIP_LENGTH,
        'validation_starts': list(VALIDATION_STARTS),
    }
