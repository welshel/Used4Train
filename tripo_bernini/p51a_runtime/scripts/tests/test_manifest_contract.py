import json
import sys
import unittest
from pathlib import Path

ROOT = Path('/fs1/private/user/baitongyuan/projects/liuzh/outputs/p51a_bernini_r13b_single_scene')
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from p51a_data_contract import (  # RED: module does not exist yet
    CLEAN_DIR,
    CONDITION_DIR,
    RAW_DIR,
    FRAME_STEMS,
    VALIDATION_STARTS,
    cyclic_indices,
    build_training_rows,
    build_inference_plan,
)


class ManifestContractTest(unittest.TestCase):
    def test_source_sequences_are_exact_f00_to_f71(self):
        expected = [f'F{i:02d}' for i in range(72)]
        self.assertEqual(FRAME_STEMS, expected)
        for directory in (CONDITION_DIR, CLEAN_DIR, RAW_DIR):
            actual = sorted(path.stem for path in directory.glob('*.png'))
            self.assertEqual(actual, expected, directory)

    def test_cyclic_clip_preserves_order_and_wraps_only_after_f71(self):
        self.assertEqual(cyclic_indices(0), list(range(33)))
        self.assertEqual(cyclic_indices(50), list(range(50, 72)) + list(range(11)))
        self.assertEqual(cyclic_indices(71), [71] + list(range(32)))

    def test_training_rows_have_paired_condition_and_clean_frame_indices(self):
        rows = build_training_rows()
        self.assertEqual(len(rows), 72)
        self.assertEqual([row['start_index'] for row in rows], list(range(72)))
        for row in rows:
            self.assertEqual(row['input_frame_indices'], row['target_frame_indices'])
            self.assertEqual(len(row['input_frame_indices']), 33)
            self.assertEqual(row['source_video_role'], 'adjusted_condition')
            self.assertEqual(row['target_video_role'], 'clean_target')

    def test_validation_starts_are_fixed_and_inference_never_uses_clean_as_condition(self):
        self.assertEqual(VALIDATION_STARTS, [0, 17, 31, 50])
        adjusted = build_inference_plan(route='adjusted')
        raw = build_inference_plan(route='raw')
        self.assertEqual(adjusted['condition_dir'], str(CONDITION_DIR))
        self.assertEqual(raw['condition_dir'], str(RAW_DIR))
        for plan in (adjusted, raw):
            self.assertNotIn(str(CLEAN_DIR), json.dumps(plan, sort_keys=True))
            self.assertEqual(plan['output_frame_indices'], list(range(72)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
