import json
import sys
import unittest
from pathlib import Path

ROOT = Path('/fs1/private/user/baitongyuan/projects/liuzh/outputs/p51a_bernini_r13b_single_scene')
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from p51a_build_dataset import build_parquet_row, concat_file_lines  # RED: absent
from p51a_data_contract import CLEAN_DIR, CONDITION_DIR, cyclic_indices


class DatasetSerializationTest(unittest.TestCase):
    def test_concat_lists_are_cyclic_and_follow_source_frame_order(self):
        lines = concat_file_lines(CONDITION_DIR, start_index=50)
        expected_indices = cyclic_indices(50)
        self.assertEqual(len(lines), 33)
        self.assertEqual(lines[0], f"file '{CONDITION_DIR / 'F50.png'}'")
        self.assertEqual(lines[-1], f"file '{CONDITION_DIR / 'F10.png'}'")
        self.assertEqual([int(line.rsplit('F', 1)[1][:2]) for line in lines], expected_indices)

    def test_renderer_row_marks_only_clean_video_as_generation_target(self):
        row = build_parquet_row(start_index=17, source_video='/tmp/condition.mkv', target_video='/tmp/clean.mkv')
        messages = json.loads(row['inputs'])
        self.assertEqual([message['type'] for message in messages], ['text', 'video', 'video_gen'])
        self.assertFalse(messages[1].get('has_loss', 0))
        self.assertEqual(messages[2].get('has_loss'), 1)
        self.assertEqual(row['videos'][0]['video_path'], '/tmp/condition.mkv')
        self.assertEqual(row['videos'][1]['video_path'], '/tmp/clean.mkv')
        self.assertEqual(row['start_index'], 17)
        self.assertEqual(row['input_frame_indices'], row['target_frame_indices'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
