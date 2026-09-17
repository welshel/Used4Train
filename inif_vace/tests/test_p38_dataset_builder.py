import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p38_dataset_builder import build_clip_records, deterministic_resize_pair


class DatasetBuilderTests(unittest.TestCase):
    def test_dataset_builder_materializes_l33_cyclic_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            rows = [
                {
                    "frame_index": index,
                    "frame_stem": f"{index:06d}",
                    "condition_rgb": str(tmp_path / f"condition_{index:06d}.png"),
                    "clean_rgb": str(tmp_path / f"clean_{index:06d}.png"),
                }
                for index in range(72)
            ]
            records = build_clip_records(rows, [60], tmp_path / "clips", split="validation")
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["sample_id"], "L33_start60_33f")
            self.assertEqual(record["frame_indices"], list(range(60, 72)) + list(range(0, 21)))
            self.assertEqual(record["condition_frame_stems"], record["clean_frame_stems"])

    def test_resize_pair_uses_lanczos_from_master_frames(self) -> None:
        pixels = np.zeros((9, 9, 3), dtype=np.uint8)
        pixels[::2, ::2] = 255
        condition = Image.fromarray(pixels, mode="RGB")
        clean = Image.fromarray(255 - pixels, mode="RGB")

        condition_out, clean_out = deterministic_resize_pair(condition, clean, (7, 7))

        expected_condition = condition.resize((7, 7), Image.Resampling.LANCZOS)
        expected_clean = clean.resize((7, 7), Image.Resampling.LANCZOS)
        self.assertTrue(np.array_equal(np.asarray(condition_out), np.asarray(expected_condition)))
        self.assertTrue(np.array_equal(np.asarray(clean_out), np.asarray(expected_clean)))


if __name__ == "__main__":
    unittest.main()
