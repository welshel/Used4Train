import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p38_validate import load_validation_records, validation_prefix, validation_seed


class ValidationContractTests(unittest.TestCase):
    def test_validation_seed_and_prefix_are_frozen(self) -> None:
        self.assertEqual(validation_seed(31, 1337), 1368)
        self.assertEqual(validation_prefix(0), "V0_start00")
        self.assertEqual(validation_prefix(50), "V3_start50")

    def test_validation_manifest_requires_fixed_33f_starts_at_640(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            rows = []
            for start in (0, 17, 31, 50):
                clean = tmp_path / f"clean_{start}.mp4"
                condition = tmp_path / f"condition_{start}.mp4"
                clean.write_bytes(b"clean")
                condition.write_bytes(b"condition")
                indices = [(start + offset) % 72 for offset in range(33)]
                stems = [f"{index:06d}" for index in indices]
                rows.append(
                    {
                        "sample_id": f"L33_start{start:02d}_33f",
                        "split": "validation",
                        "start_index": start,
                        "frame_indices": indices,
                        "condition_frame_stems": stems,
                        "clean_frame_stems": stems,
                        "video": str(clean),
                        "vace_video": str(condition),
                        "prompt": "a realistic indoor room tour",
                        "num_frames": 33,
                        "width": 640,
                        "height": 640,
                    }
                )
            manifest = tmp_path / "validation.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.assertEqual(
                [row["start_index"] for row in load_validation_records(manifest)],
                [0, 17, 31, 50],
            )


if __name__ == "__main__":
    unittest.main()
