import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p39_full72_qa import summarize_transitions, validate_video_contract


class P39Full72QATests(unittest.TestCase):
    def test_672_contract_accepts_matching_video(self) -> None:
        video = np.zeros((72, 672, 672, 3), dtype=np.float32)

        validate_video_contract(
            video,
            fps=12.0,
            label="candidate",
            path=Path("candidate.mp4"),
            width=672,
            height=672,
            expected_fps=12,
        )

    def test_672_contract_rejects_640_video(self) -> None:
        video = np.zeros((72, 640, 640, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "full72 contract failed"):
            validate_video_contract(
                video,
                fps=12.0,
                label="candidate",
                path=Path("candidate.mp4"),
                width=672,
                height=672,
                expected_fps=12,
            )

    def test_direct_method_checks_rotated_model_boundary_and_loop(self) -> None:
        clean = np.zeros((72, 16, 16, 3), dtype=np.float32)
        output = clean.copy()

        result = summarize_transitions(
            output,
            clean,
            method="DIRECT73_ROT16_KEEP72",
            model_boundary=(15, 16),
        )

        self.assertEqual(result["segment_seams"], "PASS")
        self.assertEqual(result["model_boundary"], "PASS")
        self.assertEqual(result["model_boundary_transition"], [15, 16])
        self.assertEqual(result["loop_f71_to_f00"], "PASS")


if __name__ == "__main__":
    unittest.main()
