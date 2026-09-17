import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p38_full72_qa import classify_transition, summarize_transitions, transition_metrics


class Full72QATests(unittest.TestCase):
    def test_identical_output_and_clean_have_perfect_loop_transition(self) -> None:
        clean = np.zeros((4, 16, 16, 3), dtype=np.float32)
        clean[0] = 0.25
        clean[-1] = 0.20

        result = transition_metrics(clean, clean, previous_index=3, next_index=0)

        self.assertEqual(result["transition_delta_error"], 0.0)
        self.assertEqual(result["jump_excess"], 0.0)
        self.assertEqual(classify_transition(result), "PASS")

    def test_large_generated_jump_fails_transition_guard(self) -> None:
        clean = np.zeros((4, 16, 16, 3), dtype=np.float32)
        output = clean.copy()
        output[0] = 1.0

        result = transition_metrics(output, clean, previous_index=3, next_index=0)

        self.assertEqual(classify_transition(result), "FAIL")

    def test_direct_method_has_no_segment_stitch_seams(self) -> None:
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
        self.assertEqual(
            [item["kind"] for item in result["transitions"]],
            ["model_boundary", "loop"],
        )

    def test_overlap_method_still_checks_segment_stitches(self) -> None:
        clean = np.zeros((72, 16, 16, 3), dtype=np.float32)
        output = clean.copy()
        output[36] = 1.0

        result = summarize_transitions(output, clean, method="OVERLAP_33F")

        self.assertEqual(result["segment_seams"], "FAIL")
        self.assertIsNone(result["model_boundary"])
        self.assertEqual(len(result["transitions"]), 4)


if __name__ == "__main__":
    unittest.main()
