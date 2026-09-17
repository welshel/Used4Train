import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p38_full72 import (
    canonicalize_rotated_frames,
    direct73_indices,
    inference_contract,
    overlap_windows,
    validate_condition_rows,
)


class Full72ContractTests(unittest.TestCase):
    def test_direct73_repeats_only_frame_zero_for_wan_4n_plus_1(self) -> None:
        self.assertEqual(direct73_indices(), list(range(72)) + [0])

    def test_rotated_direct73_places_model_boundary_at_selected_transition(self) -> None:
        expected = list(range(16, 72)) + list(range(0, 16)) + [16]
        self.assertEqual(direct73_indices(start_index=16), expected)
        local_frames = [f"frame_{index}" for index in expected[:-1]]
        canonical = canonicalize_rotated_frames(local_frames, start_index=16)
        self.assertEqual(canonical, [f"frame_{index}" for index in range(72)])

    def test_overlap_windows_cover_every_output_frame_once(self) -> None:
        windows = overlap_windows()
        self.assertEqual([window["segment_start"] for window in windows], [0, 18, 36, 54])
        self.assertTrue(all(len(window["context_indices"]) == 33 for window in windows))
        self.assertTrue(all(len(window["output_indices"]) == 18 for window in windows))
        covered = [index for window in windows for index in window["output_indices"]]
        self.assertEqual(covered, list(range(72)))
        self.assertEqual(windows[0]["context_indices"][:7], list(range(65, 72)))
        self.assertEqual(windows[-1]["context_indices"][-8:], list(range(0, 8)))

    def test_inference_contract_has_no_clean_target_inputs(self) -> None:
        contract = inference_contract()
        self.assertEqual(contract["scene_input"], "native_gaussian_rgb_condition")
        self.assertEqual(contract["vace_reference_image"], None)
        self.assertEqual(contract["clean_inputs"], [])
        self.assertFalse(contract["target_leakage"])

    def test_condition_manifest_rejects_clean_sidecars(self) -> None:
        rows = [
            {
                "frame_index": index,
                "frame_stem": f"{index:06d}",
                "condition_rgb": f"/condition/{index:06d}.png",
            }
            for index in range(72)
        ]
        self.assertEqual(validate_condition_rows(rows), 72)
        rows[0]["clean_rgb"] = "/forbidden/clean.png"
        with self.assertRaisesRegex(ValueError, "condition-only"):
            validate_condition_rows(rows)


if __name__ == "__main__":
    unittest.main()
