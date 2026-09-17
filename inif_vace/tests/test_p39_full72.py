import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p39_full72 import (
    canonicalize_rotated_frames,
    direct73_indices,
    inference_contract,
    validate_full72_resolution,
)


class P39Full72Tests(unittest.TestCase):
    def test_accepts_promoted_704_square_inference_resolution(self) -> None:
        self.assertEqual(validate_full72_resolution(704, 704), (704, 704))

    def test_rejects_non_square_inference_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported resolutions"):
            validate_full72_resolution(704, 672)

    def test_rotated_direct73_places_boundary_at_start16(self) -> None:
        expected = list(range(16, 72)) + list(range(0, 16)) + [16]
        self.assertEqual(direct73_indices(start_index=16), expected)
        local_frames = [f"frame_{index}" for index in expected[:-1]]
        self.assertEqual(
            canonicalize_rotated_frames(local_frames, start_index=16),
            [f"frame_{index}" for index in range(72)],
        )

    def test_inference_contract_excludes_clean_inputs(self) -> None:
        contract = inference_contract()
        self.assertEqual(contract["clean_inputs"], [])
        self.assertIsNone(contract["vace_reference_image"])
        self.assertFalse(contract["target_leakage"])


if __name__ == "__main__":
    unittest.main()
