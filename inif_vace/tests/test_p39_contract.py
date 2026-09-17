import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p38_contract import validate_spatial_resolution


class P39ContractTests(unittest.TestCase):
    def test_h704_is_a_supported_square_resolution(self) -> None:
        self.assertEqual(validate_spatial_resolution(704, 704), (704, 704))

    def test_h704_still_rejects_non_square_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported resolutions"):
            validate_spatial_resolution(704, 672)


if __name__ == "__main__":
    unittest.main()
