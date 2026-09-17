"""Focused regression checks for the exact clean-camera adapter."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("render_tripo_clean_path.py")
SPEC = importlib.util.spec_from_file_location("p48", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class ExactPathContractTests(unittest.TestCase):
    def test_floor_axis_constraint_maps_tripo_y_to_clean_z(self):
        r = MOD.axis_permutation_matrix((2, 0, 1), (1, -1, -1))
        self.assertTrue(np.allclose(r @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0]))
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0)

    def test_similarity_rotates_means_scales_and_quaternions_together(self):
        r = MOD.axis_permutation_matrix((2, 0, 1), (1, -1, -1))
        m = MOD.make_similarity(4.0, r, [3.0, -2.0, 1.0])
        self.assertTrue(np.allclose(MOD.apply_similarity(np.array([[0., 1., 0.]]), m), [[3., -2., -3.]]))
        q = MOD.transform_quaternions(np.array([[1., 0., 0., 0.]], np.float32), m)
        self.assertAlmostEqual(float(np.linalg.norm(q[0])), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
