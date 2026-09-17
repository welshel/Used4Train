#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("p49a_postprocess", Path(__file__).with_name("p49a_postprocess.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PostprocessTests(unittest.TestCase):
    def test_four_start_windows_cover_every_frame(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for start in M.STARTS:
                folder = M.validation_dir(root, 200, start) / "frames"
                folder.mkdir(parents=True)
                for pos in range(33):
                    idx = (start + pos) % M.FRAME_COUNT
                    (folder / f"F{idx:02d}.png").touch()
            original = M.image_array
            M.image_array = lambda path, size=672: np.full((2, 2, 3), int(path.stem[1:]), dtype=np.uint8)
            try:
                frames, provenance = M.assemble_best_full72(root, 200)
            finally:
                M.image_array = original
            self.assertEqual(len(frames), 72)
            self.assertTrue(all(provenance[str(i)] for i in range(72)))
            self.assertTrue(all(int(frame[0, 0, 0]) == i for i, frame in enumerate(frames)))

    def test_rank_best_requires_all_fixed_starts(self):
        records = []
        for step in M.ELIGIBLE_STEPS:
            for start in M.STARTS:
                records.append({
                    "step": step,
                    "start": start,
                    "metrics": {"mse": 1000 - step, "ssim": step / 1000, "exposure_matched_ssim": step / 1000, "hf_edge_f1": step / 1000, "gradient_error": 1 / step},
                    "temporal": {"temporal_delta": 0.1, "flicker": 1 / step, "flow_warp_residual": 1 / step},
                })
        best, summaries = M.rank_best(records)
        self.assertEqual(best, 700)
        self.assertEqual(set(summaries), set(M.ELIGIBLE_STEPS))

    def test_collect_validations_ignores_preliminary_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            validation_root = root / "07_start31_validation"
            formal = [
                "step0_base_start31",
                "step50_start31",
                "step100_start31",
            ]
            preliminary = [
                "step400_start0_prelim",
                "step400_start31_prelim",
            ]
            for name in formal + preliminary:
                folder = validation_root / name
                folder.mkdir(parents=True)
                (folder / "metrics.json").write_text(
                    '{"metrics": {}, "temporal": {}}\n'
                )
            records = M.collect_validations(root)
            self.assertEqual(len(records), len(formal))
            self.assertEqual({r["name"] for r in records}, set(formal))

    def test_pair_metrics_reports_exposure_matched_values(self):
        output = np.full((4, 4, 3), 64, dtype=np.uint8)
        target = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = M.pair_metrics(output, target)
        self.assertGreater(result["mse"], 0)
        self.assertEqual(result["exposure_matched_mse"], 0)
        self.assertAlmostEqual(result["exposure_matched_ssim"], 1.0)


if __name__ == "__main__":
    unittest.main()
