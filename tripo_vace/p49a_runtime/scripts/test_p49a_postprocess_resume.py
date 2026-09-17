#!/usr/bin/env python3
import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("p49a_postprocess", Path(__file__).with_name("p49a_postprocess.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PostprocessTests(unittest.TestCase):
    def test_combine_training_log_prefers_latest_resume_segment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / "06_checkpoints"
            (root / "05_training").mkdir(parents=True)
            ckpt.mkdir()
            fields = ["global_step", "phase", "loss", "learning_rate", "optimizer_step_sec"]
            def write_log(path, steps, loss):
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for step in steps:
                        writer.writerow({"global_step": step, "phase": "timed", "loss": loss,
                                         "learning_rate": 0.0001 if step <= 400 else 0.00005,
                                         "optimizer_step_sec": 1.0})
            write_log(ckpt / "P49A_RUNTIME_LOG.csv", range(1, 151), 1.0)
            write_log(ckpt / "P49A_RESUME_RUNTIME_LOG_FROM_150.csv", range(151, 423), 2.0)
            write_log(ckpt / "P49A_RESUME_RUNTIME_LOG_FROM_400.csv", range(401, 701), 3.0)
            rows = M.combine_training_log(root)
            self.assertEqual(len(rows), 700)
            self.assertEqual(rows[400]["global_step"], "401")
            self.assertEqual(rows[400]["loss"], "3.0")
            self.assertEqual(rows[400]["segment"], "resume_from_step400_optimizer_reinitialized")

    def test_promote_best_checkpoint_copies_selected_weights_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "06_checkpoints" / "step-350.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"checkpoint")
            result = M.promote_best_checkpoint(root, 350)
            self.assertEqual((root / "06_checkpoints" / "best.safetensors").read_bytes(), b"checkpoint")
            self.assertEqual(result["best_step"], 350)
            self.assertEqual(result["source_checkpoint"], str(source))

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


if __name__ == "__main__":
    unittest.main()
