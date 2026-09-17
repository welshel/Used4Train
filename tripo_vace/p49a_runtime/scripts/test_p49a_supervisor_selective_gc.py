#!/usr/bin/env python3
import unittest
from pathlib import Path


class SupervisorSelectiveGCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.text = (
            root
            / "outputs/p49a_tripo_vace13b_fresh/05_training/run_segmented_resume_supervisor.sh"
        ).read_text()

    def test_defaults_to_verified_tail_six(self):
        self.assertIn('gc_tail_blocks="${P49A_GC_TAIL_BLOCKS:-6}"', self.text)
        self.assertIn(
            '--gradient_checkpointing_uncheckpointed_tail_blocks "$gc_tail_blocks"',
            self.text,
        )

    def test_cuda_oom_falls_back_without_resetting_global_step(self):
        self.assertIn("CUDA out of memory", self.text)
        self.assertIn("gc_tail_blocks=4", self.text)
        self.assertIn("gc_tail_blocks=0", self.text)
        self.assertNotIn("--start_step 0", self.text)


if __name__ == "__main__":
    unittest.main()
