import unittest
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from p38_train import merge_preprocessed_batch, validate_runtime_args


class HighSpeedBatchingTests(unittest.TestCase):
    def test_batch2_accum2_preserves_effective_batch_four(self):
        validate_runtime_args(
            width=512,
            height=512,
            num_frames=33,
            batch_size=2,
            gradient_accumulation=2,
            rank=16,
        )

    def test_batch1_accum4_allows_640_resolution_curriculum(self):
        validate_runtime_args(
            width=640,
            height=640,
            num_frames=33,
            batch_size=1,
            gradient_accumulation=4,
            rank=16,
        )

    def test_allows_672_resolution_after_640(self):
        validate_runtime_args(
            width=672,
            height=672,
            num_frames=33,
            batch_size=1,
            gradient_accumulation=4,
            rank=16,
        )

    def test_rejects_unapproved_resolution(self):
        with self.assertRaisesRegex(ValueError, "supported resolutions"):
            validate_runtime_args(
                width=608,
                height=608,
                num_frames=33,
                batch_size=1,
                gradient_accumulation=4,
                rank=16,
            )

    def test_rejects_effective_batch_drift(self):
        with self.assertRaisesRegex(ValueError, "effective batch size 4"):
            validate_runtime_args(
                width=512,
                height=512,
                num_frames=33,
                batch_size=2,
                gradient_accumulation=4,
                rank=16,
            )

    def test_merge_preprocessed_batch_concatenates_model_tensors(self):
        sample_a = {
            "noise": torch.zeros(1, 2, 3),
            "input_latents": torch.ones(1, 2, 3),
            "context": torch.full((1, 4, 5), 2.0),
            "height": 512,
            "use_gradient_checkpointing": False,
            "prompt": "same prompt",
            "animate_pose_video": None,
        }
        sample_b = {
            "noise": torch.full((1, 2, 3), 3.0),
            "input_latents": torch.full((1, 2, 3), 4.0),
            "context": torch.full((1, 4, 5), 5.0),
            "height": 512,
            "use_gradient_checkpointing": False,
            "prompt": "same prompt",
            "animate_pose_video": None,
        }

        merged = merge_preprocessed_batch([sample_a, sample_b])

        self.assertEqual(merged["noise"].shape, (2, 2, 3))
        self.assertEqual(merged["input_latents"].shape, (2, 2, 3))
        self.assertEqual(merged["context"].shape, (2, 4, 5))
        self.assertTrue(torch.equal(merged["noise"][1], sample_b["noise"][0]))
        self.assertEqual(merged["height"], 512)
        self.assertEqual(merged["prompt"], "same prompt")
        self.assertIsNone(merged["animate_pose_video"])

    def test_merge_rejects_mismatched_model_tensor_shapes(self):
        with self.assertRaisesRegex(ValueError, "non-batch shape mismatch"):
            merge_preprocessed_batch(
                [
                    {"noise": torch.zeros(1, 2, 3)},
                    {"noise": torch.zeros(1, 2, 4)},
                ]
            )


if __name__ == "__main__":
    unittest.main()
