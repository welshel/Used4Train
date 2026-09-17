#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "DiffSynth-Studio"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diffsynth.core.gradient.gradient_checkpoint import should_checkpoint_block


class SelectiveGradientCheckpointingTests(unittest.TestCase):
    def test_period_one_checkpoints_every_block(self):
        self.assertEqual(
            [should_checkpoint_block(True, i, 1) for i in range(6)],
            [True, True, True, True, True, True],
        )

    def test_period_three_uses_memory_for_two_of_three_blocks(self):
        self.assertEqual(
            [should_checkpoint_block(True, i, 3) for i in range(7)],
            [True, False, False, True, False, False, True],
        )

    def test_disabled_checkpointing_never_checkpoints(self):
        self.assertFalse(should_checkpoint_block(False, 0, 1))
        self.assertFalse(should_checkpoint_block(False, 3, 3))

    def test_invalid_period_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "period"):
            should_checkpoint_block(True, 0, 0)

    def test_uncheckpointed_tail_only_uses_memory_near_chain_end(self):
        self.assertEqual(
            [
                should_checkpoint_block(
                    True, i, 1, total_blocks=6, uncheckpointed_tail_blocks=2
                )
                for i in range(6)
            ],
            [True, True, True, True, False, False],
        )

    def test_tail_larger_than_chain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tail"):
            should_checkpoint_block(
                True, 0, 1, total_blocks=4, uncheckpointed_tail_blocks=5
            )


if __name__ == "__main__":
    unittest.main()
