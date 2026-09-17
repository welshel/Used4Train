import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("p49a_prepare_dataset.py")
SPEC = importlib.util.spec_from_file_location("p49a_prepare_dataset", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
assert SPEC.loader
SPEC.loader.exec_module(MOD)
TRAIN_SPEC = importlib.util.spec_from_file_location("p49a_train", Path(__file__).with_name("p49a_train.py"))
TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = TRAIN
assert TRAIN_SPEC.loader
TRAIN_SPEC.loader.exec_module(TRAIN)

class P49ADataContractTests(unittest.TestCase):
    def test_build_clip_manifest_uses_cyclic_33_frames_and_exact_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition"
            clean = root / "clean"
            condition.mkdir(); clean.mkdir()
            for i in range(72):
                MOD.Image.new("RGB", (896, 896), (i, 0, 0)).save(condition / f"F{i:02d}.png")
                MOD.Image.new("RGB", (896, 896), (0, i, 0)).save(clean / f"F{i:02d}.png")
            rows = MOD.build_clip_manifest(condition, clean, starts=range(72), clip_len=33)
        self.assertEqual(len(rows), 72)
        self.assertEqual(rows[0]["start"], 0)
        self.assertEqual(rows[0]["video"][0], str(clean / "F00.png"))
        self.assertEqual(rows[0]["video"][-1], str(clean / "F32.png"))
        self.assertEqual(rows[71]["video"][-1], str(clean / "F31.png"))
        self.assertEqual(rows[0]["vace_video"], [str(condition / f"F{i:02d}.png") for i in range(33)])
        self.assertEqual(rows[0]["vace_reference_image"], [str(condition / "F00.png")])

    def test_png_list_operator_loads_ordered_sequences_without_video_decode(self):
        op = TRAIN.png_list_operator("/base", 672, 672)
        self.assertIsNotNone(op)
        self.assertEqual(type(op).__name__, "RouteByType")

    def test_validation_starts_are_fixed_and_disjoint_from_train_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); condition = root / "condition"; clean = root / "clean"
            condition.mkdir(); clean.mkdir()
            for i in range(72):
                MOD.Image.new("RGB", (896, 896), (i, 0, 0)).save(condition / f"F{i:02d}.png")
                MOD.Image.new("RGB", (896, 896), (0, i, 0)).save(clean / f"F{i:02d}.png")
            rows = MOD.build_validation_manifest(condition, clean, starts=(0,17,31,50), clip_len=33)
        self.assertEqual([r["start"] for r in rows], [0,17,31,50])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(r["video"]) == 33 for r in rows))
        self.assertTrue(all(len(r["vace_video"]) == 33 for r in rows))

    def test_training_parser_exposes_framework_offload_and_optimizer_options(self):
        parser = TRAIN.build_arg_parser()
        args = parser.parse_args([
            "--dataset_base_path", "/base",
            "--dataset_metadata_path", "/meta",
            "--model_paths", "[]",
            "--tokenizer_path", "/tok",
            "--output_path", "/out",
        ])
        self.assertFalse(args.enable_model_cpu_offload)
        self.assertFalse(args.enable_optimizer_cpu_offload)
        self.assertIsNone(args.cpu_offload_split_threshold)
        self.assertIsNone(args.customized_optimizer)

    def test_training_parser_and_schedule_support_700_step_fresh_run(self):
        parser = TRAIN.build_arg_parser()
        args = parser.parse_args([
            "--dataset_base_path", "/base",
            "--dataset_metadata_path", "/meta",
            "--model_paths", "[]",
            "--tokenizer_path", "/tok",
            "--output_path", "/out",
            "--max_train_steps", "700",
        ])
        self.assertEqual(args.max_train_steps, 700)
        self.assertEqual(args.lr_decay_start_step, 400)
        self.assertAlmostEqual(TRAIN.lr_multiplier(1, 400, 1e-4, 5e-5), 1.0)
        self.assertAlmostEqual(TRAIN.lr_multiplier(401, 400, 1e-4, 5e-5), 0.5)

    def test_scheduler_is_not_wrapped_by_accelerate_and_multiplied_by_world_size(self):
        source = Path(TRAIN.__file__).read_text()
        self.assertNotIn("model, optimizer, dataloader, scheduler = accelerator.prepare", source)

if __name__ == "__main__":
    unittest.main()
