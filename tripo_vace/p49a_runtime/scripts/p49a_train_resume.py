#!/usr/bin/env python3
"""P49A project-local DDP entrypoint with native ordered PNG-list inputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "DiffSynth-Studio"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadImage,
    RouteByType,
    SequencialProcess,
    ToAbsolutePath,
    ToList,
)
from diffsynth.diffusion.logger import ModelLogger
from examples.wanvideo.model_training.train import WanTrainingModule


def png_list_operator(base_path: str, height: int, width: int, max_pixels: int = 1024 * 1024):
    """Load an ordered JSON list of PNGs directly and never decode a derived MP4."""
    single = (
        ToAbsolutePath(base_path)
        >> LoadImage()
        >> ImageCropAndResize(height, width, max_pixels, 16, 16)
        >> ToList()
    )
    sequence = SequencialProcess(
        ToAbsolutePath(base_path)
        >> LoadImage()
        >> ImageCropAndResize(height, width, max_pixels, 16, 16)
    )
    return RouteByType(operator_map=[(str, single), (list, sequence)])


def build_dataset(base_path: str, metadata_path: str, height: int, width: int):
    return UnifiedDataset(
        base_path=base_path,
        metadata_path=metadata_path,
        repeat=1,
        data_file_keys=["video", "vace_video", "vace_reference_image"],
        main_data_operator=png_list_operator(base_path, height, width),
    )


class P49AWanTrainingModule(WanTrainingModule):
    """Adds a P49A-only selective checkpoint policy without changing the objective."""

    def __init__(
        self,
        *args,
        gradient_checkpointing_period: int = 1,
        gradient_checkpointing_uncheckpointed_tail_blocks: int = 0,
        **kwargs,
    ):
        self.gradient_checkpointing_period = gradient_checkpointing_period
        self.gradient_checkpointing_uncheckpointed_tail_blocks = (
            gradient_checkpointing_uncheckpointed_tail_blocks
        )
        super().__init__(*args, **kwargs)

    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        inputs_shared["gradient_checkpointing_period"] = self.gradient_checkpointing_period
        inputs_shared["gradient_checkpointing_uncheckpointed_tail_blocks"] = (
            self.gradient_checkpointing_uncheckpointed_tail_blocks
        )
        return inputs_shared, inputs_posi, inputs_nega


def lr_multiplier(step: int, decay_start_step: int, base_lr: float, decay_lr: float) -> float:
    return 1.0 if step <= decay_start_step else decay_lr / base_lr


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_base_path", required=True)
    ap.add_argument("--dataset_metadata_path", required=True)
    ap.add_argument("--height", type=int, default=672)
    ap.add_argument("--width", type=int, default=672)
    ap.add_argument("--num_frames", type=int, default=33)
    ap.add_argument("--dataset_num_workers", type=int, default=0)
    ap.add_argument("--dataset_repeat", type=int, default=1)
    ap.add_argument("--data_file_keys", default="video,vace_video,vace_reference_image")
    ap.add_argument("--model_paths", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--lr_decay_start_step", type=int, default=400)
    ap.add_argument("--lr_after_decay", type=float, default=5e-5)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--benchmark_warmup_steps", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--lora_base_model", default="vace")
    ap.add_argument("--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--extra_inputs", default="vace_video,vace_reference_image")
    ap.add_argument("--use_gradient_checkpointing", action="store_true")
    ap.add_argument(
        "--gradient_checkpointing_period",
        type=int,
        default=1,
        help="Checkpoint every Nth transformer block; 1 checkpoints all blocks.",
    )
    ap.add_argument(
        "--gradient_checkpointing_uncheckpointed_tail_blocks",
        type=int,
        default=0,
        help="Keep the final N blocks of each transformer chain without checkpointing.",
    )
    ap.add_argument("--force_gradient_checkpointing_off", action="store_true",
                    help="Override DiffSynth's conservative forced-GC default for an explicit benchmark.")
    ap.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    ap.add_argument("--enable_csv_log", action="store_true")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--enable_model_cpu_offload", action="store_true")
    ap.add_argument("--enable_optimizer_cpu_offload", action="store_true")
    ap.add_argument("--cpu_offload_split_threshold", type=int, default=None)
    ap.add_argument("--customized_optimizer", default=None)
    ap.add_argument("--lora_checkpoint", default=None,
                    help="Load LoRA weights for a same-run continuation; optimizer is reinitialized.")
    ap.add_argument("--start_step", type=int, default=0,
                    help="Completed global optimizer steps represented by lora_checkpoint.")
    ap.add_argument("--resume_note", default=None)
    return ap


def _write_runtime_row(writer, global_step, phase, loss, learning_rate, seconds):
    writer.writerow(
        {
            "global_step": global_step,
            "phase": phase,
            "loss": f"{loss:.9f}",
            "learning_rate": f"{learning_rate:.9g}",
            "optimizer_step_sec": f"{seconds:.6f}",
        }
    )


def run_training(accelerator: Accelerator, dataset, model, logger: ModelLogger, args):
    if args.enable_model_cpu_offload or args.enable_optimizer_cpu_offload:
        raise ValueError("P49A forbids CPU/model/optimizer offload for the 4xA40 training route")
    if args.use_gradient_checkpointing_offload:
        raise ValueError("P49A forbids gradient-checkpointing offload")
    if args.use_gradient_checkpointing and args.force_gradient_checkpointing_off:
        raise ValueError("gradient checkpointing cannot be both forced on and forced off")
    if args.gradient_checkpointing_period < 1:
        raise ValueError("gradient checkpointing period must be >= 1")
    if not 0 <= args.gradient_checkpointing_uncheckpointed_tail_blocks <= 15:
        raise ValueError("uncheckpointed tail blocks must be between 0 and 15")
    if args.customized_optimizer is not None:
        raise ValueError("P49A requires the standard AdamW RGB-only objective")
    if args.num_frames != 33 or args.height != 672 or args.width != 672:
        raise ValueError("P49A training contract requires 33 frames at 672x672")

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # LoRA-only continuation: AdamW is intentionally new, while LambdaLR is
    # positioned at the completed global step for the prescribed schedule.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(
            step, args.lr_decay_start_step, args.learning_rate, args.lr_after_decay
        ),
        # LambdaLR performs one initialization step in its constructor, so
        # last_epoch=start_step-1 makes the first resumed optimizer update use
        # the schedule value for global step start_step+1.
        last_epoch=args.start_step - 1,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda samples: samples[0], num_workers=args.dataset_num_workers
    )
    model.to(device=accelerator.device)
    # Keep LambdaLR local. Accelerate otherwise multiplies scheduler.step by world size
    # when it shards the DataLoader, which would decay at global step 100 instead of 400.
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    runtime_writer = None
    runtime_file = None
    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)
        runtime_path = Path(args.output_path) / (
            f"P49A_RESUME_RUNTIME_LOG_FROM_{args.start_step}.csv"
            if args.start_step else "P49A_RUNTIME_LOG.csv"
        )
        runtime_exists = runtime_path.exists() and runtime_path.stat().st_size > 0
        runtime_file = open(runtime_path, "a" if runtime_exists else "w", encoding="utf-8", newline="")
        runtime_writer = csv.DictWriter(
            runtime_file,
            fieldnames=["global_step", "phase", "loss", "learning_rate", "optimizer_step_sec"],
        )
        if not runtime_exists:
            runtime_writer.writeheader()
        runtime_file.flush()

    global_step = args.start_step
    timed_seconds = []
    completed = False
    for _epoch in range(args.num_epochs):
        for data in dataloader:
            start_time = time.perf_counter()
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            elapsed = time.perf_counter() - start_time
            global_step += 1
            loss_mean = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            lr = optimizer.param_groups[0]["lr"]
            phase = "warmup" if global_step <= args.benchmark_warmup_steps else "timed"
            if phase == "timed":
                timed_seconds.append(elapsed)
            logger.on_step_end(accelerator, model, args.save_steps, loss=loss_mean)
            if accelerator.is_main_process:
                _write_runtime_row(runtime_writer, global_step, phase, loss_mean, lr, elapsed)
                runtime_file.flush()
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                completed = True
                break
        if completed:
            break

    local_memory = torch.tensor(
        [
            torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 2),
            torch.cuda.max_memory_reserved(accelerator.device) / (1024 ** 2),
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )
    all_memory = accelerator.gather(local_memory).detach().cpu().tolist()
    accelerator.wait_for_everyone()
    logger.on_training_end(accelerator, model, args.save_steps)
    if accelerator.is_main_process:
        if runtime_file is not None:
            runtime_file.close()
        summary = {
            "world_size": accelerator.num_processes,
            "per_gpu_microbatch": 1,
            "gradient_accumulation": 1,
            "effective_batch": accelerator.num_processes,
            "global_steps_completed": global_step,
            "start_step": args.start_step,
            "resume_from_lora_checkpoint": args.lora_checkpoint,
            "optimizer_reinitialized": bool(args.start_step and args.lora_checkpoint),
            "resume_note": args.resume_note,
            "warmup_steps": args.benchmark_warmup_steps,
            "timed_steps": len(timed_seconds),
            "mean_timed_optimizer_step_sec": (
                sum(timed_seconds) / len(timed_seconds) if timed_seconds else None
            ),
            "gpu_max_memory_mib": [
                {"allocated": all_memory[idx], "reserved": all_memory[idx + 1]}
                for idx in range(0, len(all_memory), 2)
            ],
            "gradient_checkpointing_requested": args.use_gradient_checkpointing,
            "gradient_checkpointing_actual": not args.force_gradient_checkpointing_off,
            "gradient_checkpointing_period": args.gradient_checkpointing_period,
            "gradient_checkpointing_uncheckpointed_tail_blocks": (
                args.gradient_checkpointing_uncheckpointed_tail_blocks
            ),
            "main_blocks_checkpointed": len(
                range(
                    0,
                    30 - args.gradient_checkpointing_uncheckpointed_tail_blocks,
                    args.gradient_checkpointing_period,
                )
            ),
            "vace_blocks_checkpointed": len(
                range(
                    0,
                    15 - args.gradient_checkpointing_uncheckpointed_tail_blocks,
                    args.gradient_checkpointing_period,
                )
            ),
            "model_cpu_offload": False,
            "optimizer_cpu_offload": False,
            "gradient_checkpointing_offload": False,
            "fresh_lora": True,
            "old_lora_loaded": False,
            "lora_checkpoint": args.lora_checkpoint,
            "same_run_lora_continuation": bool(args.start_step and args.lora_checkpoint),
            "resume_from_checkpoint": None,
        }
        (Path(args.output_path) / "P49A_TRAIN_RUNTIME_SUMMARY.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2))


def main():
    args = build_arg_parser().parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    dataset = build_dataset(
        args.dataset_base_path, args.dataset_metadata_path, args.height, args.width
    )
    model = P49AWanTrainingModule(
        model_paths=args.model_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=None,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        gradient_checkpointing_period=args.gradient_checkpointing_period,
        gradient_checkpointing_uncheckpointed_tail_blocks=(
            args.gradient_checkpointing_uncheckpointed_tail_blocks
        ),
        extra_inputs=args.extra_inputs,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt="pipe.vace.",
        task="sft",
        device=accelerator.device,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    )
    if args.force_gradient_checkpointing_off:
        # WanTrainingModule force-enables GC during construction; the forward
        # path reads this attribute at runtime, so an explicit post-init
        # override provides a scoped, benchmarkable GC-OFF route.
        model.use_gradient_checkpointing = False
    logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt="pipe.vace.",
        enable_csv_log=args.enable_csv_log,
    )
    # Continue checkpoint numbering from the completed LoRA step.  This keeps
    # step-200/350/... semantics global rather than relative to the restart.
    logger.num_steps = args.start_step
    run_training(accelerator, dataset, model, logger, args)


if __name__ == "__main__":
    main()
