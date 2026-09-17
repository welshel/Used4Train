#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Sequence

from p38_contract import (
    EXPECTED_LORA_PARAMS,
    SUPPORTED_RESOLUTIONS,
    training_contract,
    validate_records,
    validate_spatial_resolution,
)


MODEL_BATCH_TENSOR_KEYS = frozenset(
    {"noise", "latents", "input_latents", "vace_context", "context"}
)


def merge_preprocessed_batch(samples: Sequence[dict]) -> dict:
    """Merge independently preprocessed Wan samples along model batch dimension."""

    import torch

    if not samples:
        raise ValueError("cannot merge an empty preprocessed batch")
    expected_keys = set(samples[0])
    for sample in samples[1:]:
        if set(sample) != expected_keys:
            raise ValueError("preprocessed batch keys do not match")

    merged = dict(samples[0])
    for key in MODEL_BATCH_TENSOR_KEYS.intersection(expected_keys):
        values = [sample[key] for sample in samples]
        if not all(torch.is_tensor(value) for value in values):
            raise TypeError(f"model batch field {key} must contain tensors")
        if any(value.ndim < 1 or value.shape[0] != 1 for value in values):
            raise ValueError(f"model batch field {key} must have singleton batch dimension")
        reference_shape = values[0].shape[1:]
        if any(value.shape[1:] != reference_shape for value in values[1:]):
            raise ValueError(f"model batch field {key} has non-batch shape mismatch")
        merged[key] = torch.cat(values, dim=0)
    return merged


def microbatch_budget(start_optimizer_step: int, end_optimizer_step: int, accumulation: int) -> int:
    if accumulation < 1 or end_optimizer_step < start_optimizer_step:
        raise ValueError("invalid optimizer-step range or accumulation")
    return (end_optimizer_step - start_optimizer_step) * accumulation


def checkpoint_steps_in_range(start: int, end: int, requested: Sequence[int]) -> list[int]:
    return sorted({int(step) for step in requested if start < int(step) <= end})


def validate_training_manifest(path: Path | str) -> int:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return validate_records(rows)


def classify_resume(start_step: int, resume_lora: Path | None, resume_optimizer: Path | None) -> str:
    if start_step == 0 and (resume_lora is not None or resume_optimizer is not None):
        raise ValueError("L33 primary step 0 must start fresh from the base Wan model")
    if resume_optimizer is not None and resume_lora is None:
        raise ValueError("optimizer resume requires a matching LoRA checkpoint")
    if start_step > 0 and resume_lora is None:
        raise ValueError("non-zero start step requires a LoRA checkpoint")
    if resume_lora is None:
        return "FRESH"
    if resume_optimizer is None:
        return "WEIGHT_RESUME"
    return "WEIGHT_AND_OPTIMIZER_RESUME"


def validate_runtime_args(
    *,
    width: int,
    height: int,
    num_frames: int,
    batch_size: int,
    gradient_accumulation: int,
    rank: int,
) -> None:
    if batch_size not in (1, 2):
        raise ValueError("P3.8 PRO6000 trainer supports batch size 1 or 2")
    if batch_size * gradient_accumulation != 4:
        raise ValueError("P3.8 contract requires effective batch size 4")
    if rank != 16:
        raise ValueError("P3.8 contract requires LoRA rank 16")
    validate_spatial_resolution(width, height)
    if num_frames != 33:
        raise ValueError(
            f"P3.8 contract requires 33 frames at supported resolutions {SUPPORTED_RESOLUTIONS}"
        )


def parse_steps(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def save_lora_checkpoint(accelerator, model, output_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    trainable_names = unwrapped.trainable_param_names()
    state = {}
    for name, parameter in unwrapped.named_parameters():
        if name not in trainable_names:
            continue
        clean_name = name.removeprefix("pipe.vace.")
        state[clean_name] = parameter.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output_path))


def save_optimizer_checkpoint(
    accelerator,
    optimizer,
    scheduler,
    output_path: Path,
    optimizer_step: int,
) -> None:
    import numpy as np
    import torch

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    state = {
        "optimizer_step": int(optimizer_step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all(),
        "sampler_resume": "RESTARTS_FROM_FIXED_SEED",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def restore_optimizer_checkpoint(path: Path, optimizer, scheduler, expected_step: int) -> dict:
    import numpy as np
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state["optimizer_step"]) != int(expected_step):
        raise ValueError(
            f"optimizer state step {state['optimizer_step']} does not match start step {expected_step}"
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    torch.cuda.set_rng_state_all(state["torch_cuda_random_state_all"])
    return {
        "optimizer_step": int(state["optimizer_step"]),
        "sampler_resume": state.get("sampler_resume", "UNKNOWN"),
    }


def restore_scheduler_for_weight_resume(scheduler, optimizer, start_step: int) -> dict:
    """Reconstruct ConstantLR's already-completed warm-start for weight-only resume."""

    if int(start_step) <= 0:
        raise ValueError("weight resume scheduler reconstruction requires a positive start step")
    inner = getattr(scheduler, "scheduler", scheduler)
    state = inner.state_dict()
    total_iters = int(state.get("total_iters", 0))
    base_lrs = [float(value) for value in state.get("base_lrs", [])]
    if total_iters < 1 or len(base_lrs) != len(optimizer.param_groups):
        raise ValueError("unsupported scheduler state for P3.6 weight resume")
    state["last_epoch"] = max(total_iters, int(start_step))
    state["_step_count"] = state["last_epoch"] + 1
    state["_last_lr"] = list(base_lrs)
    inner.load_state_dict(state)
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr
    return {
        "policy": "RECONSTRUCTED_POST_WARMUP_CONSTANT_LR",
        "start_step": int(start_step),
        "total_iters": total_iters,
        "learning_rates": base_lrs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-step P3.8 high-resolution Wan VACE LoRA trainer")
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--checkpoint-steps", default="50,100,200,250,500,800")
    parser.add_argument("--resume-lora", type=Path)
    parser.add_argument("--resume-optimizer", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation recomputation. Use --no-gradient-checkpointing for the PRO6000 high-speed mode.",
    )
    parser.add_argument("--gradient-checkpointing-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_checkpointing_offload and not args.gradient_checkpointing:
        raise ValueError("gradient checkpointing offload requires gradient checkpointing")
    contract = training_contract()
    validate_runtime_args(
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        rank=args.rank,
    )
    resume_classification = classify_resume(args.start_step, args.resume_lora, args.resume_optimizer)
    sample_count = validate_training_manifest(args.manifest)
    expected_microbatches = microbatch_budget(
        args.start_step, args.end_step, args.gradient_accumulation
    )
    requested_checkpoints = checkpoint_steps_in_range(
        args.start_step, args.end_step, parse_steps(args.checkpoint_steps)
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(args.diffsynth_repo))
    os.chdir(args.diffsynth_repo)

    import numpy as np
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from safetensors import safe_open
    from torch.utils.data import DataLoader

    from diffsynth.trainers.unified_dataset import UnifiedDataset
    from examples.wanvideo.model_training.train import WanTrainingModule

    class BatchedWanTrainingModule(WanTrainingModule):
        def forward(self, data, inputs=None):
            if isinstance(data, list):
                if inputs is not None:
                    raise ValueError("explicit inputs are unsupported for list batches")
                inputs = merge_preprocessed_batch(
                    [self.forward_preprocess(sample) for sample in data]
                )
                data = data[0]
            return super().forward(data, inputs=inputs)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_paths = [
        args.model_dir / "diffusion_pytorch_model.safetensors",
        args.model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        args.model_dir / "Wan2.1_VAE.pth",
    ]
    missing = [str(path) for path in model_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model files: {missing}")
    if args.resume_lora is not None:
        if not args.resume_lora.is_file():
            raise FileNotFoundError(args.resume_lora)
        with safe_open(str(args.resume_lora), framework="pt", device="cpu") as handle:
            if not list(handle.keys()):
                raise ValueError("resume LoRA has no tensors")

    dataset = UnifiedDataset(
        base_path="/",
        metadata_path=str(args.manifest),
        repeat=1,
        data_file_keys=contract["data_file_keys"],
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path="/",
            max_pixels=args.width * args.height,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )
    model = BatchedWanTrainingModule(
        model_paths=json.dumps([str(path) for path in model_paths]),
        trainable_models=None,
        lora_base_model="vace",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=args.rank,
        lora_checkpoint=str(args.resume_lora) if args.resume_lora else None,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_gradient_checkpointing_offload=args.gradient_checkpointing_offload,
        extra_inputs=contract["extra_inputs"],
    )
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    if trainable_params != EXPECTED_LORA_PARAMS:
        raise RuntimeError(
            f"trainable parameter drift: expected {EXPECTED_LORA_PARAMS}, got {trainable_params}"
        )

    generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch[0] if args.batch_size == 1 else batch,
        num_workers=args.num_workers,
        generator=generator,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    optimizer_resume_detail = None
    scheduler_resume_detail = None
    if args.resume_optimizer is not None:
        optimizer_resume_detail = restore_optimizer_checkpoint(
            args.resume_optimizer, optimizer, scheduler, args.start_step
        )
    elif resume_classification == "WEIGHT_RESUME":
        scheduler_resume_detail = restore_scheduler_for_weight_resume(
            scheduler, optimizer, args.start_step
        )
    model.train()
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = args.output_dir / "loss_curve.csv"
    checkpoint_dir = args.output_dir / "checkpoints"
    log_fields = [
        "stage",
        "branch",
        "optimizer_step",
        "microbatch_step",
        "loss",
        "learning_rate",
        "elapsed_seconds",
        "vram_allocated_gb",
        "vram_reserved_gb",
        "vram_peak_allocated_gb",
        "ram_peak_gb",
    ]
    loss_file = loss_csv.open("w", newline="")
    writer = csv.DictWriter(loss_file, fieldnames=log_fields)
    writer.writeheader()

    start_time = time.time()
    optimizer_step = args.start_step
    microbatch_step = 0
    accumulated_losses: list[float] = []
    checkpoint_paths: list[str] = []
    optimizer_checkpoint_paths: list[str] = []
    while optimizer_step < args.end_step:
        for data in dataloader:
            if optimizer_step >= args.end_step:
                break
            with accelerator.accumulate(model):
                loss = model(data)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        f"non-finite loss at microbatch {microbatch_step + 1}: {loss}"
                    )
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                did_sync = accelerator.sync_gradients
            microbatch_step += 1
            accumulated_losses.append(float(loss.detach().float().cpu()))
            if not did_sync:
                continue
            optimizer_step += 1
            mean_loss = sum(accumulated_losses) / len(accumulated_losses)
            accumulated_losses.clear()
            row = {
                "stage": args.stage,
                "branch": "l33",
                "optimizer_step": optimizer_step,
                "microbatch_step": microbatch_step,
                "loss": mean_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - start_time,
                "vram_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "vram_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "vram_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "ram_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            }
            writer.writerow(row)
            loss_file.flush()
            print("P38_STEP " + json.dumps(row), flush=True)
            if optimizer_step in requested_checkpoints or optimizer_step == args.end_step:
                checkpoint_path = checkpoint_dir / f"optimizer_step-{optimizer_step:04d}.safetensors"
                optimizer_path = checkpoint_dir / f"optimizer_state_step-{optimizer_step:04d}.pt"
                save_lora_checkpoint(accelerator, model, checkpoint_path)
                save_optimizer_checkpoint(
                    accelerator, optimizer, scheduler, optimizer_path, optimizer_step
                )
                checkpoint_paths.append(str(checkpoint_path))
                optimizer_checkpoint_paths.append(str(optimizer_path))
                print(f"P38_CHECKPOINT {checkpoint_path}", flush=True)

    loss_file.close()
    summary = {
        "stage": args.stage,
        "branch": "l33",
        "start_optimizer_step": args.start_step,
        "end_optimizer_step": optimizer_step,
        "optimizer_steps_completed": optimizer_step - args.start_step,
        "microbatches_completed": microbatch_step,
        "expected_microbatches": expected_microbatches,
        "gradient_accumulation": args.gradient_accumulation,
        "batch_size": args.batch_size,
        "sample_count": sample_count,
        "rank": args.rank,
        "trainable_params": trainable_params,
        "reference_adapter_params": contract["reference_adapter_params"],
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "data_file_keys": list(contract["data_file_keys"]),
        "extra_inputs": contract["extra_inputs"],
        "precision": "bf16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_offload": args.gradient_checkpointing_offload,
        "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "peak_ram_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "elapsed_seconds": time.time() - start_time,
        "checkpoints": checkpoint_paths,
        "optimizer_checkpoints": optimizer_checkpoint_paths,
        "resume_lora": str(args.resume_lora) if args.resume_lora else None,
        "resume_optimizer": str(args.resume_optimizer) if args.resume_optimizer else None,
        "resume_classification": resume_classification,
        "optimizer_resume_detail": optimizer_resume_detail,
        "scheduler_resume_detail": scheduler_resume_detail,
        "sampler_resume": "RESTARTS_FROM_FIXED_SEED",
    }
    (args.output_dir / "runtime_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("P38_RUNTIME_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
