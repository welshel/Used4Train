import torch


def should_checkpoint_block(
    use_gradient_checkpointing: bool,
    block_id: int,
    checkpointing_period: int = 1,
    total_blocks: int | None = None,
    uncheckpointed_tail_blocks: int = 0,
):
    """Checkpoint one block per period; period=1 preserves legacy behavior."""
    if checkpointing_period < 1:
        raise ValueError("gradient checkpointing period must be >= 1")
    if uncheckpointed_tail_blocks < 0:
        raise ValueError("uncheckpointed tail block count must be >= 0")
    if uncheckpointed_tail_blocks:
        if total_blocks is None:
            raise ValueError("total_blocks is required for an uncheckpointed tail")
        if uncheckpointed_tail_blocks > total_blocks:
            raise ValueError("uncheckpointed tail cannot exceed total blocks")
        if block_id >= total_blocks - uncheckpointed_tail_blocks:
            return False
    return bool(use_gradient_checkpointing and block_id % checkpointing_period == 0)


try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ModuleNotFoundError:
    _HAS_DEEPSPEED = False


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def create_custom_forward_use_reentrant(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward


def judge_args_requires_grad(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return True
    return False


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing and _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured():
        all_args = args + tuple(kwargs.values())
        if not judge_args_requires_grad(*all_args):
            # get the first grad_enabled tensor from un_checkpointed forward
            model_output = model(*args, **kwargs)
        else:
            model_output = deepspeed.checkpointing.checkpoint(
                create_custom_forward_use_reentrant(model),
                *all_args,
            )
        return model_output
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            model_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
