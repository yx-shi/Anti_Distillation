from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    """Runtime information shared by the trainer and evaluation loop."""

    use_fsdp: bool
    world_size: int
    rank: int
    local_rank: int
    device: torch.device


def setup_distributed() -> DistributedContext:
    """Initialize torch.distributed when launched by torchrun and choose the current device."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_fsdp = world_size > 1

    if use_fsdp:
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP training requires CUDA, but no visible GPU was found.")

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return DistributedContext(
        use_fsdp=use_fsdp,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed(runtime: DistributedContext) -> None:
    """Tear down the process group so repeated launches do not leak distributed state."""

    if runtime.use_fsdp and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(runtime: DistributedContext) -> bool:
    """Only rank 0 should print logs or do one-off side effects."""

    return runtime.rank == 0


def reduce_mean(value: torch.Tensor, runtime: DistributedContext) -> torch.Tensor:
    """Average one scalar tensor across ranks so logs reflect the whole job instead of one rank."""

    if not runtime.use_fsdp:
        return value.detach()

    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= runtime.world_size
    return reduced
