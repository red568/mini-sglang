"""分布式集合通信的实现层。

提供统一的 `DistributedCommunicator` 门面，底层支持两种实现（插件）：
- `TorchDistributedImpl`：基于 `torch.distributed` 的 all_reduce/all_gather。
- `PyNCCLDistributedImpl`：基于自定义 PyNCCL 内核，性能更优。

`DistributedCommunicator.plugins` 是一个插件栈，通信时取栈顶实现；
`enable_pynccl_distributed` 会把 PyNCCL 实现压栈以启用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    """集合通信实现的抽象接口。"""

    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """基于 torch.distributed 的默认实现（单卡时直接返回原张量）。"""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        # 沿第 0 维拼接各 rank 的张量
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """基于 PyNCCL 内核的实现，内部维护一个通信器句柄。"""

    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """集合通信的门面：持有实现栈，始终委托给栈顶实现。"""

    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """启用 PyNCCL 通信：初始化通信器并压入插件栈。单卡无需启用。"""
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """清空所有通信插件（在释放 NCCL 资源后调用）。"""
    DistributedCommunicator.plugins = []
