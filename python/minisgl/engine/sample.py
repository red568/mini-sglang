"""Sampler：采样器。

根据请求的采样参数（greedy / temperature / top_k / top_p）从 logits 中采样
下一个 token。greedy 用 argmax；非 greedy 委托给 flashinfer 的采样实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    """一批请求的采样参数（已打包成设备端张量）。

    - `temperatures` 为 None 表示整批都是 greedy 采样。
    - `top_k` / `top_p` 仅在存在对应约束时为非 None。
    """

    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """把 Python 列表转成 pinned 内存后异步搬到设备。"""
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """底层采样：先 softmax 归一化，再按是否启用 top_k/top_p 选对应采样函数。"""
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """把 batch 中每个请求的采样参数打包成设备端张量（greedy 走快速路径）。"""
        params = [r.sampling_params for r in batch.reqs]
        # 整批 greedy：temperatures 置 None，采样时直接 argmax
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        # top_k < 1 视为不约束（等价于全词表）
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        # 仅当确实存在 top_k/top_p 约束时才打包对应张量
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """执行采样：greedy 用 argmax，否则转 float32 后走 sample_impl。"""
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
