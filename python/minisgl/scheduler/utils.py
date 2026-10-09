"""Scheduler 相关的轻量数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    """处于等待状态的请求：已收到但尚未（完全）进入 prefill 调度。

    - `chunked_req`：若请求被 chunked prefill 拆分为多段，这里记录已入队的前一段，
      后续调度时从中断位置继续。
    """

    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        """原始输入 token 数。"""
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        """期望生成的最大输出 token 数。"""
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    """一次调度产出的结果：待处理请求及其对应的输出索引。"""

    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
