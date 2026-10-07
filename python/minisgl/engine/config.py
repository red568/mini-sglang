from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

"""
是「一个 TP worker 的引擎」需要的所有配置。
"""
@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages


    """
    @cached_property：只执行一次，之后返回缓存值。适合计算开销大、结果不变的场景。
    加载 HuggingFace 的 PretrainedConfig（读 JSON/下载，慢）
    """
    @cached_property
    def hf_config(self): 
        return cached_load_hf_config(self.model_path)

    """
    转成项目自己的 ModelConfig
    ModelConfig.from_hf 做的是「字段翻译」：把 HF 里各种命名不统一的字段，统一成项目需要的固定字段
    """
    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
