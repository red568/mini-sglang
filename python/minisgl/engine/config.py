"""EngineConfig：单个 TP worker 的引擎配置。

继承层级：`EngineConfig` 是引擎底层配置；`SchedulerConfig` 在此基础上补充
调度相关参数（见 scheduler/config.py）。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str  # 模型权重/HF 配置路径
    tp_info: DistributedInfo  # 张量并行信息（rank / size）
    dtype: torch.dtype  # 计算 dtype
    max_running_req: int = 256  # 最大同时运行的请求数
    attention_backend: str = "auto"  # attention 后端，auto 时按硬件自动选择
    moe_backend: str = "auto"  # MoE 后端，auto 时按模型自动选择
    cuda_graph_bs: List[int] | None = None  # 显式指定要捕获的 CUDA Graph 批大小
    cuda_graph_max_bs: int | None = None  # 未指定时按显存估算的最大捕获批大小
    page_size: int = 1  # 分页 KV Cache 的页大小（token 数）
    memory_ratio: float = 0.9  # 可用于 KV Cache 的显存比例
    distributed_timeout: float = 60.0  # 分布式初始化超时（秒）
    use_dummy_weight: bool = False  # 是否用随机权重（测试/基准）
    use_pynccl: bool = True  # 是否使用 pynccl 做跨卡通信
    max_seq_len_override: int | None = None  # 覆盖最大序列长度
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        """加载 HuggingFace 的 PretrainedConfig（读 JSON/下载，较慢）。

        `cached_property`：只执行一次，之后返回缓存值，适合计算开销大、
        结果不变的场景。
        """
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        """转成项目自己的 ModelConfig。

        `ModelConfig.from_hf` 做的是「字段翻译」：把 HF 里命名不统一的字段，
        统一成项目需要的固定字段。
        """
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        """最大序列长度：优先用覆盖值，否则取模型 rotary 配置的最大位置。"""
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        """单次 forward 的最大长度，直接复用 max_seq_len。"""
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        """分布式进程组的 init_method 地址。"""
        return "tcp://127.0.0.1:2333"
