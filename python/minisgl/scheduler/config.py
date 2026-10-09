"""Scheduler 配置：在 `EngineConfig` 基础上补充调度相关参数与进程间通信地址。"""

from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    """用当前进程 PID 生成唯一后缀，避免多个进程同时运行时 IPC 地址冲突。"""
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    # prefill 阶段单次 forward 允许扩展的最大 token 数（即 chunked prefill 的预算上限）
    max_extend_tokens: int = 8192
    # 前缀缓存类型，例如 "radix"（基于 radix tree 的前缀缓存）
    cache_type: str = "radix"
    # 离线模式：跳过 tokenizer 通信链路，直接本地驱动（用于测试/benchmark）
    offline_mode: bool = False

    # networking config
    # 唯一后缀用于隔离不同进程的 ZMQ IPC 地址
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        """tokenizer -> backend 的消息队列地址（请求入口）。"""
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        """backend -> detokenizer 的消息队列地址（结果出口）。"""
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """TP 多 rank 场景下，rank0 向其它 rank 广播消息的 pub/sub 地址。"""
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        """单次 forward 的最大长度，直接复用 prefill 扩展预算。"""
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """backend 是否需要主动创建到 detokenizer 的连接。"""
        return True
