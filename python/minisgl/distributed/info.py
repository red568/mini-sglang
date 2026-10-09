"""DistributedInfo：张量并行（TP）的进程身份信息及其全局读写。

`_TP_INFO` 是进程级全局单例，在引擎初始化时由 `set_tp_info` 设置一次，
之后各模块通过 `get_tp_info` 读取，用于按 rank 切分模型/KV 等。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int  # 当前 rank 编号
    size: int  # TP 总 rank 数

    def __post_init__(self):
        # 校验 rank 在合法范围内
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        """是否为 rank0（主 rank）。"""
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    """设置全局 TP 信息（仅允许设置一次，重复设置报错）。"""
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """读取全局 TP 信息；未设置时抛错。"""
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """安全读取全局 TP 信息；未设置时返回 None。"""
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
