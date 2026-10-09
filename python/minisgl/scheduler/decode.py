"""DecodeManager：管理处于 decode 阶段（自回归生成）的请求集合。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    # 当前可继续 decode 的请求集合（每个请求每步只生成一个 token）
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """合并新请求并过滤掉不可 decode 的请求（如已结束或 chunked 请求）。"""
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        """从运行集合中移除指定请求（正常结束或被释放）。"""
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """按 uid 查找并移除请求，返回被移除的请求；未找到返回 None。"""
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """当前在飞请求预估占用的 token 数，用于 prefill 阶段的资源预留。

        每个请求额外预留 `page_size - 1`（一页）的空间，避免因页对齐造成的碎片。
        """
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """生成下一个 decode batch：按 uid 排序以稳定各 TP rank 间的顺序。"""
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        """是否存在可继续 decode 的请求。"""
        return len(self.running_reqs) > 0
