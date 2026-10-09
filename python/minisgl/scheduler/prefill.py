"""PrefillManager / PrefillAdder：处理请求的 prefill（前缀填充）阶段调度。

prefill 阶段将输入的 prompt token 送入模型计算 KV，并支持 chunked prefill：
当单个请求的扩展长度超过单次 token 预算时，拆分为多段依次调度，避免长序列
挤占其它请求的调度机会。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """chunked prefill 中尚未处理完的请求占位：禁止采样、不进入 decode 集合。"""

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    """负责在单次 prefill 调度中逐个尝试加入请求，并管理预算与资源预留。"""

    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """尝试为一个新请求分配资源（前缀缓存匹配 + 请求表槽位）。

        返回 `(cache_handle, table_idx)`，若资源不足则返回 None。
        """
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        # 匹配前缀缓存，命中部分无需重新计算
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        # 预估该请求最终占用的 token 数 = 待扩展长度 + 输出长度
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # 资源不足则放弃（需同时预留 decode 在飞请求的空间）
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        # 加锁后二次校验，避免并发/竞争导致超额分配
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # 分配请求表槽位
        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            # 命中缓存的部分：把已缓存的 token id 与页表直接搬入该请求的槽位
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """根据预算确定本段 chunk 大小，构造对应的 Req（或 ChunkedReq）。"""
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        # 为未完成部分 + 输出预留空间，供后续资源检查使用
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        # 仅搬入本段 token id；物理页由 scheduler 的 allocate_paged 统一分配
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """尝试加入一个请求：预算耗尽或资源不足时返回 None。

        若是 chunked 请求的后续段，则复用之前分配的 handle/table_idx 继续。
        """
        if self.token_budget <= 0:
            return None

        # 续接之前被拆分的请求：复用已有的 handle 与槽位
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # 全新请求：先分配资源再构造
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    # 等待 prefill 的请求队列
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """将新到达的用户请求加入待处理队列。"""
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """从待处理队列中尽可能多地选出请求组成 prefill batch。

        以 token 预算为上限，并在资源估算时预留 decode 在飞请求的空间。
        """
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    # 记录被拆分的请求，供下次调度续接
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        # 被拆分的请求移到队首，其余保留原顺序，保证 chunked 请求优先续接
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        """按 uid 取消一个待处理请求，返回其已入队的 chunked 段（若有）。"""
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        """是否存在等待 prefill 的请求。"""
        return len(self.pending_list) > 0
