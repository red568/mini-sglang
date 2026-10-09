"""CacheManager：管理分页 KV Cache 的分配/回收，以及与前缀缓存（prefix cache）的交互。

核心职责：
- 维护一块按「页」（page）对齐的空闲页池 `free_slots`。
- 通过 `prefix_cache` 复用相同前缀的已计算 KV，避免重复计算。
- 在请求生命周期中协调「空闲页分配」与「前缀缓存插入/驱逐」之间的页流转。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        # free_slots 以页对齐方式记录空闲页的起始 token 偏移（每项代表一整页）。
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        # 前缀缓存：用于匹配/插入共享前缀的 KV
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        """在前缀缓存中匹配请求的前缀，返回匹配结果（含命中 handle）。

        注意只匹配前 `input_len - 1` 个 token：最后一个 token 留给本次 prefill 计算。
        """
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """当前可用 token 数 = 可驱逐缓存 + 空闲页，用于 prefill 的资源估算。"""
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        """锁定前缀缓存句柄，防止其在请求使用期间被驱逐。"""
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """解锁前缀缓存句柄，允许后续驱逐。"""
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        """为一批请求分配其扩展段（未命中缓存的 token）所需的物理页。

        对每个请求，计算从 `cached_len`（已命中前缀的长度）到 `device_len`
        （需要驻留设备的总长度）之间的页数，一次性批量分配并写回 page_table。
        """
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            # 已缓存部分占用的页数
            first_page = div_ceil(req.cached_len, self.page_size)
            # 完整驻留所需的页数
            last_page = div_ceil(req.device_len, self.page_size)
            # 仅当扩展段跨越了新页时才需要分配
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            # 分配页，再展开为 token 级偏移，写回每个请求的 page_table 行
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """将请求已计算的 KV 写入前缀缓存，并释放不再需要的页。

        依据请求是否 finished 决定尾部页的去留：
        - 未完成：保留尾部，更新 handle 并加锁，等待下一轮 decode。
        - 已完成：释放尾部，回收全部资源。
        """
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        # 将 [0, cached_len) 的 token 序列与对应页写入前缀缓存，返回实际命中长度与新句柄
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        # 该段已被其它请求插入缓存，释放本请求持有的页，避免泄漏
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        """校验缓存页与空闲页的总数守恒，以及空闲页的对齐性。"""
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        """临时将 `_free` 替换为「延迟回收」，在上下文中产生的释放操作最后统一合并。

        用于在批量处理多个请求时，避免频繁的 tensor concat，减少碎片化与开销。
        """

        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            # 恢复原始 _free，并将延迟释放的页一次性合并回空闲池
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        """分配指定数量的页，若空闲不足则从前缀缓存驱逐以腾出空间。"""
        if needed_pages > (free_pages := len(self.free_slots)):
            # 按 token 粒度驱逐，再折算为页
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        """释放 token 级索引对应的页（按页对齐取每页首地址）。"""
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        """把页索引展开为页内所有 token 的偏移。page_size == 1 时无需展开。"""
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    """将分配的物理页 token 偏移写回每个请求对应的 page_table 行。

    `allocation_info` 每项为 `(table_idx, first_page, last_page)`，表示该请求
    需要覆盖的页区间；`allocated` 是连续排列的已分配 token 偏移。
    """
    needed_tokens = len(allocated)
    # 用 pinned memory 先在 CPU 侧构造 (table_idx, 位置) 索引对，再一次性搬移到 GPU
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    # 通过高级索引一次性写入：page_table[table_idx, position] = allocated
    page_table[table_idxs, offsets] = allocated
