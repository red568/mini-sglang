"""TableManager：管理请求表（request table）槽位的分配与回收。

每个请求占用一个 `table_idx`（请求表行号），对应 `page_table` 与 `token_pool`
中的一整行。这里的「表」是运行时请求槽位的抽象，与 KV Cache 的分页表不同。
"""

import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        # 最多同时运行的请求数，即请求表的行数
        self._max_running_reqs = max_running_reqs
        # 空闲槽位池，pop() 分配、append() 回收
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # token_pool 保存每个槽位当前的 token id，与 page_table 同形状；初始化为 0，
        # 保证 dummy 请求读取到合法 token id。
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        """当前可分配的槽位数量。"""
        return len(self._free_slots)

    def allocate(self) -> int:
        """分配一个槽位，返回其索引（从空闲池尾部弹出）。"""
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """回收一个槽位，重新放回空闲池。"""
        self._free_slots.append(slot)
