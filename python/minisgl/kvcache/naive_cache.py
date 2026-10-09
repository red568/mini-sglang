"""NaivePrefixCache：不做任何前缀缓存的「空实现」。

用于关闭前缀复用场景（如基准对比或调试）：匹配恒不命中、插入恒返回空、
驱逐不支持。提供与其它实现一致的接口，保证上层逻辑无需分支。
"""

import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    """naive 句柄：命中长度恒为 0，命中索引为空张量。"""

    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        self.device = device
        # 共享一个空张量作为所有句柄的命中索引，避免反复分配
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass  # 无缓存，无需加锁

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle())  # 恒不命中

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle())  # 恒不插入

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass  # 无内部状态，无需校验
