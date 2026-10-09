"""RadixPrefixCache：基于 radix tree（前缀树）的前缀缓存。

原理：
- 树的每个节点存一段 token 序列（key）及其对应的 KV 页索引（value）。
- 共享前缀的请求复用树上同一条路径，避免重复计算。
- 通过 `ref_count` 表示节点被多少请求「锁定」使用；ref_count == 0 且为叶子时
  可被驱逐（配合时间戳实现近似 LRU）。
- key 以「页」为粒度对齐：`key_fn` 取每页前 page_size 个 token 作为分叉键，
  保证节点的边界与分页对齐。
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0  # 被锁定引用的次数；0 表示可驱逐
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()  # 最近访问时间，用于 LRU

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """设置节点对应的 token 序列（key）与 KV 页索引（value）。"""
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        """挂到父节点下，并以本节点 key 的前缀作为分叉键。"""
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        # 返回本节点 key 与 input_ids 的最长公共前缀长度
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """在 pos 处把本节点一分为二：前半段成为新节点，后半段保留在本节点。

        返回新的前半段节点，本节点变成其子节点。
        """
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count  # 继承引用计数

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        # 供最小堆按时间戳排序：越早访问越优先驱逐
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        """从命中节点沿父链回溯到根，拼接得到命中的页索引序列。"""
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()  # 根 -> 叶，恢复原始顺序
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        # 分叉键函数：按页对齐取前缀作为子节点索引
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """锁定/解锁句柄对应路径上的所有节点。

        锁定使 ref_count +1（变为受保护，不可驱逐），解锁则 -1；
        当某节点 ref_count 归零时，其长度从 protected_size 移回 evictable_size。
        """
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        # 按页对齐截断，仅缓存完整页的部分
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            # 仅插入未命中的后半段，作为新节点挂到当前节点下
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        """驱逐至少 `size` 个 token 的缓存（按时间戳 LRU 优先驱逐最旧叶子）。

        返回被驱逐的页索引；可能驱逐超过请求量（以整节点为单位）。
        """
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        # 收集所有可驱逐的叶子节点，按时间戳构建最小堆
        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            # 父节点若变为叶子且无引用，则也可继续驱逐
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """广度遍历整棵树，收集所有 ref_count == 0 的叶子节点。"""
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """沿树向下匹配 input_ids 的最长前缀，返回 (末端节点, 已匹配长度)。

        匹配过程中遇到部分匹配的节点会将其拆分，并刷新访问时间戳（实现 LRU）。
        """
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len  # 无子节点，匹配到此为止
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            # 未完全匹配则拆分节点，返回拆分后的前半段节点
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    """根据页大小返回分叉键函数：page_size==1 用单个 token，否则用整页 token 元组。"""
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
