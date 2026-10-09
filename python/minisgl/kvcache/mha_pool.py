"""MHAKVCache：多头注意力（MHA）的 KV Cache 存储池。

以分页方式分配一块大 buffer（形状 [2, 层数, 页数, 页大小, KV头数, 头维]），
其中第 0 维分别存 K 与 V。跨 TP 时，KV 头被切分到各 rank 本地。
"""

from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        # TP 切分后本地持有的 KV 头数
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # [K/V, 层, 页, 页内 token, 本地 KV 头, 头维] 的一块大 buffer
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        # 展平后供 kernel 按 token 偏移索引的形状
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 K 缓存（按层索引）。"""
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 V 缓存（按层索引）。"""
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """将第 layer_id 层的 K/V 按 `out_loc` 指定的 token 偏移写入缓存。"""
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
