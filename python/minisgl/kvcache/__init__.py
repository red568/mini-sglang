"""kvcache 子包：KV Cache 存储池与前缀缓存（prefix cache）的实现与工厂。

对外暴露两个核心创建入口：
- `create_kvcache_pool`：创建 KV Cache 存储池（如 MHAKVCache）。
- `create_prefix_cache`：按类型名创建前缀缓存（如 "naive" / "radix"）。

前缀缓存实现通过 `SUPPORTED_CACHE_MANAGER` 注册表按名字查找。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    """前缀缓存工厂的函数签名协议：接收 device，返回 BasePrefixCache。"""

    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    """按模型配置创建 KV Cache 存储池。当前仅支持 MHA（多头注意力）。"""
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    """naive 前缀缓存：不做任何缓存，仅提供空实现。"""
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    """radix 前缀缓存：基于 radix tree 实现前缀共享与 LRU 驱逐。"""
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    """按类型名从注册表中实例化前缀缓存。"""
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
