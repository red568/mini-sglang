# Step 19：KV Cache 物理存储

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 19。
> 核心文件：[kvcache/base.py](python/minisgl/kvcache/base.py)、[kvcache/mha_pool.py](python/minisgl/kvcache/mha_pool.py)。
>
> 这一 Step 回答：**每一层算出来的 K/V，最终被存到哪块显存里？`out_loc` 这个 batch 字段到底在指示什么？**

---

## 一、这个 Step 要解决什么

Step 18 讲到 `attn_backend.forward(q, k, v, ...)` 会「写 KV cache」。本 Step 打开这个「KV cache 池子」看它长什么样、怎么写。

核心认知：**KV cache 就是一块巨大的连续显存**，按 `(层, 物理token位置, KV头, 头维度)` 组织。所有请求的 K/V 都往这个池子里写，`out_loc` 决定「每个新 token 写到池子的哪个位置」。

---

## 二、核心逻辑

### 2.1 `BaseKVCachePool`：KV 池的接口

```python
class BaseKVCachePool(ABC):
    def k_cache(self, index): ...    # 取第 index 层的 K 缓存
    def v_cache(self, index): ...    # 取第 index 层的 V 缓存
    def store_kv(self, k, v, out_loc, layer_id): ...   # 把本层算出的 K/V 写进去
    @property device / dtype / num_layers
```

接口就三件正事：**按层取 K/V 缓存**（给注意力后端读）、**写 K/V**（`store_kv`）。加几个只读属性。

### 2.2 `MHAKVCache`：一个 6 维大张量

```python
class MHAKVCache(BaseKVCachePool):
    def __init__(self, num_kv_heads, num_layers, head_dim, num_pages, page_size, dtype, device):
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)   # TP 切分
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device, dtype=dtype)
        self._k_buffer = self._kv_buffer[0]    # K 是第 0 维
        self._v_buffer = self._kv_buffer[1]    # V 是第 1 维
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
```

这个 6 维张量的每个维度：

| 维度 | 大小 | 含义 |
|---|---|---|
| 0 | `2` | K 和 V 两份 |
| 1 | `num_layers` | 每一层一份 |
| 2 | `num_pages` | 页数（PagedAttention 的分页） |
| 3 | `page_size` | 每页的 token 数 |
| 4 | `local_kv_heads` | 本 rank 的 KV head 数（TP 切分后） |
| 5 | `head_dim` | 每个 head 的维度 |

`k_cache(index)` / `v_cache(index)` 就是 `self._k_buffer[index]`——取第 `index` 层的 K/V 缓存，形状 `(num_pages, page_size, local_kv_heads, head_dim)`。

### 2.3 `store_kv`：把新算的 K/V 写进池子

```python
def store_kv(self, k, v, out_loc, layer_id):
    from minisgl.kernel import store_cache
    store_cache(
        k_cache=self._k_buffer[layer_id].view(self._storage_shape),
        v_cache=self._v_buffer[layer_id].view(self._storage_shape),
        indices=out_loc,
        k=k, v=v)
```

关键两步：

1. **`view(self._storage_shape)`**：把 `(num_pages, page_size, local_kv_heads, head_dim)` 摊平成 `(num_pages * page_size, local_kv_heads, head_dim)`——把「页 × 页内」两个维度**合并成一个连续的 token 维度**。
2. **`indices=out_loc`**：`out_loc`（Step 8 里 `batch.out_loc = page_table[input_mapping]`）是**每个新 token 要写到的物理位置**（在 `num_pages * page_size` 这个 token 空间里的下标）。`store_cache` kernel 按这个下标，把 `k`/`v` 里对应 token 的 K/V 写到池子的正确位置。

一句话：**`out_loc` 是把「逻辑 token」映射到「物理池子位置」的钥匙**，`store_kv` 靠它把每个 token 的 K/V 精准落位。

---

## 三、难点解析

### 难点 1：为什么要 `view` 成 3 维再 `store_cache`？

`_kv_buffer` 是 6 维（含 K/V 和层），但取到某一层后是 4 维 `(num_pages, page_size, local_kv_heads, head_dim)`。其中「页 × 页内」其实是**一个统一的 token 空间**——第 `p` 页第 `i` 个 token = token 空间里的 `p * page_size + i`。

`out_loc` 存的就是「token 空间里的扁平下标」（Step 13 里 `_page_to_token` 算出来的正是这个扁平位置）。所以要把页维度摊平，让 `store_cache` 能直接用 `out_loc` 做一维索引。

### 难点 2：`out_loc` 和 `page_table` 的关系

回顾 Step 13 的链条：

```
page_table[table_idx, pos]  →  物理 KV 位置（扁平 token 下标）
batch.out_loc = page_table[input_mapping]  →  本 batch 每个 token 的物理位置
```

`page_table` 是「逻辑（table_idx, pos）→ 物理」的映射，`out_loc` 只是把「本 batch 要算的那些 token」的物理位置**取出来**。`store_kv` 拿到 `out_loc` 后，就知道每个新 token 的 K/V 该写进池子的哪个格子。

### 难点 3：`local_kv_heads` 为什么要除以 `tp_size`？

KV cache 也是**张量并行切分**的（和 Step 17 的权重切分一致）：每个 rank 只存 `num_kv_heads / tp_size` 个 KV head。所以 `_kv_buffer` 的第 4 维是 `local_kv_heads` 而不是完整的 `num_kv_heads`。

`allow_replicate=True`（Step 17 讲过）处理 KV head 不够分的情况。

### 难点 4：K/V 为什么拼成第 0 维，而不是分开两个张量？

`_kv_buffer = torch.empty((2, ...))` 把 K 和 V 放在同一个张量的第 0 维。好处：

1. **一次分配**，连续显存，避免两个张量各自对齐浪费；
2. `self._k_buffer = self._kv_buffer[0]` / `[1]` 是零拷贝的 view，逻辑上分开、物理上连续；
3. 传给 kernel 时，K/V 的地址可以算出来（offset）。

这是「一个物理张量、多个逻辑视图」的常见做法。

---

## 四、注意事项

1. **`store_kv` 是「写」不是「读」**：它由注意力后端在每次 forward 后调用（Step 18 的 `attn_backend.forward` 内部），把本层刚算出的 K/V 落盘。读 KV 是在下一次 forward 的 attention 里做。
2. **`k_cache`/`v_cache` 按 `layer_id` 索引**：因为 KV cache 是**按层独立**的，第 `layer_id` 层的 K/V 只写/读到 `_kv_buffer[layer_id]`。
3. **`_storage_shape` 在 `__init__` 就算好**：`(num_pages * page_size, local_kv_heads, head_dim)`，`store_kv` 里反复 `view` 成这个形状，避免重复计算。
4. **`num_pages` 是 Step 14 里 `_determine_num_pages` 算出来的**：KV 池的大小由显存决定，池子大小 = 页数 × 页大小 × KV head × head dim × 层数 × 2（K+V）。
5. **`store_cache` 是自定义 kernel**（[kernel/](python/minisgl/kernel/)）：用 `out_loc` 做 gather/scatter 式的写，比 Python 循环快得多。

---

## 五、反思题

1. `_kv_buffer` 的 6 个维度分别是什么？如果 `page_size` 从 1 变成 2，哪些维度的大小会变？
2. `store_kv` 里为什么要 `view(self._storage_shape)`？不 view 直接传 4 维张量给 kernel 会怎样？
3. `out_loc` 里的值范围是多少？（提示：它索引的是 `num_pages * page_size` 的 token 空间）超出范围会发生什么？
4. 为什么 KV cache 要按 `layer_id` 分开，而不能所有层共用一个 buffer？（提示：每层的 K/V 内容完全不同）
5. `local_kv_heads` 和完整 `num_kv_heads` 什么时候相等？（提示：`tp_size=1` 时）

---

## 六、示意图

### 6.1 `_kv_buffer` 的 6 维结构

```
  _kv_buffer: (2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
     │
     ├─ [0] = K buffer  ──► (num_layers, num_pages, page_size, ...)
     └─ [1] = V buffer  ──► (num_layers, num_pages, page_size, ...)
                                  │
                                  └─ 第 layer_id 层: (num_pages, page_size, local_kv_heads, head_dim)
```

### 6.2 `store_kv` 的落位过程

```
  attention 算出 k, v（形状: [num_tokens, local_kv_heads, head_dim]）
        │
        │ out_loc = [12, 30, 7, 45, ...]（每个 token 的物理位置）
        ▼
  store_cache(k_cache.view(storage_shape), indices=out_loc, k=k, v=v)
        │
        │ 把 k[i] 写到 pool[out_loc[i]]，v[i] 写到 pool[out_loc[i]]
        ▼
  pool（token 空间，num_pages * page_size 行）
  ┌──────────────┐
  │ ...          │
  │ pool[7]  ◄── k/v of token #2
  │ pool[12] ◄── k/v of token #0
  │ pool[30] ◄── k/v of token #1
  │ pool[45] ◄── k/v of token #3
  └──────────────┘
```

### 6.3 逻辑 token → 物理位置 → 池子格子

```
  逻辑:  (table_idx, pos)   ──page_table──►  物理位置（扁平下标）
         batch.out_loc = page_table[input_mapping]
                                              │
                                              ▼
  池子:  _kv_buffer[layer_id].view(num_pages*page_size, ...)
                                              │ store_cache 按 out_loc 写
                                              ▼
                                        KV 落位
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [kvcache/base.py](python/minisgl/kvcache/base.py) | KV 池 + 前缀缓存接口 | `BaseKVCachePool`、`store_kv`、`BasePrefixCache` |
| [kvcache/mha_pool.py](python/minisgl/kvcache/mha_pool.py) | MHA KV 池实现 | `MHAKVCache` |

**下一步**：进入 Step 20（Radix Cache 前缀复用），看「前缀共享」是怎么靠一棵 Radix 树做到「共享前缀的 K/V 只算一次」的。
