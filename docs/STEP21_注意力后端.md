# Step 21：注意力后端接口与实现

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 21。
> 核心文件：[attention/base.py](python/minisgl/attention/base.py)、[attention/fa.py](python/minisgl/attention/fa.py)、[attention/fi.py](python/minisgl/attention/fi.py)、[attention/utils.py](python/minisgl/attention/utils.py)。
>
> 这一 Step 回答：**`attn_backend.forward` 到底怎么算注意力？prefill 和 decode 为什么用不同的 kernel？`prepare_metadata` 算出来的 `cu_seqlens` 是什么？**

---

## 一、这个 Step 要解决什么

Step 18 说注意力层把重活「甩给后端」。本 Step 打开后端这个黑盒。

注意力是最吃性能的部分，而且 **prefill 和 decode 的计算特征完全不同**：prefill 是「长序列、一次算一大段」，decode 是「短序列、每请求 1 个 token」。所以后端被设计成「按 phase 分发」，各自用最优 kernel。

---

## 二、核心逻辑

### 2.1 `BaseAttnBackend`：五个接口

```python
class BaseAttnBackend(ABC):
    def forward(self, q, k, v, layer_id, batch): ...       # 真算注意力
    def prepare_metadata(self, batch): ...                  # 前向前的元数据准备
    def init_capture_graph(self, max_seq_len, bs_list): ... # CUDA graph 初始化
    def prepare_for_capture(self, batch): ...               # 捕获前准备
    def prepare_for_replay(self, batch): ...                # replay 前准备
```

后面三个是给 CUDA graph（Step 22）用的，本 Step 聚焦前两个：`prepare_metadata`（算元数据）和 `forward`（算注意力）。

### 2.2 `HybridBackend`：按 phase 分发

```python
class HybridBackend(BaseAttnBackend):
    def __init__(self, prefill_backend, decode_backend):
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(self, q, k, v, layer_id, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)
```

`--attn fa,fi` 就是「prefill 用 FlashAttention、decode 用 FlashInfer」。`HybridBackend` 只是按 `batch.is_prefill` 二选一，把请求转发给对应的后端。注意 `init_capture_graph` / `prepare_for_capture` / `prepare_for_replay` 只调 `decode_backend`（因为 CUDA graph 只用于 decode）。

### 2.3 `prepare_metadata`：算 `cu_seqlens`（题眼）

以 [fa.py](python/minisgl/attention/fa.py) 为例：

```python
def prepare_metadata(self, batch):
    reqs = batch.padded_reqs
    seqlens_q = [req.extend_len for req in reqs]   # 每个请求「要新算」的长度
    seqlens_k = [req.device_len for req in reqs]   # 每个请求「全部 KV」的长度
    cached_lens = [req.cached_len for req in reqs]
    max_seqlen_k = max(seqlens_k)
    max_seqlen_q = max(seqlens_q)

    cu_seqlens_k = torch.tensor([0] + seqlens_k).cumsum_(0)   # K 的累积长度

    if max_seqlen_q == 1:                          # ① decode
        cu_seqlens_q = torch.arange(0, padded_size + 1)
    elif all(l == 0 for l in cached_lens):         # ② 纯 prefill（无缓存命中）
        cu_seqlens_q = cu_seqlens_k
    else:                                          # ③ 部分命中的 extend prefill
        cu_seqlens_q = torch.tensor([0] + seqlens_q).cumsum_(0)
```

`cu_seqlens`（cumulative sequence lengths）=「累积序列长度」，是**变长序列打包**的关键：多个不同长度的序列被拼成一个连续张量，`cu_seqlens` 记录每个序列的起止边界（`[0, len1, len1+len2, ...]`）。

三种情况（Q 的打包方式不同）：

| 情况 | 场景 | Q 是什么 |
|---|---|---|
| ① `max_seqlen_q == 1` | decode | 每请求 1 个新 token，`cu_seqlens_q = [0,1,2,...]` |
| ② 全 `cached_lens == 0` | 纯 prefill 无命中 | Q 和 K 一样长（整段 prompt 都算） |
| ③ 其他 | 部分命中 extend prefill | Q 只有 `extend_len`，K 是 `device_len` |

### 2.4 `forward`：先写 KV，再算注意力

```python
def forward(self, q, k, v, layer_id, batch):
    metadata = batch.attn_metadata
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)   # ① 先把本层 K/V 写进缓存
    return _fa_sgl_impl(
        q=q,
        k_cache=self.kvcache.k_cache(layer_id),            # ② 从缓存读历史 K/V
        v_cache=self.kvcache.v_cache(layer_id),
        page_table=metadata.page_table,
        cache_seqlens=metadata.cache_seqlens,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        ...)
```

**顺序很关键**：先 `store_kv` 把**这一层刚算出的** K/V 写进缓存，再从缓存读**完整的** K/V（历史 + 刚写的）算注意力。这样新 token 能看到之前所有 token 的 KV（自回归）。

`_fa_sgl_impl` 最终调 `sgl_kernel.flash_attn_with_kvcache`，把 Q、KV 缓存、page_table、cu_seqlens 全丢给 CUDA kernel 一次性算完。

### 2.5 FlashInfer 后端：`wrapper.run`

[fi.py](python/minisgl/attention/fi.py) 的思路类似，但用 FlashInfer 的 `wrapper`：

```python
def forward(self, q, k, v, layer_id, batch):
    metadata = batch.attn_metadata
    self._initialize_metadata_once(metadata)   # 第一次调用时 plan（预计算 kernel 计划）
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
    return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
```

FlashInfer 用 `wrapper.plan(...)` 预计算 kernel 启动计划（`indptr`/`indices`/`last_page_len` 等），`wrapper.run` 时直接跑。`_initialize_metadata_once` 保证 plan 只做一次（`metadata.initialized` 标志）。

---

## 三、难点解析

### 难点 1：为什么 prefill 和 decode 要分开？（性能本质）

- **prefill**：一次处理整段 prompt（比如 1000 token），`Q` 是长序列，**算力密集**（大量矩阵乘法），需要 FlashAttention 这类高吞吐 kernel。
- **decode**：每请求只生成 1 个 token，`Q` 只有 1 行，但 `K/V` 很长，**访存密集**（要读整个历史 KV），需要 FlashInfer 这类针对「短 Q 长 KV」优化的 kernel。

「长序列算力密集」用 FA、「短序列访存密集」用 FI，`HybridBackend` 让两者各取所长。

### 难点 2：`cu_seqlens` 到底解决了什么问题？

注意力 kernel 处理的是**一个 batch 里多个不同长度的序列**。如果逐序列调用 kernel，会有 N 次启动开销。`cu_seqlens` 把所有序列**拼成一个大张量**，kernel 内部根据 `cu_seqlens` 的边界知道「第 i 个序列从哪到哪」，一次 kernel 算完整个 batch。

这就是「变长序列打包」（variable-length batching）的标准技巧，`cu_seqlens` 就是打包的「索引表」。

### 难点 3：三种 `cu_seqlens_q` 情况对应的物理含义

关键是区分「Q 的长度」和「K 的长度」：

- decode：每请求 Q=1（只算新 token），但 K=device_len（全部历史）。所以 `cu_seqlens_q = [0,1,2,...]`，`cu_seqlens_k = [0, len1, len1+len2, ...]`，两者**不同**。
- 纯 prefill：Q 和 K 一样长（整段 prompt 都是新的），所以 `cu_seqlens_q = cu_seqlens_k`。
- 部分命中：Q 只有「新算的 extend_len」，K 是「全部 device_len」，所以 Q 和 K 的累积长度也不同，要**分别算**。

### 难点 4：`page_table` 在 FA 里为什么要 `div_(page_size)`？

全局 `page_table`（Step 13）存的是 **token 位置**（`page_size=1` 语义，即每个位置一个 token）。但 FA 的 `flash_attn_with_kvcache` 期望的是 **page 号**（`page_size>1` 时）。

所以 [fa.py](python/minisgl/attention/fa.py) 里：

```python
new_page_table = torch.stack([page_table[req.table_idx, :max_seqlen_k:page_size] for req in reqs])
if self.page_size > 1:
    new_page_table.div_(self.page_size, rounding_mode="floor")
```

先按 `page_size` 步长采样（取每页起点），再除以 `page_size` 把「token 位置」转成「page 号」。FlashInfer 则干脆要求 `page_size=1`（`assert self.page_size == 1`），省掉这个转换。

### 难点 5：`get_last_indices` 是给谁用的？

`BaseAttnMetadata.get_last_indices(bs)` 返回「每个序列最后一个 token 的位置」。它在 [embedding.py](python/minisgl/layers/embedding.py) 的 `ParallelLMHead.forward` 里，prefill 时被调用：

```python
if batch.is_prefill:
    indices = batch.attn_metadata.get_last_indices(bs)
    x = x[indices].contiguous()   # prefill 只取每个序列最后一个 token 的 logits
```

因为 prefill 阶段只需要「最后一个 token」的 logits 来采样，前面 token 的 logits 直接丢弃。

---

## 四、注意事项

1. **`store_kv` 必须先于读缓存**：顺序反了的话，新 token 看不到自己的 KV，注意力会算错。
2. **`prepare_metadata` 在 `_prepare_batch` 里被调用**（Step 8 的第 7 步），生成 `batch.attn_metadata`，之后 `forward` 直接读它。
3. **`FAMetadata` 的 `cu_seqlens_k` 是 `[0] + seqlens_k` 再 cumsum**：别漏了开头的 0，否则第一个序列的边界错位。
4. **FlashInfer 要求 `page_size=1`**：`FIMetadata.__post_init__` 里 `assert self.page_size == 1`，所以 `--attn fi` 时 page_size 会被强制为 1（或报错）。
5. **`_initialize_metadata_once` 的 `last_event.synchronize()`**：FlashInfer 的 plan 复用一个 pinned host buffer，plan 前要等上一个异步 H2D 拷贝完成，避免覆盖。

---

## 五、反思题

1. 用「算力密集 vs 访存密集」解释为什么 prefill 和 decode 要用不同 kernel。哪个阶段更依赖 `cu_seqlens` 的打包？
2. `cu_seqlens_q` 和 `cu_seqlens_k` 什么时候相等、什么时候不等？分别在什么场景？
3. 三种 `cu_seqlens_q` 情况里，`max_seqlen_q == 1` 和 `all(cached_lens == 0)` 的**判断顺序**能换吗？（提示：decode 时 cached_lens 可能也是 0）
4. `store_kv` 如果挪到 `forward` 的最后（算完注意力再写），会发生什么？
5. FA 里 `page_table` 为什么要 `div_(page_size)`，而 FI 里不 div？两者对 `page_size` 的要求各是什么？

---

## 六、示意图

### 6.1 `HybridBackend` 的分发

```
  AttentionLayer.forward
        │ ctx.attn_backend.forward(q,k,v,layer_id,batch)
        ▼
  HybridBackend
   ├─ batch.is_prefill ──► prefill_backend（FlashAttention）
   └─ batch.is_decode  ──► decode_backend（FlashInfer）
```

### 6.2 `cu_seqlens` 的打包

```
  三个请求：len = [3, 2, 4]
  拼成连续张量: [r0 r0 r0 | r1 r1 | r2 r2 r2 r2]
  cu_seqlens = [0, 3, 5, 9]
                  ▲     ▲     ▲
                 r0    r1    r2 的起止边界
```

### 6.3 `forward` 的两步：先写后读

```
  attention 算出 k, v
        │ ① store_kv(k, v, out_loc, layer_id)  ──► 写进 KV 池
        ▼
  读 k_cache / v_cache（历史 + 刚写的完整 KV）
        │ ② flash_attn_with_kvcache / wrapper.run
        ▼
  注意力输出 o
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [attention/base.py](python/minisgl/attention/base.py) | 后端接口 + Hybrid | `BaseAttnBackend`、`HybridBackend` |
| [attention/fa.py](python/minisgl/attention/fa.py) | FlashAttention 后端 | `FlashAttentionBackend`、`_fa_sgl_impl` |
| [attention/fi.py](python/minisgl/attention/fi.py) | FlashInfer 后端 | `FlashInferBackend`、`FIMetadata` |
| [attention/utils.py](python/minisgl/attention/utils.py) | 捕获数据基类 | `BaseCaptureData` |

**下一步**：进入 Step 22（CUDA Graph），看 decode 的固定形状怎么被「录」成一张图，一次 replay 替代成千上万次 kernel 启动。
