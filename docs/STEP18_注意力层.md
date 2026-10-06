# Step 18：注意力层

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 18。
> 核心文件：[layers/attention.py](python/minisgl/layers/attention.py) 的 `AttentionLayer`。
>
> 这一 Step 回答：**Q/K/V 投影之后，注意力层做了什么？它为什么不自己算注意力，而是把重活甩给后端？**

---

## 一、这个 Step 要解决什么

Step 17 讲了 Q/K/V 投影矩阵（`LinearQKVMerged`）怎么按 head 切分。投影完得到的 `qkv` 张量，下一步就进入 `AttentionLayer`。

但 `AttentionLayer` 其实**不直接算注意力**——它只做三件「轻活」：**切分 Q/K/V → 可选 Q/K norm → RoPE**，然后把「真正算注意力」这件事交给 `ctx.attn_backend`（Step 21）。理解这个「分层」，就知道为什么 attention 能灵活切换 FlashAttention / FlashInfer 后端。

---

## 二、核心逻辑

### 2.1 `AttentionLayer`：一个无参数的「路由层」

```python
class AttentionLayer(StateLessOP):
    def __init__(self, layer_id, num_qo_heads, num_kv_heads, head_dim, rotary_config, q_norm=None, k_norm=None):
        assert num_qo_heads % num_kv_heads == 0          # GQA：Q head 必须是 KV head 的整数倍
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)                    # 切分后的 Q head 数
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)  # 切分后的 KV head 数
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(head_dim=head_dim, rotary_dim=rotary_config.rotary_dim, ...)  # RoPE
        self.q_norm = q_norm
        self.k_norm = k_norm
```

注意 `AttentionLayer` 继承 `StateLessOP`（Step 16）——它自己**没有任何可学习参数**（参数都在 `qkv_proj`、`o_proj`、`q_norm`/`k_norm` 里），只是一个「把张量变个形、转个身」的中间层。

### 2.2 `forward`：三件轻活 + 一次甩锅

```python
def forward(self, qkv: torch.Tensor) -> torch.Tensor:
    ctx = get_global_ctx()
    # ① 切分 Q/K/V
    q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
    # ② 可选 Q/K 归一化
    if self.q_norm is not None:
        self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
    if self.k_norm is not None:
        self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
    # ③ RoPE
    q, k = self.rotary.forward(ctx.batch.positions, q, k)
    q = q.view(-1, self.num_qo_heads, self.head_dim)
    # ④ 甩给后端
    o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
    return o.view(-1, self.qo_attn_dim)
```

逐行理解：

- **① `split`**：`LinearQKVMerged` 拼出来的 `qkv` 是 `[q | k | v]` 三段连在一起，`split` 按 `[qo_attn_dim, kv_attn_dim, kv_attn_dim]` 切开。
- **② Q/K norm**：只有某些模型（Qwen2）才有 `q_norm`/`k_norm`，Llama 默认没有（`None`）。用 `forward_inplace` 就地归一化，省一次拷贝。
- **③ RoPE**：`rotary.forward(positions, q, k)` 用 Step 8 里 `_make_positions` 算出的 `positions` 给 Q/K 加旋转位置编码。
- **④ 甩锅**：`ctx.attn_backend.forward(q, k, v, layer_id, batch)` 才是真正算注意力 + 写 KV cache 的地方（Step 21）。

---

## 三、难点解析

### 难点 1：GQA（Grouped Query Attention）的 `assert`

```python
assert num_qo_heads % num_kv_heads == 0
```

GQA 的意思是 **Q head 数 > KV head 数，且 KV head 被多组 Q head 共享**。比如 Qwen3 常见 `num_qo_heads=16, num_kv_heads=2`，即 8 个 Q head 共享 1 个 KV head。

`num_qo_heads % num_kv_heads == 0` 保证「能整除」，即每个 KV head 被**整数个** Q head 共享。不能整除的话 GQA 无法分组。

### 难点 2：`kv_attn_dim` 为什么是 `num_kv_heads * head_dim` 而不是 `num_qo_heads * head_dim`？

因为 K 和 V 只有 `num_kv_heads` 个头（GQA 下比 Q 少）。所以 `qkv` 总维度 = `qo_attn_dim + kv_attn_dim + kv_attn_dim` = `(num_qo + 2*num_kv) * head_dim`，正好对应 Step 17 里 `LinearQKVMerged` 的 `full_osize`。

切分时 `split([qo_attn_dim, kv_attn_dim, kv_attn_dim])` 把这段连续张量切成「Q 占前段、K 占中段、V 占后段」。

### 难点 3：注意力层为什么「不自己算注意力」？

注意力（softmax(QK^T/√d)V + 写 KV cache）是整个推理**最吃计算/访存、也最有优化空间**的部分。不同场景有不同最优 kernel：

- **prefill**（长序列、算力密集）→ FlashAttention；
- **decode**（短序列、访存密集）→ FlashInfer；
- 不同硬件（SM90/SM100）→ TRT-LLM。

如果 `AttentionLayer` 自己写死一种注意力实现，就没法灵活切换。所以它只做「**把 Q/K/V 准备好、变形好**」这种**后端无关**的活，把「怎么算注意力」抽象成 `ctx.attn_backend.forward`（Step 21 的 `HybridBackend` 再按 phase 分发给 fa/fi）。

### 难点 4：`positions` 从哪来、为什么 RoPE 需要它？

`ctx.batch.positions` 是 Step 8 里 `_make_positions` 算出来的：每个要新算的 token 的**绝对位置**（`torch.arange(cached_len, device_len)`）。

RoPE 是「旋转位置编码」，需要知道每个 token 的位置才能旋转对应的角度。所以 `rotary.forward(positions, q, k)` 把位置传进去，由 flashinfer 的 `apply_rope_with_cos_sin_cache_inplace` 就地旋转 Q 和 K。

### 难点 5：`q.view(-1, num_qo_heads, head_dim)` 的三次变形

注意 `forward` 里有多次 `view`：

- `q_norm` 前：`q.view(-1, num_qo_heads, head_dim)`——把 `[batch, dim]` 摊成 `[batch, heads, head_dim]`，norm 按 head 做；
- RoPE 前：q 是扁平的（`-1, dim`），RoPE 内部再按 head 处理；
- RoPE 后：`q.view(-1, num_qo_heads, head_dim)`——准备给后端的 3D 形状。

这些 `view` 都是**零拷贝**（只改 shape 不改内存布局），是「同一个张量在不同阶段被不同的下游看成不同形状」的体现。

---

## 四、注意事项

1. **`AttentionLayer` 是 `StateLessOP`**：它没有参数，权重都在外层（`qkv_proj`/`o_proj`），所以它在 state_dict 里「透明」（Step 16）。
2. **`q_norm`/`k_norm` 用 `forward_inplace`**：就地修改，不产生新张量，省显存；但意味着输入的 q/k 会被原地改掉。
3. **`split` 是沿 `dim=-1`**：即最后一维（隐藏维），因为 `qkv` 是 `[q | k | v]` 在最后一维拼接的。
4. **`allow_replicate=True` 在切 KV head 时又出现**：和 Step 17 的 `LinearQKVMerged` 一致，KV head 不够分时复制。
5. **`layer_id` 传给后端**：因为 KV cache 是**按层**存的（Step 19），后端要知道当前算的是第几层，才能把 K/V 写到正确的 layer 位置。

---

## 五、反思题

1. GQA 的 `assert num_qo_heads % num_kv_heads == 0` 如果去掉，用 `num_qo_heads=16, num_kv_heads=3` 会发生什么？（提示：分组怎么分？）
2. `qkv.split([qo_attn_dim, kv_attn_dim, kv_attn_dim], dim=-1)` 为什么是这三个数？和 `LinearQKVMerged` 的 `full_osize` 有什么关系？
3. 如果把注意力直接写在 `AttentionLayer.forward` 里（不抽象后端），切换 FlashAttention/FlashInfer 时要改哪些地方？
4. `q_norm.forward_inplace` 就地修改 q，如果后面还要用「未归一化的 q」，会有 bug 吗？为什么这里能安全就地改？
5. `layer_id` 为什么要传给 `attn_backend.forward`？它最终会被用来做什么（联系 Step 19 的 KV cache 按层存储）？

---

## 六、示意图

### 6.1 `AttentionLayer.forward` 的数据流

```
  qkv = [ q | k | v ]        （来自 LinearQKVMerged，dim = qo + 2*kv）
        │ split(dim=-1)
        ├──► q [qo_attn_dim] ──► (可选 q_norm) ──► RoPE(positions) ──┐
        ├──► k [kv_attn_dim] ──► (可选 k_norm) ──► RoPE(positions) ──┤
        └──► v [kv_attn_dim] ─────────────────────────────────────────┤
                                                                      ▼
                                    ctx.attn_backend.forward(q, k, v, layer_id, batch)
                                                                      │
                                                                      ▼
                                                          o [qo_attn_dim]
```

### 6.2 GQA 的 head 共享

```
  num_qo_heads=8, num_kv_heads=2（每个 KV head 被 4 个 Q head 共享）
  Q heads:  h0 h1 h2 h3 | h4 h5 h6 h7
             \    /        \    /
              KV0            KV1
```

### 6.3 注意力层的「职责边界」

```
  LinearQKVMerged（投影，TP 切分）
        │ qkv
        ▼
  AttentionLayer（切分 + norm + RoPE，后端无关）
        │ q,k,v + positions
        ▼
  attn_backend（真算注意力 + 写 KV cache，后端相关）
        │ o
        ▼
  LinearOProj（输出投影 + all-reduce）
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [layers/attention.py](python/minisgl/layers/attention.py) | 注意力层（路由） | `AttentionLayer` |
| [layers/rotary.py](python/minisgl/layers/rotary.py) | RoPE | `RotaryEmbedding.forward`、`get_rope` |

**下一步**：进入 Step 19（KV Cache 物理存储），看 `attn_backend.forward` 算完注意力后，K/V 是怎么被写进那个「按层、按 token 位置」的大池子里的。
