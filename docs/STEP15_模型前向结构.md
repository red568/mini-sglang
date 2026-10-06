# Step 15：模型前向结构（以 Llama 为例）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 15。
> 核心文件：[models/base.py](python/minisgl/models/base.py)、[models/llama.py](python/minisgl/models/llama.py)、[models/utils.py](python/minisgl/models/utils.py)、[layers/norm.py](python/minisgl/layers/norm.py)。
>
> 这一 Step 回答：**`model.forward()` 不带任何参数，是怎么从全局 ctx 拿到输入，一层层算出 logits 的？残差流是怎么在层间传递的？**

---

## 一、这个 Step 要解决什么

Step 7 说了「模型 `forward()` 不带参数的秘密是全局 ctx」，Step 14 说了模型在 meta 设备上建出来。本 Step 落到具体的**前向结构**：以 Llama 为例，看一个标准 decoder-only Transformer 的每一层长什么样、残差流（residual stream）怎么走。

这是进入 Step 17（TP 线性层）和 Step 18（注意力层）之前，先建立「骨架」的一步。

---

## 二、核心逻辑

### 2.1 `BaseLLMModel`：抽象的「无参 forward」

```python
class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...
```

注意 `forward(self)` **没有任何参数**——输入不靠参数传，而是靠子类实现里自己去 `get_global_ctx().batch.input_ids` 拿（Step 7 的机制）。这就是项目里所有模型前向的统一约定。

### 2.2 `LlamaForCausalLM.forward`：入口

```python
class LlamaForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(..., tie_word_embeddings=config.tie_word_embeddings,
                                      tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None)

    def forward(self):
        output = self.model.forward(get_global_ctx().batch.input_ids)   # 从全局拿输入
        logits = self.lm_head.forward(output)
        return logits
```

顶层分两块：`LlamaModel`（embedding + N 层 + 最终 norm）+ `lm_head`（词表投影）。`tie_word_embeddings=True` 时 `lm_head` **复用** `embed_tokens` 的权重（不单独存一份，省显存）。

### 2.3 `LlamaModel.forward`：embedding → N 层 → norm

```python
def forward(self, input_ids):
    x = self.embed_tokens.forward(input_ids)     # ① token → hidden
    residual = None
    for layer in self.layers.op_list:            # ② N 个 decoder layer
        x, residual = layer.forward(x, residual)
    return self.norm.forward(x, residual)[0]     # ③ 最终 norm，取归一化输出
```

### 2.4 `LlamaDecoderLayer.forward`：一层的主干（题眼）

```python
def forward(self, x, residual=None):
    x, residual = self.input_layernorm.forward(x, residual)       # ① pre-norm（融合残差加）
    x = self.self_attn.forward(x)                                 # ② 注意力
    x, residual = self.post_attention_layernorm.forward(x, residual)  # ③ 再 norm
    x = self.mlp.forward(x)                                       # ④ MLP
    return x, residual
```

一层 = **两个子块**（attention 和 MLP），每个子块前面各一个 norm，标准的 **pre-LayerNorm** 结构。

### 2.5 `RMSNormFused`：残差流的关键（融合了「加残差」和「归一化」）

```python
def forward(self, x, residual=None):
    if residual is None:
        return self.rmsnorm(x, self.weight, self.eps), x      # 第一层：x 自己就是残差
    self.fused_add_rmsnorm(x, residual, self.weight, self.eps) # 融合：residual += x；x = rmsnorm(residual)
    return x, residual
```

`fused_add_rmsnorm(x, residual, ...)` 一个 kernel 干两件事：**把 `x` 加进 `residual`（残差累加），再对 `residual` 做 RMSNorm**，结果写回 `x`。这样残差流 `residual` 一路累加、`x` 一路是「归一化后的残差」。

---

## 三、难点解析

### 难点 1：残差流（residual stream）是怎么走的？

这是理解整个 Llama 前向的钥匙。跟踪 `residual` 变量：

```
进入层:  x = 上一层的 mlp 输出（未经 norm），residual = 上一层的累计残差
  input_layernorm.forward(x, residual):
       residual = residual + x           # attention 的输入加进残差流
       x = RMSNorm(residual)             # 归一化后的残差喂给 attention
  self_attn(x) → a                       # attention 输出
  post_attention_layernorm.forward(a, residual):
       residual = residual + a           # attention 输出也加进残差流
       x = RMSNorm(residual)             # 归一化后喂给 MLP
  mlp(x) → m                             # MLP 输出，作为下一层的 x
返回:  (m, residual)                     # 注意 m 还没进残差流！
```

**关键点**：每个子块的输出（`a`、`m`）不是立刻加进残差，而是**在下一个 norm 处才加**。所以层结束返回的 `residual` 已经包含了 attention 输出，但 MLP 输出 `m` 要等**下一层的 `input_layernorm`** 才加进去。这就是 pre-norm 残差流的经典写法。

### 难点 2：为什么 `fused_add_rmsnorm` 要「融合」？

如果分开写，是 `residual = residual + x; x = rmsnorm(residual)`——两个 kernel，中间还有一个「读 residual、写 residual、再读 residual」的往返。

`fused_add_rmsnorm` 一个 kernel 完成，省掉一次显存往返。这种「把加法和归一化融成一个 kernel」是推理优化的常见套路（flashinfer 提供）。对每一层都省一点，几十层累积起来就很可观。

### 难点 3：`LlamaModel.forward` 最后的 `[0]` 是什么意思？

`self.norm.forward(x, residual)` 返回 `(x_norm, residual)` 元组，但最终只取 `[0]`（归一化后的 `x`）喂给 `lm_head`。因为最后不需要再传残差了，`residual` 到这里使命结束。

### 难点 4：`RopeAttn` 和 `GatedMLP` 的命名

`models/utils.py` 里 `RopeAttn`（`LlamaAttn`）和 `GatedMLP`（`LlamaMLP`）是 Llama 的注意力块和 MLP 块：

```python
class RopeAttn(BaseOP):
    def forward(self, x):
        qkv = self.qkv_proj.forward(x)       # Q/K/V 合并成一个矩阵（Step 17）
        o = self.attn.forward(qkv)           # 真正的注意力（Step 18）
        return self.o_proj.forward(o)        # 输出投影

class GatedMLP(BaseOP):
    def forward(self, x):
        gate_up = self.gate_up_proj.forward(x)   # gate/up 合并（Step 17）
        y = self.act_fn(gate_up)                 # SiLU/GELU 激活 + 门控
        return self.down_proj.forward(y)         # down 投影（带 all-reduce）
```

命名里的 `Merged`（合并）和 `RowParallel`（带 all-reduce）都是 Step 17 张量并行的伏笔，这里先记住「Q/K/V 合一个矩阵、gate/up 合一个矩阵」即可。

---

## 四、注意事项

1. **`forward()` 真的不带参数**：`LlamaForCausalLM.forward(self)` 里 `self.model.forward(get_global_ctx().batch.input_ids)`，输入是运行时从全局拿的，不是构造/调用时传的。
2. **`residual` 初始为 `None`**：第一层的 `input_layernorm` 走 `if residual is None` 分支，`x` 自己成为初始残差。
3. **`tie_word_embeddings` 时 `lm_head` 不存独立权重**：`tied_embedding=self.model.embed_tokens`，权重共享，改 embedding 权重会同步影响 lm_head。
4. **`RopeAttn` 里的 `has_qk_norm`**：某些模型（Qwen2）有 Q/K 归一化，Llama 默认没有，对应 `self.q_norm = self.k_norm = None`。
5. **每个 layer 的 `_layer_id`**：`@nvtx_annotate("Layer_{}", layer_id_field="_layer_id")` 用 layer id 给 NVTX 打标，方便 profile 时定位到具体哪一层。

---

## 五、反思题

1. 用一句话描述 `residual` 变量在一层 forward 里的「两次累加」分别发生在哪两个 norm 处，各自累加的是什么。
2. 如果 `RMSNormFused.forward` 里 `residual is None` 分支写错（比如返回 `None`），第一层之后会发生什么？
3. `LlamaModel.forward` 最后的 `[0]` 去掉会怎样？为什么这里能安全取 `[0]`？
4. `tie_word_embeddings=True` 和 `False` 各有什么利弊？什么场景下必须 False？
5. 为什么「Q/K/V 合并成一个矩阵」是张量并行的前置条件？（提示：想想 Q/K/V 分别按什么维度切，Step 17 展开）

---

## 六、示意图

### 6.1 Llama 完整前向结构

```
  input_ids ──► embed_tokens ──► [ x ]
                                   │
        ┌──────────────────────────┴───────────────────────────┐
        │                    × N 层 DecoderLayer                │
        │  input_layernorm → self_attn → post_attn_norm → mlp  │
        └──────────────────────────┬───────────────────────────┘
                                   ▼
                                norm ──► lm_head ──► logits
```

### 6.2 一层内的残差流（题眼图）

```
  x(m 上一层 mlp 输出) ──┐
                        │ residual += x
  residual ─────────────┴─► RMSNorm ──► self_attn ──► a
                              │                          │
                              │                          │ residual += a
                              │                          ▼
                              └──────── RMSNorm ◄────────┘
                                          │
                                          ▼
                                        mlp ──► m ──► (下一层 x)
  （此时 residual 已含 a，但还没含 m）
```

### 6.3 `fused_add_rmsnorm` 的「融合」

```
  分开写:  residual = residual + x     （kernel 1）
           x = rmsnorm(residual)       （kernel 2）

  融合写:  fused_add_rmsnorm(x, residual, w, eps)   （kernel 1，一个 kernel 干完）
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [models/base.py](python/minisgl/models/base.py) | 模型基类 | `BaseLLMModel` |
| [models/llama.py](python/minisgl/models/llama.py) | Llama 结构 | `LlamaForCausalLM`、`LlamaDecoderLayer`、`LlamaModel` |
| [models/utils.py](python/minisgl/models/utils.py) | 注意力/MLP 块 | `RopeAttn`、`GatedMLP` |
| [layers/norm.py](python/minisgl/layers/norm.py) | RMSNorm（融合残差） | `RMSNormFused`、`RMSNorm` |

**下一步**：进入 Step 16（BaseOP 与权重加载），看这套「类 nn.Module」的轻量层系统怎么支撑 meta 建图和精确权重加载。
