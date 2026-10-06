# Step 17：张量并行线性层（TP 精华）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 17。
> 核心文件：[layers/linear.py](python/minisgl/layers/linear.py) 的 `LinearReplicated`、`LinearColParallelMerged`、`LinearQKVMerged`、`LinearOProj`、`LinearRowParallel`。
>
> 这一 Step 回答：**一张权重矩阵在多个 GPU 上是怎么「切开」的？为什么有的层切完要 all-reduce、有的不用？**

---

## 一、这个 Step 要解决什么

`--tp 4` 时，模型的每一层权重被切到 4 张卡上，每张卡只算自己那一份。切法的核心问题只有一个：**这个矩阵按「列」切（输出维度切）还是按「行」切（输入维度切）？** 这决定了 forward 之后需不需要 all-reduce。

一句话口诀：**按列切（输出分片）不用 all-reduce；按行切（输入分片）必须 all-reduce。**

---

## 二、核心逻辑

### 2.1 基类 `_LinearTPImpl`：区分「完整尺寸」和「本地尺寸」

```python
class _LinearTPImpl(BaseOP):
    def __init__(self, full_isize, full_osize, local_isize, local_osize, has_bias):
        self.full_input_size  = full_isize    # 完整输入维度
        self.full_output_size = full_osize    # 完整输出维度
        self.local_input_size = local_isize   # 本 rank 的输入维度
        self.local_output_size = local_osize  # 本 rank 的输出维度
        self.weight = torch.empty(local_osize, local_isize)   # 只存本地那一份
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

每个线性层的 weight 形状是 `(local_osize, local_isize)`——**只存切分后的本地份**，不存完整矩阵。`full_*` 只是记账，记录「完整权重本该多大」，用于理解切分关系。

### 2.2 五种线性层

| 类 | 切法 | 用途 | 需要 all-reduce？ |
|---|---|---|---|
| `LinearReplicated` | 不切（每 rank 存完整） | embedding、router gate | 否（每个 rank 都有完整结果） |
| `LinearColParallelMerged` | 按列切（输出分片） | MLP 的 gate/up | 否（输出已分片） |
| `LinearQKVMerged` | 按列切（按 head 分片） | Q/K/V 投影 | 否（输出已分片） |
| `LinearOProj` | 按行切（输入分片） | attention 输出投影 | **是** |
| `LinearRowParallel` | 按行切（输入分片） | MLP 的 down | **是** |

### 2.3 按列切：`LinearColParallelMerged`

```python
class LinearColParallelMerged(_LinearTPImpl):
    def __init__(self, input_size, output_sizes, has_bias):
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]  # 输出各切一份
        output_size = sum(output_sizes)              # 完整输出 = gate + up
        tp_output_size = sum(tp_output_sizes)        # 本地输出 = 各切一半
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)
```

MLP 第一层是 `gate_proj` 和 `up_proj` 两个矩阵，`output_sizes = [intermediate, intermediate]`。`Merged` 把两个矩阵**拼成一个** `[2*intermediate, hidden]`，一次 `F.linear` 算出 gate 和 up 两路。按列切时，每个 rank 拿 `intermediate/tp` 列的输出，输出天然是分片的——**不需要 all-reduce**（每个 rank 的输出就是它该负责的那片）。

### 2.4 按 head 切：`LinearQKVMerged`

```python
class LinearQKVMerged(_LinearTPImpl):
    def __init__(self, hidden_size, head_dim, num_qo_heads, num_kv_heads, has_bias):
        tp_info = get_tp_info()
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)   # 注意 allow_replicate
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim     # Q + K + V 拼一起
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(hidden_size, full_osize, hidden_size, local_osize, has_bias)
```

Q/K/V 三个投影**合并成一个矩阵** `[(qo + 2*kv) * head_dim, hidden]`，一次算出 Q、K、V。切分按 **head 数**切：每个 rank 负责 `num_qo/tp` 个 Q head + `num_kv/tp` 个 KV head。也是按列切，输出分片，**不需要 all-reduce**。

### 2.5 按行切：`LinearOProj` 和 `LinearRowParallel`（需要 all-reduce）

```python
class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size, output_size, has_bias):
        tp_info = get_tp_info()
        local_isize = div_even(input_size, tp_info.size)   # 输入维度切分
        local_osize = output_size                          # 输出维度完整
        self._comm = DistributedCommunicator()
        super().__init__(input_size, output_size, local_isize, local_osize, has_bias)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)   # 关键：部分和 → all-reduce 求和
        return y
```

`LinearRowParallel` 结构完全一样（`local_input_size = div_even(input_size, tp_size)`）。它们按**行切**（输入维度切）：每个 rank 只算「部分输入 × 部分权重」的**部分和**，要得到完整输出，必须把各 rank 的部分和 **all-reduce 求和**。

---

## 三、难点解析

### 难点 1：为什么「按列切」不需要 all-reduce？（题眼）

设权重 `W: [out, in]`，输入 `x: [in]`，输出 `y = W x`。

**按列切**（切 `out`）：`W = [W_0; W_1]`（上下叠），`y_0 = W_0 x`、`y_1 = W_1 x`。每个 rank 算出的 `y_i` **本来就是完整输出的不同片段**，拼起来就是完整 `y`，中间没有任何「求和」需求——所以不用通信。

**按行切**（切 `in`）：`W = [W_0 | W_1]`（左右并），`x = [x_0; x_1]`，`y = W_0 x_0 + W_1 x_1`。每个 rank 算出的 `W_i x_i` 只是**部分和**，必须 all-reduce 加起来才是完整 `y`——所以需要通信。

一句话：**切输出 = 各算一段，天然拼接；切输入 = 各算一部分，必须求和。**

### 难点 2：为什么 MLP 的两层一个按列、一个按行？

MLP = `down(gate_up(x))`，两个矩阵相乘。为了中间不通信：

- 第一层 `gate_up`（hidden → intermediate）**按列切**：输出分片，不用通信。
- 第二层 `down`（intermediate → hidden）**按行切**：输入正好是上一层的分片输出，各 rank 算部分和，最后 **all-reduce 一次** 得到完整 hidden。

这样整个 MLP 只 all-reduce **一次**（在 down 之后），而不是每层都通信。这是 TP 的标准模式。

### 难点 3：`div_even(..., allow_replicate=True)` 什么时候用？

```python
def div_even(a, b, allow_replicate=False):
    if allow_replicate and b > a:        # 分不开（b > a）
        assert b % a == 0                # 但要能整除，比如 4 个 rank 分 2 个 KV head
        return 1                         # 每个 rank 存 1 个，但会「复制」
    assert a % b == 0
    return a // b
```

当 **KV head 数 < TP 数**时（如 GQA 模型 2 个 KV head、`--tp 4`），KV head 不够分。`allow_replicate=True` 允许「复制」：每个 rank 存 1 个 KV head，但 4 个 rank 总共只有 2 个不同的 KV head，于是**某些 rank 存的是重复的 KV head**。

Q head 不能复制（`local_num_qo = div_even(num_qo_heads, tp)` 没开 allow_replicate），因为 Q head 通常远多于 TP 数。只有 KV head 可能少到不够分。

### 难点 4：`Merged`（合并）的意义

`LinearQKVMerged` 把 Q/K/V 三个矩阵拼成一个，`LinearColParallelMerged` 把 gate/up 两个矩阵拼成一个。为什么合并？

1. **一次 `F.linear` 替代两次/三次**，减少 kernel 启动次数；
2. **权重连续存储**，切分时按「合并后的输出维度」切，边界干净；
3. **减少权重搬运**，load 时一个 key 对应一块连续内存。

代价是「合并不了不同 input size 的矩阵」，但 Q/K/V 和 gate/up 的 input size 恰好相同，所以能合并。

---

## 四、注意事项

1. **`local_osize`/`local_isize` 才是真实 weight 形状**：`weight = torch.empty(local_osize, local_isize)`，`full_*` 只是记账，别在 forward 里用 full 尺寸。
2. **`LinearOProj` 和 `LinearRowParallel` 的 `_comm` 是 `DistributedCommunicator`**：它用 `plugins[-1]`（最后注册的实现，通常是 pynccl，Step 14 的 `enable_pynccl_distributed` 注册）做 all-reduce。
3. **`all_reduce` 前有 `if self._tp_size > 1` 判断**：单卡时不通信，直接返回。
4. **`allow_replicate=True` 只在 KV head 分不开时触发**：Q head、MLP 中间维度都不开，因为它们分不开就该报错（说明模型和 TP 数不匹配）。
5. **embedding 也是 TP 的**（[embedding.py](python/minisgl/layers/embedding.py)）：`VocabParallelEmbedding` 按 vocab 切（`div_ceil`），forward 后 `all_reduce`；`ParallelLMHead` 是它的反向（`all_gather`）。这俩不在 linear.py 里，但同属 TP 通信。

---

## 五、反思题

1. 用「切输出 vs 切输入」的矩阵视角，分别画 `LinearColParallelMerged` 和 `LinearRowParallel` 的权重切分图，标出 all-reduce 的位置。
2. 为什么 `LinearQKVMerged` 按 head 切，而不是按「隐藏维」切？按 head 切对后面的注意力计算有什么好处？
3. `div_even(2, 4, allow_replicate=True)` 返回什么？`div_even(2, 4)`（不传）会怎样？为什么 KV head 能复制、Q head 不能？
4. `LinearOProj` 和 `LinearRowParallel` 代码几乎一样，为什么分成两个类？（提示：语义用途不同，未来可能有不同优化）
5. 如果 MLP 的 `down` 层改成「按列切」而不是「按行切」，会发生什么？（提示：它输出的 hidden 会被切碎吗？）

---

## 六、示意图

### 6.1 按列切 vs 按行切

```
【按列切】（切输出 out，无需 all-reduce）
      in                 in
   ┌────────┐         ┌────────┐
out│  W_0   │  rank0  out_0│ W_0 x │  ← 各算一段，天然拼接
   ├────────┤
   │  W_1   │  rank1  out_1│ W_1 x │
   └────────┘

【按行切】（切输入 in，必须 all-reduce）
      in_0   in_1          in_0  in_1
   ┌───────┬───────┐    rank0: W_0 x_0  ┐
out│  W_0  │  W_1  │    rank1: W_1 x_1  ┘─► all_reduce 求和 → y
   └───────┴───────┘
```

### 6.2 一个 Transformer 层里的 TP 通信位置

```
  hidden (replicated)
     │
  ┌──▼───────────────┐   ┌─────────────────┐
  │ QKV (列切)        │   │ gate_up (列切)   │   ← 输出分片，不通信
  └──┬───────────────┘   └────────┬────────┘
     │ Q/K/V 分片                  │ 中间分片
     ▼                             ▼
  Attention                     act_fn
     │                             │
     ▼                             ▼
  OProj (行切) ──all_reduce──►  down (行切) ──all_reduce──►
     │                             │
     └──────────► hidden (replicated) ◄──────────┘
```

### 6.3 Q/K/V 合并矩阵的 head 切分

```
  LinearQKVMerged weight: [(qo + 2*kv) * head_dim, hidden]
  按 head 切:
  rank0: Q_heads[0..qo/tp-1] + KV_heads[0..kv/tp-1]
  rank1: Q_heads[qo/tp..]    + KV_heads[kv/tp..]
  ...
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [layers/linear.py](python/minisgl/layers/linear.py) | 五种 TP 线性层 | `LinearColParallelMerged`、`LinearQKVMerged`、`LinearOProj`、`LinearRowParallel` |
| [layers/embedding.py](python/minisgl/layers/embedding.py) | 词表并行（顺带理解） | `VocabParallelEmbedding`、`ParallelLMHead` |
| [distributed/impl.py](python/minisgl/distributed/impl.py) | all-reduce/all-gather 实现 | `DistributedCommunicator` |

**下一步**：进入 Step 18（注意力层），看 Q/K/V 分片后，注意力层怎么切分、RoPE、再把重活甩给注意力后端。
