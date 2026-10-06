# Step 22：CUDA Graph（decode 加速）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 22。
> 核心文件：[engine/graph.py](python/minisgl/engine/graph.py) 的 `GraphRunner`。
>
> 这一 Step 回答：**decode 阶段每一小步的几千次 kernel 启动，怎么被「录」成一张图、一次 replay 就全跑完？`dummy_req` 和静态 buffer 是干嘛的？**

---

## 一、这个 Step 要解决什么

decode 每生成一个 token，都要跑一遍完整的模型 forward（几十层 × 每层多个 kernel）。这些 kernel 的**启动本身**有 CPU 开销——每次 `F.linear`、`flash_attn` 都要 CPU 发指令给 GPU。

CUDA Graph 的思路：**把这些 kernel 启动序列「录制」下来，之后一次 `replay` 就全部重放**，把成千上万次 CPU→GPU 启动压成一次。前提是「每次的 kernel 序列和形状完全一样」——decode 恰好满足（每请求固定 1 个 token）。

---

## 二、核心逻辑

### 2.1 `GraphCaptureBuffer`：静态 buffer

```python
@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    def copy_from(self, batch):
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids   # 把真实 batch 数据灌进静态 buffer
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
```

CUDA graph 捕获时，**张量的内存地址是固定的**（静态 buffer）。replay 时不能换张量，只能**把新数据 `copy_from` 进这个固定 buffer**，然后重放。四个字段正好是 forward 的输入（`input_ids`/`out_loc`/`positions`）和输出（`logits`）。

### 2.2 `_capture_graphs`：对每个 batch size 录一张图

```python
def _capture_graphs(self, max_seq_len, vocab_size, model):
    self.graph_map = {}                     # bs → CUDAGraph
    if self.max_graph_bs == 0:
        return logger.info_rank0("CUDA graph is disabled.")

    self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

    pool = None
    for bs in sorted(self.graph_bs_list, reverse=True):   # 从大到小录
        graph = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")   # 用 dummy_req 造 batch
        batch.padded_reqs = batch.reqs
        self.attn_backend.prepare_for_capture(batch)      # 后端准备静态 metadata
        self.buffer.set_batch(batch)                      # batch 的输入指向静态 buffer
        with get_global_ctx().forward_batch(batch):
            self.buffer.logits[:bs] = model.forward()     # ① warmup（分配内存）
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                self.buffer.logits[:bs] = model.forward() # ② 真正捕获
        if pool is None:
            pool = graph.pool()                          # 复用内存池
        self.graph_map[bs] = graph
```

几个关键点：

- **`dummy_req`**：捕获时用 `[dummy_req] * bs` 造一个「假的 decode batch」，形状和真实 decode 一样（每请求 1 个 token）。
- **两次 forward**：第一次是 **warmup**（让 CUDA 分配好内存），第二次才在 `torch.cuda.graph(...)` 上下文里**真正捕获**。
- **`pool` 复用**：第一次捕获后拿 `graph.pool()`（内存池），后续 bs 复用同一个池，**大幅省显存**（否则每个 bs 都重新分配一套内存）。

### 2.3 `pad_batch`：用 dummy 补齐到捕获尺寸

```python
def pad_batch(self, batch):
    padded_size = (
        next(bs for bs in self.graph_bs_list if bs >= batch.size)   # 找 ≥ 真实大小的最近捕获尺寸
        if self.can_use_cuda_graph(batch)
        else batch.size)
    batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
```

CUDA graph 是**按固定 batch size 录的**（`graph_bs_list = [1,2,4,8,16,...]`），真实 batch 不可能每次都刚好等于这些数。`pad_batch` 把真实 batch 补到**最近的捕获尺寸**，多出来的位置用 `dummy_req` 填（Step 7 讲的 `padded_reqs`）。

### 2.4 `replay`：灌数据 → 重放 → 取 logits

```python
def can_use_cuda_graph(self, batch):
    return batch.is_decode and batch.size <= self.max_graph_bs

def replay(self, batch):
    self.buffer.copy_from(batch)                 # ① 把真实数据灌进静态 buffer
    g = self.graph_map[batch.padded_size]        # ② 取对应尺寸的图
    self.attn_backend.prepare_for_replay(batch)  # ③ 后端把 metadata 拷进静态 buffer
    g.replay()                                   # ④ 一次重放整个前向
    return self.buffer.logits[: batch.size]      # ⑤ 取真实部分的 logits（丢弃 dummy）
```

`replay` 是 decode 时 `engine.forward_batch` 里的快路径（Step 14 的 `can_use_cuda_graph` 判定）。`logits[: batch.size]` 只取真实请求的结果，dummy 部分丢弃。

---

## 三、难点解析

### 难点 1：为什么只有 decode 能用 CUDA graph？

CUDA graph 要求「**每次重放时 kernel 序列和形状完全一致**」。decode 恰好满足：

- 每请求固定算 **1 个新 token**（`extend_len == 1`）；
- 所以整个 forward 的每个张量形状都是固定的（只随 batch size 变，而 batch size 已被 pad 到固定值）。

prefill 不满足：prompt 长度千变万化，`extend_len` 不确定，形状每次不同，没法录成一张固定图。所以 `can_use_cuda_graph` 要求 `batch.is_decode`。

### 难点 2：为什么需要 warmup（第一次 forward）？

CUDA graph 捕获时，所有中间张量**必须分配在固定地址**（因为图里记录的是内存地址）。第一次 forward 让 PyTorch 分配好这些中间张量并「固定」下来，第二次在 `torch.cuda.graph` 上下文里捕获时，kernel 就能稳定引用这些地址。

没有 warmup 的话，捕获时 PyTorch 的动态内存分配会干扰图的地址稳定性，导致捕获失败或 replay 时地址错乱。

### 难点 3：`pool` 复用为什么省显存？

每个 `torch.cuda.CUDAGraph` 默认有自己的内存池，录 N 个 bs 就是 N 份内存。`pool = graph.pool()` 拿到第一个图的内存池后，传给后续 `torch.cuda.graph(..., pool=pool)`，让**所有 bs 的图共用同一个内存池**。

这样显存开销从「N × 单图内存」降到「1 × 最大图内存」（因为最大的 bs 图需要的内存最大，其他 bs 可以复用它的池子）。

### 难点 4：`dummy_req` 的完整链路

`dummy_req`（Step 14 里 `table_idx = max_running_req`）贯穿整个 CUDA graph 机制：

1. **捕获时**：`Batch(reqs=[dummy_req]*bs)` 造假 batch；
2. **补齐时**：`pad_batch` 把真实 batch 补到 `padded_size`；
3. **采样时**：`forward_batch` 里 `logits[: batch.size]` 只取真实部分，dummy 的 logits 丢弃。

`dummy_req` 的 `table_idx` 指向 dummy page（Step 14 的 `page_table.fill_(num_tokens)`），它的 KV 写到专门留的 dummy 页，不影响真实请求。

### 难点 5：`_determine_cuda_graph_bs` 的 bs 列表为什么是 `[1,2,4,8,16,...]`

```python
return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
```

小 bs（1/2/4）单独列（小 batch 常见，精确匹配省显存），大 bs 按 8 的步长列（`8, 16, 24, ...`）。这是「精确度 vs 图数量」的权衡：步长越小，pad 越少（浪费越少），但要录的图越多（显存越大）。

---

## 四、注意事项

1. **`can_use_cuda_graph` 有两个条件**：`is_decode` 且 `size <= max_graph_bs`。超过 `max_graph_bs` 的 decode batch 走正常 forward（不 replay）。
2. **`pad_batch` 只在能用 graph 时才 pad**：不能用 graph 时 `padded_size = batch.size`，不 pad（避免无意义的 dummy）。
3. **`copy_from` 和 `set_batch` 是反向操作**：`copy_from` 把 batch → 静态 buffer（replay 前），`set_batch` 把静态 buffer → batch（捕获时让 batch 指向静态 buffer）。
4. **`destroy_cuda_graphs` 必须在释放 NCCL 资源前调用**：注释明确写了「must be called before freeing NCCL resources to prevent program hang」，因为 graph 里可能记录了通信算子。
5. **`max_graph_bs = 0`（如 `--cuda-graph-max-bs 0`）时完全禁用**：`_capture_graphs` 直接 return，`can_use_cuda_graph` 恒 False。

---

## 五、反思题

1. 为什么 prefill 不能用 CUDA graph，而 decode 能？用「kernel 序列和形状是否固定」解释。
2. `pad_batch` 把 5 个真实请求补到 8，那 3 个 dummy 的 forward 结果去哪了？采样时怎么排除它们？
3. `pool = graph.pool()` 复用内存池，为什么能省显存？如果不复用会多占多少？
4. 捕获时为什么 `sorted(graph_bs_list, reverse=True)`（从大到小）？顺序有影响吗？（提示：pool 的复用）
5. 如果 `--cuda-graph-max-bs` 设太小，导致 decode batch 常超过 `max_graph_bs`，性能会怎样？为什么？

---

## 六、示意图

### 6.1 捕获流程

```
  对每个 bs（从大到小）:
  Batch([dummy_req]*bs, phase="decode")
        │
        ▼
  buffer.set_batch(batch)      batch 输入指向静态 buffer
        │
        ▼
  model.forward()              ← ① warmup（分配内存）
  torch.cuda.graph(graph):
      model.forward()          ← ② 捕获（录 kernel 序列）
        │
        ▼
  graph_map[bs] = graph
```

### 6.2 replay 流程

```
  decode batch（真实 size=5，pad 到 8）
        │
        ▼
  buffer.copy_from(batch)      把 5 个真实请求数据灌进静态 buffer
        │
        ▼
  graph_map[8].replay()        一次重放整个 forward
        │
        ▼
  buffer.logits[:5]            取真实 5 个 logits，丢弃 3 个 dummy
```

### 6.3 CUDA graph 省了什么

```
  无 graph：decode 一步 = 几十层 × 每层多个 kernel = 上千次 CPU→GPU 启动
  有 graph：decode 一步 = 1 次 g.replay()（重放已录好的上千个 kernel）
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [engine/graph.py](python/minisgl/engine/graph.py) | CUDA graph 捕获/重放 | `GraphRunner`、`GraphCaptureBuffer`、`pad_batch`、`replay` |
| [attention/utils.py](python/minisgl/attention/utils.py) | 捕获数据基类 | `BaseCaptureData` |

**下一步**：进入 Step 23（采样），看 logits 怎么变成下一个 token，greedy 和随机采样分别走哪条路。
