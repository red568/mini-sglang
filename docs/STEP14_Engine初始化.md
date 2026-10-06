# Step 14：Engine 初始化（TP worker 的心脏）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 14。
> 核心文件：[engine/engine.py](python/minisgl/engine/engine.py) 的 `Engine`。
>
> 这一 Step 回答：**一个 Scheduler rank 里，真正干活的 `Engine` 是怎么从零把「通信、模型、KV cache、页表、后端、采样、CUDA graph」一步步搭起来的？**

---

## 一、这个 Step 要解决什么

前面 Step 8-13 一直在用 `self.engine.page_table`、`self.engine.forward_batch`、`self.engine.graph_runner`，但从没打开过 `Engine` 这个黑盒。本 Step 打开它。

`Engine.__init__` 是**整个项目初始化逻辑最密集的函数之一**，按顺序干七件大事。理解它的关键是把握「**meta 设备建图 → 一次性填权重 → 按显存算 KV 页数**」这条主线。

---

## 二、核心逻辑

### 2.1 `__init__` 的七步

```python
def __init__(self, config: EngineConfig):
    assert not torch.cuda.is_initialized()          # 前置：CUDA 还没初始化
    set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
    _adjust_config(config)                           # 自动补全 attention/moe 后端

    self.device = torch.device(f"cuda:{config.tp_info.rank}")
    torch.cuda.set_device(self.device)
    torch.manual_seed(42)
    self.stream = torch.cuda.Stream()                # engine 自己的 stream
    torch.cuda.set_stream(self.stream)
    self.dtype = config.dtype
    self.ctx = Context(config.page_size)
    set_global_ctx(self.ctx)                         # 全局 ctx（Step 7）

    self.tp_cpu_group = self._init_communication(config)     # ① 通信
    init_free_memory = self._sync_get_memory()[1]

    # ② meta 建图
    set_rope_device(self.device)
    with torch.device("meta"), torch_dtype(config.dtype):
        self.model = create_model(config.model_config)
    # ③ 填权重
    self.model.load_state_dict(self._load_weight_state_dict(config))

    # ④ 按显存算 KV 页数
    self.num_pages = self._determine_num_pages(init_free_memory, config)
    num_tokens = self.num_pages * config.page_size
    self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
        model_config=config.model_config,
        num_pages=self.num_pages + 1,                # +1 dummy page
        page_size=config.page_size, device=self.device, dtype=self.dtype)

    # ⑤ 页表初始化
    self.max_seq_len = min(config.max_seq_len, num_tokens)
    aligned_max_seq_len = _align_up_32(self.max_seq_len)
    self.ctx.page_table = self.page_table = torch.zeros(
        (config.max_running_req + 1, aligned_max_seq_len),  # +1 dummy request
        dtype=torch.int32, device=self.device)

    # ⑥ 注意力/MoE 后端
    self.ctx.attn_backend = self.attn_backend = create_attention_backend(...)
    if config.model_config.is_moe:
        self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

    # ⑦ 采样器 + CUDA graph
    self.sampler = Sampler(self.device, config.model_config.vocab_size)
    self.dummy_req = Req(input_ids=torch.tensor([0], ...), table_idx=config.max_running_req, ...)
    self.page_table[self.dummy_req.table_idx].fill_(num_tokens)   # 指向 dummy page
    self.graph_runner = GraphRunner(stream=self.stream, model=self.model, ...)
```

### 2.2 ① `_init_communication`：gloo + pynccl 还是 nccl

```python
def _init_communication(self, config):
    if config.tp_info.size == 1 or config.use_pynccl:
        torch.distributed.init_process_group(backend="gloo", ...)   # CPU 控制面
        tp_cpu_group = torch.distributed.group.WORLD
        max_bytes = config.max_forward_len * hidden_size * dtype.itemsize
        enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)  # GPU 数据面
    else:
        torch.distributed.init_process_group(backend="nccl", ...)   # GPU 数据面
        tp_cpu_group = torch.distributed.new_group(backend="gloo")  # CPU 控制面
    return tp_cpu_group
```

两套后端的分工：**gloo 跑 CPU 控制面**（Step 10 的广播消息条数、barrier），**nccl/pynccl 跑 GPU 数据面**（模型前向里的 all-reduce）。单卡 `size==1` 时也用 gloo，是因为哪怕只有一个 rank 也要有 `tp_cpu_group` 来做 `broadcast`/`barrier`。

### 2.3 ②③ meta 建图 + 一次性填权重

```python
with torch.device("meta"), torch_dtype(config.dtype):
    self.model = create_model(config.model_config)      # meta：不占真实显存
self.model.load_state_dict(self._load_weight_state_dict(config))  # 一次性搬真权重
```

- **meta 设备**：PyTorch 的「假设备」，建图时 tensor 只有形状/dtype、不分配内存。几十层模型瞬间建完。
- `load_state_dict` 把真实权重（或 `--dummy-weight` 的随机权重）**一次性**灌进去，同时把 meta tensor 物化成真实 CUDA tensor。

`_load_weight_state_dict`：

```python
if config.use_dummy_weight:
    return {k: torch.randn_like(v, device=self.device) for k, v in self.model.state_dict().items()}
else:
    return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}
```

### 2.4 ④ `_determine_num_pages`：按显存算 KV 页数

```python
cache_per_page = (
    2                               # key + value
    * config.model_config.head_dim
    * div_even(num_kv_heads, tp_size, allow_replicate=True)   # TP 切分后的 KV head 数
    * config.page_size
    * self.dtype.itemsize
    * config.model_config.num_layers
)
num_pages = config.num_page_override
if num_pages is None:
    model_memory = old_free_memory - new_free_memory
    available_memory = int(config.memory_ratio * old_free_memory) - model_memory
    num_pages = available_memory // cache_per_page
```

`cache_per_page` 是「一页 KV cache 占多少字节」：K 和 V 各一份 × head_dim × KV head 数 × page_size × dtype 字节 × 层数。**注意 KV head 数要除以 `tp_size`**——张量并行下每个 rank 只存自己那一片 KV。

`num_page_override` 未设时，用「可用显存 ÷ 每页大小」反推页数。

### 2.5 ⑤ 页表初始化的两个 `+1`

```python
self.ctx.kv_cache = create_kvcache_pool(num_pages=self.num_pages + 1, ...)   # +1 dummy page
self.ctx.page_table = torch.zeros((config.max_running_req + 1, aligned_max_seq_len), ...)  # +1 dummy req
```

两个 `+1` 都是给 **dummy 请求**留的位置（见 Step 22 CUDA graph）：`dummy_req` 的 `table_idx = max_running_req`（最后一个槽位），指向 `num_tokens` 处的 dummy page。

### 2.6 `forward_batch`：前向 + 采样 + 异步拷贝

```python
def forward_batch(self, batch, args):
    assert torch.cuda.current_stream() == self.stream
    with self.ctx.forward_batch(batch):
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)   # decode 用 CUDA graph
        else:
            logits = self.model.forward()              # prefill 正常 forward
    for req in batch.reqs:
        req.complete_one()                              # 更新三个长度（Step 6）
    next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)   # 异步拷回 CPU
    copy_done_event = torch.cuda.Event()
    copy_done_event.record(self.stream)
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

这就是 Step 8/9 里 `_forward` 调用的那个方法。`complete_one()` 在采样后统一执行，`copy_done_event` 是 Step 9 里 `copy_done.synchronize()` 等的那个事件。

---

## 三、难点解析

### 难点 1：为什么用 meta 设备建图？

正常 `nn.Module` 建图时每个参数都立即分配 GPU 显存。但这里要**先知道模型长什么样，再决定 KV cache 用多少显存**（`_determine_num_pages` 需要「模型占了多少显存」这个数）。

meta 设备让建图「零成本」——拿到 `state_dict` 的形状信息，却不占显存。之后 `load_state_dict` 才真正把参数物化到 CUDA。这也是为什么 `_load_weight_state_dict` 里能 `for k, v in self.model.state_dict().items()` 拿到所有参数的**形状**（dummy 模式按形状生成随机权重）。

### 难点 2：`cache_per_page` 里的 `div_even(..., allow_replicate=True)`

KV head 数要除以 `tp_size`（每个 rank 只存一部分 KV head）。但如果 KV head 数**不够分**（比如 2 个 KV head、4 个 rank），`div_even` 的 `allow_replicate=True` 允许**复制**而不是报错——某些 rank 存重复的 KV head。

`allow_replicate=True` 是在「分不开」时优雅降级的开关（Step 17 张量并行会详细展开）。

### 难点 3：`_sync_get_memory` 为什么用 all-reduce 取 min？

```python
free_mem_tensor = torch.tensor([free_memory, -free_memory], ...)
torch.distributed.all_reduce(free_mem_tensor, op=MIN, group=self.tp_cpu_group)
min_free_memory = int(free_mem_tensor[0].item())      # 所有 rank 的最小值
max_free_memory = -int(free_mem_tensor[1].item())     # 取负后再取反 = 最大值
```

一个巧妙技巧：把 `free_memory` 和 `-free_memory` 打包，用 `MIN` all-reduce 一次拿到「最小值」和「最大值」。取 min 是因为**所有 rank 必须用同样大小的 KV cache**（否则页数不一致，page_table 形状对不上），按「显存最紧张的 rank」来定。

### 难点 4：`_adjust_config` 用 `object.__setattr__` 绕过 frozen

```python
def override(attr, value):
    object.__setattr__(config, attr, value)   # 绕过 frozen dataclass 的 __setattr__
```

`EngineConfig` 是 `frozen=True`（Step 2），正常 `config.attention_backend = "fi"` 会抛异常。`object.__setattr__` 直接改底层 `__dict__`，在「自动补全默认值」这种**初始化期一次性设置**场景下是刻意的绕过。注释自己都写了 `this is dangerous, use with caution`。

---

## 四、注意事项

1. **`assert not torch.cuda.is_initialized()`**：Engine 必须在 CUDA 上下文建立前初始化，否则 stream/device 的设置会被污染。
2. **`self.stream` 和 Scheduler 的 `self.stream` 是两条不同的 stream**：Engine 里这条是计算流（`forward_batch` 断言 `current_stream == self.stream`），Scheduler 里那条是元数据准备流（Step 9）。
3. **`max_seq_len = min(config.max_seq_len, num_tokens)`**：最长序列长度受限于 KV cache 能装下的 token 总数，两者取小。
4. **`page_table` 用 `aligned_max_seq_len = _align_up_32(max_seq_len)`**：对齐到 32，为了满足 kernel 的内存对齐要求（128 字节）。
5. **`dummy_req` 的 `sampling_params=None`、`cache_handle=None`**：它是占位请求，不参与真实采样和缓存，所以这两项置空（`# type: ignore`）。

---

## 五、反思题

1. 为什么 `_determine_num_pages` 要「先建模型、测显存、再算页数」，而不是反过来？如果反过来会有什么问题？
2. `cache_per_page` 公式里每一项各代表什么？为什么 `num_kv_heads` 要除以 `tp_size`，而 `num_layers` 不除？
3. meta 设备建图 + `load_state_dict` 相比直接 `nn.Module().cuda()`，省掉了什么？dummy 模式为什么能拿到「形状」？
4. `_sync_get_memory` 打包 `[free, -free]` 的技巧，如果只 all-reduce `free`（不取负），能拿到什么？拿不到什么？
5. `dummy_req` 的 `table_idx = max_running_req`、`page_table.fill_(num_tokens)` 指向 dummy page，这个 dummy 请求在 Step 22 的 CUDA graph 里扮演什么角色？

---

## 六、示意图

### 6.1 `Engine.__init__` 七步流程

```
  __init__
   │
   ├─ ① _init_communication      gloo(控制面) + nccl/pynccl(数据面)
   ├─ ② create_model(meta)       零成本建图，拿到形状
   ├─ ③ load_state_dict          一次性灌真/随机权重
   ├─ ④ _determine_num_pages     按显存算 KV 页数
   ├─ ⑤ page_table / kv_cache    初始化页表 + KV 池（+1 dummy）
   ├─ ⑥ attention / moe 后端     选 fa/fi/trtllm、fused 等
   └─ ⑦ Sampler + GraphRunner    采样器 + CUDA graph 捕获
```

### 6.2 显存的三段账

```
  总显存
  ├─ 模型权重（meta 建图后 load_state_dict 才占用）
  ├─ KV cache（_determine_num_pages 按剩余显存反推页数）
  └─ 其他（激活值、CUDA graph 内存池、采样器等）
```

### 6.3 meta 建图 → 填权重的两阶段

```
  阶段1: torch.device("meta")        阶段2: load_state_dict
  create_model ──► 只有形状/dtype ──► 物化成真实 CUDA tensor
  不占显存、瞬间完成                   一次性搬权重（或 randn_like dummy）
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [engine/engine.py](python/minisgl/engine/engine.py) | Engine 组装 + 前向 | `Engine.__init__`、`_determine_num_pages`、`forward_batch` |
| [engine/graph.py](python/minisgl/engine/graph.py) | CUDA graph 捕获（Step 22 展开） | `GraphRunner` |
| [engine/sample.py](python/minisgl/engine/sample.py) | 采样（Step 23 展开） | `Sampler` |

**下一步**：进入 Step 15（模型前向结构），看 meta 建出来的 Llama 模型 `forward()` 是怎么一层层算下去的。
