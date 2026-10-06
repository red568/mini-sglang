# Step 8：Scheduler 主循环（`normal_loop`）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 8。
> 核心文件：[scheduler/scheduler.py](python/minisgl/scheduler/scheduler.py) 的 `Scheduler`、`normal_loop`、`_prepare_batch`、`_forward`。
>
> 这一 Step 回答：**一个 batch 从「凑齐请求」到「算完 logits、把结果送回 tokenizer」的完整编排是怎么一圈圈转起来的？**

---

## 一、这个 Step 要解决什么

前面 Step 6/7 讲了 `Req` 和 `Batch` 这两个「数据」，Step 8 讲的是把它们**用起来的「引擎」**——Scheduler。它像工厂流水线的调度员，每一轮干四件事：**收消息 → 凑 batch → 前向 → 处理结果**。这四步的编排逻辑就是 `normal_loop`。

理解 Scheduler 的关键，是看清它手里握着的两样东西：

1. **一个 `Engine`**：真正算 logits 的地方（模型、KV cache、sampler、CUDA graph 都在里面）。
2. **四个 manager**：`TableManager`（逻辑槽位）、`CacheManager`（KV 页）、`PrefillManager`（待 prefill 请求）、`DecodeManager`（正在 decode 请求）。

---

## 二、核心逻辑

### 2.1 `Scheduler.__init__`：组装零件

```python
def __init__(self, config: SchedulerConfig):
    self.engine = Engine(config)                 # ① 计算核心

    self.device = self.engine.device
    self.stream = torch.cuda.Stream(device=self.device)   # ② 一条新 stream（给 CPU 侧元数据用）
    self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
    torch.cuda.set_stream(self.stream)                     # ③ 当前 stream 切成 self.stream

    self.table_manager  = TableManager(config.max_running_req, self.engine.page_table)
    self.cache_manager  = CacheManager(self.engine.num_pages, config.page_size, ...)
    self.decode_manager = DecodeManager(config.page_size)
    self.prefill_manager = PrefillManager(self.cache_manager, self.table_manager, self.decode_manager)

    self.finished_reqs: Set[Req] = set()
    self.tokenizer = load_tokenizer(config.model_path)
    self.token_pool = self.table_manager.token_pool      # ④ 别名，GPU 上的 token 池
    self.prefill_budget = config.max_extend_tokens

    super().__init__(config, self.engine.tp_cpu_group)   # ⑤ I/O mixin，接 ZMQ
```

要点：Scheduler 自己开了一条 **`self.stream`**，与 `engine.stream` 是两条不同的 CUDA stream。这条 stream 专门给「CPU 侧准备元数据 + H2D 拷贝」用，`engine.stream` 给「真正算 logits」用。两条 stream 是 Step 9（overlap 调度）的伏笔，Step 8 的 `normal_loop` 里暂时用不上这个区分。

### 2.2 `normal_loop`：一轮流水线的四步

```python
def normal_loop(self) -> None:
    blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):   # ① 收消息
        self._process_one_msg(msg)

    forward_input = self._schedule_next_batch()       # ② 凑 batch + 准备元数据
    ongoing_data = None
    if forward_input is not None:
        ongoing_data = (forward_input, self._forward(forward_input))  # ③ 前向

    self._process_last_data(ongoing_data)             # ④ 处理结果
```

四步一一拆解：

| 步骤 | 函数 | 干什么 |
|---|---|---|
| ① 收消息 | `receive_msg` | 从 tokenizer 拉 `UserMsg`/`AbortBackendMsg`，逐个 `_process_one_msg` |
| ② 凑 batch | `_schedule_next_batch` | prefill 优先、decode 其次，选出 `Batch` 并填好 `positions`/`out_loc` 等 |
| ③ 前向 | `_forward` | 取 `input_ids`、跑 `engine.forward_batch`、把 `next_tokens_gpu` 写回 token_pool |
| ④ 处理结果 | `_process_last_data` | 等 GPU→CPU 拷贝完成，`append_host`、判 finished、发 `DetokenizeMsg` |

### 2.3 `_process_one_msg`：三路分拣消息

```python
def _process_one_msg(self, msg):
    if isinstance(msg, BatchBackendMsg):      # 批量包装 → 拆开逐个处理
        for msg in msg.data: self._process_one_msg(msg)
    elif isinstance(msg, ExitMsg):            # 退出信号 → 抛异常终止
        raise KeyboardInterrupt
    elif isinstance(msg, UserMsg):            # 新请求 → 校验长度 → 进 prefill 队列
        input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
        max_output_len = max_seq_len - input_len
        if max_output_len <= 0:
            return logger.warning_rank0(...)  # 输入太长，直接丢弃
        if msg.sampling_params.max_tokens > max_output_len:
            msg.sampling_params.max_tokens = max_output_len  # 超上限则截断
        self.prefill_manager.add_one_req(msg)
    elif isinstance(msg, AbortBackendMsg):    # 中止请求 → 从两个 manager 里揪出来释放
        req_to_free = self.prefill_manager.abort_req(msg.uid)
        req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
        if req_to_free is not None:
            self._free_req_resources(req_to_free)
```

注意 `UserMsg` 进来不是直接变成 `Req`，而是先存成 `PendingReq`（见 [utils.py](python/minisgl/scheduler/utils.py)）塞进 `PrefillManager.pending_list`，等下一轮 `_schedule_next_batch` 才真正分配资源。

### 2.4 `_schedule_next_batch`：prefill 优先

```python
def _schedule_next_batch(self) -> ForwardInput | None:
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)  # 先 prefill
        or self.decode_manager.schedule_next_batch()                    # 再 decode
    )
    return self._prepare_batch(batch) if batch else None
```

- **prefill 优先**：只要有待 prefill 的请求，就优先做 prefill（新请求的「冷启动」优先级高于老请求的「继续吐 token」）。
- `PrefillManager.schedule_next_batch` 受 `prefill_budget`（= `max_extend_tokens`）限制，一轮最多扩展这么多 token，装不下就**分块（chunk）**，剩下的留到下轮。
- `DecodeManager.schedule_next_batch` 直接 `sorted(running_reqs, key=uid)` 把正在 decode 的请求按 uid 排序打包，全量一起算（decode 每步每请求只算 1 个 token，通常都能装下）。

### 2.5 `_prepare_batch`：把 `Batch` 的空字段填满

```python
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    self.engine.graph_runner.pad_batch(batch)      # ① 补齐到 CUDA graph 尺寸（填 padded_reqs）
    self.cache_manager.allocate_paged(batch.reqs)  # ② 分配 KV 页
    batch.positions = _make_positions(batch, self.device)   # ③ 每个 token 的位置（RoPE 用）
    input_mapping = _make_input_tuple(batch, self.device)   # ④ (table_idx, positions) 映射
    write_mapping = _make_write_tuple(batch, self.device)   # ⑤ (table_idx, seq_len|-1) 写回映射
    batch.out_loc = self.engine.page_table[input_mapping]   # ⑥ 逻辑位置 → 物理 KV 位置
    self.engine.attn_backend.prepare_metadata(batch)        # ⑦ 注意力后端元数据
    return ForwardInput(batch=batch,
                        sample_args=self.engine.sampler.prepare(batch),
                        input_tuple=input_mapping,
                        write_tuple=write_mapping)
```

这七步正好对应 Step 7 那张「字段填充时序图」，把 `Batch` 里所有 `init=False` 的字段一个个填上。三个 `_make_*` 是纯 index 计算，见难点 2。

### 2.6 `_forward`：取输入 → 算 → 写回

```python
def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
    batch, sample_args, input_mapping, output_mapping = forward_input
    batch.input_ids = self.token_pool[input_mapping]          # ① 从 token_pool 取输入
    forward_output = self.engine.forward_batch(batch, sample_args)  # ② 模型前向 + 采样
    self.token_pool[output_mapping] = forward_output.next_tokens_gpu  # ③ 新 token 写回池
    self.decode_manager.filter_reqs(forward_input.batch.reqs)  # ④ 更新 running_reqs
    return forward_output
```

这就是 Step 7 说的「`batch.input_ids` 在 `_forward` 里才被赋值」。`forward_batch` 内部（见 [engine.py](python/minisgl/engine/engine.py)）会 `with self.ctx.forward_batch(batch)` 把 batch 放进全局 ctx，再 `model.forward()` 或 `graph_runner.replay(batch)`。

---

## 三、难点解析

### 难点 1：`blocking` 参数在控制什么？

```python
blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
```

`receive_msg(blocking=...)`：当 `blocking=True`（即**没有任何可算的东西**）时，`receive_msg` 会调 `run_when_idle()` 然后 **`recv.get()` 阻塞等待**下一条消息，避免空转烧 CPU。当有活可干时 `blocking=False`，就**非阻塞**地扫一遍队列（`while not empty(): get()`），拿到多少算多少，绝不干等——因为手头还有 batch 要 forward。

这是「忙轮询 + 空闲阻塞」的经典结合。

### 难点 2：三个 `_make_*` 的 index 计算（最烧脑的地方）

这是把「逻辑请求」翻译成「GPU 上一维张量」的核心。关键概念：GPU 上所有 token 平铺成一个池，`positions` / `input_mapping` 都是按「每个 req 贡献 `extend_len` 个」来展开的。

**`_make_positions`**：给每个要新算的 token 一个位置编号。

```python
needed_size = sum(r.extend_len for r in batch.padded_reqs)
for req in batch.padded_reqs:
    torch.arange(req.cached_len, req.device_len, out=indices_host[offset:offset+length])
    offset += length
```

每个 req 贡献 `[cached_len, device_len)` 这段位置（= 它要新算的那 `extend_len` 个 token 的位置），拼到一起就是本 batch 全部 token 的 RoPE 位置。

**`_make_input_tuple`**：返回 `(token_mapping, positions)`，`token_mapping` 每个元素是「这个 token 属于哪个 `table_idx`」。

```python
mapping_host[offset:offset+length].fill_(req.table_idx)
```

配合 `batch.input_ids = token_pool[input_mapping]`——`token_pool[table_idx]` 能取到该请求存在池里的 token，`input_mapping` 就是「第 i 个 token 去池里哪个槽位取」。

**`_make_write_tuple`**：返回 `(req_mapping, seq_lens)`，告诉采样结果往哪写。

```python
mapping_list = [req.table_idx for req in batch.reqs]           # 每个真实 req 的槽位
write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
```

`seq_lens = -1` 表示这个请求**不能 decode**（比如 chunked prefill 请求），新 token 不写回。否则写到 `token_pool[table_idx][device_len]`（下一个空位）。

一句话总结：**`input_mapping` 管「从哪读」，`write_mapping` 管「往哪写」，`positions` 管「算什么位置」。**

### 难点 3：`ChunkedReq` 是什么？为什么它在 `_process_last_data` 里被跳过？

`PrefillManager.schedule_next_batch` 里，如果一个请求 `extend_len` 超过 `prefill_budget`，就把它切成一个 `ChunkedReq`（[prefill.py](python/minisgl/scheduler/prefill.py)），只算前 `budget` 个 token，剩下的留到下轮。

`ChunkedReq` 重写了两个方法：

```python
class ChunkedReq(Req):
    def append_host(self, next_token): raise NotImplementedError("ChunkedReq should not be sampled")
    @property
    def can_decode(self): return False  # 避免被加进 decode manager
```

它是「算到一半的 prefill」，不采样、不进 decode。所以在 `_process_last_data` 里 `if isinstance(req, ChunkedReq): continue` 直接跳过，不生成 `DetokenizeMsg`。

### 难点 4：`finished_reqs` 集合是干嘛的？

这是 Step 9 的伏笔，但数据结构在这里定义：`self.finished_reqs: Set[Req] = set()`。

`_process_last_data` 里 `if finished and req not in self.finished_reqs:` 才做释放。原因是 **overlap 调度下同一个请求可能被处理两次**（上一轮的 last_data 和这一轮的 last_data 可能重叠），用这个集合去重，防止 `free` 两次。`Req` 之所以要 `eq=False`（Step 6 讲的），正是因为这里要用**对象身份**做 `set` 判重。

---

## 四、注意事项

1. **`receive_msg` 的 blocking 只在「完全空闲」时为 True**：只要 prefill 或 decode 有任一可算，就非阻塞，避免为了等新消息而让 GPU 空着。
2. **`_process_last_data(ongoing_data)` 里 `ongoing_data` 可能为 None**：本轮没凑出 batch 时，直接跳过结果处理（没有结果可处理）。
3. **`input_ids` 在 `_forward` 里才 `token_pool[input_mapping]` 取**：`_prepare_batch` 阶段 `batch.input_ids` 还是 `init=False` 的空字段，别在 prepare 里提前读它。
4. **`filter_reqs` 会重建 `running_reqs`**：`self.running_reqs = {req for req in ... if req.can_decode}`，把 `can_decode=False`（生成完/chunked）的请求从 decode 集合里剔出去。
5. **`_free_req_resources` 是「释放」的统一入口**：`table_manager.free(table_idx)`（还槽位）+ `cache_manager.cache_req(req, finished=True)`（还 KV 页），abort 和 finished 都走这里。

---

## 五、反思题

1. `normal_loop` 四步里，如果「收消息」这步收到的都是 `UserMsg`，这些请求是**本轮**就被 forward，还是**下一轮**？从代码里找出依据（提示：看 `add_one_req` 和 `_schedule_next_batch` 的先后）。
2. `_make_input_tuple` 返回的 `token_mapping` 和 `_make_write_tuple` 返回的 `req_mapping`，长度一样吗？分别等于多少？为什么一个按 `padded_reqs` 展开、一个按 `reqs` 展开？
3. 为什么 `blocking` 的判断条件是 `not (prefill.runnable or decode.runnable)`，而不是「是否有待处理消息」？两者有什么区别？
4. `ChunkedReq.can_decode` 返回 False 会连锁影响哪些地方？（提示：`filter_reqs`、`_make_write_tuple` 的 `-1`、`_process_last_data` 的 `continue`）
5. 如果 `_process_one_msg` 忘了处理 `BatchBackendMsg` 的拆包，直接当未知类型抛异常，会发生什么？

---

## 六、示意图

### 6.1 `normal_loop` 一轮的完整数据流

```
                    ┌─────────────────────────────────────────────┐
                    │            normal_loop（一轮）                 │
                    └─────────────────────────────────────────────┘
  tokenizer ──UserMsg──┐
                       ▼
  ① receive_msg ──► _process_one_msg ──► PrefillManager.pending_list
       (blocking 视空闲而定)                    │
                                               ▼
  ② _schedule_next_batch
       prefill 优先 ──► schedule_next_batch(budget) ──► Batch(reqs, "prefill")
         (或 decode) ──► schedule_next_batch()       ──► Batch(reqs, "decode")
                                               │
                                               ▼
       _prepare_batch：pad → 分页 → positions → input/write tuple → out_loc → metadata
                                               │
                                               ▼
  ③ _forward：token_pool[input_mapping] → engine.forward_batch → token_pool[write_mapping]
                                               │
                                               ▼
  ④ _process_last_data：copy_done.synchronize → append_host → 判 finished
                                               │
                                               ▼
                        send_result ──DetokenizeMsg──► detokenizer
```

### 6.2 `token_pool` 的读写映射

```
  token_pool（GPU 上一维大池，按 table_idx 分段）
  ┌──────────┬──────────┬──────────┬──────────┐
  │ table 0  │ table 1  │ table 2  │  ...     │
  └────▲─────┴────▲─────┴──────────┴──────────┘
       │读         │写
       │           │
  input_mapping   write_mapping
  (第 i 个 token   (第 j 个 req
   去哪个槽位取)    写到槽位 device_len 处)
       │
  batch.input_ids = token_pool[input_mapping]   ← _forward 里读
  token_pool[write_mapping] = next_tokens_gpu    ← _forward 里写
```

### 6.3 prefill 分块（chunk）示意

```
  一个长请求 input_len=10，prefill_budget=6
  ┌──────────────────────────────┐
  │ 0 1 2 3 4 5 6 7 8 9          │  ← 10 个输入 token
  └──────────────────────────────┘
  第 1 轮：算 [0,6) → ChunkedReq(cached=0, device=6)
  第 2 轮：算 [6,10) → Req(cached=6, device=10) → 之后进入 decode
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [scheduler/scheduler.py](python/minisgl/scheduler/scheduler.py) | 主循环 + 编排 | `Scheduler`、`normal_loop`、`_prepare_batch`、`_forward` |
| [scheduler/prefill.py](python/minisgl/scheduler/prefill.py) | prefill 调度 | `PrefillManager`、`ChunkedReq`、`PrefillAdder` |
| [scheduler/decode.py](python/minisgl/scheduler/decode.py) | decode 调度 | `DecodeManager` |
| [scheduler/io.py](python/minisgl/scheduler/io.py) | ZMQ 收发 | `receive_msg`、`send_result` |
| [scheduler/utils.py](python/minisgl/scheduler/utils.py) | 中间数据 | `PendingReq`、`ScheduleResult` |

**下一步**：进入 Step 9（Overlap 调度），看两条 CUDA stream 怎么把「处理上一轮结果」和「算这一轮 batch」叠起来，隐藏 CPU 延迟。
