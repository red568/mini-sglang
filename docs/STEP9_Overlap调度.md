# Step 9：Overlap 调度（`overlap_loop` + 双 CUDA stream）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 9。
> 核心文件：[scheduler/scheduler.py](python/minisgl/scheduler/scheduler.py) 的 `overlap_loop`、`_process_last_data`、`run_forever`。
>
> 这一 Step 回答：**怎么让「处理上一轮结果」的 CPU 活，和「算这一轮 batch」的 GPU 活，重叠起来，把 CPU 延迟藏掉？**

---

## 一、这个 Step 要解决什么

Step 8 的 `normal_loop` 是「串行」的：先 forward，再处理结果，下一轮再 forward……问题在于 `_process_last_data` 里有大量 **CPU 活**（`append_host` 拼 token、判 finished、组 `DetokenizeMsg`、发 ZMQ），这些活干的时候 GPU 是闲着的。

`overlap_loop` 的解法是：**把「处理上一轮结果」挪到「这一轮 forward 之后、下一轮 forward 之前」的空档里做**，让 CPU 处理结果和 GPU 算 logits 在时间上重叠。

---

## 二、核心逻辑

### 2.1 两条 CUDA stream

这是整个 Step 的地基，先看 `Scheduler.__init__` 里已经埋下的两条 stream：

| stream | 谁拥有 | 干什么 |
|---|---|---|
| `self.stream`（scheduler 的） | Scheduler | CPU 侧准备元数据（`_make_positions`、H2D 拷贝 `positions`/`mapping`） |
| `engine.stream`（engine 的） | Engine | 真正跑模型 forward（`model.forward` / `graph_runner.replay`） |

`torch.cuda.set_stream(self.stream)` 让 scheduler 默认在当前进程的 `self.stream` 上做 CPU 侧的准备工作；`engine_stream_ctx = torch.cuda.stream(self.engine.stream)` 是一个上下文管理器，`with` 进去就切到 engine 的 stream 上跑计算。

### 2.2 `run_forever`：两种模式的入口

```python
@torch.inference_mode()
def run_forever(self) -> NoReturn:
    if ENV.DISABLE_OVERLAP_SCHEDULING:
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            while True:
                self.normal_loop()          # 关掉 overlap：纯串行
    else:
        assert torch.cuda.current_stream() == self.stream
        data = None
        while True:
            data = self.overlap_loop(data)  # 默认：overlap 模式
```

- `DISABLE_OVERLAP_SCHEDULING`（[env.py](python/minisgl/env.py)，默认 False）是**消融开关**：设为 True 就退回 Step 8 的串行模式，方便对比性能差异、排查 overlap 引入的 bug。
- 默认走 overlap：`overlap_loop(data)` 每轮**吃上一轮的 `data`、吐这一轮的 `data`**，形成一个「数据在轮与轮之间接力」的闭环。

### 2.3 `overlap_loop`：关键的一步挪移

对比两个 loop（只差最后一步的顺序）：

```python
def normal_loop(self):                       # 串行版
    ...
    forward_input = self._schedule_next_batch()
    if forward_input is not None:
        ongoing_data = (forward_input, self._forward(forward_input))   # ③ 算这一轮
    self._process_last_data(ongoing_data)                             # ④ 处理这一轮结果

def overlap_loop(self, last_data):           # overlap 版
    blocking = not (last_data is not None
                    or self.prefill_manager.runnable
                    or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    forward_input = self._schedule_next_batch()
    ongoing_data = None
    if forward_input is not None:
        with self.engine_stream_ctx:                     # 切到 engine 的 stream
            self.engine.stream.wait_stream(self.stream)  # 等 scheduler stream 的元数据就绪
            ongoing_data = (forward_input, self._forward(forward_input))  # ③ 算这一轮（GPU）

    self._process_last_data(last_data)   # ④ 处理【上一轮】的结果（CPU）
    return ongoing_data
```

**核心区别**：`overlap_loop` 处理的是 `last_data`（**上一轮**的 batch 结果），而不是 `ongoing_data`（这一轮的）。顺序变成：

```
 算这一轮（GPU 上跑 forward）──►  处理上一轮（CPU 上处理结果）
```

于是当 GPU 还在算这一轮 batch 时，CPU 可以同时去处理上一轮的结果——两者重叠，CPU 延迟被 GPU 计算「盖住」。

### 2.4 `engine.stream.wait_stream(self.stream)` 在等什么？

`scheduler.stream` 上的 `_make_positions` / `_make_input_tuple` 等做了 H2D 拷贝（`indices_host.to(device, non_blocking=True)`）。这些拷贝是**异步**的，如果 engine.stream 立刻开始 forward 读这些 tensor，可能读到还没拷完的数据。

`wait_stream(self.stream)` 让 engine.stream **等到 scheduler.stream 的所有已提交操作完成**才开始执行，保证 `positions`/`input_mapping`/`out_loc` 已经就绪。这是双 stream 协同的经典同步点。

### 2.5 `_process_last_data`：处理上一轮结果（含双重 free 防护）

```python
def _process_last_data(self, last_data: ForwardData | None) -> None:
    if last_data is None:
        return

    batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
    copy_done.synchronize()                    # ① 等 GPU→CPU 的 token 拷贝完成
    reply: List[DetokenizeMsg] = []
    new_finished_reqs: Set[Req] = set()
    with self.cache_manager.lazy_free_region():
        for i, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                continue                        # ② chunked prefill 不采样，跳过
            next_token = next_tokens_cpu[i]
            req.append_host(next_token.unsqueeze(0))     # ③ 把新 token 拼到 CPU 侧 input_ids
            next_token = int(next_token.item())
            finished = not req.can_decode                 # ④ 生成满 max_tokens
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id  # ⑤ 命中 EOS
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            if finished and req not in self.finished_reqs:   # ⑥ 双重 free 防护
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                new_finished_reqs.add(req)
            elif batch.is_prefill:                           # ⑦ 非 chunk prefill：缓存前缀
                self.cache_manager.cache_req(req, finished=False)

    self.finished_reqs = new_finished_reqs
    self.send_result(reply)
```

逐段理解：

- **① `copy_done.synchronize()`**：`next_tokens_cpu` 是 `next_tokens_gpu.to("cpu", non_blocking=True)` 的异步拷贝结果。`copy_done` 是一个记录在 engine.stream 上的 `torch.cuda.Event`，`synchronize()` 阻塞直到这个拷贝真正完成，才能安全读 CPU 上的 token。
- **④⑤ 判 finished**：`can_decode`（= `remain_len > 0`）为 False 表示吐够了；`next_token == eos_token_id` 表示命中结束符。两者任一成立即 finished。
- **⑥ `finished_reqs` 双重 free 防护**：overlap 模式下，一个请求可能在「上一轮的 last_data」里被标 finished，同时又在别处被处理，导致 `free` 被调用两次。`req not in self.finished_reqs` 保证同一轮内只 free 一次。这里依赖 `Req` 的 `eq=False`（对象身份判等，Step 6）。
- **⑦ prefill 缓存前缀**：非 chunk 的 prefill 完成后，把它的前缀写进 Radix Cache，供后续请求复用（Step 20 展开）。

---

## 三、难点解析

### 难点 1：为什么 overlap 能「藏」CPU 延迟？

画一张时间轴对比：

```
【串行 normal_loop】
  时间 →
  |── forward A ──|── 处理 A(CPU) ──|── forward B ──|── 处理 B(CPU) ──|
                              ▲
                      这段 CPU 活里 GPU 闲着

【overlap_loop】
  |── forward A ──|── forward B ──|── forward C ──|
        |── 处理 A(CPU) ──|── 处理 B(CPU) ──|── 处理 C(CPU) ──|
                              ▲
                      CPU 处理 和 GPU forward 重叠了
```

关键前提：`_process_last_data(last_data)` 处理的是**上一轮**结果，而上一轮的 `next_tokens_cpu` 已经在上一轮 forward 里发起了异步拷贝。等轮到这一轮处理它时，`copy_done.synchronize()` 大概率已经完成（不用真等），CPU 活就「顺手」在 GPU 算下一轮时做完了。

### 难点 2：`copy_done.synchronize()` 会不会又把重叠抵消掉？

会部分抵消——如果 GPU 太快、CPU 太慢，`synchronize()` 仍会阻塞等拷贝。但 overlap 的目标是「在拷贝和 forward 的**大部分**时间上重叠」，而不是完全消除等待。真实的收益来自：`append_host` / 判 finished / 组 ZMQ 消息这些**纯 CPU 活**（不依赖 GPU）不再独占时间片，而是和下一轮 forward 并行。

### 难点 3：`blocking` 判断为什么多了 `last_data is not None`？

```python
blocking = not (last_data is not None
                or self.prefill_manager.runnable
                or self.decode_manager.runnable)
```

相比 Step 8 的 `normal_loop`，这里多了一个 `last_data is not None`。原因：**只要上一轮还有结果待处理**（`last_data` 非空），这轮就绝不能阻塞等消息——因为处理 `last_data` 本身就是一件必须立刻做的活。只有「既没有结果要处理、又没有请求要算」的真空闲状态，才允许阻塞。

### 难点 4：`_forward` 里的 `assert torch.cuda.current_stream() == self.stream`

[engine.py](python/minisgl/engine/engine.py) 的 `forward_batch` 第一行：

```python
assert torch.cuda.current_stream() == self.stream   # self.stream 是 engine 的 stream
```

这解释了为什么 `overlap_loop` 要 `with self.engine_stream_ctx:` 包住 `_forward`——因为 `forward_batch` 断言「当前必须在 engine 的 stream 上」。串行模式 `run_forever` 里用 `with self.engine_stream_ctx:` 包住整个 `while True` 也是同理。这个 assert 是双 stream 设计里防「在错误 stream 上算」的护栏。

---

## 四、注意事项

1. **`last_data` 接力必须精确**：`overlap_loop` 返回的 `ongoing_data` 会作为**下一轮**的 `last_data`。如果某轮没凑出 batch（`ongoing_data=None`），下一轮的 `last_data` 就是 None，`_process_last_data` 直接跳过——所以「这一轮没算」对应「下一轮没结果可处理」，时序要对齐。
2. **`finished_reqs` 每轮重建**：`self.finished_reqs = new_finished_reqs` 是整集合替换，不是累加。这样防 free 的「黑名单」只在本轮 `_process_last_data` 内有效。
3. **`ChunkedReq` 在 `_process_last_data` 里 `continue`**：chunk 请求不算完、不采样、不生成 reply，直接跳过，但它的资源要等后续 chunk 完成后才释放。
4. **`lazy_free_region()` 是 contextmanager**：把多个 `free` 合并成一次延迟释放，减少碎片化（Radix Cache 相关，Step 20 展开）。
5. **消融验证**：怀疑 overlap 有 bug 时，设 `DISABLE_OVERLAP_SCHEDULING=1` 跑串行版对照，是最快的定位手段。

---

## 五、反思题

1. 把 `overlap_loop` 里的 `self._process_last_data(last_data)` 改成 `self._process_last_data(ongoing_data)`，会发生什么？（提示：CPU 处理和 GPU forward 还重叠吗？结果会被正确处理吗？）
2. 如果删掉 `engine.stream.wait_stream(self.stream)` 这一行，可能出什么 bug？在什么情况下能观察到？（提示：异步 H2D 拷贝 + 数据竞争）
3. `copy_done.synchronize()` 和 `engine.stream.wait_stream(self.stream)` 分别等的是什么？为什么两个都需要？
4. `finished_reqs` 为什么是「每轮重建」而不是「一直累加」？如果改成累加（`self.finished_reqs |= new_finished_reqs`）会有什么问题？
5. overlap 模式一定比串行快吗？什么时候它反而可能更慢？（提示：CPU 活很少、GPU 本来就不忙时）

---

## 六、示意图

### 6.1 双 stream 的协作时序

```
  scheduler.stream（CPU 准备元数据）
  ── _make_positions ── H2D 拷贝 positions/mapping ──►
        │
        │ wait_stream（engine 等 scheduler 就绪）
        ▼
  engine.stream（GPU 算 logits）
  ── model.forward / graph.replay ──► next_tokens_gpu ── 异步 to("cpu") ──►
                                        │
                                        │ copy_done.record(engine.stream)
                                        ▼
  CPU 读 next_tokens_cpu ── copy_done.synchronize() 保证拷贝完成
```

### 6.2 overlap 模式的时间重叠

```
  时间 ──────────────────────────────────────────────►

  GPU（engine.stream）:
  |─ forward batch#1 ─|─ forward batch#2 ─|─ forward batch#3 ─|

  CPU（scheduler）:
     |─ 处理 batch#0 结果 ─|─ 处理 batch#1 结果 ─|─ 处理 batch#2 结果 ─|
                          ▲
                    CPU 活被 GPU forward 盖住，吞吐更高
```

### 6.3 `last_data` 的接力闭环

```
  run_forever:
      data = None
      while True:
          data = overlap_loop(data)   ← 吃上一轮 data，吐这一轮 data

  轮 0: overlap_loop(None)      → 处理 None（跳过）→ 返回 data#1
  轮 1: overlap_loop(data#1)    → 处理 batch#1 结果 → 返回 data#2
  轮 2: overlap_loop(data#2)    → 处理 batch#2 结果 → 返回 data#3
  ...
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [scheduler/scheduler.py](python/minisgl/scheduler/scheduler.py) | overlap 调度 + 结果处理 | `overlap_loop`、`_process_last_data`、`run_forever` |
| [engine/engine.py](python/minisgl/engine/engine.py) | 前向 + 异步拷贝事件 | `forward_batch`、`copy_done_event` |
| [env.py](python/minisgl/env.py) | 消融开关 | `DISABLE_OVERLAP_SCHEDULING` |

**下一步**：进入 Step 10（Engine 与 Page Table），看 `Engine` 内部怎么组织模型、页表、KV cache 和 CUDA graph。
