# Step 4：API Server 前端（异步 + 流式）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 4。
> 核心文件：[server/api_server.py](python/minisgl/server/api_server.py)。
>
> 这一 Step 回答：**HTTP 请求进来后，FastAPI 是怎么把它变成 `TokenizeMsg` 发出去、又是怎么把生成结果流式吐回给用户的？**

---

## 一、这个 Step 要解决什么

API Server 是整个系统的「门面」，跑在主进程里，用 `asyncio` + FastAPI + 异步 ZMQ。它要做两件异步的事：

1. **去路**：收 HTTP 请求 → 分配 `uid` → 发 `TokenizeMsg`。
2. **回路**：收 `UserReply` → 按 `uid` 分发给等待中的 HTTP 连接 → 流式返回。

---

## 二、核心逻辑

### 2.1 `FrontendManager`：前端的总管家

```python
@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]   # 去 tokenizer
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]    # 从 detokenizer
    uid_counter: int = 0
    ack_map: Dict[int, List[UserReply]] = ...             # uid → 攒到的回复
    event_map: Dict[int, asyncio.Event] = ...             # uid → 唤醒事件
```

- `new_user()`：分配自增 `uid`，初始化 `ack_map[uid] = []` 和 `event_map[uid] = asyncio.Event()`。
- `listen()`：一个**后台常驻任务**，死循环从 `recv_tokenizer` 拉 `UserReply`，塞进 `ack_map[uid]`，并 `event_map[uid].set()` 唤醒等待者。
- `wait_for_ack(uid)`：一个 async 生成器，`await event.wait()` 等被唤醒，然后逐个 `yield` 攒到的回复，直到收到 `finished=True`。

### 2.2 两个核心端点

- `/v1/chat/completions`：OpenAI 兼容。`req.stream` 为真走 `stream_chat_completions`（SSE），为假则 `async for` 收完拼成完整 JSON。
- `/generate`：更简单的文本接口。
- `/v1/models`：返回模型卡片。

### 2.3 SSE 流式响应

`stream_chat_completions` 逐 chunk 拼出 OpenAI 格式：

```python
chunk = {"id": f"cmpl-{uid}", "object": "text_completion.chunk",
         "choices": [{"delta": delta, "index": 0, "finish_reason": None}]}
yield f"data: {json.dumps(chunk)}\n\n".encode()
```

最后一个 chunk 带 `finish_reason: "stop"`，再 `yield b"data: [DONE]\n\n"`。

### 2.4 客户端断开 → abort

`stream_with_cancellation` 包裹生成器，每次 yield 前检查 `await request.is_disconnected()`，断了就 `raise asyncio.CancelledError`，然后在 `except` 里 `asyncio.create_task(self.abort_user(uid))` 异步发 `AbortMsg` 下去。

---

## 三、难点解析

### 难点 1：异步 HTTP 与「按 uid 分发」的桥接

HTTP 连接是「一个请求一个协程」，而 `UserReply` 是从**一个** ZMQ 通道流进来的、混合了所有请求的结果。怎么把混在一起的回复，精准送回各自等待的协程？

答案是 **`uid` 作为 key 的 `ack_map` + `event_map`**：

- `listen()` 是唯一从 ZMQ 拉消息的地方，按 `uid` 把回复分到各自的 `ack_map`。
- 每个请求的协程 `wait_for_ack(uid)` 用 `asyncio.Event` 等待自己的回复，`event.set()` 就是「有你的一条回复了」的信号。

这是典型的「单一消费者 + 按 key 分发」的异步模式。

### 难点 2：`wait_for_ack` 里 `event.wait()` / `event.clear()` 的配合

```python
async def wait_for_ack(self, uid: int):
    event = self.event_map[uid]
    while True:
        await event.wait()      # 等有新回复
        event.clear()           # 清掉，准备下次等
        pending = self.ack_map[uid]
        self.ack_map[uid] = []  # 取走攒到的回复
        for ack in pending:
            yield ack
        if ack and ack.finished:
            break
    del self.ack_map[uid]; del self.event_map[uid]
```

`clear()` 必须在 `wait()` 返回后立刻做，否则下次 `wait()` 会「瞬间通过」（事件还处于 set 状态）。这是一个容易写错的竞态细节。

### 难点 3：`listen()` 为什么只 `_create_listener_once` 一次

`listen()` 是个死循环后台任务，必须全局只启动一个（否则多个消费者抢同一个 ZMQ 通道，回复会被随机分走）。所以用 `initialized` 标志保证只 `asyncio.create_task` 一次。

### 难点 4：abort 为什么要 `asyncio.sleep(0.1)`

`abort_user` 先睡 0.1 秒再发 `AbortMsg`。原因是客户端断开后，可能还有已经在途的 `UserReply` 在飞，直接删 `ack_map` 可能漏收或冲突。睡一小段时间给在途消息「落地」的机会，再清理状态、发 abort。这是个务实的经验值。

---

## 四、注意事项

1. **非流式 `stream=false` 是「先攒后发」**：`full_content` 累加所有 `incremental_output`，最后一次性返回完整 JSON，所以用户要等生成全部结束才能拿到结果。
2. **`stream_with_cancellation` 里的 `raise asyncio.CancelledError` 是刻意的**：用异常来中断生成器，并触发 abort 清理。
3. **shell 模式不走 uvicorn**：`run_api_server` 里 `if run_shell: asyncio.run(shell())`，直接在内嵌协程里跑交互，且 shell 结束时会用 `psutil` 杀掉所有子进程。
4. **`uid_counter` 从 0 单调递增**，不回收，所以 `uid` 在整个进程生命周期内唯一。
5. **`--dummy-weight` 时 shell 模式会 assert**：`assert not config.use_dummy_weight`。

---

## 五、反思题

1. 如果 `listen()` 被启动两次，会发生什么？`_create_listener_once` 解决了什么问题？
2. `wait_for_ack` 里 `event.clear()` 为什么必须紧跟 `event.wait()` 之后？如果删掉会怎样？
3. `stream=false` 时为什么不用 `stream_generate` 而是直接 `async for ack in wait_for_ack`？两者差别在哪？
4. `abort_user` 为什么用 `asyncio.create_task` 而不是直接 `await`？（提示：在异常处理里，不能被二次异常打断）
5. `ack_map` 和 `event_map` 为什么是两个 dict 而不是合并成一个结构？

---

## 六、示意图

### 6.1 异步分发的数据流

```
          HTTP 连接（每个请求一个协程）
   req1 ◄──── wait_for_ack(uid=1) ◄──┐
   req2 ◄──── wait_for_ack(uid=2) ◄──┤ 各等各的 event
   req3 ◄──── wait_for_ack(uid=3) ◄──┤
                                     │
                                     │
   ZMQ ──► listen()（唯一消费者）─────┘
                 │
                 │ 按 uid 分拣
                 ▼
        ack_map: {1:[...], 2:[...], 3:[...]}
        event_map: {1:Event, 2:Event, 3:Event}
                 │
                 │ event.set() 唤醒对应协程
                 ▼
         wait_for_ack 逐个 yield 给 HTTP
```

### 6.2 请求往返时序（单请求）

```
 用户        API Server            tokenizer/detokenizer
  │  ① POST    │                        │
  ├───────────►│ ② new_user() → uid=0   │
  │            │ ③ TokenizeMsg(uid=0)   │
  │            ├───────────────────────►│
  │            │   （用户等着，SSE 保持连接）│
  │            │   ④ UserReply(uid=0)   │
  │            │◄───────────────────────┤
  │            │ ⑤ event.set() 唤醒      │
  │ ⑥ SSE chunk│                        │
  │◄───────────┤                        │
  │   ...循环 ④⑤⑥ 直到 finished...       │
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [server/api_server.py](python/minisgl/server/api_server.py) | FastAPI 前端 + 异步分发 | `FrontendManager`、`v1_completions`、`wait_for_ack` |

**下一步**：进入 Step 5（Tokenize 与 Detokenize），看消息到 tokenizer 进程后，文本和 token 是怎么互相转换的。
