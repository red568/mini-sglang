# Step 3：消息系统与轻量序列化

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 3。
> 核心文件：[message/backend.py](python/minisgl/message/backend.py)、[message/frontend.py](python/minisgl/message/frontend.py)、[message/tokenizer.py](python/minisgl/message/tokenizer.py)、[message/utils.py](python/minisgl/message/utils.py)、[utils/mp.py](python/minisgl/utils/mp.py)。
>
> 这一 Step 回答：**进程之间是靠什么「语言」对话的？一个 `torch.Tensor` 是怎么跨进程传过去的？**

---

## 一、这个 Step 要解决什么

进程间通信分两类：**控制消息**（谁要做什么）走 ZMQ，**张量数据**（模型权重、KV cache）走 NCCL/pynccl。本 Step 只讲前者：消息数据类长什么样、怎么被序列化成字节流、ZMQ 队列又是怎么封装的。

---

## 二、核心逻辑

### 2.1 三类消息 = 三段链路的协议

| 文件 | 消息类 | 在哪段链路之间传递 |
|---|---|---|
| [message/tokenizer.py](python/minisgl/message/tokenizer.py) | `TokenizeMsg` / `DetokenizeMsg` / `AbortMsg` | API Server ⇄ tokenizer/detokenizer |
| [message/backend.py](python/minisgl/message/backend.py) | `UserMsg` / `AbortBackendMsg` / `ExitMsg` | tokenizer → Scheduler |
| [message/frontend.py](python/minisgl/message/frontend.py) | `UserReply` | detokenizer → API Server |

每类都有 `Batch*Msg` 包装，用于一次传多条（减少 ZMQ 收发次数）。

### 2.2 序列化是「两段式」

看 [utils/mp.py](python/minisgl/utils/mp.py) 的 `ZmqPushQueue.put`：

```python
event = msgpack.packb(self.encoder(obj), use_bin_type=True)
self.socket.send(event, copy=False)
```

- 第一步 `self.encoder(obj)`：`encoder` 就是 `serialize_type`（见 [message/utils.py](python/minisgl/message/utils.py)），把消息对象转成**纯 dict**。
- 第二步 `msgpack.packb`：把这个 dict 打包成字节流。

`serialize_type` 的核心：给每个对象塞一个 `__type__` 标记（类名或 `"Tensor"`），然后递归处理字段：

- 普通类型（int/float/str/bool/None/bytes）直接返回；
- dict/list/tuple 递归；
- `torch.Tensor` **只支持 1D**，转成 `numpy().tobytes()` 存进 `buffer` 字段，`dtype` 用字符串存；
- 其他 dataclass 递归调用 `serialize_type`。

反序列化 `deserialize_type` 反向做：根据 `__type__` 找到类，`Tensor` 则用 `np.frombuffer` + `.copy()` 重建张量。

### 2.3 ZMQ 队列封装

[utils/mp.py](python/minisgl/utils/mp.py) 封装了 6 个队列类，统一了「bind/connect + 序列化」：

```
ZmqPushQueue / ZmqPullQueue   —— 同步 PUSH/PULL（tokenizer、scheduler 用）
ZmqPubQueue  / ZmqSubQueue    —— 同步 PUB/SUB（rank0 广播给其他 rank）
ZmqAsyncPushQueue / ZmqAsyncPullQueue —— 异步版本（API Server 用，配合 asyncio）
```

`create=True` → `bind`（服务端，谁先建谁 bind）；`create=False` → `connect`（客户端）。

---

## 三、难点解析

### 难点 1：为什么不直接用 pickle 或 msgpack 传对象？

- **msgpack 不认识 `torch.Tensor` 和自定义 dataclass**，直接 pack 会报错。
- **pickle 慢、且反序列化任意对象不安全**（执行任意代码），而且 pickle 的字节流依赖 Python 版本/类定义。
- 所以项目采用「自定义 `serialize_type` 转成纯 dict + msgpack 打包」的两段式：**msgpack 负责高效二进制打包，`serialize_type` 负责把 Python 对象降维成 msgpack 认识的类型**。

### 难点 2：`__type__` 标记 + `globals()` 类映射

`serialize_type` 给每个对象写 `__type__` 记录类名；反序列化时 `deserialize_type(globals(), json)` 用 `globals()` 作为「类名字符串 → 类对象」的映射表。

所以 `BaseBackendMsg.decoder` 里传的是 `globals()`——意味着**只有 `backend.py` 这个模块里定义的消息类能被反序列化**。这也是为什么三类消息要分三个文件、各自维护自己的 `globals()`。

### 难点 3：Tensor 为什么只支持 1D？

序列化用 `numpy().tobytes()`，虽然多维也技术上可行，但这里的消息里的 tensor 都是 `input_ids`（1D token 序列），为了简单明确就限制成 1D，`assert self.dim() == 1` 直接拦下误用。

反序列化时 `np.frombuffer` 得到的是**共享内存的 view**，所以必须 `.copy()` 成独立张量，否则 `buffer` 字节被回收后 tensor 会失效。

### 难点 4：bind 与 connect 的语义

- `bind`：谁「拥有」这个端点，必须先于 connect 就位。
- `connect`：谁「去连」它，ZMQ 里 connect 到一个还没 bind 的端点**不会报错**，消息会静默排队等待。

这也是 Step 1 里为什么「主进程先 bind、再 spawn 子进程去 connect」的原因。

---

## 四、注意事项

1. **消息类是 dataclass，字段名和类型必须和 `serialize_type` 能处理的类型匹配**。加字段时别用自定义对象（除非也实现序列化）。
2. **`globals()` 作为类映射的局限**：跨模块的消息类不能互相反序列化，新增消息类要加在对应文件的 `globals()` 可见位置。
3. **`copy=False`** 表示 ZMQ 直接发送缓冲区，省一次拷贝，但要求 `msgpack.packb` 产生的 bytes 在 `send` 期间不被回收。
4. **`raw=False`**（`msgpack.unpackb(event, raw=False)`）让字符串反序列化成 `str` 而不是 `bytes`，否则中文会出问题。
5. **异步队列和同步队列不能混用**：API Server 是 `asyncio` 环境，必须用 `ZmqAsync*`，否则阻塞事件循环。

---

## 五、反思题

1. 如果要支持序列化 2D 的 `torch.Tensor`，`serialize_type` 需要怎么改？（提示：除了 buffer 还要存 shape）
2. `np.frombuffer` 返回的是 view，为什么必须 `.copy()`？不 copy 会有什么 bug？
3. 为什么 `deserialize_type` 的类映射用 `globals()` 而不是显式传一个 dict？这有什么隐患？
4. 三类消息为什么分成三个文件？如果合并成一个文件会有什么影响？
5. `Batch*Msg` 包装的意义是什么？如果每次都单条单条传，性能上会怎样？

---

## 六、示意图

### 6.1 序列化与反序列化全流程

```
发送方进程                              接收方进程
┌───────────────────┐                  ┌───────────────────┐
│ 消息对象 (dataclass)│                  │ 消息对象 (dataclass)│
│ 含 torch.Tensor    │                  │ 含 torch.Tensor    │
└─────────┬─────────┘                  └─────────▲─────────┘
          │ serialize_type                        │ deserialize_type
          ▼ (纯 dict)                             │ (从 dict 重建)
┌───────────────────┐                  ┌───────────────────┐
│ {"__type__":"User│                  │ {"__type__":"User│
│  "uid":3,        │                  │  "uid":3,        │
│  "input_ids":     │                  │  "input_ids":     │
│   {"__type__":    │                  │   {"__type__":    │
│    "Tensor",      │                  │    "Tensor",      │
│    "buffer":bytes │                  │    "buffer":bytes │
│    "dtype":"int32"│                  │    "dtype":"int32"│
│  }                │                  │  }                │
└─────────┬─────────┘                  └─────────▲─────────┘
          │ msgpack.packb                        │ msgpack.unpackb
          ▼ (bytes)                             │ (bytes)
┌───────────────────┐   ──── ZMQ 传输 ────►   ┌───────────────────┐
│  字节流 (bytes)     │                        │  字节流 (bytes)     │
└───────────────────┘                        └───────────────────┘
```

### 6.2 消息在链路里的流动

```
 API Server ──TokenizeMsg──► tokenizer ──UserMsg──► Scheduler
     ▲                          │                    │
     │                          │ (detokenize 时)     │
     └────UserReply───────── detokenizer ◄──DetokenizeMsg──┘
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [message/backend.py](python/minisgl/message/backend.py) | tokenizer→scheduler 消息 | `UserMsg`、`AbortBackendMsg`、`ExitMsg` |
| [message/tokenizer.py](python/minisgl/message/tokenizer.py) | 前后端消息 | `TokenizeMsg`、`DetokenizeMsg`、`AbortMsg` |
| [message/frontend.py](python/minisgl/message/frontend.py) | detokenizer→API Server 消息 | `UserReply` |
| [message/utils.py](python/minisgl/message/utils.py) | 序列化/反序列化 | `serialize_type`、`deserialize_type` |
| [utils/mp.py](python/minisgl/utils/mp.py) | ZMQ 队列封装 | `ZmqPushQueue`、`ZmqPubQueue` 等 |

**下一步**：进入 Step 4（API Server 前端），看这些消息在异步的 FastAPI 里怎么被生产和消费。
