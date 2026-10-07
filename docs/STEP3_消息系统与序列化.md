# Step 3：消息系统与轻量序列化

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 3。
> 核心文件：[message/backend.py](python/minisgl/message/backend.py)、[message/frontend.py](python/minisgl/message/frontend.py)、[message/tokenizer.py](python/minisgl/message/tokenizer.py)、[message/utils.py](python/minisgl/message/utils.py)、[utils/mp.py](python/minisgl/utils/mp.py)。
>
> 这一 Step 回答：**进程之间是靠什么「语言」对话的？一个 `torch.Tensor` 是怎么跨进程传过去的？**

> **本文阅读顺序**：从整体图景（第一节）→ 三个核心概念（第二节）→ 逐个展开实现细节（第三～七节）→ 如何应用（第八节）→ 难点与注意事项（第九～十节）。先建立心智模型，再钻代码，最后落到怎么用。

---

## 一、整体图景：进程之间靠什么对话

一个推理服务由多个进程组成（API Server、tokenizer、Scheduler、detokenizer），它们不共享内存，只能靠**发消息**协作。进程间通信分两类：

| 通信 | 传什么 | 走什么 | 本 Step 是否覆盖 |
|---|---|---|---|
| 控制消息 | 「谁要做什么」：tokenize / 生成 / 中止 | ZMQ | **是（本 Step 主题）** |
| 张量数据 | 模型权重、KV cache 等大块数据 | NCCL / pynccl | 否（见 Step 2） |

本 Step 只讲前者，回答三个递进的问题：

1. **消息长什么样** → 一个 `@dataclass` 对象（[第三节](#三消息的具体形态三类消息--三段链路协议)）。
2. **怎么变成字节、再变回来** → 两段式序列化（[第四节](#四序列化的实现对象怎么变成字节)）。
3. **怎么在进程间收发** → ZMQ 队列封装（[第五～七节](#五zmq-队列封装)）。

先记住这张全局拓扑图，后面每个细节都能挂回这张图上：

```
 API Server ──(tokenize)──► tokenizer ──(backend)──► Scheduler(rank0..N-1)
     ▲                          │                        │
     │                    (detokenize)              (detokenize 结果)
     └────(frontend)────── detokenizer ◄──────────────┘
```

把结论先说清楚（后面逐步展开）：

- 三类消息（`BaseTokenizerMsg` / `BaseBackendMsg` / `BaseFrontendMsg`）分别对应三段链路上的协议。
- 序列化是「自定义 `serialize_type` 降维成纯 dict → `msgpack.packb` 打包」的**两段式**。
- ZMQ 队列在 [utils/mp.py](python/minisgl/utils/mp.py) 里统一封装成 6 个类，屏蔽了 `bind/connect`、编解码、同步/异步的差异。

---

## 二、三个核心概念（先建立心智模型）

在深入代码之前，用最少的篇幅把三件事钉牢。整个 Step 其实是这三个概念的展开，其余都是它们的细节。

### 2.1 消息 = 一个 `@dataclass` 对象

进程之间传递的「一句话」就是一个 Python `@dataclass` 对象，**字段就是协议**。比如「把这段文字 tokenize 一下」这条消息，字段是 `uid`（哪个请求）、`text`（文字内容）、`sampling_params`（采样参数）。

选 dataclass 的原因：它能被自动遍历字段（`__dict__`），这让后面的序列化/反序列化**全自动**——不需要为每条消息手写 encode/decode。

### 2.2 队列 = 一条 ZMQ 管道

消息靠 **ZMQ 队列**在进程间搬运，队列是「一端 push 进、另一端 pull 出」的管道（PUSH/PULL 模式）。你只需要知道两点：

- **bind vs connect**：`bind` = 「我拥有这个端点」，`connect` = 「我去连它」。约定接收端先 bind，发送端后 connect。
- **同步 vs 异步**：跑在独立进程死循环里的用同步队列；跑在 `asyncio` 事件循环里的用异步队列（`ZmqAsync*`），否则会冻住事件循环。

### 2.3 序列化 = 对象 ⇄ 字节（两段式）

ZMQ 只能搬**字节流**，不能直接传 Python 对象。所以发送前把对象「降维」成字节，接收后再「重建」回对象：

```
消息对象 ──serialize_type──► 纯 dict ──msgpack.packb──► 字节流
                                                          │ ZMQ 传输
消息对象 ◄─deserialize_type── 纯 dict ◄─msgpack.unpackb──┘
```

- 第一段 `serialize_type`：把对象（含 `torch.Tensor`、嵌套 dataclass）转成**纯 dict**（`int/float/str/bool/None/bytes/list/dict`），因为 msgpack 只认这些类型。
- 第二段 `msgpack.packb`：把这个 dict 打包成二进制。

这就是「轻量序列化」的全部：**不用 pickle（慢且不安全），而是自定义降维函数 + msgpack 打包**。

> 一句话总结：**消息（dataclass）经序列化（两段式）变成字节，在队列（ZMQ）里传输**。下面几节逐一展开成具体代码。

---

## 三、消息的具体形态：三类消息 = 三段链路协议

（进程拓扑见[第一节](#一整体图景进程之间靠什么对话)的图。）

每一段链路上跑一种消息，各自定义在一个文件里：

| 文件 | 基类 | 消息类 | 在哪段链路之间传递 | 字段 |
|---|---|---|---|---|
| [message/tokenizer.py](python/minisgl/message/tokenizer.py) | `BaseTokenizerMsg` | `TokenizeMsg` | API Server → tokenizer | `uid`、`text`、`sampling_params` |
| | | `DetokenizeMsg` | Scheduler → detokenizer | `uid`、`next_token`、`finished` |
| | | `AbortMsg` | API Server → tokenizer（中止请求） | `uid` |
| [message/backend.py](python/minisgl/message/backend.py) | `BaseBackendMsg` | `UserMsg` | tokenizer → Scheduler | `uid`、`input_ids`、`sampling_params` |
| | | `AbortBackendMsg` | tokenizer → Scheduler（转发中止） | `uid` |
| | | `ExitMsg` | tokenizer → Scheduler（退出信号） | （无字段） |
| [message/frontend.py](python/minisgl/message/frontend.py) | `BaseFrontendMsg` | `UserReply` | detokenizer → API Server | `uid`、`incremental_output`、`finished` |

**要点拆解：**

1. **消息本质是 `@dataclass`**。字段就是协议。比如一次 tokenize 请求就是 `TokenizeMsg(uid, text, sampling_params)`，tokenizer 处理后回一条 `UserMsg(uid, input_ids, sampling_params)` 给调度器。`sampling_params` 是 [core.py](python/minisgl/core.py) 里的 `SamplingParams`（`temperature/top_k/top_p/ignore_eos/max_tokens`），因为也是 dataclass，所以能被递归序列化。

2. **每个基类自带 `encoder` / `decoder`**：

   ```python
   # backend.py
   @dataclass
   class BaseBackendMsg:
       def encoder(self) -> Dict:          # 实例方法，self 就是消息本身
           return serialize_type(self)

       @staticmethod
       def decoder(json: Dict) -> BaseBackendMsg:
           return deserialize_type(globals(), json)
   ```

   - `encoder` 负责「对象 → dict」，`decoder` 负责「dict → 对象」。
   - 注意一个**不一致的小细节**：`BaseBackendMsg.encoder` 是实例方法（`def encoder(self)`），而 `BaseTokenizerMsg.encoder` / `BaseFrontendMsg.encoder` 是 `@staticmethod def encoder(msg)`。两者在调用点都写作 `BaseXxxMsg.encoder`，传给队列后以 `self.encoder(obj)` 调用，所以都能工作——但你照抄时要留意别把静态方法写岔。

3. **为什么拆成三个文件？** 见 [难点 2](#难点-2type-标记--globals-类映射) 和 [反思题 4](#十一反思题)。一句话：每个文件的 `decoder` 用自己模块的 `globals()` 当「类名 → 类对象」映射，所以**只有本文件里定义的消息类能被反序列化**。拆开天然做了隔离。

4. **`Batch*Msg` 批量包装**：每个文件还有一个 `BatchXxxMsg(data: List[BaseXxxMsg])`。收到一次可以传多条，减少 ZMQ 收发次数。接收端统一用 `_unwrap_msg` 拆开：

   ```python
   def _unwrap_msg(msg):
       if isinstance(msg, BatchTokenizerMsg):
           return msg.data
       return [msg]
   ```

   发送端也有对应的「单条就不包 Batch」的小优化（见 [tokenizer/server.py](python/minisgl/tokenizer/server.py) 里 `if len(batch_output.data) == 1: batch_output = batch_output.data[0]`）。

### 消息流向与 bind/connect

三个基类各走一段链路，下面是共享模式（默认 `--num-tokenizer 0`）下的完整流向。`C` = connect（`create=False`，发送方连过去），`B` = bind（`create=True`，接收方拥有端点）：

```
  ① BaseTokenizerMsg（TokenizeMsg / AbortMsg）            API Server → tokenizer
     send_tokenizer         PUSH ──C──► [tokenizer_addr]  ◄──B── PULL  recv_listener

  ② BaseBackendMsg（UserMsg / AbortBackendMsg / ExitMsg） tokenizer → Scheduler
     send_backend           PUSH ──C──► [backend_addr]    ◄──B── PULL  _recv_from_tokenizer

  ③ BaseTokenizerMsg（DetokenizeMsg）                     Scheduler → detokenizer
     _send_into_tokenizer   PUSH ──C──► [detokenizer_addr]◄──B── PULL  recv_listener

  ④ BaseFrontendMsg（UserReply）                          detokenizer → API Server
     send_frontend          PUSH ──C──► [frontend_addr]   ◄──B── PULL  recv_tokenizer
```

规律：共享模式下**接收端总是 bind、发送端总是 connect**——「谁收谁 bind」。①和③的接收端是同一个 `recv_listener`（共享模式下 `tokenizer_addr == detokenizer_addr`），靠 `isinstance` 分流 `TokenizeMsg` / `DetokenizeMsg`。

独立模式（`--num-tokenizer N>0`）下，tokenizer 和 detokenizer 拆成两个进程，①③的接收端随之分开，且部分链路的 bind/connect 归属翻转（谁先起来谁 bind），见[第七节](#七完整接线表谁-bind谁-connect)的拓扑图与对照表。

---

## 四、序列化的实现：对象怎么变成字节

看 [utils/mp.py](python/minisgl/utils/mp.py) 的 `ZmqPushQueue.put`：

```python
event = msgpack.packb(self.encoder(obj), use_bin_type=True)
self.socket.send(event, copy=False)
```

- 第一步 `self.encoder(obj)`：`encoder` 就是 `serialize_type`（见 [message/utils.py](python/minisgl/message/utils.py)），把消息对象转成**纯 dict**。
- 第二步 `msgpack.packb`：把这个 dict 打包成字节流。

**`serialize_type` 的完整规则**（[message/utils.py:20-35](python/minisgl/message/utils.py#L20-L35)）：

```python
def serialize_type(self):
    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        return {"__type__": "Tensor",
                "buffer": self.numpy().tobytes(),
                "dtype": str(self.dtype)}

    serialized = {"__type__": self.__class__.__name__}
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized
```

`_serialize_any` 递归处理字段：

- `dict` → 逐 value 递归；
- `list` / `tuple` → 保持原容器类型，逐元素递归；
- `int / float / str / None / bool / bytes` → 原样返回（msgpack 原生支持的类型）；
- 其他 → 交给 `serialize_type`（即嵌套的 dataclass 或 Tensor）。

**`deserialize_type` 反向重建**（[message/utils.py:52-69](python/minisgl/message/utils.py#L52-L69)）：

```python
def deserialize_type(cls_map, data):
    type_name = data["__type__"]
    if type_name == "Tensor":
        np_dtype = getattr(np, data["dtype"].replace("torch.", ""))
        np_tensor = np.frombuffer(data["buffer"], dtype=np_dtype)
        return torch.from_numpy(np_tensor.copy())   # 必须 copy，见难点 3

    cls = cls_map[type_name]
    kwargs = {k: _deserialize_any(cls_map, v) for k, v in data.items() if k != "__type__"}
    return cls(**kwargs)   # 直接用字段名作为关键字参数构造 dataclass
```

**一个完整字节流长这样**（`UserMsg` 为例）：

```json
{
  "__type__": "UserMsg",
  "uid": 3,
  "input_ids": {"__type__": "Tensor", "buffer": "<bytes>", "dtype": "int32"},
  "sampling_params": {"__type__": "SamplingParams", "temperature": 0.0, "top_k": -1, "top_p": 1.0, "ignore_eos": false, "max_tokens": 1024}
}
```

---

## 五、ZMQ 队列封装

[utils/mp.py](python/minisgl/utils/mp.py) 封装了 6 个队列类，统一了「bind/connect + 序列化」：

| 队列类 | Socket 类型 | 同步/异步 | 主要方法 | 谁在用 |
|---|---|---|---|---|
| `ZmqPushQueue` | PUSH | 同步 | `put` | tokenizer（发 backend/frontend）、scheduler（发 detokenizer） |
| `ZmqPullQueue` | PULL | 同步 | `get` / `get_raw` / `decode` / `empty` | tokenizer（收）、scheduler（收 backend） |
| `ZmqAsyncPushQueue` | PUSH | 异步 | `await put` | API Server（发 tokenizer） |
| `ZmqAsyncPullQueue` | PULL | 异步 | `await get` | API Server（收 frontend） |
| `ZmqPubQueue` | PUB | 同步 | `put` / `put_raw` | scheduler rank0（广播给其他 rank） |
| `ZmqSubQueue` | SUB | 同步 | `get` / `empty` | scheduler 非 rank0（订阅 rank0） |

**统一构造函数签名**：

```python
ZmqPushQueue(addr, create, encoder)   # 发送端：create=True 才 bind，否则 connect
ZmqPullQueue(addr, create, decoder)   # 接收端：create=True 才 bind，否则 connect
```

- `create=True` → `bind`（「谁拥有这个端点」）；`create=False` → `connect`（「谁去连它」）。见 [难点 4](#难点-4bind-与-connect-的语义)。
- 发送端要传 `encoder`，接收端要传 `decoder`；构造时只存引用，真正收发时才调用。

**`bind` 的动作发生在构造函数里**：六个队列类都是同一句，`create` 就是那个开关。

```python
class ZmqAsyncPullQueue(Generic[T]):
    def __init__(self, addr, create, decoder):
        self.context = zmq.asyncio.Context()          # ① 建 context
        self.socket = self.context.socket(zmq.PULL)   # ② 建 socket
        self.socket.bind(addr) if create else self.socket.connect(addr)  # ③ bind 或 connect
        self.decoder = decoder
```

所以 `create=True` → **构造这个队列对象的瞬间**，`socket.bind(addr)` 就执行了；没有单独的「启动/监听」步骤，`bind` 是 ZMQ 提供的一个立即返回的调用。这也意味着「谁先 bind」完全由「谁先被 `new` 出来」决定——启动顺序见 [第七节](#七完整接线表谁-bind谁-connect)。

`bind` 在底层做的事：

- `ipc:///tmp/minisgl_3.pid=…`：libzmq 创建一个 **Unix domain socket 文件**，让 socket 进入可 accept 状态（类似 `listen()`）。
- `tcp://127.0.0.1:xxxx`（分布式用）：**打开监听端口**，等价于 POSIX 的 `bind + listen`。
- `bind` 立即返回、不阻塞；之后对端 `connect` 上来的连接由 ZMQ 在后台 accept。

**几个值得注意的实现细节：**

1. **`copy=False`**：`self.socket.send(event, copy=False)` 告诉 ZMQ 直接发送这块缓冲区，省一次内存拷贝。前提是 `msgpack.packb` 产出的 bytes 在 `send` 期间不被回收（这里作为临时变量，安全）。

2. **`raw=False`**：接收端 `msgpack.unpackb(event, raw=False)` 让字符串反序列化成 `str` 而不是 `bytes`，否则中文等非 ASCII 会出问题。

3. **`SUB` 必须订阅**：`ZmqSubQueue.__init__` 里 `self.socket.setsockopt_string(zmq.SUBSCRIBE, "")`，空串表示「订阅所有主题」，否则 SUB 默认收不到任何消息。

4. **`get_raw` / `put_raw` / `decode` 这套裸字节接口是为 TP 多 rank 设计的**（见 [难点 5](#难点-5tp-多-rank-时为什么用-get_raw--put_raw-转发裸字节)）。

**同步 vs 异步：`ZmqPullQueue` 与 `ZmqAsyncPullQueue` 的区别**

这两个类（以及对应的 `ZmqPushQueue` / `ZmqAsyncPushQueue`）本质差别就一个字：**Context 和 `recv/send` 是同步还是异步**。

```python
# 同步版
class ZmqPullQueue(Generic[T]):
    def __init__(self, addr, create, decoder):
        self.context = zmq.Context()            # 普通 Context
        self.socket = self.context.socket(zmq.PULL)
        ...
    def get(self) -> T:
        event = self.socket.recv()              # 阻塞当前线程
        return self.decoder(msgpack.unpackb(event, raw=False))

# 异步版
class ZmqAsyncPullQueue(Generic[T]):
    def __init__(self, addr, create, decoder):
        self.context = zmq.asyncio.Context()    # asyncio Context
        self.socket = self.context.socket(zmq.PULL)
        ...
    async def get(self) -> T:
        event = await self.socket.recv()        # 挂起当前协程，不阻塞事件循环
        return self.decoder(msgpack.unpackb(event, raw=False))
```

| 维度 | `ZmqPullQueue` | `ZmqAsyncPullQueue` |
|---|---|---|
| Context | `zmq.Context()` | `zmq.asyncio.Context()` |
| `get()` | `def`，`socket.recv()` **阻塞线程** | `async def`，`await socket.recv()` **挂起协程** |
| 额外方法 | `get_raw()` / `decode()` / `empty()` | 只有 `get()` |
| 使用场景 | 多进程 worker（tokenizer、scheduler） | API Server（`asyncio` + FastAPI） |
| 谁在用 | `tokenize_worker` 的 `recv_listener`、scheduler 的 `_recv_from_tokenizer` | `FrontendManager.recv_tokenizer` |

**为什么不能混用：**

- `socket.recv()` 会**卡住当前 OS 线程**直到有消息。`tokenize_worker` / scheduler 是独立进程，跑 `while True: msg = recv.get()` 这种死循环，阻塞没关系——它们就这一件事。
- API Server 是**单线程事件循环**（uvicorn + asyncio）同时服务一堆 HTTP 连接。如果在这里用同步 `recv()`，收到消息前整个事件循环都会被冻住，所有请求一起卡死。所以必须用 `zmq.asyncio.Context`，让 `await recv()` 在等待时把控制权交还给事件循环。

**一个附带差异**：`ZmqPullQueue` 多出的 `get_raw()/decode()/empty()` 是给 scheduler 的 TP 多 rank 广播用的（rank0 要拿裸字节转发、要非阻塞 `empty()` 轮询），这些场景只出现在同步的调度器进程里，所以异步版没提供——API Server 只需要一条条 `await get()`。

**选择规则**：凡是跑在 `asyncio` 事件循环里的（API Server、shell 模式）用 `ZmqAsync*`；凡是独立进程里 `while True` 阻塞循环的（tokenizer、scheduler）用同步 `Zmq*`。两边用错了，不是报错就是卡死事件循环。

---

## 六、地址约定

所有 ZMQ 地址由 `_unique_suffix = f".pid={os.getpid()}"` 拼出，避免同一台机器起多个实例时端口/文件冲突。地址用 `ipc://`（进程内通信，走 Unix domain socket 的等价物，只在本机有效）：

| 属性 | 地址 | 链路 | 定义处 |
|---|---|---|---|
| `zmq_backend_addr` | `ipc:///tmp/minisgl_0.pid=…` | tokenizer → Scheduler | [scheduler/config.py](python/minisgl/scheduler/config.py) |
| `zmq_detokenizer_addr` | `ipc:///tmp/minisgl_1.pid=…` | Scheduler → detokenizer | 同上 |
| `zmq_scheduler_broadcast_addr` | `ipc:///tmp/minisgl_2.pid=…` | scheduler rank0 → 其他 rank | 同上 |
| `zmq_frontend_addr` | `ipc:///tmp/minisgl_3.pid=…` | detokenizer → API Server | [server/args.py](python/minisgl/server/args.py) |
| `zmq_tokenizer_addr` | `ipc:///tmp/minisgl_4.pid=…` | API Server → tokenizer（仅独立 tokenizer 时） | 同上 |

> 分布式初始化（NCCL/pynccl）用的是 `tcp://`（`distributed_addr`），因为要跨机；控制消息只用 `ipc://`，因为全在本机进程之间。

---

## 七、完整接线表（谁 bind、谁 connect）

「谁先 bind、谁后 connect」由启动顺序决定。启动顺序是（见 [launch.py](python/minisgl/server/launch.py) 与 [api_server.py](python/minisgl/server/api_server.py)）：

1. `run_api_server` 先建 `FrontendManager`（bind `frontend_addr` 的 PULL；若独立 tokenizer 还 bind `tokenizer_addr` 的 PUSH）。
2. `start_backend()` 再 spawn 各 scheduler / tokenizer / detokenizer 子进程去 connect。

这解释了 [launch.py:127](python/minisgl/server/launch.py#L127) 那句注释：`start_subprocess` 要作为回调传入 API server，因为 **ZMQ 必须先 bind 再 spawn 子进程 connect**。

| 地址 | 谁 bind（`create=True`） | 谁 connect（`create=False`） |
|---|---|---|
| `backend_addr` | scheduler rank0 的 `ZmqPullQueue` | tokenizer/detokenizer 的 `ZmqPushQueue`（`send_backend`） |
| `detokenizer_addr`（共享模式） | detokenizer 的 `ZmqPullQueue` | scheduler 的 `ZmqPushQueue` + API Server 的 `ZmqAsyncPushQueue` |
| `detokenizer_addr`（独立模式） | scheduler 的 `ZmqPushQueue` | detokenizer 的 `ZmqPullQueue` |
| `broadcast_addr` | scheduler rank0 的 `ZmqPubQueue` | 其他 rank 的 `ZmqSubQueue` |
| `frontend_addr` | API Server 的 `ZmqAsyncPullQueue` | tokenizer/detokenizer 的 `ZmqPushQueue`（`send_frontend`） |
| `tokenizer_addr`（独立模式） | API Server 的 `ZmqAsyncPushQueue` | tokenizer 的 `ZmqPullQueue` |

**拓扑示意图**（图例：`B` = bind，即 `create=True`、拥有端点；`C` = connect，即 `create=False`、主动连过去。箭头 = 数据流向，永远是 PUSH → PULL）：

```
  共享模式（默认 --num-tokenizer 0）：tokenizer 与 detokenizer 同进程

  ┌─────────────────────────────────────────────┐
  │            API Server（主进程）              │
  │  send_tokenizer  PUSH ──C──► [1]            │
  │  recv_tokenizer  PULL ◄──B── [3]            │
  └───────────────┬──────────────────▲──────────┘
                  │ [1] detokenizer_addr     │ [3] frontend_addr
                  │                  │
                  ▼                  │
  ┌──────────────────────────────────┴──────────┐
  │        tokenizer/detokenizer（同进程）        │
  │  recv_listener  PULL ◄──B── [1]             │
  │  send_frontend  PUSH ──C──► [3]             │
  │  send_backend   PUSH ──C──► [0]             │
  └───────────────────────────────┬─────────────┘
                                  │ [0] backend_addr
                                  ▼
  ┌─────────────────────────────────────────────┐
  │             Scheduler rank0                  │
  │  _recv_from_tokenizer PULL ◄──B── [0]       │
  │  _send_into_tokenizer PUSH ──C──► [1]       │
  └─────────────────────────────────────────────┘
```

```
  独立模式（--num-tokenizer N>0）：额外 N 个 tokenizer + 1 个 detokenizer，bind/connect 归属翻转

  ┌─────────────────────────────────────────────┐
  │            API Server（主进程）              │
  │  send_tokenizer  PUSH ──B──► [4]            │
  │  recv_tokenizer  PULL ◄──B── [3]            │
  └───────────────┬──────────────────▲──────────┘
                  │ [4] tokenizer_addr      │ [3] frontend_addr
                  ▼                  │
  ┌──────────────────────────────┐   │
  │     tokenizer × N（独立）      │   │
  │  recv_listener PULL ◄──C── [4]│   │
  │  send_backend  PUSH ──C──► [0]│   │
  └───────────────┬──────────────┘   │
                  ▼                  │
  ┌──────────────────────────────┐   │
  │      detokenizer（独立）       │   │
  │  recv_listener PULL ◄──C── [1]│   │
  │  send_frontend PUSH ──C──► [3]│───┘
  └───────────────┬──────────────┘
                  ▼
  ┌─────────────────────────────────────────────┐
  │             Scheduler rank0                  │
  │  _recv_from_tokenizer PULL ◄──B── [0]       │
  │  _send_into_tokenizer PUSH ──B──► [1]       │
  └─────────────────────────────────────────────┘
```

两张图对照着看，能看出「独立模式」和「共享模式」最关键的差异：`[1] detokenizer_addr` 的 owner 从 detokenizer 变成了 scheduler（`_send_into_tokenizer` 从 `C` 变 `B`），而 detokenizer 那边从 `B` 变 `C`——因为独立模式下 scheduler 是「先起来」的那一方。

「共享模式 vs 独立模式」由 `--num-tokenizer` 决定：`num_tokenizer == 0`（默认）时 tokenizer 与 detokenizer 是同一个进程、共用 `detokenizer_addr`；`> 0` 时额外起 N 个 tokenizer 进程，用独立的 `tokenizer_addr`。对应的 `*_create_*` 布尔属性在 [server/args.py](python/minisgl/server/args.py) 里统一切换。

---

## 八、如何使用

### 8.1 定义一条新消息

按「它在哪段链路」选一个文件，继承对应的 `Base*`，字段用 dataclass 标注：

```python
# 假设要加一条「暂停请求」：API Server → tokenizer
@dataclass
class PauseMsg(BaseTokenizerMsg):
    uid: int
    reason: str
```

只要字段是 `int/float/str/bool/None/bytes/list/dict/tuple`、`torch.Tensor`（1D）或其它 dataclass，就能被自动序列化。**不需要手写任何 encode/decode**——`serialize_type` 会遍历 `__dict__` 拿到字段，`deserialize_type` 用 `cls(**kwargs)` 按字段名重建。

> 注意：新类必须定义在该文件的 `globals()` 可见位置（模块顶层），否则 `deserialize_type(globals(), …)` 找不到它。跨模块复用一个消息类是不行的。

### 8.2 起一个发送端 / 接收端

```python
from minisgl.utils import ZmqPushQueue, ZmqPullQueue
from minisgl.message import BaseTokenizerMsg, TokenizeMsg

# 接收端：bind（create=True）
recv = ZmqPullQueue("ipc:///tmp/example", create=True, decoder=BaseTokenizerMsg.decoder)
# 发送端：connect（create=False）
send = ZmqPushQueue("ipc:///tmp/example", create=False, encoder=BaseTokenizerMsg.encoder)

send.put(TokenizeMsg(uid=0, text="hello", sampling_params=SamplingParams()))
msg = recv.get()          # 阻塞，直到收到一条
print(msg.uid, msg.text)
```

要点：

- 先 bind 后 connect（先 `recv` 后 `send`）。若顺序反了，ZMQ 的 connect 不会报错，会静默排队（见 [难点 4](#难点-4bind-与-connect-的语义)）。
- `send.put()` 传的是**消息对象**，不是 bytes；编解码对使用者透明。
- 异步环境（`asyncio`）换 `ZmqAsyncPushQueue` / `ZmqAsyncPullQueue`，用 `await put()/get()`，否则会阻塞事件循环。

### 8.3 在现有链路里加一条消息的完整流程

以「API Server → tokenizer」方向为例，需要动三处：

1. 定义 `TokenizeMsg`（或新类）于 [message/tokenizer.py](python/minisgl/message/tokenizer.py)（继承 `BaseTokenizerMsg`）。
2. tokenizer 侧（[tokenizer/server.py](python/minisgl/tokenizer/server.py)）的 `while True` 循环里，`isinstance(m, XxxMsg)` 分流并处理，把结果 `put` 到 `send_backend` / `send_frontend`。
3. API Server 侧（[server/api_server.py](python/minisgl/server/api_server.py)）调用 `send_one(...)` / `await self.send_tokenizer.put(...)` 发出。

### 8.4 单条 vs 批量

发送端（[tokenizer/server.py](python/minisgl/tokenizer/server.py)）的惯用写法：攒满一批包成 `Batch*Msg`，只有一条时直接发单条：

```python
batch_output = BatchBackendMsg(data=[...])       # 多条
if len(batch_output.data) == 1:
    batch_output = batch_output.data[0]          # 单条就不套 Batch，省一层
send_backend.put(batch_output)
```

接收端统一 `_unwrap_msg` 拆包即可。这样对端不用关心收到的是单条还是 Batch。

---

## 九、难点解析

### 难点 1：为什么不直接用 pickle 或 msgpack 传对象？

- **msgpack 不认识 `torch.Tensor` 和自定义 dataclass**，直接 pack 会报错。
- **pickle 慢、且反序列化任意对象不安全**（可执行任意代码），而且 pickle 的字节流依赖 Python 版本/类定义。
- 所以项目采用「自定义 `serialize_type` 转成纯 dict + msgpack 打包」的两段式：**msgpack 负责高效二进制打包，`serialize_type` 负责把 Python 对象降维成 msgpack 认识的类型**（`int/float/str/bool/None/bytes/list/dict`）。

### 难点 2：`__type__` 标记 + `globals()` 类映射

`serialize_type` 给每个对象写 `__type__` 记录类名；反序列化时 `deserialize_type(globals(), json)` 用 `globals()` 作为「类名字符串 → 类对象」的映射表。

所以 `BaseBackendMsg.decoder` 里传的是 `globals()`——意味着**只有 `backend.py` 这个模块里定义的消息类能被反序列化**。这也是为什么三类消息要分三个文件、各自维护自己的 `globals()`。

隐患：`globals()` 是隐式传入的，等于「反序列化能构造出当前模块里任意一个类的对象」。只要模块里有别的类（比如导入了 `SamplingParams`），理论上也能被外部恶意 bytes 触发构造。这个项目是本地 IPC、进程可信，风险可控；但换到暴露公网的场景就要小心。

### 难点 3：Tensor 为什么只支持 1D？

序列化用 `numpy().tobytes()`，虽然多维也技术上可行，但这里的消息里的 tensor 都是 `input_ids`（1D token 序列），为了简单明确就限制成 1D，`assert self.dim() == 1` 直接拦下误用。

反序列化时 `np.frombuffer` 得到的是**共享内存的 view**，所以必须 `.copy()` 成独立张量，否则 `buffer` 字节被回收后 tensor 会失效（指向已释放内存，读到脏数据）。

### 难点 4：bind 与 connect 的语义

- `bind`：谁「拥有」这个端点，必须先于 connect 就位。
- `connect`：谁「去连」它，ZMQ 里 connect 到一个还没 bind 的端点**不会报错**，消息会静默排队等待。

这也是 Step 1 里为什么「主进程先 bind、再 spawn 子进程去 connect」的原因（对应 [launch.py](python/minisgl/server/launch.py) 把 `start_subprocess` 作为回调、在 `run_api_server` 建好队列之后再执行）。

**两个延伸理解：**

1. ZMQ 的 `bind/connect` **不是 POSIX 那种「必须先 listen 才能 connect」**。对端 `connect` 一个还没 `bind` 的端点不会报错、消息静默排队。所以「先 bind 后 connect」在这里是**约定/顺序习惯**，不是硬性要求——反过来 connect 先发生也能工作，只是消息会先排队等 bind 就位。
2. 地址写死了 `ipc:///tmp/...`（Unix domain socket 路径）。在 Windows 上 libzmq 对 `ipc://` 的处理和 Linux 不同，若要本地起服务验证需留意。

### 难点 5：TP 多 rank 时为什么用 `get_raw` / `put_raw` 转发裸字节？

在 [scheduler/io.py](python/minisgl/scheduler/io.py) 的 `_recv_msg_multi_rank0` 里，rank0 从 tokenizer 收到的是**字节流**，它做两件事：

```python
raw = self._recv_from_tokenizer.get_raw()   # 1. 拿原始 bytes
self._send_into_ranks.put_raw(raw)          # 2. 原样 PUB 广播给其他 rank
pending_msgs.append(self._recv_from_tokenizer.decode(raw))  # 3. 自己 decode
```

- 为什么不 `decode` 成对象再 `put`？那样其他 rank 收到对象后，rank0 还得**再序列化一次**（`put` 会 `encoder`），白白多一次 encode。`put_raw` 直接把已收到的 bytes 转发，零拷贝、零重复编码。
- 更重要的是**一致性**：所有 rank decode 的是**同一份 bytes**，保证 rank0 和 rank1 拿到完全一致的消息内容。而各 rank 收到的消息**条数**用 `torch.distributed.broadcast` 对齐（`_recv_msg_multi_rank1` 里先广播长度，再按长度循环 `get`）。

---

## 十、注意事项

1. **消息类是 dataclass，字段名和类型必须和 `serialize_type` 能处理的类型匹配**。加字段时别用自定义对象（除非也实现序列化，或本身就是 dataclass）。
2. **`globals()` 作为类映射的局限**：跨模块的消息类不能互相反序列化，新增消息类要加在对应文件的 `globals()` 可见位置。
3. **`copy=False`** 表示 ZMQ 直接发送缓冲区，省一次拷贝，但要求 `msgpack.packb` 产生的 bytes 在 `send` 期间不被回收。
4. **`raw=False`**（`msgpack.unpackb(event, raw=False)`）让字符串反序列化成 `str` 而不是 `bytes`，否则中文会出问题。
5. **异步队列和同步队列不能混用**：API Server 是 `asyncio` 环境，必须用 `ZmqAsync*`，否则阻塞事件循环。
6. **`SUB` 必须 `SUBSCRIBE`**：忘了 `setsockopt_string(zmq.SUBSCRIBE, "")` 会一条消息都收不到。
7. **启动顺序不能反**：必须先 `run_api_server`（bind）再 `start_backend`（spawn 子进程 connect）。

---

## 十一、反思题

1. 如果要支持序列化 2D 的 `torch.Tensor`，`serialize_type` 需要怎么改？（提示：除了 buffer 还要存 shape）
2. `np.frombuffer` 返回的是 view，为什么必须 `.copy()`？不 copy 会有什么 bug？
3. 为什么 `deserialize_type` 的类映射用 `globals()` 而不是显式传一个 dict？这有什么隐患？
4. 三类消息为什么分成三个文件？如果合并成一个文件会有什么影响？
5. `Batch*Msg` 包装的意义是什么？如果每次都单条单条传，性能上会怎样？
6. TP 多 rank 时 rank0 为什么用 `put_raw` 转发，而不是 `decode` 后再 `put`？两种方式结果有何不同？

---

## 十二、示意图

### 12.1 序列化与反序列化全流程

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

### 12.2 消息在链路里的流动

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
| [message/backend.py](python/minisgl/message/backend.py) | tokenizer→scheduler 消息 | `UserMsg`、`AbortBackendMsg`、`ExitMsg`、`BaseBackendMsg`、`BatchBackendMsg` |
| [message/tokenizer.py](python/minisgl/message/tokenizer.py) | 前后端消息 | `TokenizeMsg`、`DetokenizeMsg`、`AbortMsg`、`BaseTokenizerMsg`、`BatchTokenizerMsg` |
| [message/frontend.py](python/minisgl/message/frontend.py) | detokenizer→API Server 消息 | `UserReply`、`BaseFrontendMsg`、`BatchFrontendMsg` |
| [message/utils.py](python/minisgl/message/utils.py) | 序列化/反序列化 | `serialize_type`、`deserialize_type`、`_serialize_any`、`_deserialize_any` |
| [utils/mp.py](python/minisgl/utils/mp.py) | ZMQ 队列封装 | `ZmqPushQueue`、`ZmqPullQueue`、`ZmqPubQueue`、`ZmqSubQueue`、`ZmqAsyncPushQueue`、`ZmqAsyncPullQueue` |
| [scheduler/io.py](python/minisgl/scheduler/io.py) | scheduler 侧收发、TP 广播 | `SchedulerIOMixin`、`_recv_msg_multi_rank0`、`_reply_tokenizer_rank0` |
| [scheduler/config.py](python/minisgl/scheduler/config.py) | 后端地址定义 | `zmq_backend_addr`、`zmq_detokenizer_addr`、`zmq_scheduler_broadcast_addr` |
| [server/args.py](python/minisgl/server/args.py) | 前端地址定义 | `zmq_frontend_addr`、`zmq_tokenizer_addr` |

**下一步**：进入 Step 4（API Server 前端），看这些消息在异步的 FastAPI 里怎么被生产和消费。
