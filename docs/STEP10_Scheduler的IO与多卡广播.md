# Step 10：Scheduler 的 I/O 与多卡广播

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 10。
> 核心文件：[scheduler/io.py](python/minisgl/scheduler/io.py) 的 `SchedulerIOMixin`。
>
> 这一 Step 回答：**单卡和多卡下，Scheduler 是怎么从 tokenizer 收消息、怎么把结果发回去的？为什么多卡时 rank0 要「既广播消息、又广播一个长度」？**

---

## 一、这个 Step 要解决什么

Step 8/9 讲了 Scheduler 主循环里调用了 `receive_msg` 和 `send_result`，但没说这两个方法在**单卡和多卡下是完全不同的实现**。本 Step 补上这块：Scheduler 的网络层（I/O mixin）。

核心矛盾：**只有 rank0 对外收发消息**（Step 0.3 的主线），但**所有 rank 都要跑同一套调度逻辑、看到同样的请求顺序**。多卡时怎么把 rank0 收到的消息「原样、同序」同步给其他 rank？这就是本 Step 的题眼。

---

## 二、核心逻辑

### 2.1 `SchedulerIOMixin.__init__`：按 rank 分叉

```python
def __init__(self, config: SchedulerConfig, tp_cpu_group):
    tp_info = config.tp_info
    self.tp_cpu_group = tp_cpu_group
    if config.offline_mode:                      # 离线模式：无 ZMQ，直接 return
        self.receive_msg = self.offline_receive_msg
        self.send_result = self.offline_send_result
        return

    if tp_info.is_primary():                     # rank0：直接连 tokenizer/detokenizer
        self._recv_from_tokenizer = ZmqPullQueue(config.zmq_backend_addr, create=True, ...)
        self._send_into_tokenizer = ZmqPushQueue(config.zmq_detokenizer_addr, create=True, ...)

    recv = self._recv_msg_single_rank            # 默认：单卡版
    send = self._reply_tokenizer_rank0
    if tp_info.size > 1:                         # 多卡：换成广播版
        if tp_info.is_primary():
            recv = self._recv_msg_multi_rank0
            self._send_into_ranks = ZmqPubQueue(config.zmq_scheduler_broadcast_addr, create=True, ...)
        else:
            recv = self._recv_msg_multi_rank1
            send = self._reply_tokenizer_rank1   # rank1..N 不回结果
            self._recv_from_rank0 = ZmqSubQueue(config.zmq_scheduler_broadcast_addr, create=False, ...)

    self.receive_msg = recv
    self.send_result = send
```

注意 `is_primary()` 就是 `rank == 0`。三种队列的组合：

| rank | 收消息来源 | 发结果去向 |
|---|---|---|
| 单卡 rank0 | PULL（bind `minisgl_0`） | PUSH（bind `minisgl_1`） |
| 多卡 rank0 | PULL（bind `minisgl_0`）+ 再 PUB 广播 | PUSH（bind `minisgl_1`） |
| 多卡 rank1..N | SUB（connect `minisgl_2`） | no-op |

### 2.2 单卡收消息 `_recv_msg_single_rank`

```python
def _recv_msg_single_rank(self, blocking=False):
    pending_msgs = []
    if blocking:
        self.run_when_idle()                      # 空闲时做后台任务
        pending_msgs.append(self._recv_from_tokenizer.get())   # 阻塞等一条
    while not self._recv_from_tokenizer.empty():
        pending_msgs.append(self._recv_from_tokenizer.get())   # 非阻塞扫空
    return pending_msgs
```

`blocking=True`（完全空闲时）先阻塞 `get()` 等一条；否则非阻塞扫空队列。配合 Step 8 讲的「忙轮询 + 空闲阻塞」。

### 2.3 多卡 rank0 收消息 `_recv_msg_multi_rank0`

```python
def _recv_msg_multi_rank0(self, blocking=False):
    pending_msgs = []
    if blocking:
        self.run_when_idle()
        raw = self._recv_from_tokenizer.get_raw()      # 拿到原始 bytes
        self._send_into_ranks.put_raw(raw)             # 原样广播给所有 rank
        pending_msgs.append(self._recv_from_tokenizer.decode(raw))

    pending_raw_msgs = []
    while not self._recv_from_tokenizer.empty():
        pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

    # 广播「还有多少条」给所有 rank
    src_tensor = torch.tensor(len(pending_raw_msgs))
    self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

    for raw in pending_raw_msgs:
        self._send_into_ranks.put_raw(raw)
        pending_msgs.append(self._recv_from_tokenizer.decode(raw))
    return pending_msgs
```

关键动作：rank0 每收到一条消息，**用 `put_raw` 把原始字节原样 PUB 出去**（不 decode 不重编码，保证 byte-level 一致），再自己 `decode`。

### 2.4 多卡 rank1 收消息 `_recv_msg_multi_rank1`

```python
def _recv_msg_multi_rank1(self, blocking=False):
    pending_msgs = []
    if blocking:
        self.run_when_idle()
        pending_msgs.append(self._recv_from_rank0.get())   # 从 SUB 收一条

    dst_tensor = torch.tensor(-1)
    self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()  # 收「还有多少条」
    dst_length = int(dst_tensor.item())

    for _ in range(dst_length):
        pending_msgs.append(self._recv_from_rank0.get())   # 按数量继续收
    return pending_msgs
```

### 2.5 发结果 `_reply_tokenizer_rank0` / `_reply_tokenizer_rank1`

```python
def _reply_tokenizer_rank0(self, reply):
    if num_reply == 1: self._send_into_tokenizer.put(reply[0])
    elif num_reply > 1: self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))

def _reply_tokenizer_rank1(self, reply):
    _ = reply  # 非 rank0 什么都不发
```

只有 rank0 回结果，rank1..N 是 no-op——这是「rank0 对外、全员对内」分工的直接体现。

---

## 三、难点解析

### 难点 1：为什么 `put_raw` 而不是 `put`？

`put` 会先把对象 `serialize_type` + `msgpack.packb` 再发；`put_raw` 直接把**已经序列化好的 bytes** 原样发出。

rank0 从 tokenizer 收到的是 bytes（`get_raw`），要广播给 rank1..N。如果 rank0 先 `decode` 成对象、再 `put`（重新 encode），中间多一次序列化/反序列化，且**可能引入编码差异**。直接用 `put_raw` 转发原始字节，既省开销，又保证「所有 rank 拿到的字节流一模一样」。

### 难点 2：为什么要 CPU broadcast 一个「消息条数」张量？

这是最微妙的地方。rank1 的 SUB 通道是**异步**的，它不知道这一轮 rank0 到底发了多少条。如果 rank1 一直 `get()` 到 empty 就停，会因为时序竞态而**少收或多收**消息，导致各 rank 看到的请求顺序不一致。

所以用 `tp_cpu_group.broadcast`（gloo 后端、CPU 侧）做一次**同步屏障**：rank0 广播 `pending_raw_msgs` 的数量，rank1 收到这个数后 `for _ in range(dst_length)` 精确取这么多条。这个 broadcast 同时起到了「对齐」作用——让所有 rank 在同一个时间点知道「这一轮共 N 条消息」。

对应 [commit `9a91cfa`](https://github.com/xxx/mini-sglang/commit/9a91cfa) 修复的 decode 顺序问题：多卡时各 rank 的请求顺序必须严格一致，否则各自算出来的 batch 对不上，NCCL all-reduce 会死锁或算错。

### 难点 3：rank1 的 SUB 为什么 `create=False`？

`minisgl_2`（广播地址）由 rank0 的 `ZmqPubQueue(create=True)` 来 bind，rank1..N 的 `ZmqSubQueue(create=False)` 去 connect。这跟 Step 1 讲的「主进程先 bind、子进程再 connect」是同一套 bind/connect 语义：PUB 是服务端，SUB 是客户端。

### 难点 4：`offline_mode` 为什么提前 return？

离线模式下没有 tokenizer 进程，也就不需要 ZMQ 队列。`receive_msg`/`send_result` 被替换成 `offline_*` 版本（当前代码里是 `raise NotImplementedError`，说明离线模式是预留接口，尚未完整实现）。

---

## 四、注意事项

1. **单卡时 `receive_msg` 的默认绑定**：`recv = _recv_msg_single_rank` 是初始值，只有 `tp_info.size > 1` 才覆盖成 multi 版本。
2. **`is_primary()` 判断的是 `rank == 0`**：多卡下只有 rank0 创建 `_recv_from_tokenizer`（PULL）和 `_send_into_tokenizer`（PUSH），rank1 只有 `_recv_from_rank0`（SUB）。
3. **`get_raw` / `put_raw` 绕过序列化**：`raw` 是 msgpack 打包后的 bytes，`decode(raw)` 才还原成消息对象。转发用 raw，本地用 decode。
4. **`broadcast(...).wait()` 是同步的**：`.wait()` 等 gloo broadcast 完成才继续，确保所有 rank 都拿到一致的 `dst_length` 再进入取消息循环。
5. **`BatchTokenizerMsg` 包装**：结果多于 1 条时打成 `Batch*Msg` 一次发，减少 ZMQ 收发次数（Step 3 讲过的批量化）。

---

## 五、反思题

1. 如果 `_recv_msg_multi_rank0` 忘记 `put_raw` 广播，只自己 decode，会发生什么？rank1 会怎样？
2. 为什么广播「条数」用的是 `tp_cpu_group`（gloo）而不是 ZMQ？两者各负责什么？（提示：gloo 是同步屏障，ZMQ 是单向数据流）
3. `_recv_msg_multi_rank1` 里 `blocking` 分支收的那一条，和 `for range(dst_length)` 收的那批，各自对应 rank0 里的哪段代码？为什么数量能对上？
4. 为什么 rank0 要「先 `get_raw` 再 `decode`」而不是「先 `get` 出对象再 `put` 广播」？两种做法结果一样吗？
5. `offline_mode` 下 `send_result` 是 `offline_send_result`，它应该做什么？（提示：没有 detokenizer 进程，结果直接攒在本地）

---

## 六、示意图

### 6.1 单卡 vs 多卡的 I/O 拓扑

```
【单卡】                      【多卡 tp=4】
tokenizer ──UserMsg──►        tokenizer ──UserMsg──►
   ▲          │                 ▲          │
   │PULL(bind)│                 │PULL(bind)│
   │          ▼                 │          ▼
 rank0                    rank0 ──PUB──► rank1/2/3 (SUB connect)
   │          │                 │            ▲
   │PUSH(bind)│                 │PUSH(bind)   │
   ▼          ▼                 ▼            │
detokenizer ◄─DetokenizeMsg─ detokenizer      └── 结果 no-op
```

### 6.2 多卡一轮收消息的时序（rank0 + rank1 对齐）

```
  tokenizer        rank0                    rank1
     │  msg#1        │                        │
     ├──────────────►│ put_raw(msg#1) ───────►│ get() 收到 msg#1
     │  msg#2,msg#3  │                        │
     ├──────────────►│ get_raw×2 攒着         │
     │               │ broadcast(count=2) ◄──►│ broadcast 收 count=2
     │               │ put_raw(msg#2,msg#3)─► │ get()×2 收 msg#2,msg#3
     │               │                        │
     │               │ 两边都拿到 [1,2,3] 同序  │
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [scheduler/io.py](python/minisgl/scheduler/io.py) | Scheduler 网络层 | `SchedulerIOMixin`、`_recv_msg_multi_rank0`、`_recv_msg_multi_rank1` |
| [scheduler/config.py](python/minisgl/scheduler/config.py) | ZMQ 地址定义 | `zmq_backend_addr`、`zmq_scheduler_broadcast_addr` |

**下一步**：进入 Step 11（PrefillManager 与 Chunked Prefill），看长 prompt 怎么被切成多块、准入控制怎么防爆显存。
