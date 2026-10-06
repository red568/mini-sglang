# Mini-SGLang 源码学习路线图（Step by Step）

> 用法：从 **Step 0** 开始，按顺序往下走。每一步都指向具体源码文件和函数，先读源码、再想「检查点」里的问题，最后（可选）做「动手」里的实验。全项目约 5000 行 Python，全部读完是可行的。
>
> 主线：**一个请求从进来到出去，经历了哪些进程、哪些函数、哪些数据结构。**

---

## Step 0：跑起来 + 建立全局地图

### 0.1 先跑通

> ⚠️ 平台前提：项目依赖 Linux 专属 CUDA kernel（`sgl-kernel`/`flashinfer`），**Windows/macOS 不能直接跑**，请用 **WSL2** 或 **Docker**（见 [README.md](README.md)）。只想读逻辑可加 `--dummy-weight`（随机权重，不下载模型）。

```bash
python -m minisgl --model Qwen/Qwen3-0.6B --shell          # 交互聊天
python -m minisgl --model Qwen/Qwen3-0.6B --dummy-weight --shell  # 无网/dummy 权重
```

### 0.2 进程拓扑（背下来，后面每个 Step 都在往里填细节）

**0.2.1 启动时 spawn 出哪些进程**

```
python -m minisgl   （主进程，你在终端里跑的那个）
│
├── API Server ───────────── 不 spawn，主进程自己接着跑 FastAPI（uvicorn）
│
├── Scheduler × N ─────────── N = --tp 的值，每个 GPU 一个，入口 _run_scheduler()
│     ├─ minisgl-TP0-scheduler    (rank 0)
│     └─ minisgl-TP1-scheduler    (rank 1)   ← 仅 --tp ≥ 2 时存在
│
└── tokenizer/detokenizer ──── 默认合并成 1 个进程，入口 tokenize_worker()
      （--num-tokenizer 0 时共享；否则拆成 独立 tokenizer×n + detokenizer×1）
```

**0.2.2 单卡拓扑（--tp 1，默认情况）**

三个进程靠三条 ZMQ 链路连起来，箭头 = 消息流向：

```
           ① HTTP 请求           ② TokenizeMsg            ③ UserMsg
用户 ──────────────► ┌────────────┐ ───────────► ┌────────────────────┐ ───────────► ┌──────────────┐
                    │ API Server │              │ tokenizer/detokenizer│              │  Scheduler   │
用户 ◄────────────── │  (主进程)   │ ◄─────────── │ (子进程, 默认共享1个) │ ◄─────────── │  (rank 0)    │
           ⑧ SSE流式  └────────────┘   ⑦ UserReply │                    │ ④ DetokenizeMsg│  (GPU 0)     │
                                                    └────────────────────┘              └──────────────┘
```

另外两步在图外：**⑤ 模型前向** 发生在 Scheduler 框内部（GPU 上跑 CUDA kernel，不走 ZMQ）；**⑥ rank0 收集 token** 也在 Scheduler 内部，把结果打包成 ④ 发出去。

**0.2.3 多卡拓扑（--tp 4）**

多了「Scheduler 之间」的通信，其余和单卡一样：

```
                 ┌─────────── API Server ───────────┐
                 │   (只跟 rank 0 收发，和单卡一致)     │
                 ▼                                  │
         ┌─────────────────────┐                    │
         │ tokenizer/detokenizer│                   │
         └─────────┬───────────┘                    │
                   │ UserMsg                        │
                   ▼                                │
        ┌──────────────────────┐    DetokenizeMsg   │
        │  Scheduler rank 0    │ ───────────────────┘
        │     (GPU 0)          │
        └─────────┬────────────┘
                  │ PUB/SUB 广播（ZMQ，把 UserMsg 转发给所有 rank）
       ┌──────────┼──────────┬──────────┐
       ▼          ▼          ▼          ▼
  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
  │ rank 1   ││ rank 2   ││ rank 3   ││ rank 4   │  各自 GPU 上跑同一份调度逻辑、算自己那一片权重
  │ (GPU 1)  ││ (GPU 2)  ││ (GPU 3)  ││ (GPU 4)  │
  └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
       └──────┴─────┴──────┴─────┴──────┴────┘
              NCCL / pynccl（GPU 间 all-reduce，只在模型前向时通信）
```

要点：**只有 rank 0 对外收发消息**，rank 1..N 只做「收广播 + 各自前向 + NCCL 通信」。

**0.2.4 ZMQ 通道速查表**

| 通道（ipc:///tmp/ 前缀） | 方向 | 传什么 | 谁 bind / 谁 connect |
|---|---|---|---|
| `minisgl_4`（share 时 = `minisgl_1`） | API Server → tokenizer | `TokenizeMsg` | tokenizer bind / API Server connect |
| `minisgl_0` | tokenizer → rank0 | `UserMsg`、`AbortBackendMsg` | rank0 bind / tokenizer connect |
| `minisgl_1` | rank0 → detokenizer | `DetokenizeMsg` | detokenizer bind / rank0 connect |
| `minisgl_3` | detokenizer → API Server | `UserReply` | API Server bind / detokenizer connect |
| `minisgl_2` | rank0 → rank1..N | 广播 `UserMsg` 等 | rank0 PUB bind / rank1..N SUB connect |

> 关键点：默认共享模式下，**`minisgl_1` 一个地址同时收两类消息**（API Server 来的 `TokenizeMsg` + rank0 来的 `DetokenizeMsg`），所以 `tokenize_worker` 主循环里用 `isinstance` 把消息分拣成 tokenize / detokenize / abort 三类——这就是 tokenizer 和 detokenizer 能合并成一个进程的原因。
>
> 术语：`bind` = 谁创建通信端点（服务端），`connect` = 谁去连它（客户端），对应代码里的 `create=True / False`。序列化是两段式：先 `serialize_type` 把对象（含 1D `torch.Tensor`）转成纯 dict，再交给 **msgpack** 打包成字节流跨进程传。

### 0.3 请求生命周期（这是贯穿全文的主线）

**时序图**（时间从上往下，箭头 = 一条消息）：

```
 Client        API Server       tokenizer/detok      Scheduler r0      other ranks
   │               │                  │                   │                 │
   │ ① POST         │                  │                   │                 │
   │ /v1/chat/completions             │                   │                 │
   ├──────────────►│                  │                   │                 │
   │               │ ② TokenizeMsg    │                   │                 │
   │               ├─────────────────►│                   │                 │
   │               │                  │ ③ UserMsg         │                 │
   │               │                  ├──────────────────►│                 │
   │               │                  │                   │ ④ 广播(仅多卡)   │
   │               │                  │                   ├────────────────►│
   │               │                  │                   │ ⑤ 调度+前向      │
   │               │                  │                   │  (各rank各自算)  │
   │               │                  │                   │◄──NCCL reduce──►│
   │               │                  │ ⑥ DetokenizeMsg   │                 │
   │               │                  │◄──────────────────┤                 │
   │               │ ⑦ UserReply      │                   │                 │
   │               │◄─────────────────┤                   │                 │
   │ ⑧ SSE 流式返回  │                  │                   │                 │
   │◄──────────────┤                  │                   │                 │
```

**关键：prefill 一次 + decode 循环**

- ①→③ 是**一次性**的：请求进来、tokenize、进入 prefill 队列。
- ⑤ 分两个阶段：**prefill**（一次性吃下整段 prompt，可能被 chunked prefill 切成多块）→ 之后进入 **decode 循环**。
- **decode 循环**里，⑤⑥⑦⑧ 会**重复很多次**——每循环一次生成一个 token，直到命中 EOS 或到达 `max_tokens`。所以「一个请求」实际是「一次 prefill + N 次 decode」。

**每步对应的源码**：

| 步骤 | 发生在哪 | 消息/动作 | 关键函数 | 文件 |
|---|---|---|---|---|
| ① | 用户 → API Server | HTTP POST | `v1_completions` | [server/api_server.py](python/minisgl/server/api_server.py) |
| ② | API Server → tokenizer | `TokenizeMsg` | `new_user` / `send_one` | [server/api_server.py](python/minisgl/server/api_server.py) |
| ③ | tokenizer → rank0 | `UserMsg` | `tokenize` / `send_backend.put` | [tokenizer/tokenize.py](python/minisgl/tokenizer/tokenize.py)、[tokenizer/server.py](python/minisgl/tokenizer/server.py) |
| ④ | rank0 → 其他 rank | ZMQ PUB/SUB 广播 | `_recv_msg_multi_rank0` | [scheduler/io.py](python/minisgl/scheduler/io.py) |
| ⑤ | 各 rank 本地 | 调度 + 模型前向 + 采样 | `_schedule_next_batch` → `_forward` → `forward_batch` | [scheduler/scheduler.py](python/minisgl/scheduler/scheduler.py)、[engine/engine.py](python/minisgl/engine/engine.py) |
| ⑥ | rank0 → detokenizer | `DetokenizeMsg` | `_process_last_data` → `send_result` | [scheduler/scheduler.py](python/minisgl/scheduler/scheduler.py) |
| ⑦ | detokenizer → API Server | `UserReply` | `detokenize` / `send_frontend.put` | [tokenizer/detokenize.py](python/minisgl/tokenizer/detokenize.py)、[tokenizer/server.py](python/minisgl/tokenizer/server.py) |
| ⑧ | API Server → 用户 | SSE 流式 | `stream_chat_completions` | [server/api_server.py](python/minisgl/server/api_server.py) |

> 关键认知：**每个 Scheduler rank 都跑同一套调度逻辑、算自己那一片权重**（张量并行切分），只有 rank0 负责对外收发消息。这套「rank0 对外、全员对内」的分工贯穿全项目。

---

## 预备知识自查（缺什么补什么）

| 知识点 | 必要度 | 用到哪个 Step |
|---|---|---|
| Python `dataclass` / `abc` / 类型注解 | 必备 | 全程 |
| PyTorch 张量 / `stream` / `Event` / `inference_mode` | 必备 | Step 8+ |
| 多进程 + ZMQ（`PUSH/PULL/PUB/SUB`） | 必备 | Step 1、3、10 |
| LLM 推理常识：prefill vs decode、KV Cache、自回归、batch | 必备 | Step 11+ |
| CUDA stream/event/graph | 用到再补 | Step 22 |
| 张量并行 / all-reduce | 用到再补 | Step 17 |
| 注意力 Q/K/V、RoPE、GQA | 用到再补 | Step 18、21 |

---

## Part A · 启动与进程（Step 1–4）

### Step 1：项目入口与总装配线

**源码**：[python/minisgl/__main__.py](python/minisgl/__main__.py) → [python/minisgl/server/launch.py](python/minisgl/server/launch.py)

**读什么**：
- `__main__.py` 只有一行 `launch_server()`，看它怎么被触发。
- `launch.py` 的 `launch_server()` → `start_subprocess()`：谁 `spawn` 了哪些进程、每个进程的 `target` 入口是什么。
- `_run_scheduler()`：一个 Scheduler 进程的生命周期（`Scheduler(args)` → `run_forever()`）。
- `tokenize_worker` 被拉起几次、参数里 `local_bs` / `create` / `tokenizer_id` 的含义。

**学什么**：为什么主进程要 `mp.set_start_method("spawn")`（CUDA 上下文不能 fork）；`ack_queue` 怎么保证所有子进程就绪后才对外服务。

**检查点**：能说出启动后一共有几个进程、每个进程的入口函数名。

---

### Step 2：CLI 参数与三层 Config 继承

**源码**：[python/minisgl/server/args.py](python/minisgl/server/args.py)、[python/minisgl/scheduler/config.py](python/minisgl/scheduler/config.py)、[python/minisgl/engine/config.py](python/minisgl/engine/config.py)

**读什么**：
- `args.py` 的 `parse_args()`：每个 `--xxx` 参数对应哪个字段（`dest`）。
- 三层 dataclass 继承：`ServerArgs → SchedulerConfig → EngineConfig`，各定义了哪些字段。
- `engine/config.py` 的 `model_config`（`cached_property`）：HuggingFace config 怎么被转成项目自己的 `ModelConfig`。
- `scheduler/config.py` 的 ZMQ 地址 `zmq_*_addr`：为什么带 pid 后缀。

**学什么**：dataclass 继承 + `frozen=True`；配置在进程间如何传递（每个 rank 拿一份 `DistributedInfo(rank, size)`）。

**检查点**：能解释 `--tp 2` 之后，两个 scheduler 进程的 `tp_info` 分别是什么。

---

### Step 3：消息系统与轻量序列化

**源码**：[python/minisgl/message/backend.py](python/minisgl/message/backend.py)、[python/minisgl/message/frontend.py](python/minisgl/message/frontend.py)、[python/minisgl/message/tokenizer.py](python/minisgl/message/tokenizer.py)、[python/minisgl/message/utils.py](python/minisgl/message/utils.py)

**读什么**：
- 三类消息分别是谁和谁之间的协议：`backend.py`（tokenizer→scheduler）、`tokenizer.py`（api_server→tokenizer + scheduler→detokenizer）、`frontend.py`（detokenizer→api_server）。
- 每个数据类：`TokenizeMsg` / `UserMsg` / `DetokenizeMsg` / `UserReply` / `AbortMsg` / `AbortBackendMsg` 里各有什么字段。
- `message/utils.py` 的 `serialize_type` / `deserialize_type`：把对象先转成纯 dict（`torch.Tensor` 只支持 1D，转成 numpy bytes），再交给 msgpack 打包（见 `utils/mp.py` 的 `put`）。

**学什么**：这套消息就是「进程之间的语言」。理解它，后面每看到一次 `put`/`get` 就知道传的是什么。

**检查点**：能说出 `UserMsg` 里 `input_ids` 的 dtype 和所在设备（CPU）。

---

### Step 4：API Server 前端（异步 + 流式）

**源码**：[python/minisgl/server/api_server.py](python/minisgl/server/api_server.py)

**读什么**：
- `FrontendManager`：`new_user`（分配 `uid`）、`listen`（后台任务收 `UserReply`）、`wait_for_ack`（生成器，逐 token 产出）、`stream_chat_completions`（拼 SSE chunk）。
- `/v1/chat/completions` 端点：`req.stream` 为真/假时两条不同分支。
- `ack_map` / `event_map` 两个 dict 的作用。
- 客户端断开时 `stream_with_cancellation` → `abort_user` 怎么一路把 `AbortMsg` 发下去。

**学什么**：FastAPI 的 `StreamingResponse` + SSE 格式；`asyncio.Event` 做「有结果就唤醒」的跨协程通知。

**动手**：用 `curl ... -d '{"stream":true}'` 抓一次 SSE 原始输出，对照 `stream_chat_completions` 的 chunk 拼装逻辑。

---

## Part B · Tokenizer 与核心数据结构（Step 5–7）

### Step 5：Tokenize 与 Detokenize（文本⇄token）

**源码**：[python/minisgl/tokenizer/tokenize.py](python/minisgl/tokenizer/tokenize.py)、[python/minisgl/tokenizer/detokenize.py](python/minisgl/tokenizer/detokenize.py)、[python/minisgl/tokenizer/__init__.py](python/minisgl/tokenizer/__init__.py)

**读什么**：
- `tokenize.py` 的 `TokenizeManager.tokenize`：`apply_chat_template` 什么时候用、`encode` 怎么转成 int32 张量。
- `detokenize.py` 的 `DetokenizeManager.detokenize`（**重点**）：`DecodeStatus` 里 `read_offset` / `surr_offset` / `sent_offset` 三个偏移量各管什么；`find_printable_text` 怎么处理中文字符和不完整词。
- `__init__.py` 的 `tokenize_worker` 主循环：一个进程如何同时 pull tokenize 和 detokenize 两类消息。

**学什么**：流式 detokenize 的本质——不能等整句再解码，要按「完整词 / 中文字符」边界增量吐字。

**动手**：发一个中英混排 prompt，在 `detokenize` 里打印三个 offset 的变化。

---

### Step 6：核心数据结构 `Req`（全项目最重要的类）

**源码**：[python/minisgl/core.py](python/minisgl/core.py) 的 `Req`

**读什么**：`Req` 的三个长度字段和它们的关系：

| 字段 | 含义 |
|---|---|
| `cached_len` | 已被 KV cache 复用/缓存到的长度 |
| `device_len` | 当前在 GPU 上的长度 |
| `max_device_len` | 总长度上限（input + output） |

- 派生属性：`extend_len = device_len - cached_len`（本次要新算）、`remain_len = max_device_len - device_len`（还能生成多少）。
- `complete_one()`：decode 每步 `device_len += 1`。
- `append_host()`：把新 token 拼到 CPU 侧的 `input_ids`。
- `can_decode` / `__post_init__` 里的断言。

**学什么**：这个三个长度是调度器、缓存、注意力**共用的坐标系**，读懂了它，后面所有 index 计算都迎刃而解。

**检查点**：能手画一个请求从 `cached_len=0, device_len=5` 到 decode 几步之后的长度变化。

---

### Step 7：`Batch` 与 `Context`（全局上下文）

**源码**：[python/minisgl/core.py](python/minisgl/core.py) 的 `Batch` 和 `Context`

**读什么**：
- `Batch`：`reqs` / `phase`（prefill/decode）/ `input_ids` / `positions` / `out_loc` / `padded_reqs` / `attn_metadata`，哪些是构造时给、哪些是 `init=False` 后填。
- `Context`：`page_table` / `attn_backend` / `moe_backend` / `kv_cache` / `_batch`，以及 `forward_batch` 上下文管理器。
- `set_global_ctx` / `get_global_ctx`：为什么要做成全局单例。

**学什么**：模型 `forward()` **不带参数**的秘密——batch 通过全局 `Context` 传递，前向时 `get_global_ctx().batch` 拿输入。

**检查点**：能解释「为什么 `forward_batch` 用 `contextmanager` 而不是直接赋值」。

---

## Part C · 调度器（Step 8–13）

### Step 8：Scheduler 主循环（normal_loop）

**源码**：[python/minisgl/scheduler/scheduler.py](python/minisgl/scheduler/scheduler.py)

**读什么**：
- `Scheduler.__init__`：怎么把 `Engine` + 四个 manager 组装起来（先扫一眼，细节留到 Step 14）。
- `run_forever`：`ENV.DISABLE_OVERLAP_SCHEDULING` 决定走 `normal_loop` 还是 `overlap_loop`。
- `normal_loop` 的四个动作：`receive_msg` → `_schedule_next_batch` → `_forward` → `_process_last_data`。
- `_process_one_msg`：`UserMsg`（进 prefill）/ `AbortBackendMsg`（中止）/ `ExitMsg`（退出）三路分发。
- `_prepare_batch` 和 `_forward`：batch 里的 `positions` / `input_ids` / `out_loc` 是怎么被填上的。

**学什么**：一个调度 step 的完整流水线；`token_pool`（按 table_idx 存的输入 token）的作用。

**检查点**：能默写 `normal_loop` 的控制流，并说明每一步的输入输出。

---

### Step 9：Overlap 调度（核心性能技巧）

**源码**：[python/minisgl/scheduler/scheduler.py](python/minisgl/scheduler/scheduler.py) 的 `overlap_loop` 和 `_process_last_data`

**读什么**：
- `overlap_loop`：为什么「先 `_schedule_next_batch` + `_forward`（在 engine stream 上），再 `_process_last_data(last_data)`」，和 `normal_loop` 顺序相反。
- `self.stream`（CPU 处理流）和 `self.engine.stream`（计算流）两个 CUDA stream 怎么切换、`wait_stream` 干什么。
- `_process_last_data`：拿到 `next_tokens_cpu` 后，怎么 `append_host`、判断 `finished`、构造 `DetokenizeMsg`、`send_result`；`lazy_free_region` 是干嘛的。

**学什么**：把 CPU 上的调度/结果处理，藏进 GPU 前向的阴影里。

**动手**：`MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 开/关各跑一次，对比吞吐。

---

### Step 10：Scheduler 的 I/O 与多卡广播

**源码**：[python/minisgl/scheduler/io.py](python/minisgl/scheduler/io.py)

**读什么**：
- `SchedulerIOMixin.__init__`：rank0 和 rank1 各自创建哪些 ZMQ 队列（`Pull/Push/Pub/Sub`）。
- `_recv_msg_single_rank` vs `_recv_msg_multi_rank0` vs `_recv_msg_multi_rank1`：为什么多卡时要用 `torch.distributed.broadcast` 同步「消息条数」。
- `_reply_tokenizer_rank0` vs `_reply_tokenizer_rank1`（后者 no-op）。

**学什么**：PUB/SUB 广播 + CPU 侧 broadcast 保证「所有 rank 看到同样的请求顺序」（对应 commit `9a91cfa` 修复的 decode 顺序问题）。

**检查点**：能解释 rank0 为什么既要 `put_raw` 广播、又要 CPU broadcast 一个长度张量。

---

### Step 11：PrefillManager 与 Chunked Prefill

**源码**：[python/minisgl/scheduler/prefill.py](python/minisgl/scheduler/prefill.py)

**读什么**：
- `PrefillAdder.try_add_one` / `_try_allocate_one`：准入控制的三个条件（`table` 有槽位、cache 有空间、token_budget 够）。
- `_add_one_req`：`chunk_size = min(token_budget, remain_len)`，`chunk_size < remain_len` 时构造 `ChunkedReq`。
- `ChunkedReq`：`can_decode` 恒 False、`append_host` 抛异常——为什么 chunk 不能采样。
- `schedule_next_batch`：`chunked_list` 和 `pending_list` 怎么重组。

**学什么**：长 prompt 切成多个 batch 处理，避免一次占满显存、也避免长 prefill 饿死 decode。

**动手**：`--max-prefill-length 64` 发一个长 prompt，观察它被切成几个 `ChunkedReq`。

---

### Step 12：DecodeManager 与 TableManager

**源码**：[python/minisgl/scheduler/decode.py](python/minisgl/scheduler/decode.py)、[python/minisgl/scheduler/table.py](python/minisgl/scheduler/table.py)

**读什么**：
- `DecodeManager`：`running_reqs`（正在生成的请求集合）、`filter_reqs`、`schedule_next_batch`（按 `uid` 排序，保证顺序稳定）。
- `inflight_tokens`：为什么除了 `sum(remain_len)` 还要额外 reserve `(page_size-1) * len(reqs)`。
- `TableManager`：`_free_slots` 是 `table_idx` 的池子；`token_pool`（shape 同 page_table）存的是输入 token id。

**学什么**：`table_idx` 是「逻辑请求槽位」，与物理 KV 页通过 page_table 解耦。

**检查点**：能说清 `token_pool` 和 `page_table` 两个张量分别存什么、形状为何一致。

---

### Step 13：CacheManager 分页分配

**源码**：[python/minisgl/scheduler/cache.py](python/minisgl/scheduler/cache.py)

**读什么**：
- `allocate_paged`：对每个 req 算 `first_page`/`last_page`，把新页写到 `page_table[table_idx, position]`。
- `_page_to_token` / `_write_page_table`：page 号怎么映射成连续 token 位置。
- `_allocate`：不够页时先 `evict` 再取。
- `cache_req`：那段长注释画的「合法缓存区 vs 已分配缓存区」边界；`insert_prefix` 之后为什么有两处 `_free`。
- `lazy_free_region`：为什么要把 `_free` 临时换成 `lazy_free` 攒着最后一次性 `cat`。

**学什么**：页表（page table）是 PagedAttention 的核心——逻辑位置→物理页的映射。

**检查点**：能解释 `page_size` 从 1 变 2 时，`allocate_paged` 的行为差异。

---

## Part D · 引擎与模型（Step 14–18）

### Step 14：Engine 初始化（TP worker 的心脏）

**源码**：[python/minisgl/engine/engine.py](python/minisgl/engine/engine.py)

**读什么**：`Engine.__init__` 按顺序做的七件事：
1. `_init_communication`（gloo + pynccl / nccl）
2. `create_model`（**meta device** 建图）
3. `_load_weight_state_dict`（`load_state_dict` 填权重 / dummy 权重）
4. `_determine_num_pages`（按显存算 KV cache 页数）
5. page_table 初始化
6. `create_attention_backend` / `create_moe_backend`
7. `Sampler` + `GraphRunner`（CUDA graph 捕获）

再看 `forward_batch`：`can_use_cuda_graph` 决定 replay 还是正常 forward，然后 `sample` + 异步拷回 CPU。

**学什么**：为什么模型用 `meta` 设备建、再一次性 `load_state_dict`；`dummy_req` 和 dummy page 是干嘛的。

**检查点**：能说出 `_determine_num_pages` 里 `cache_per_page` 是怎么算出来的（key+value × head_dim × kv_heads × page_size × dtype × layers）。

---

### Step 15：模型前向结构（以 Llama 为例）

**源码**：[python/minisgl/models/base.py](python/minisgl/models/base.py)、[python/minisgl/models/llama.py](python/minisgl/models/llama.py)

**读什么**：
- `BaseLLMModel.forward()` **不带参数**——靠全局 ctx。
- `LlamaDecoderLayer.forward`：`input_layernorm → self_attn → post_attention_layernorm → mlp`，注意 `residual` 是怎么在层间传递的（residual stream）。
- `LlamaModel.forward`：`embed_tokens → N 层 → norm`。
- `LlamaForCausalLM.forward`：`lm_head`，`tie_word_embeddings` 时复用 embedding 权重。

**学什么**：一个标准的 decoder-only Transformer；RMSNorm + 残差流的写法。

**动手**：在 `LlamaDecoderLayer.forward` 打印每层输入输出 shape。

---

### Step 16：BaseOP 与权重加载

**源码**：[python/minisgl/layers/base.py](python/minisgl/layers/base.py)

**读什么**：
- `BaseOP`：为什么不用 `nn.Module`？它的 `state_dict` / `load_state_dict` 怎么用 `__dict__` 递归。
- `StateLessOP`：无参数的层（如 RMSNorm 若无可学习参数）。
- `OPList`：把 layer 列表也纳入 state_dict 递归。

**学什么**：这套轻量「类 nn.Module」是模型能 `meta` 建图 + 精确加载权重的关键。

**检查点**：能解释 `load_state_dict` 里 `_internal` 参数和最后 `state_dict` 非空报错的作用。

---

### Step 17：张量并行线性层（TP 精华）

**源码**：[python/minisgl/layers/linear.py](python/minisgl/layers/linear.py)

**读什么**：五种线性层，重点看**哪些 forward 里有 `all_reduce`**：
- `LinearQKVMerged`：Q/K/V 合并成一个矩阵，按列切（每 rank 算一部分 head）。
- `LinearColParallelMerged`：MLP 第一层，按列切。
- `LinearRowParallel` / `LinearOProj`：按行切 + 输出 `all_reduce`。
- `LinearReplicated`：每 rank 都存完整权重（如 embedding）。

**学什么**：TP 的核心公式——按列切的不需要 all-reduce，按行切的需要 all-reduce；`div_even(..., allow_replicate=True)` 什么时候用（KV head 不够分时复制）。

**动手**：`--tp 2` 起模型，在 `LinearRowParallel.forward` 打印 all-reduce 前后 shape。

---

### Step 18：注意力层

**源码**：[python/minisgl/layers/attention.py](python/minisgl/layers/attention.py)

**读什么**：`AttentionLayer.forward`：
1. `qkv.split([qo_attn_dim, kv_attn_dim, kv_attn_dim])`
2. 可选 q_norm / k_norm
3. `rotary.forward(positions, q, k)`（RoPE）
4. `ctx.attn_backend.forward(q, k, v, layer_id, batch)`（真正算注意力的地方）

**学什么**：注意力层只做「切分 + RoPE」，把重活甩给后端（Step 21）。`positions` 从哪来（Step 8 里 `_make_positions`）。

---

## Part E · KV Cache 与 Radix（Step 19–20）

### Step 19：KV Cache 物理存储

**源码**：[python/minisgl/kvcache/base.py](python/minisgl/kvcache/base.py)、[python/minisgl/kvcache/mha_pool.py](python/minisgl/kvcache/mha_pool.py)

**读什么**：
- `BaseKVCachePool` 接口：`k_cache` / `v_cache` / `store_kv`。
- `MHAKVCache`：K/V 各存成一个 `[num_layers, num_pages*page_size, num_kv_heads, head_dim]` 的大张量，`store_kv` 按 `out_loc` 把本层算出的 K/V 写进去。
- `BasePrefixCache` / `BaseCacheHandle`：前缀缓存的接口（match/insert/evict/lock）。

**学什么**：KV cache 就是「按 layer、按 token 位置」存历史 K/V，`store_kv` 由注意力后端在每次 forward 后调用。

**检查点**：能说清 `out_loc` 这个 batch 字段在 `store_kv` 里的作用（指示写到哪里）。

---

### Step 20：Radix Cache 前缀复用

**源码**：[python/minisgl/kvcache/radix_cache.py](python/minisgl/kvcache/radix_cache.py)、[python/minisgl/kvcache/naive_cache.py](python/minisgl/kvcache/naive_cache.py)

**读什么**：
- `RadixTreeNode`：`children`（按 key_fn 分叉）、`_key`/`_value`（token 序列 + 对应 KV 页 index）、`ref_count`、`timestamp`。
- `match_prefix` → `_tree_walk`：沿树找最长公共前缀，必要时 `split_at` 劈开节点。
- `insert_prefix`：把新前缀插进树。
- `evict`：只驱逐 `ref_count == 0` 的叶子节点（timestamp + 最小堆 = LRU 近似）。
- `lock_handle`：沿 parent 链改 `ref_count`，维护 `evictable_size` / `protected_size`。
- 对照 `naive_cache.py`：没有前缀复用的朴素实现，反衬 Radix 的价值。

**学什么**：共享前缀（system prompt / few-shot）的 K/V 只算一次。

**动手**：shell 连续问同一个人设，在 `match_prefix` 打印 `cached_len`，第二次应 > 0；切 `--cache naive` 对比。

---

## Part F · 注意力后端与性能（Step 21–23）

### Step 21：注意力后端接口与实现

**源码**：[python/minisgl/attention/base.py](python/minisgl/attention/base.py)、[python/minisgl/attention/fa.py](python/minisgl/attention/fa.py)、[python/minisgl/attention/fi.py](python/minisgl/attention/fi.py)、[python/minisgl/attention/utils.py](python/minisgl/attention/utils.py)

**读什么**：
- `BaseAttnBackend` 接口：`forward` / `prepare_metadata` / `init_capture_graph` / `prepare_for_capture` / `prepare_for_replay`。
- `HybridBackend`：prefill 和 decode 各用一个后端（`--attn fa,fi`）。
- `fa.py` / `fi.py`：`prepare_metadata` 怎么根据 batch 的 `positions`/`out_loc` 构造给 kernel 的 metadata；`forward` 怎么调 FlashAttention/FlashInfer 并 `store_kv`。

**学什么**：prefill 是「长序列、算力密集」，decode 是「短序列、访存密集」，不同 kernel 各取所长。

**动手**：`--attn fi` 和 `--attn fa,fi` 各跑一次对比。

---

### Step 22：CUDA Graph（decode 加速）

**源码**：[python/minisgl/engine/graph.py](python/minisgl/engine/graph.py)

**读什么**：
- `GraphRunner._capture_graphs`：对每个 batch size 捕获一张图；`torch.cuda.graph(graph, pool=...)` 怎么复用内存池。
- `GraphCaptureBuffer`：静态 buffer（`input_ids`/`out_loc`/`positions`/`logits`），replay 时 `copy_from` 把数据灌进去。
- `pad_batch`：用 `dummy_req` 把真实 batch 补到最近的捕获尺寸。
- `replay`：`copy_from → prepare_for_replay → g.replay() → 取 logits`。
- `_determine_cuda_graph_bs`：按显存自动挑 batch size 列表。

**学什么**：把一长串 kernel 启动「录」成一张图，decode 时一次 replay 替代成千上万次 CPU→GPU launch。

**动手**：`--cuda-graph-max-bs 0` 关掉 CUDA graph，对比 decode 吞吐。

---

### Step 23：采样

**源码**：[python/minisgl/engine/sample.py](python/minisgl/engine/sample.py)

**读什么**：
- `Sampler.prepare`：为什么「全 greedy」的 batch 特判返回 `temperatures=None`。
- `Sampler.sample`：greedy 走 `torch.argmax`；随机采样走 `sample_impl`。
- `sample_impl`：flashinfer 的 `softmax` + `top_k/top_p` 采样 kernel 组合。

**学什么**：logits → 下一个 token 的最后一步；batch 里不同请求可以有不同的采样参数（`temperatures` 是逐请求的张量）。

---

### Step 24：完整生命周期串讲 + 进阶地图

**串讲**（把 24 步连成一条线）：对着 Step 0.3 的 8 步，逐一说出「这一步在哪个进程、哪个函数、传了什么消息、改了什么数据结构」。能一口气讲下来，就说明主线打通了。

**进阶地图**（可选，按兴趣深入）：
- **MoE**：[models/qwen3_moe.py](python/minisgl/models/qwen3_moe.py)、[moe/](python/minisgl/moe/)、[kernel/triton/fused_moe.py](python/minisgl/kernel/triton/fused_moe.py)
- **自定义 CUDA kernel + JIT**：[kernel/](python/minisgl/kernel/)（`tvm-ffi` 绑定、`.cu` 源、`radix.cpp` 的 `fast_compare_key`）
- **通信**：[kernel/pynccl.py](python/minisgl/kernel/pynccl.py)、[distributed/impl.py](python/minisgl/distributed/impl.py)
- **离线 Python 接口**：[llm/llm.py](python/minisgl/llm/llm.py)

---

## 二次开发实战（难度递增，标注对应 Step）

| 项目 | 难度 | 改哪里 | 依赖 Step | 验证 |
|---|---|---|---|---|
| 加日志 / 改默认参数 | L1 | `scheduler.py`、`args.py` | 2、8 | 能读懂日志每个字段 |
| 加 `min_p` 采样参数 | L2 | `core.py`、`sample.py`、`api_server.py` | 6、23、4 | `min_p=0.05` 过滤低概率 token |
| 实现 `stop` 字符串 | L3 | `core.py`、`scheduler.py`、`detokenize.py` | 6、9、5 | `stop:["\n"]` 提前停 |
| decode 优先调度策略 | L4 | `scheduler.py` 的 `_schedule_next_batch` | 8、11、12 | 对比 TTFT vs TPOT |
| 加一个新模型架构 | L5 | `models/` + `register.py` | 15、16 | 新模型跑通、对齐 HF |
| 更聪明的驱逐策略 | L6 | `radix_cache.py` 的 `evict` | 20 | 提升共享前缀命中率 |
| Radix 匹配 benchmark + 微优化 | L7 | `kernel/radix.py`、`kernel/csrc/src/radix.cpp` | 20、进阶 | 给出优化前后耗时 |
| 加一个 attention 后端 | L8 | `attention/` + `engine/config.py` | 21 | 与 `fi`/`fa` 对比延迟 |

---

## 调试与验证技巧

- **最小复现**：先 `--dummy-weight --shell`，逻辑通了再上真实权重。
- **看进程**：`ps aux | grep minisgl` 确认拓扑。
- **加日志**：`logger.info_rank0` / `debug_rank0`（只在 rank0 打，避免多卡刷屏）。
- **profile**：代码里大量 `@nvtx_annotate("...")`，用 `nsys` / Nsight Systems 看每算子耗时。
- **消融开关**：[env.py](python/minisgl/env.py) 的 `MINISGL_*`（如 `MINISGL_DISABLE_OVERLAP_SCHEDULING`）。
- **跑测试**：[tests/](tests/) 下 core/kernel/misc 三组；改完代码先跑相关测试。

---

## 常见坑 / FAQ

- **Windows 直接安装失败？** 正常，依赖 Linux 专属 CUDA kernel，用 WSL2 或 Docker。
- **`--dummy-weight` 是干嘛的？** 随机权重，只测链路不测质量，无需下载模型。
- **`ipc:///tmp/minisgl_...` 是什么？** ZMQ 进程间通信地址，pid 后缀防多实例冲突。
- **`--attn fa,fi` 什么意思？** prefill 用 fa、decode 用 fi；`auto` 按架构自动选。
- **改模型代码输出没变？** 模型继承 `BaseOP` 不是 `nn.Module`，检查新层有没有被 `state_dict`/`load_state_dict` 覆盖。
- **多卡时 rank1 为什么不回结果？** 只有 rank0 对外通信，rank1 只参与计算（`_reply_tokenizer_rank1` 是 no-op）。

---

## 参考资源

**项目内**：[README.md](README.md)、[docs/structures.md](docs/structures.md)、[docs/features.md](docs/features.md)

**项目外**（由浅入深）：
- 推理基础：Hugging Face *How to generate text*；KV cache 图解。
- SGLang 官方：`sgl-project/sglang` + LMSYS 博客（RadixAttention、overlap scheduler 两篇必读）。
- vLLM：*PagedAttention* 论文（连续批处理源头）。
- 论文：Sarathi-Serve（chunked prefill）、NanoFlow（overlap）、FlashAttention 1/2/3、FlashInfer。
- CUDA：官方 Programming Guide 的 stream/event/graph 三章。

---

## 附：模块速查表

| 模块 | 文件 | 职责 | 对应 Step |
|---|---|---|---|
| 入口/装配 | `server/launch.py`、`server/args.py` | 解析参数、拉起子进程 | 1、2 |
| 前端 | `server/api_server.py` | FastAPI + SSE + shell | 4 |
| 消息 | `message/*.py` | 消息类 + 序列化 | 3 |
| 数据模型 | `core.py` | `Req`/`Batch`/`Context` | 6、7 |
| 调度 | `scheduler/scheduler.py` | 主循环、overlap | 8、9 |
| 调度 manager | `scheduler/{prefill,decode,table,cache}.py` | 队列、槽位、分页 | 11–13 |
| 引擎 | `engine/{engine,graph,sample}.py` | 组装、CUDA graph、采样 | 14、22、23 |
| 模型 | `models/*.py` | 各架构 + 权重 | 15 |
| 层 | `layers/*.py` | BaseOP + TP 线性层 + 注意力层 | 16–18 |
| KV Cache | `kvcache/*.py` | KV 池 + Radix/Naive | 19、20 |
| 注意力后端 | `attention/*.py` | fa/fi/trtllm/hybrid | 21 |
| 通信 | `distributed/*.py`、`kernel/pynccl.py` | TP 通信 | 17、进阶 |
| 自定义 kernel | `kernel/*.py`、`kernel/csrc/*` | JIT CUDA kernel | 进阶 |
