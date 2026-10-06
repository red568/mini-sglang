# Step 1：项目入口与总装配线

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 1。
> 核心文件：[python/minisgl/__main__.py](python/minisgl/__main__.py)、[python/minisgl/server/launch.py](python/minisgl/server/launch.py)，以及支撑的 [distributed/info.py](python/minisgl/distributed/info.py)、[utils/logger.py](python/minisgl/utils/logger.py)。
>
> 这一 Step 回答一个问题：**`python -m minisgl` 这一条命令，最后是怎么变成「一堆协作的进程」的？**

---

## 一、这个 Step 要解决什么

在请求还没进来之前，系统必须先「长出来」。本 Step 只关心**进程是怎么被创建和启动的**，不关心请求进来后怎么处理（那是 Step 2~10 的事）。

一句话主线：`__main__.py` → `launch_server()` → 解析参数 → `start_subprocess()` 拉起所有子进程 → 等它们就绪 → 主进程开始跑 FastAPI。

---

## 二、核心逻辑（按执行顺序读）

### 2.0 全景：一张图看懂启动调用链

```
                    python -m minisgl
                          │
                          ▼
        ┌────────────────────────────────────────┐
        │  __main__.py                           │
        │  assert __name__ == "__main__"         │
        │  launch_server()                       │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │  launch_server()           [launch.py] │
        │  ① parse_args() → server_args          │
        │     （tp_info 暂为 rank 0）              │
        │  ② 定义 start_subprocess() 闭包（不执行） │
        │  ③ run_api_server(args, 回调)           │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │  run_api_server()      [api_server.py] │
        │  先创建 FrontendManager（bind ZMQ 端点） │
        │  再调用 start_backend()                │
        └───────────────────┬────────────────────┘
                            │
                            ▼
              start_subprocess()   ← 真正 spawn 的地方
```

### 2.1 入口链：`__main__.py`

```python
from .server import launch_server

assert __name__ == "__main__"

launch_server()
```

- 只有三行。`python -m minisgl` 时，Python 把 `minisgl/__main__.py` 当作 `__main__` 模块执行。
- `assert __name__ == "__main__"` 是一个**防御性断言**：这个文件只允许作为入口被执行，不允许被别人 `import`（否则 `__name__` 是 `minisgl.__main__`，直接报错）。

### 2.2 `launch_server()` 总流程（[launch.py](python/minisgl/server/launch.py)）

```python
def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)   # ① 解析 CLI
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:                                 # ② 定义（还没执行！）
        ...

    run_api_server(server_args, start_subprocess, run_shell=run_shell)  # ③ 真正启动
```

三个关键点：

1. **`parse_args` 返回的 `server_args` 里，`tp_info` 暂时是 `DistributedInfo(0, tp_size)`**（即 rank0）。真正的 rank 是后面 `spawn` 时用 `replace` 逐个注入的。
2. **`start_subprocess` 只是一个闭包，此刻还没执行**。真正 spawn 子进程发生在 `run_api_server` 内部。
3. **函数内 import**（`from .api_server import ...`）是刻意的：把重模块的加载推迟到真正需要时，避免模块被 import 时就拉起一堆依赖（`torch`、`scheduler` 等都在各自的函数内才 import）。

### 2.3 `start_subprocess()` 进程装配（[launch.py](python/minisgl/server/launch.py) 核心）

```python
def start_subprocess() -> None:
    import multiprocessing as mp
    from minisgl.tokenizer import tokenize_worker

    mp.set_start_method("spawn", force=True)          # ① 必须用 spawn

    world_size = server_args.tp_info.size
    ack_queue: mp.Queue[str] = mp.Queue()              # ② 就绪确认队列

    for i in range(world_size):                        # ③ 每个 GPU 一个 Scheduler
        new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
        mp.Process(target=_run_scheduler, args=(new_args, ack_queue),
                   daemon=False, name=f"minisgl-TP{i}-scheduler").start()

    num_tokenizers = server_args.num_tokenizer
    mp.Process(target=tokenize_worker, kwargs={...},    # ④ detokenizer（1 个）
               name="minisgl-detokenizer-0").start()
    for i in range(num_tokenizers):                     # ⑤ tokenizer（n 个，默认 0）
        mp.Process(target=tokenize_worker, kwargs={...},
                   name=f"minisgl-tokenizer-{i}").start()

    for _ in range(num_tokenizers + 2):                 # ⑥ 等所有子进程就绪
        logger.info(ack_queue.get())
```

`world_size` = `--tp` 的值。默认 `--tp 1` 时只 spawn 1 个 scheduler；`--tp 4` 就 spawn 4 个。

**装配流程一览**（对应上面代码的 ①~⑥）：

```
start_subprocess()
      │
      ├─ ① mp.set_start_method("spawn", force=True)   ← 必须最先，且必须 spawn
      │
      ├─ ② ack_queue = mp.Queue()                     ← 就绪确认队列
      │
      ├─ ③ for i in range(world_size):                ← 每个 GPU 一个
      │        new_args = replace(server_args, tp_info=DistributedInfo(i, N))
      │        mp.Process(target=_run_scheduler, args=(new_args, ack_queue))
      │        → 进程名 minisgl-TP{i}-scheduler
      │
      ├─ ④ mp.Process(target=tokenize_worker, kwargs={...})  ← detokenizer，1 个
      │        → 进程名 minisgl-detokenizer-0
      │
      ├─ ⑤ for i in range(num_tokenizers):            ← tokenizer，默认 0 个
      │        mp.Process(target=tokenize_worker, kwargs={...})
      │        → 进程名 minisgl-tokenizer-{i}
      │
      └─ ⑥ for _ in range(num_tokenizers + 2):        ← 阻塞等就绪
               ack_queue.get()   （收齐 N+2 条 "ready" 才继续）
```

### 2.4 `_run_scheduler()` 单个 Scheduler 进程的一生（[launch.py](python/minisgl/server/launch.py)）

```python
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)          # 初始化（Step 14 再展开）
        scheduler.sync_all_ranks()           # 所有 rank 同步

        if args.tp_info.is_primary():        # 只有 rank0 上报就绪
            ack_queue.put("Scheduler is ready")

        try:
            scheduler.run_forever()          # 死循环：收消息→调度→前向
        except KeyboardInterrupt:
            scheduler.shutdown()
```

- 每个 rank 进程都独立跑这段，各自初始化自己的 `Scheduler`（含模型、KV cache、GPU 资源）。
- `run_forever()` 是无限循环，进程不会自然结束，靠 `KeyboardInterrupt` 触发优雅退出。

---

## 三、难点解析（每个都值得想明白「为什么」）

### 难点 1：为什么必须 `mp.set_start_method("spawn", force=True)`？

`multiprocessing` 在 Linux 上默认用 **fork**：子进程直接复制父进程的内存。这在 CUDA 程序里是致命的：

- CUDA 上下文、GPU 显存、以及各种驱动锁的状态都会被复制一份，子进程里继续用会**报错或死锁**。
- fork 出来的进程和父进程共享同一个 CUDA 上下文，两个进程同时往 GPU 上跑会互相踩。

**spawn** 则是启动一个全新的 Python 解释器，重新 import 模块、重新初始化一切，CUDA 状态干净。代价是启动慢、且 `target` 函数必须能被 pickle。

> `force=True` 表示「覆盖已经设置过的启动方式」，保证不管前面有没有人调过 `set_start_method`，这里一定是 spawn。

### 难点 2：`dataclasses.replace` 怎么给每个 rank 注入不同的身份？

`ServerArgs` 是 `@dataclass(frozen=True)`（[args.py](python/minisgl/server/args.py)），frozen 意味着字段不可直接改。

```python
new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
```

`replace` 创建一个**新对象**，只替换 `tp_info` 这一个字段，其余字段原样复制。于是同一个 `server_args` 模板被复制出 `world_size` 份，每份的 `tp_info.rank` 不同（`i = 0, 1, 2, ...`）。

这样每个 scheduler 进程都知道「我是第几个 rank、总共有几个 rank」。

```
          server_args（模板，frozen dataclass）
              tp_info = DistributedInfo(rank=0, size=N)
                         │
       ┌─────────────────┼─────────────────┐
       │ replace(...)    │ replace(...)    │ replace(...)
       ▼                 ▼                 ▼
 DistributedInfo   DistributedInfo   DistributedInfo
 (rank=0,size=N)   (rank=1,size=N)   (rank=2,size=N)
       │                 │                 │
       ▼                 ▼                 ▼
 Scheduler TP0     Scheduler TP1     Scheduler TP2
 （进程名带 0）      （进程名带 1）      （进程名带 2）
```

### 难点 3：ack 计数为什么是 `num_tokenizers + 2`？

这是最容易懵的一行。逐个进程数「谁会 ack」：

| 进程 | 数量 | 谁 ack |
|---|---|---|
| Scheduler | world_size 个 | **只有 rank0** ack（`is_primary()` 才 `put`） |
| tokenizer | num_tokenizers 个 | 每个都 ack |
| detokenizer | 1 个 | ack |

所以总 ack 数 = `1 (rank0) + num_tokenizers + 1 (detokenizer) = num_tokenizers + 2`。

验证：默认 `--num-tokenizer 0` 时，tokenizer/detokenizer 合并成一个进程，此时 ack = `1 (rank0) + 0 + 1 (detokenizer) = 2`，正好等于 `0 + 2`。✓

主进程靠 `ack_queue.get()` **阻塞等待**，收够 `num_tokenizers + 2` 条「xx is ready」才继续，保证所有子进程都起来了才对外服务。

### 难点 4：为什么只有 rank0 ack，却能让主进程确认「所有 rank 就绪」？

看 `_run_scheduler` 里的顺序：

```python
scheduler = Scheduler(args)
scheduler.sync_all_ranks()      # ← 关键
if args.tp_info.is_primary():
    ack_queue.put("Scheduler is ready")
```

`scheduler.sync_all_ranks()` 内部是 `tp_cpu_group.barrier()`（CPU 侧的集合通信，见 [scheduler/io.py](python/minisgl/scheduler/io.py)），这是一个**全体同步点**：rank0 只有等到 rank1、rank2…… 全都到达 barrier，它自己才会通过 barrier。

所以当 rank0 走到 `ack_queue.put(...)` 这一行时，**必然意味着其他所有 rank 也都已经初始化完了**。rank0 的 ack 就等价于「全员就绪」。

> 这也是为什么多卡时只让 rank0 对外、其他 rank 沉默——用一次 barrier + 一次 ack 就完成了全局同步。

**就绪握手时序图**：

```
 主进程          Scheduler r0       Scheduler r1..N      tokenizer/detok
   │                 │                   │                    │
   │  spawn ────────►│  spawn ─────────►│  spawn ───────────►│
   │                 │                   │                    │
   │                 │ Scheduler(args)   │ Scheduler(args)    │ load_tokenizer
   │                 │ sync_all_ranks()  │ sync_all_ranks()   │
   │                 │◄──── barrier ────►│                    │
   │                 │                   │                    │
   │                 │ ack "ready"       │ （不 ack）          │ ack "ready"
   │◄────────────────┴───────────────────┴────────────────────┤
   │      ack_queue.get() 收够 num_tokenizers+2 条才继续        │
   │                 │                   │                    │
   │ 主进程开始跑 FastAPI，对外服务                               │
```

### 难点 5：为什么真正的 spawn 放在 `run_api_server` 里（回调），而不是 `launch_server` 直接做？

顺序是 ZMQ 的关键。ZMQ 里：

- `bind` = 谁创建端点（服务端，必须先就位）。
- `connect` = 谁去连（客户端，连不上时会静默排队）。

[api_server.py](python/minisgl/server/api_server.py) 的 `run_api_server` 会**先**创建 `FrontendManager`（里面 bind 了 `zmq_frontend_addr` 等端点），**然后**才调用 `start_backend()` 去 spawn 子进程。这样能保证：子进程起来后去 `connect` 这些端点时，主进程已经在 `bind` 监听了，不会出现「connect 到不存在的端点」。

把 `start_subprocess` 作为回调传进去，正是为了控制这个先后顺序。

### 难点 6：函数内 import 是为了什么？

注意 [launch.py](python/minisgl/server/launch.py) 顶部只 `import` 了轻量模块（`multiprocessing`、`sys`、`dataclasses`），而 `torch`、`Scheduler`、`tokenize_worker` 都是在**函数体内部**才 import：

- **加快模块加载**：`import minisgl.server.launch` 时不会连带拉起 torch 这种大依赖。
- **避免循环 import**：`launch` ↔ `scheduler` ↔ `api_server` 之间如果都在顶层互相 import，容易形成环；延迟到函数内能解开。
- **spawn 友好**：子进程重新 import 主模块时，只执行顶层语句，重的东西不会被无谓加载。

---

## 四、注意事项（容易踩的坑）

1. **`set_start_method("spawn", force=True)` 必须在创建任何子进程之前调用**，且要放在会真正 spawn 的那段代码里（这里是 `start_subprocess`）。
2. **ack 数量必须精确**。少 ack → 主进程永久卡在 `ack_queue.get()`；多 ack → 队列里残留消息，逻辑错乱。改动子进程时最容易忘改这里的计数。
3. **子进程的 target 必须是模块级顶层函数**（`_run_scheduler`、`tokenize_worker`），不能是 lambda 或嵌套闭包——spawn 要通过 pickle 把 target 传过去。
4. **`daemon=False`**（默认值，这里显式写出是强调）：子进程不是守护进程，主进程不会自动清理它们。所以 shell 模式结束时 [api_server.py](python/minisgl/server/api_server.py) 里要显式用 `psutil` 杀掉所有子进程。
5. **日志里的 rank 前缀有「时间差」**：主进程（`launch_server` 阶段）还没 `set_tp_info`，所以日志不带 rank；子进程在 `Engine.__init__` 里 `set_tp_info` 之后，日志才会带上 `|core|rank=N` 前缀（见 [utils/logger.py](python/minisgl/utils/logger.py) 的 `try_get_tp_info`）。
6. **`--shell` 会改参数**：`parse_args` 里 shell 模式会强制 `cuda_graph_max_bs=1`、`max_running_req=1`、`silent_output=True`，别在 shell 下测吞吐。

---

## 五、反思题（先自己想，再回源码验证）

1. 把 `mp.set_start_method("spawn")` 换成 `"fork"`，跑一个带 CUDA 的模型会怎样？从「CUDA 上下文能不能跨进程复制」的角度解释为什么 spawn 是必须的。
2. 手推 `num_tokenizers + 2` 这个 ack 计数：假设 `--tp 4 --num-tokenizer 2`，一共会有几个进程、几个 ack、主进程要 `get()` 几次？
3. 为什么 rank0 的 ack 能代表「所有 rank 都就绪」？如果删掉 `scheduler.sync_all_ranks()` 这一行，会发生什么？
4. `ServerArgs` 是 `frozen=True`，为什么不能直接 `server_args.tp_info = ...`？`replace` 在背后做了什么？
5. `start_subprocess` 为什么被定义成闭包、再作为回调传给 `run_api_server`？如果直接在 `launch_server` 里 spawn，ZMQ 的 bind/connect 顺序会出什么问题？
6. `__main__.py` 里是 `assert __name__ == "__main__"` 而不是 `if __name__ == "__main__":`，两者语义差别在哪？这个 assert 在 spawn 子进程时会被触发吗？（提示：spawn 重新 import 的是 `minisgl` 包，不是把 `__main__.py` 再执行一遍）
7. `_run_scheduler` 里 `with torch.inference_mode():` 包住整个 scheduler 生命周期，目的是什么？（提示：推理不需要 autograd）

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [python/minisgl/__main__.py](python/minisgl/__main__.py) | 入口 | `launch_server` |
| [python/minisgl/server/launch.py](python/minisgl/server/launch.py) | 总装配线 | `launch_server`、`start_subprocess`、`_run_scheduler` |
| [python/minisgl/server/args.py](python/minisgl/server/args.py) | 参数解析 | `parse_args`、`ServerArgs` |
| [python/minisgl/distributed/info.py](python/minisgl/distributed/info.py) | rank 身份 | `DistributedInfo`、`set_tp_info`、`is_primary` |
| [python/minisgl/utils/logger.py](python/minisgl/utils/logger.py) | 日志 + rank0 后缀 | `init_logger`、`info_rank0` |
| [python/minisgl/server/api_server.py](python/minisgl/server/api_server.py) | 触发 spawn 的前端 | `run_api_server` |

**下一步**：进入 Step 2（CLI 参数与三层 Config 继承），弄懂 `ServerArgs → SchedulerConfig → EngineConfig` 的字段是怎么一层层传递到每个进程的。
