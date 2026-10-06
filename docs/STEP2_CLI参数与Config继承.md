# Step 2：CLI 参数与三层 Config 继承

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 2。
> 核心文件：[server/args.py](python/minisgl/server/args.py)、[scheduler/config.py](python/minisgl/scheduler/config.py)、[engine/config.py](python/minisgl/engine/config.py)、[models/config.py](python/minisgl/models/config.py)。
>
> 这一 Step 回答：**命令行参数是怎么变成每个进程手里的配置对象的？**

---

## 一、这个 Step 要解决什么

Step 1 里 `launch_server` 第一行是 `parse_args(sys.argv[1:], run_shell)`。本 Step 就是拆这条链：从「用户在命令行敲的字符串」到「子进程读到的 `ServerArgs` 对象」，中间经过了哪些变换，以及配置对象为什么分成三层继承。

---

## 二、核心逻辑

### 2.1 三层 Config 继承关系

```
EngineConfig       (最底层，模型/引擎通用字段)
    ▲
SchedulerConfig    (加调度字段 + ZMQ 地址)
    ▲
ServerArgs         (加 HTTP 服务字段)
```

- [engine/config.py](python/minisgl/engine/config.py)：`model_path`、`dtype`、`tp_info`、`page_size`、`memory_ratio`、`max_running_req`、`attention_backend`、`moe_backend`、`cuda_graph_bs` 等 —— 是「一个 TP worker 的引擎」需要的所有配置。
- [scheduler/config.py](python/minisgl/scheduler/config.py)：`max_extend_tokens`（chunked prefill 预算）、`cache_type`、`offline_mode`，以及 `zmq_backend_addr` / `zmq_detokenizer_addr` / `zmq_scheduler_broadcast_addr` 三个 ZMQ 地址 property。
- [server/args.py](python/minisgl/server/args.py)：`server_host`、`server_port`、`num_tokenizer`、`silent_output`，以及 `zmq_frontend_addr` / `zmq_tokenizer_addr` 两个地址 property。

> 三层都标 `@dataclass(frozen=True)`。frozen 意味着创建后字段不可改（`Step 1` 里 `replace` 就是为这个存在的）。

### 2.2 `parse_args` 的处理流程

```
命令行字符串 → argparse 解析成 kwargs dict
      │
      ├─ ① 处理 --shell-mode：run_shell |= True，且强制
      │      cuda_graph_max_bs=1, max_running_req=1, silent_output=True
      │
      ├─ ② model_path 展开 "~"
      │
      ├─ ③ --model-source modelscope 时 snapshot_download 下载
      │
      ├─ ④ --dtype auto：从 HF config 读原始 dtype 字符串
      │
      ├─ ⑤ DTYPE_MAP 把字符串映射成 torch.dtype
      │
      ├─ ⑥ --tensor-parallel-size → tp_info = DistributedInfo(0, size)
      │
      └─ ⑦ ServerArgs(**kwargs)  构造最终对象
```

关键点：第 ⑥ 步 `tp_info` 先固定成 rank0，真正的 rank 由 `launch.py` 在 spawn 时用 `replace` 逐个注入（见 Step 1）。

### 2.3 从 HF 配置到 `ModelConfig`

[engine/config.py](python/minisgl/engine/config.py) 里有三个 `cached_property`：

```python
@cached_property
def hf_config(self):          # 加载 HuggingFace 的 PretrainedConfig（读 JSON/下载，慢）
    return cached_load_hf_config(self.model_path)

@cached_property
def model_config(self):       # 转成项目自己的 ModelConfig
    return ModelConfig.from_hf(self.hf_config)
```

[models/config.py](python/minisgl/models/config.py) 的 `ModelConfig.from_hf` 做的是「字段翻译」：把 HF 里各种命名不统一的字段，统一成项目需要的固定字段：

- `num_kv_heads`：GQA 时可能叫 `num_key_value_heads`，兜底用 `num_attention_heads`。
- `head_dim`：有的模型直接给，否则 `hidden_size // num_attention_heads`。
- `rope_theta`：Llama/Qwen 是直接属性，Mistral 藏在 `rope_scaling` 字典里。
- `num_experts` / `num_experts_per_tok` / `moe_intermediate_size`：MoE 字段，非 MoE 模型兜底为 0。
- `text_config`：多模态模型配置嵌套在 `text_config` 里，需要先解包。

---

## 三、难点解析

### 难点 1：为什么用 dataclass 继承，而不是一个 dict 或普通类？

- **类型安全 + IDE 补全**：字段是显式声明的，写错字段名立刻报错。
- **`frozen=True`**：配置是「创建后不可变」的，避免某处代码偷偷改配置导致多进程间状态不一致。
- **继承让字段分层清晰**：引擎层只管引擎、调度层只管调度、服务层只管服务，各层职责分离。

### 难点 2：`cached_property` 在 frozen dataclass 上为什么能工作？

frozen dataclass 会覆盖 `__setattr__` 让它抛异常。但 `functools.cached_property` 的实现是直接写 `instance.__dict__[name] = value`，**绕过了 `__setattr__`**，所以能正常缓存。

这正是它和普通 `@property` 的区别：

| | `@property` | `@cached_property` |
|---|---|---|
| 每次访问 | 重新计算 | 只算一次，结果缓存进 `__dict__` |
| 适用场景 | 便宜的计算（如 `max_seq_len`） | 贵的计算（如加载 HF config） |

`hf_config` 要读磁盘/下载，很慢，所以用 `cached_property` 只算一次。

### 难点 3：`from_hf` 里的「字段兜底」逻辑为什么这么绕？

因为不同模型架构（Llama / Qwen / Mistral / 多模态）的 HF 配置字段名不一致。`from_hf` 用大量 `getattr(config, name, default)` 做兼容，把「五花八门的输入」归一成「项目内部固定的 `ModelConfig`」。

这是真实工程里最典型的「适配层」：对外兼容各种输入，对内统一格式。

### 难点 4：`tp_info` 为什么初始是 rank0，后面再 `replace`？

`parse_args` 阶段只有一个进程（主进程），还不知道自己会是几个 rank、是第几个 rank。所以先给一个占位的 `DistributedInfo(0, size)`，等 `launch.py` 真正 spawn 时，再给每个进程 `replace` 成各自的 rank。这是「配置模板 + 运行时注入」的模式。

---

## 四、注意事项

1. **`frozen=True` 的配置不能用 `obj.field = x` 修改**，要用 `dataclasses.replace` 或 `object.__setattr__`（后者在 [engine.py](python/minisgl/engine/engine.py) 的 `_adjust_config` 里用了，注释还特意标了「dangerous, use with caution」）。
2. **`--shell` 会静默改参数**：`cuda_graph_max_bs=1`、`max_running_req=1`、`silent_output=True`。在 shell 下测性能毫无意义。
3. **`--dtype auto` 依赖网络/本地 HF config**：读不到会直接报错。
4. **ZMQ 地址带 pid 后缀**（`scheduler/config.py` 的 `_unique_suffix`），保证同一个机器上同时跑多个实例互不干扰。
5. **`--model-source modelscope` 只在 modelscope 且本地无目录时才下载**，`use_dummy_weight` 时会忽略权重文件加速下载。

---

## 五、反思题

1. 三层 config 为什么是「继承」而不是「组合」（比如 `ServerArgs` 里放一个 `EngineConfig` 字段）？各自有什么取舍？
2. `cached_property` 的结果存在哪？如果换成普通 `@property`，程序会多做什么工作？
3. `from_hf` 里 `rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]`，如果两个都没有会怎样？这反映了什么隐患？
4. `parse_args` 里为什么把 `--tensor-parallel-size` 转成 `tp_info` 后要 `del kwargs["tensor_parallel_size"]`？
5. `ServerArgs(**kwargs)` 如果 kwargs 里混进一个不认识的 key 会怎样？（提示：dataclass `__init__` 不接受未知参数）

---

## 六、示意图

### 6.1 三层 Config 字段归属

```
┌─────────────────────────────────────────────────────────────┐
│ ServerArgs  (server/args.py)                                │
│   server_host, server_port, num_tokenizer, silent_output    │
│   + property: zmq_frontend_addr, zmq_tokenizer_addr          │
│   └── 继承 ──►                                               │
│        ┌──────────────────────────────────────────────────┐  │
│        │ SchedulerConfig  (scheduler/config.py)           │  │
│        │   max_extend_tokens, cache_type, offline_mode    │  │
│        │   + property: zmq_backend_addr, zmq_detokenizer… │  │
│        │   └── 继承 ──►                                   │  │
│        │        ┌──────────────────────────────────────┐  │  │
│        │        │ EngineConfig  (engine/config.py)     │  │  │
│        │        │   model_path, dtype, tp_info,        │  │  │
│        │        │   page_size, attention_backend, …    │  │  │
│        │        │   + cached_property: hf_config,      │  │  │
│        │        │     model_config                      │  │  │
│        │        └──────────────────────────────────────┘  │  │
│        └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 配置从命令行到进程的流转

```
命令行 --tp 4 --model xxx --dtype auto ...
        │
        ▼
 parse_args()  →  ServerArgs(tp_info=DistributedInfo(0,4), ...)
        │
        │  launch.py 里循环 replace
        ▼
 ┌──────────────┬──────────────┬──────────────┬──────────────┐
 │ rank=0,size=4│ rank=1,size=4│ rank=2,size=4│ rank=3,size=4│
 │ Scheduler TP0│ Scheduler TP1│ Scheduler TP2│ Scheduler TP3│
 └──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [server/args.py](python/minisgl/server/args.py) | CLI 解析 + `ServerArgs` | `parse_args` |
| [scheduler/config.py](python/minisgl/scheduler/config.py) | `SchedulerConfig` + ZMQ 地址 | `SchedulerConfig` |
| [engine/config.py](python/minisgl/engine/config.py) | `EngineConfig` + cached_property | `EngineConfig`、`model_config` |
| [models/config.py](python/minisgl/models/config.py) | HF config → `ModelConfig` | `ModelConfig.from_hf` |

**下一步**：进入 Step 3（消息系统与轻量序列化），看这些配置和请求是怎么被打包成消息、跨进程传递的。
