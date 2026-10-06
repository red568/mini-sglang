# Step 7：`Batch` 与 `Context`（全局上下文）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 7。
> 核心文件：[core.py](python/minisgl/core.py) 的 `Batch`、`Context` 和 `set_global_ctx` / `get_global_ctx`。
>
> 这一 Step 回答：**多个 `Req` 怎么打包成一个 batch 一起算？模型 `forward()` 为什么可以不带任何参数？**

---

## 一、这个 Step 要解决什么

上一 Step 讲了单个 `Req`。实际推理时是「一批请求一起喂给 GPU」。`Batch` 就是这个「打包容器」；而 `Context` 是「全局单例」，让模型和注意力后端不用层层传参，直接从全局拿当前 batch。

---

## 二、核心逻辑

### 2.1 `Batch`：一个打包容器

```python
@dataclass
class Batch:
    reqs: List[Req]                    # 本批的请求（真实请求）
    phase: Literal["prefill", "decode"]
    input_ids: torch.Tensor = field(init=False)   # 下面这些字段由调度器/后端填
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    attn_metadata: BaseAttnMetadata = field(init=False)
```

关键：`reqs` 和 `phase` 是构造时给的，其余都是 `init=False` 的「延迟填充」字段——它们由不同的人在不同阶段填：

| 字段 | 谁来填 | 填什么 |
|---|---|---|
| `input_ids` | Scheduler（`_forward`） | 从 `token_pool` 取的本批输入 token |
| `positions` | Scheduler（`_make_positions`） | 每个 token 的位置编号（RoPE 用） |
| `out_loc` | Scheduler（`_prepare_batch`） | 每个 token 要写入 KV cache 的物理位置 |
| `padded_reqs` | `GraphRunner.pad_batch` | 补齐到 CUDA graph 尺寸后的请求列表（含 dummy） |
| `attn_metadata` | 注意力后端 `prepare_metadata` | 后端专用元数据 |

派生属性：`is_prefill` / `is_decode` / `size`（= `len(reqs)`）/ `padded_size`（= `len(padded_reqs)`）。

### 2.2 `Context`：全局单例

```python
@dataclass
class Context:
    page_size: int
    page_table: torch.Tensor = field(init=False)    # 逻辑位置 → 物理 KV 位置
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)
```

`Context` 把「一次推理需要的所有全局资源」都挂在一起：页表、注意力后端、MoE 后端、KV cache 池，以及**当前正在 forward 的 batch**。

### 2.3 全局访问机制

```python
_GLOBAL_CTX: Context | None = None

def set_global_ctx(ctx):   # 只能设一次
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx

def get_global_ctx():      # 任何地方都能拿
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
```

`Engine.__init__` 里 `set_global_ctx(self.ctx)`，之后模型、注意力层、KV cache 都能 `get_global_ctx().batch` 拿到当前 batch——这就是模型 `forward()` 不带参数的秘密。

---

## 三、难点解析

### 难点 1：为什么模型 `forward()` 不带参数？（全局 ctx 的设计动机）

正常 PyTorch 模型是 `model(input_ids)` 显式传参。但这里模型层数深（几十层），如果每层都显式传 batch/positions/out_loc，签名会非常啰嗦，而且这些信息其实是「同一时刻全局唯一」的。

于是设计成：**同一时刻只有一个 batch 在 forward**，把它放进全局 `Context`，谁需要谁 `get_global_ctx().batch` 拿。代价是「全局单例」——不能同时 forward 两个 batch（但推理本来就是串行的，无所谓）。

### 难点 2：`forward_batch` 为什么要用 `contextmanager` 而不是直接赋值？

```python
@contextmanager
def forward_batch(self, batch):
    assert self._batch is None, "Nested forward_batch is not allowed"
    try:
        self._batch = batch
        yield
    finally:
        self._batch = None
```

- `assert self._batch is None` 防止**嵌套 forward**（前向里又触发前向，说明有 bug）。
- `finally: self._batch = None` 保证**异常时也能清理**，不会把上一个 batch 残留到下一次。

直接 `self._batch = batch; ...; self._batch = None` 如果中间抛异常，`_batch` 就永远脏了。contextmanager 用 try/finally 兜住了。

### 难点 3：`init=False` 字段的意义

`field(init=False)` 表示这个字段**不参与 `__init__` 构造**。这样 `Batch(reqs=..., phase=...)` 时不用传这些还没算出来的字段，它们由后续流程逐步填充。

这是 dataclass 里「声明了字段、但延迟赋值」的惯用法，比「先 `= None` 再改」更明确表达「这个字段一定会被填，只是不是构造时填」。

### 难点 4：`padded_reqs` 和 `reqs` 的区别

- `reqs`：真实的、需要采样的请求。
- `padded_reqs`：为了适配 CUDA graph 固定 batch size，在 `reqs` 后面补了 `dummy_req` 的列表（Step 22）。

前向时用 `padded_reqs`（GPU 上按固定尺寸算），但采样只取 `batch.size`（= `len(reqs)`）个真实结果，丢弃 dummy 部分。

---

## 四、注意事项

1. **全局 ctx 只能 set 一次**：`set_global_ctx` 有 `assert _GLOBAL_CTX is None`，重复设置会报错。一个进程里只有一个 `Engine`、一个 `Context`。
2. **`get_global_ctx` 在没 set 前调用会报错**：所以主进程（还没建 Engine）里不能碰 `get_global_ctx`。
3. **`batch.input_ids` 在 `_forward` 里才被赋值**：调度器准备 batch 时，`input_ids` 还没填，要等 `_forward` 从 `token_pool` 取。
4. **`attn_metadata` 依赖 `prepare_metadata` 先执行**：在 `_prepare_batch` 最后一步，注意力后端才会读 batch 的 positions/out_loc 生成元数据。

---

## 五、反思题

1. 全局 ctx 的「同一时刻只有一个 batch」是前提假设。如果未来要支持「同时 forward 两个不同 batch」（比如 pipeline 并行），这个设计会怎么被打破？
2. `forward_batch` 的 `assert self._batch is None` 能捕获哪类 bug？给出一个会触发它的错误场景。
3. 如果把 `field(init=False)` 改成 `field(default=None, init=False)`，行为有区别吗？为什么这里能不加 default？
4. `Batch.size` 和 `padded_size` 什么时候相等、什么时候不等？
5. 模型层里 `get_global_ctx().batch.input_ids` 这么拿输入，和显式传参相比，各有什么优缺点？

---

## 六、示意图

### 6.1 `Batch` 字段的填充时序

```
 构造 Batch(reqs, phase)
        │
        ▼
 [Scheduler._prepare_batch]
   1. graph_runner.pad_batch(batch)  → 填 padded_reqs
   2. cache_manager.allocate_paged    → 分配 KV 页
   3. _make_positions                 → 填 positions
   4. _make_input_tuple               → 算 input_mapping
   5. batch.out_loc = page_table[...] → 填 out_loc
   6. attn_backend.prepare_metadata   → 填 attn_metadata
        │
        ▼
 [Scheduler._forward]
   batch.input_ids = token_pool[...]  → 填 input_ids
        │
        ▼
 [Engine.forward_batch → model.forward]
   读 batch.input_ids / positions / out_loc / attn_metadata
```

### 6.2 全局 Context 的访问路径

```
 Engine.__init__ ── set_global_ctx(ctx) ──► _GLOBAL_CTX（全局单例）
                                                 ▲
        ┌──────────────┬──────────────┬──────────┴─────────┐
        │              │              │                    │
   model.forward   AttentionLayer  KV Cache 后端    Sampler
   (拿 input_ids)  (拿 batch/位置)  (拿 kv_cache)  (拿 logits 后续)
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [core.py](python/minisgl/core.py) | `Batch` + `Context` + 全局访问 | `Batch`、`Context`、`forward_batch`、`get_global_ctx` |

**下一步**：进入 Step 8（Scheduler 主循环），看这些数据结构在 `normal_loop` 里是怎么被编排起来、一轮一轮跑起来的。
