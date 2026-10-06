# Step 6：核心数据结构 `Req`

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 6。
> 核心文件：[core.py](python/minisgl/core.py) 的 `SamplingParams` 和 `Req`。
>
> 这一 Step 回答：**一个请求进入 Scheduler 后，它的一生（还剩多少要算、已经缓存了多少、还能生成多少）是怎么被精确描述的？**

---

## 一、这个 Step 要解决什么

`Req` 是**全项目最重要的数据结构**——调度器、KV cache、注意力、采样全都围绕它运转。尤其是它的三个长度字段，构成了理解所有 index 计算的「坐标系」。这个 Step 务必吃透。

---

## 二、核心逻辑

### 2.1 `SamplingParams`（用户想要什么）

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

`is_greedy` 判断是不是「贪心采样」：温度 ≤ 0 或 top_k=1，且 top_p=1。贪心时后面采样走 `argmax`（Step 23 展开）。

### 2.2 `Req` 的三个长度字段（题眼）

```python
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor      # CPU tensor，输入 token
    table_idx: int               # 逻辑槽位（见 Step 12）
    cached_len: int              # 已被 KV cache 复用/缓存到的长度
    output_len: int              # 还要生成多少个 token
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self):
        self.device_len = len(self.input_ids)         # 当前在 GPU 上的长度
        self.max_device_len = len(self.input_ids) + self.output_len  # 上限
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
```

三个长度的含义：

| 字段 | 含义 | 谁在变 |
|---|---|---|
| `cached_len` | 前缀里已被 KV cache 缓存（可复用）的长度 | prefill 缓存、complete_one 时变 |
| `device_len` | 当前在 GPU 上的 token 总数 | 每生成一个 token +1 |
| `max_device_len` | 总长度上限 = 输入 + 输出 | 创建后不变 |

派生属性（这是理解一切的关键）：

```python
@property
def remain_len(self):   # 还能生成多少
    return self.max_device_len - self.device_len

@property
def extend_len(self):   # 本次 forward 要「新算」的长度
    return self.device_len - self.cached_len

def complete_one(self):   # decode 每步：缓存长度追平、设备长度 +1
    self.cached_len = self.device_len
    self.device_len += 1

def append_host(self, next_token):   # 把新 token 拼到 CPU 侧 input_ids
    self.input_ids = torch.cat([self.input_ids, next_token])
```

### 2.3 一个形象的比喻

把 `Req` 想成一根「正在变长的香肠」：

- `cached_len`：已经「烤熟」（算过 KV 且存进缓存）的部分；
- `device_len`：已经「串在签子上」（在 GPU 上占位）的总长度；
- `max_device_len`：这根香肠最终要串多长。

`extend_len = device_len - cached_len` 就是「这次还要新烤的长度」；`remain_len = max_device_len - device_len` 就是「还能再串多长」。

---

## 三、难点解析

### 难点 1：为什么是「三个」长度，而不是两个？

如果只有「已算长度」和「总长度」两个，就无法表达「**前缀复用了，但复用部分和已算部分不是一回事**」这个状态。

有了 `cached_len`，才能区分：
- `[0, cached_len)`：前缀复用区（别人的 KV 直接拿来用）；
- `[cached_len, device_len)`：本次要新算的部分（`extend_len`）。

Radix Cache（Step 20）之所以能复用前缀，靠的就是 `cached_len` 标出「从哪里开始是新的」。

### 难点 2：`__post_init__` 的断言为什么是 `cached_len < device_len` 而不是 `<=`？

`assert 0 <= self.cached_len < self.device_len <= self.max_device_len` 里是严格小于。

因为 `cached_len == device_len` 意味着「这次 forward 什么都没新算」（`extend_len == 0`），这是一个没意义的空请求——一个请求至少要算一个 token 才有存在的价值。用严格 `<` 在构造时就排除这种非法状态。

### 难点 3：`complete_one` 为什么「先追平 cached_len，再 device_len+1」？

```python
def complete_one(self):
    self.cached_len = self.device_len   # 刚才这步算的 token，KV 已经写进缓存了
    self.device_len += 1                 # 准备算下一个 token
```

decode 阶段每步只算一个新 token。算完后，这个 token 的 KV 已经落进 cache（`cached_len` 追平旧 `device_len`），然后 `device_len` 前进一位，表示「下一个 token 位置就绪」。两步顺序不能反。

### 难点 4：`eq=False` 的 dataclass

`@dataclass(eq=False)` 让 `Req` 用**对象身份**（`is`）判等，而不是按字段值判等。因为调度器里要频繁用 `set[Req]` / `discard(req)` 去重（见 Step 9 的 `finished_reqs`），必须靠身份而不是内容。

---

## 四、注意事项

1. **`input_ids` 必须是 CPU tensor**：`__post_init__` 里 `assert self.input_ids.is_cpu`。GPU 上的真实输入存在 `token_pool`（Step 8），`Req.input_ids` 只是 CPU 侧的「账本」。
2. **`append_host` 用 `torch.cat`**：每次拼一个 token 都会新建 tensor（O(n) 拷贝），decode 阶段每步都调用，但因为是 CPU 小 tensor，代价可接受。
3. **`complete_one` 不检查边界**：调用方要保证 `remain_len > 0` 才 `complete_one`，否则 `device_len` 会超过 `max_device_len`。
4. **`can_decode = remain_len > 0`**：判断是否还能继续生成，是 `DecodeManager` 剔除请求的依据。

---

## 五、反思题

1. 用「一根变长的香肠」类比，分别说清 `cached_len` / `device_len` / `max_device_len` / `extend_len` / `remain_len` 各对应什么。
2. 如果 `cached_len` 和 `device_len` 合并成一个字段，Radix Cache 的前缀复用还做得成吗？为什么？
3. `__post_init__` 的 `cached_len < device_len` 为什么是严格小于？构造一个 `cached_len == device_len` 的请求会发生什么？
4. `complete_one` 的两步如果顺序颠倒（先 `device_len += 1` 再 `cached_len = device_len`），会有什么 bug？
5. 为什么 `Req` 要 `eq=False`？在哪些地方（Step 9 的 `set` 去重）依赖了对象身份判等？

---

## 六、示意图

### 6.1 `Req` 长度坐标轴

```
位置:    0 ─────────── cached_len ───────── device_len ───── max_device_len
        │                    │                     │              │
        │◄── 前缀复用区 ────►│◄── extend_len ─────►│◄─ remain_len ►│
        │  (别人的 KV 直接拿来用) │  (本次 forward 新算)  │  (还能生成)   │
        │                    │                     │              │
        └────────────────────┴─────────────────────┴──────────────┘
                             input_ids 长度         + output_len
```

### 6.2 一个请求的长度变化（prefill 后进入 decode 循环）

```
初始:   cached=0, device=5, max=10   (input 5 个 token, output 5 个)
  prefill 算完 5 个输入 token:
        cached=5, device=5           (extend_len = 5-0 = 5，一次算 5 个)
  decode 第 1 步:
        complete_one → cached=5, device=6   (算第 6 个 token)
        complete_one 后 cached 追平 5，device 变 6
  decode 第 2 步:
        complete_one → cached=6, device=7
  ...
  直到 remain_len = max(10) - device = 0，停止
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [core.py](python/minisgl/core.py) | `Req` + `SamplingParams` | `Req`、`SamplingParams` |

**下一步**：进入 Step 7（`Batch` 与 `Context`），看多个 `Req` 怎么打包成一个 batch，以及为什么模型前向「不带参数」。
