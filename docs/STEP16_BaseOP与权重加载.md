# Step 16：BaseOP 与权重加载

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 16。
> 核心文件：[layers/base.py](python/minisgl/layers/base.py) 的 `BaseOP`、`StateLessOP`、`OPList`。
>
> 这一 Step 回答：**项目为什么不用 `nn.Module`？模型在 meta 设备上建出来后，权重是怎么精确、完整地被灌进去的？**

---

## 一、这个 Step 要解决什么

Step 14 讲了模型在 **meta 设备**上建图、再 `load_state_dict` 一次性填权重。但 `nn.Module` 自带的 `state_dict`/`load_state_dict` 是为「常规训练」设计的，这里项目要的是：**极轻量 + meta 建图 + 精确到字节的权重对齐**。

于是自己写了一套 `BaseOP`，用 **`__dict__` 递归**把「对象树」映射成「`name → tensor` 字典」。这是整个模型能 meta 建图、能精确加载 HuggingFace 权重的根基。

---

## 二、核心逻辑

### 2.1 `BaseOP`：三个方法的抽象基类

```python
class BaseOP:
    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def state_dict(self, *, prefix="", result=None):
        result = result if result is not None else {}
        for name, param in self.__dict__.items():        # 遍历自己的属性
            if name.startswith("_"):
                continue                                  # 跳过下划线开头的（buffer/内部态）
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param   # 叶子：直接收集
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)  # 递归
        return result
```

`state_dict` 把对象树**递归展开**：遇到 `torch.Tensor` 就记到 `result`（key 是 `prefix.name`），遇到 `BaseOP` 子对象就带上前缀继续递归。

### 2.2 `load_state_dict`：反向填充

```python
def load_state_dict(self, state_dict, *, prefix="", _internal=False):
    for name, param in self.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(param, torch.Tensor):
            item = state_dict.pop(_concat_prefix(prefix, name))   # 从 dict 里「取走」
            assert isinstance(item, torch.Tensor)
            assert param.shape == item.shape and param.dtype == item.dtype   # 严格对齐
            setattr(self, name, item)                              # 用真权重替换 meta 张量
        elif isinstance(param, BaseOP):
            param.load_state_dict(state_dict, prefix=_concat_prefix(prefix, name), _internal=True)

    if not _internal and state_dict:
        raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
```

关键动作：`state_dict.pop(...)` **取走** key，`setattr(self, name, item)` 用加载到的真实 tensor **替换**掉原来 meta 设备上的空壳 tensor。`assert shape/dtype` 是「对齐校验」——meta 建图时算出的形状必须和 HF 权重完全一致，否则立刻报错。

### 2.3 `StateLessOP`：无参数层

```python
class StateLessOP(BaseOP):
    def state_dict(self, *, prefix="", result=None):
        return result if result is not None else {}      # 不收集任何权重

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: ...")  # 只做耗尽检查
```

有些层**没有可学习参数**（比如 `AttentionLayer`、`RotaryEmbedding`），它们 `state_dict` 为空、`load_state_dict` 不消费任何 key。它们在模型树里是「透明的」——既不提供权重，也不阻挡递归。

### 2.4 `OPList`：把 layer 列表纳入递归

```python
class OPList(BaseOP, Generic[T]):
    def __init__(self, ops):
        self.op_list = ops
    def state_dict(self, *, prefix="", result=None):
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)   # 用索引做前缀
        return result
    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)
        if not _internal and state_dict:
            raise RuntimeError(...)
```

`self.layers` 是个 `list[LlamaDecoderLayer]`，`list` 不是 `BaseOP`，普通的 `__dict__` 递归会漏掉它。`OPList` 专门处理「一列同构 layer」，用 **索引 `str(i)` 做前缀**（对应 HF 的 `model.layers.0`、`model.layers.1`...）。

---

## 三、难点解析

### 难点 1：为什么不用 `nn.Module`？

`nn.Module` 功能重（hook、参数注册、梯度、buffer 管理、`.cuda()`/`.to()` 等），而推理只需要「前向 + 加载权重」。几个具体原因：

1. **meta 建图**：`nn.Module` 也能 meta 建图，但它的参数注册机制（`register_parameter`）会引入额外开销和状态。
2. **精确控制加载**：这里要 `pop` + `assert shape/dtype` + 最后「没耗尽就报错」，用 `__dict__` 递归最直白可控。
3. **轻量**：`BaseOP` 就是「一个 `forward` 方法 + 两个递归方法」，没有 `nn.Module` 那一堆状态机。

代价是失去了 `nn.Module` 生态（比如不能直接 `model.cuda()`、不能用 `torch.save` 保存），但项目要的就是这种「我全都要自己掌控」的精确性。

### 难点 2：`_internal` 参数的作用（最容易写错的地方）

`load_state_dict` 最后有 `if not _internal and state_dict: raise`。为什么递归时要传 `_internal=True`？

因为 `state_dict` 是**全模型共享**的一个 dict。最外层调用时，它应该被**完全消费**（每个 key 都被某个叶子 pop 掉），所以最外层检查「是否耗尽」。但**中间层不能检查**——中间层遍历完自己子树后，`state_dict` 里**还剩下兄弟层的 key**（比如 `layer 0` 处理完，`layer 1..N` 的 key 还在），如果中间层也检查就会误报。

所以约定：**只有最外层（`_internal=False`）检查耗尽，递归层一律 `_internal=True` 静默**。

### 难点 3：`setattr` 替换 vs `copy_` 拷贝

`load_state_dict` 用的是 `setattr(self, name, item)` **整体替换**，而不是 `param.copy_(item)`。

因为 meta 设备上的 tensor 是「空壳」（有 shape/dtype，无真实内存），`copy_` 没法把数据拷进一个不存在的内存。`setattr` 直接把 `self.name` 从 meta 张量换成 HF 加载来的真实 CUDA 张量。这也是为什么 Step 14 里 meta 建图 + `load_state_dict` 能「无痛物化」。

### 难点 4：`_` 前缀 = buffer 约定

`state_dict` 和 `load_state_dict` 都 `if name.startswith("_"): continue`。所以**任何不想参与权重加载的张量，都用 `_` 前缀命名**。

典型例子是 `RotaryEmbedding._cos_sin_cache`：它是**运行时算出来的 cos/sin 缓存**（buffer），不是从 HF 加载的权重，所以加 `_` 前缀跳过。这是「参数」和「缓存/buffer」的区分手段。

### 难点 5：`RotaryEmbedding` 为什么不能 meta 建图？

[rotary.py](python/minisgl/layers/rotary.py) 里 `get_rope` 有一个特殊检查：

```python
t = torch.tensor([])
if t.device == torch.device("meta"):
    if _ROPE_DEVICE is None:
        raise RuntimeError("We cannot use meta device for rope. Please call set_rope_device() first.")
    with torch.device(_ROPE_DEVICE):
        return _get_rope(...)
```

RoPE 的 cos/sin 缓存是**真实数值**（`base ** (...)`、`freqs.cos()`），必须在真实设备上算，不能在 meta 设备上「只造形状」。所以 Step 14 在 meta 建图前先 `set_rope_device(self.device)`，让 rope 缓存绕过 meta 直接在 CUDA 上算。这是「meta 建图」的一个例外——**不是所有层都能 meta，含真实数值的缓存层要特殊处理**。

---

## 四、注意事项

1. **`__dict__` 遍历依赖「属性名 = 权重名」**：字段命名必须和 HF 的 state_dict key 对齐（比如 `self.weight` → `xxx.weight`），否则 `pop` 不到。
2. **`state_dict` 用 `pop` 不是 `get`**：`pop` 既取值又删 key，最后靠「是否还有剩余」来检测多出来的 key。
3. **`assert shape == item.shape and dtype == item.dtype`**：形状或 dtype 不一致会立刻炸，这是「对齐 HF」的第一道防线。
4. **`StateLessOP.load_state_dict` 里 `_internal` 检查**：无参数层也会做「最外层耗尽检查」，所以最外层混用 `BaseOP` 和 `StateLessOP` 不会漏检。
5. **`OPList` 用 `str(i)` 而非 `op.name`**：对应 HF 的 `layers.{i}` 命名，索引就是层号。

---

## 五、反思题

1. `state_dict` 递归里，为什么 `torch.Tensor` 和 `BaseOP` 之外的类型（比如 `int`、`str`、`list`）既不被收集也不报错？如果某个属性是个「参数化的 list」，会漏掉什么？
2. 如果去掉 `_internal` 参数，把 `load_state_dict` 改成每层都检查耗尽，会报什么错？为什么？
3. `setattr` 替换 meta 张量的前提是什么？（提示：meta 张量为什么不能 `copy_`）
4. 为什么 `_cos_sin_cache` 要用 `_` 前缀，而不是像 `weight` 一样参与 state_dict？如果它参与了，会发生什么？
5. `OPList` 的前缀用 `str(i)`，但如果模型某处有个**命名不规则**的 layer 列表（不是 `layers.0` 这种），`OPList` 还适用吗？你会怎么改？

---

## 六、示意图

### 6.1 `state_dict` 的递归收集

```
  model (BaseOP)
   ├─ embed_tokens.weight  ──────────► "embed_tokens.weight"
   ├─ layers (OPList)
   │    ├─ op_list[0] (LlamaDecoderLayer)
   │    │    ├─ self_attn.qkv_proj.weight ──► "layers.0.self_attn.qkv_proj.weight"
   │    │    └─ mlp.down_proj.weight    ──► "layers.0.mlp.down_proj.weight"
   │    └─ op_list[1] ...
   └─ norm.weight                 ──► "norm.weight"
```

### 6.2 `load_state_dict` 的「取走」语义

```
  state_dict = { "embed_tokens.weight": W1, "layers.0...": W2, ... }
        │
        ▼ 递归 load_state_dict
  每个叶子 pop 一个 key，setattr 替换 meta 张量
        │
        ▼
  最外层检查：state_dict 若还有剩余 → 报「Unexpected keys」
```

### 6.3 `_internal` 参数的传递

```
  最外层 load_state_dict(_internal=False)
      │ 递归时全部传 _internal=True
      ├─ embed_tokens.load_state_dict(_internal=True)   ← 不检查耗尽
      ├─ layers.load_state_dict(_internal=True)         ← 不检查耗尽
      │    └─ 每个 layer.load_state_dict(_internal=True)
      └─ norm.load_state_dict(_internal=True)
      │
      ▼
  只有最外层回到 _internal=False 时，检查 state_dict 是否耗尽
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [layers/base.py](python/minisgl/layers/base.py) | 轻量层基类 + 权重递归 | `BaseOP`、`StateLessOP`、`OPList` |
| [layers/rotary.py](python/minisgl/layers/rotary.py) | RoPE（StateLessOP 例子） | `RotaryEmbedding`、`get_rope`、`set_rope_device` |

**下一步**：进入 Step 17（张量并行线性层），看这五个线性层怎么按「列切 / 行切」分摊权重、哪些需要 all-reduce。
