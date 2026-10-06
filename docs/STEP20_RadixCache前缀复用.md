# Step 20：Radix Cache 前缀复用

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 20。
> 核心文件：[kvcache/radix_cache.py](python/minisgl/kvcache/radix_cache.py)、[kvcache/naive_cache.py](python/minisgl/kvcache/naive_cache.py)。
>
> 这一 Step 回答：**「共享前缀的 KV 只算一次」是怎么靠一棵 Radix 树实现的？`ref_count`、`timestamp`、`split_at` 各自在管什么？**

---

## 一、这个 Step 要解决什么

多个请求常常共享同一个前缀（system prompt、few-shot 示例）。如果每个请求都从头算一遍前缀的 KV，就浪费了。Radix Cache 的想法：**把「token 序列 → 对应 KV 页 index」的映射存成一棵树**，新请求进来先沿树找最长公共前缀，命中的部分直接复用 KV，只算新的后缀。

这是 SGLang 的招牌优化（RadixAttention）。本 Step 拆解这棵树的数据结构和四个核心操作。

---

## 二、核心逻辑

### 2.1 `RadixTreeNode`：树的节点

```python
class RadixTreeNode:
    def __init__(self, key_fn, tic=None):
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}   # 按 key_fn 分叉
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0                         # 引用计数
        self.timestamp = tic or time.monotonic_ns()    # 最近访问时间
        self._key: torch.Tensor    # 这段 token 序列
        self._value: torch.Tensor  # 这段 token 对应的 KV 页 index
        self._length: int
```

每个节点存一段**连续的 token 序列**（`_key`）和它对应的 **KV 页 index**（`_value`），两者等长。`ref_count` = 有多少个正在跑的请求引用这个节点；`timestamp` = 最近一次访问时间（用于 LRU 驱逐）。

### 2.2 `key_fn`：分叉的钥匙

```python
def _get_key_fn(page_size):
    if page_size == 1:
        return lambda x: x[0].item()            # 用第一个 token 作 key
    return lambda x: tuple(x[:page_size].tolist())   # 用前 page_size 个 token 作 key
```

`children` 是一个 dict，`key_fn(input_ids)` 算出「这段序列的开头特征」作为 dict 的 key。因为 KV 是**按页**管理的，分叉也按「页」对齐：`page_size=1` 用单个 token 分叉，`page_size>1` 用前 `page_size` 个 token 分叉。

### 2.3 `match_prefix` → `_tree_walk`：最长前缀匹配

```python
def _tree_walk(self, input_ids):
    prefix_len = 0
    node = self.root_node
    tic = time.monotonic_ns()
    while prefix_len < len(input_ids):
        child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
        if child_node is None:                      # 没有匹配的分叉 → 到此为止
            return node, prefix_len
        node = child_node
        match_len = node.get_match_len(input_ids[prefix_len:])   # 节点内找最长匹配
        match_len = align_down(match_len, self.page_size)        # 对齐到页
        prefix_len += match_len
        if match_len != node.length:                # 只匹配了一部分 → 劈开节点
            node = node.split_at(match_len)
            node.timestamp = tic
            return node, prefix_len
        node.timestamp = tic                         # 更新 LRU 时间戳
    return node, prefix_len
```

`get_match_len` 用 `fast_compare_key`（C++ kernel）找「节点 token 和输入 token 的第一个不同位置」。如果没完全匹配（`match_len != node.length`），说明新序列在节点**中间**分叉了，要 `split_at` 把节点劈开。

### 2.4 `split_at`：在节点中间劈一刀

```python
def split_at(self, pos):
    parent = self.parent
    new_node = RadixTreeNode(self.key_fn, self.timestamp)   # 前半段新节点
    new_node.set_key_value(self._key[:pos], self._value[:pos])
    new_node.set_parent(parent)                              # 挂到原 parent 下
    new_node.ref_count = self.ref_count                      # 继承引用计数
    self.set_key_value(self._key[pos:], self._value[pos:])   # 原节点变后半段
    self.set_parent(new_node)
    return new_node
```

劈开后：原节点一分为二，前半段是新节点（继续挂在 parent 下），后半段还叫原节点（挂到新节点下）。这样树就能在「任意 token 位置」分叉，而不用整节点替换。

### 2.5 `insert_prefix`：插入新前缀

```python
def insert_prefix(self, input_ids, indices):
    insert_len = align_down(len(input_ids), self.page_size)   # 只缓存整页
    input_ids, indices = input_ids[:insert_len], indices[:insert_len]
    node, prefix_len = self._tree_walk(input_ids)             # 找已匹配部分
    if prefix_len != insert_len:                              # 有新的后缀
        new_node = RadixTreeNode(self.key_fn)
        new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
        new_node.set_parent(node)
        self.evictable_size += new_node.length
        node = new_node
    return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))
```

只把「整页对齐」的部分插入树（`align_down`）。返回的 `InsertResult.cached_len = prefix_len`——**插入前已经在缓存里的长度**（Step 13 的 `cache_req` 就靠这个数释放「复用别人 KV 而多占的页」）。

### 2.6 `evict`：LRU 驱逐

```python
def evict(self, size):
    leave_nodes = self._collect_leave_nodes_for_evict()   # 收集 ref_count==0 的叶子
    heapq.heapify(leave_nodes)                             # 按 timestamp 建最小堆
    while evicted_size < size:
        node = heapq.heappop(leave_nodes)                  # 弹最旧（LRU）
        evicted_size += node.length
        evicted_indices.append(node.value)
        self.evictable_size -= node.length
        parent = node.parent
        del parent.children[self.key_fn(node._key)]        # 从父节点摘除
        if parent.is_leaf() and parent.ref_count == 0:     # 父节点也变成可驱逐叶子
            heapq.heappush(leave_nodes, parent)
    return torch.cat(evicted_indices)
```

只驱逐 **`ref_count == 0` 的叶子节点**（没人引用、且没有孩子）。用最小堆按 `timestamp` 弹出最久未访问的（LRU 近似）。驱逐一个节点后，如果它的父节点「也成了叶子且无人引用」，继续放进堆（可以一路驱逐上去）。

### 2.7 `lock_handle`：维护引用计数

```python
def lock_handle(self, handle, unlock=False):
    node = handle.node
    if unlock:
        while not node.is_root():
            node.ref_count -= 1
            if node.ref_count == 0:
                self.evictable_size += node.length     # 无人引用 → 可驱逐
                self.protected_size -= node.length
            node = node.parent
    else:
        while not node.is_root():
            if node.ref_count == 0:
                self.evictable_size -= node.length     # 被引用 → 受保护
                self.protected_size += node.length
            node.ref_count += 1
            node = node.parent
```

沿**从 handle 节点到 root 的整条祖先链**，逐个 `ref_count += 1`（lock）或 `-= 1`（unlock）。`ref_count > 0` = 受保护（不可驱逐），`ref_count == 0` = 可驱逐。`evictable_size` / `protected_size` 随时据此更新。

---

## 三、难点解析

### 难点 1：Radix 树 vs 普通前缀树（Trie）

普通 Trie 每个节点存**一个 token**，路径「a → b → c」表示序列 abc。Radix 树把**连续的一段 token**压缩成一个节点，只在真正分叉的地方才拆节点。

好处是**省节点数**：一个 1000 token 的共享前缀，Trie 要 1000 个节点，Radix 树可能只要 1 个节点（如果没人在这 1000 token 中间分叉）。代价是「匹配」和「插入」要做 `split_at` 这种节点拆分。

### 难点 2：`ref_count` 和 `timestamp` 分工

- `ref_count` 管**「能不能驱逐」**：> 0 说明有请求正在用，绝对不能驱逐；== 0 才可驱逐。
- `timestamp` 管**「驱逐谁」**：多个可驱逐节点里，驱逐**最久没访问**的那个（LRU）。

两者独立：`ref_count` 是「安全性」，`timestamp` 是「效率」。一个节点可能 `ref_count=0`（安全可驱逐）但 timestamp 很新（刚用过），驱逐时会优先赶走更旧的。

### 难点 3：`split_at` 为什么继承 `ref_count`？

```python
new_node.ref_count = self.ref_count
```

劈开时，原来引用这个节点的请求，现在**同时**引用前半段和后半段（它们的 `get_matched_indices` 会沿 parent 链收集整条路径的 value）。所以前半段新节点要继承原节点的 `ref_count`，保证引用计数不丢失——否则前半段会错误地变成 `ref_count=0` 被驱逐。

### 难点 4：`get_matched_indices` 为什么要沿 parent 链收集？

```python
def get_matched_indices(self):
    while not node.is_root():
        value_list.append(node.value)
        node = node.parent
    value_list.reverse()
    return torch.cat(value_list)
```

因为一段前缀可能跨了**多个节点**（被分叉拆开）。要拿到「从 root 到 handle 节点」的完整 KV 页 index，必须沿 parent 链一路收集每个节点的 `_value`，反转后拼起来。这对应 Step 13 里 `page_entry.copy_(handle.get_matched_indices())`——把复用的 KV 页 index 拷到新请求的 page_table。

### 难点 5：对照 `naive_cache` 看 Radix 的价值

[naive_cache.py](python/minisgl/kvcache/naive_cache.py) 的 `NaivePrefixCache`：

- `match_prefix` 永远返回 `cached_len=0`（**从不复用前缀**）；
- `insert_prefix` 永远返回 `InsertResult(0, ...)`（不真正插入）；
- `evict` 直接抛 `NotImplementedError`（不支持驱逐）。

它是「没有前缀复用」的退化实现，`--cache naive` 时每个请求都从头算前缀。和 Radix 对照，能直观看出 Radix 的价值：**共享前缀的 KV 只算一次，之后所有请求直接 `cached_len > 0` 复用**。

---

## 四、注意事项

1. **`insert_len = align_down(len, page_size)`**：只有「整页」的部分能进缓存，尾部的零头不缓存（因为 KV 按页分配）。
2. **`_tree_walk` 里的 `align_down(match_len, page_size)`**：匹配长度也对齐到页，避免「半页匹配」破坏分页。
3. **root 永远 protected**：`root_node.ref_count = 1`，且 `evict` 里 `assert not node.is_root()`，root 不会被驱逐。
4. **`insert_prefix` 里 `indices[prefix_len:].clone()`**：必须 clone，因为 `indices` 是调用方（Step 13 的 `page_indices`）的 view，不 clone 的话后续释放会互相干扰。
5. **`evict` 可能驱逐得比请求的 size 多**：base.py 的 docstring 明确写了「actual evict size may be larger」，因为驱逐按节点整段走，不能切半页。

---

## 五、反思题

1. 一个 100 token 的共享前缀，被 3 个请求共享，但其中一个请求在 token 50 处有不同后缀。这棵树会长成什么样？画出来（提示：在哪 `split_at`）。
2. `ref_count` 和 `timestamp` 分别解决什么问题？如果只用 `timestamp` 不用 `ref_count`，会出什么 bug？
3. `split_at` 里 `new_node.ref_count = self.ref_count` 如果漏了，会导致什么？（提示：前半段会不会被误驱逐？）
4. `get_matched_indices` 为什么要 `reverse()`？不 reverse 会得到什么顺序的 index？
5. 为什么 `insert_prefix` 只缓存 `align_down(len, page_size)` 的部分？尾部零头的 KV 去哪了？

---

## 六、示意图

### 6.1 Radix 树的节点结构

```
  root (ref_count=1, 永远 protected)
   │
   ├─ "You are a helpful" ──► 节点 A（value: [页1, 页2]）
   │                              │
   │                              ├─ " assistant." ──► 节点 B
   │                              └─ " doctor."    ──► 节点 C（在 A 中间分叉）
   │
   └─ "The capital of" ──► 节点 D
```

### 6.2 `_tree_walk` 的匹配 + `split_at`

```
  节点 A: "hello world"（length=11）
  新输入: "hello there"

  _tree_walk: key_fn("hello...") 命中 A，get_match_len 找到第 6 个字符分叉
              match_len=6 ≠ 11 → split_at(6)
  结果:
        "hello "（前半，新节点 A'）
            │
            ├─ "world"（原节点 A 后半）
            └─ "there"（新插入的节点）
```

### 6.3 `lock_handle` 的祖先链

```
  lock_handle(handle → 节点 C)
        │ 沿 parent 链往上，逐个 ref_count += 1
  C ──► B ──► A ──► root（root 不动）
  整条链从 evictable 变成 protected
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [kvcache/radix_cache.py](python/minisgl/kvcache/radix_cache.py) | Radix 前缀缓存 | `RadixTreeNode`、`RadixPrefixCache`、`split_at`、`evict` |
| [kvcache/naive_cache.py](python/minisgl/kvcache/naive_cache.py) | 无复用的退化实现 | `NaivePrefixCache` |
| [kvcache/base.py](python/minisgl/kvcache/base.py) | 前缀缓存接口 | `BasePrefixCache`、`BaseCacheHandle` |

**下一步**：进入 Step 21（注意力后端接口与实现），看 `attn_backend.forward` 怎么根据 prefill/decode 分发给 FlashAttention / FlashInfer。
