# Step 12：DecodeManager 与 TableManager

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 12。
> 核心文件：[scheduler/decode.py](python/minisgl/scheduler/decode.py)、[scheduler/table.py](python/minisgl/scheduler/table.py)。
>
> 这一 Step 回答：**进入 decode 阶段的请求是怎么被管理的？`table_idx`（逻辑槽位）和物理 KV 页是怎么解耦的？**

---

## 一、这个 Step 要解决什么

Step 11 讲了 prefill 怎么把请求「准入」进来。prefill 完成后，请求进入 decode 阶段——每轮生成一个 token，可能要生成几百上千轮。这时的管理由 `DecodeManager` 负责。

同时，一个请求「住在」哪个逻辑位置，由 `TableManager` 分配——它维护的 `table_idx` 是连接「请求」和「KV 物理存储」的中间层。理解了这个解耦，才能理解后面 PagedAttention（Step 13/19）为什么能自由换页。

---

## 二、核心逻辑

### 2.1 `DecodeManager`：正在生成的请求集合

```python
@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs):
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req):
        self.running_reqs.discard(req)

    def abort_req(self, uid):
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req); return req
        return None

    def schedule_next_batch(self):
        if not self.runnable: return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self): return len(self.running_reqs) > 0
```

要点：

- **`running_reqs` 是一个 `Set[Req]`**：依赖 `Req` 的 `eq=False`（对象身份判等，Step 6）做去重。
- **`filter_reqs` 每轮重建集合**：把「prefill 刚完成的请求」并进来，再过滤掉 `can_decode=False` 的（生成完的、chunk 未完成的）。
- **`schedule_next_batch` 按 `uid` 排序**：保证所有 rank 看到同样的请求顺序（对应 Step 10 的多卡顺序一致性问题），decode batch 的组成稳定可复现。

### 2.2 `inflight_tokens`：为什么多预留一截？

```python
@property
def inflight_tokens(self):
    tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 每请求预留 page_size-1
    return sum(req.remain_len for req in self.running_reqs) + tokens_reserved
```

`sum(remain_len)` 是「这些请求将来还会生成的总 token 数」。但为什么还要额外 `(page_size - 1) * len(reqs)`？

因为 KV 是**按页（page）分配**的，每个请求的最后一页可能只用了 1 个 token、其余 `page_size - 1` 个位置空着但页已经被占用。最坏情况下，每个请求都「浪费」`page_size - 1` 个 token 位置。这截预留就是把这个「页内碎片」算进去，让 prefill 准入（Step 11）更准确地估算空间。

### 2.3 `TableManager`：逻辑槽位池

```python
class TableManager:
    def __init__(self, max_running_reqs, page_table):
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))   # 槽位池
        self.page_table = page_table
        # NOTE: dummy request 也用这个池取 input_ids，所以 token_pool 初始化为 0
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self): return len(self._free_slots)

    def allocate(self): return self._free_slots.pop()   # 取一个槽位
    def free(self, slot): self._free_slots.append(slot)  # 还回槽位
```

极简，但有几个关键点：

- `_free_slots` 就是一个 `list[int]`，`allocate()` 从尾部 `pop`，`free()` 从尾部 `append`。槽位号用完即回收，**复用**。
- **`token_pool` 和 `page_table` 形状完全一致**（都是 `(max_running_req+1, max_seq_len)`），但存的东西不同：

| 张量 | 形状 | `[table_idx, pos]` 存什么 |
|---|---|---|
| `token_pool` | `(max_running_req+1, max_seq_len)` | 该槽位第 `pos` 个 token 的 **id** |
| `page_table` | `(max_running_req+1, max_seq_len)` | 该槽位第 `pos` 个 token 的 **物理 KV 位置** |

- `token_pool` 初始化为 `0`（不是随机值）：因为 `dummy_req` 也会用这个池取 `input_ids`，必须保证读到的是合法 token id。

---

## 三、难点解析

### 难点 1：`table_idx` 是「逻辑位置」，与「物理 KV 位置」解耦

一个请求被分配一个 `table_idx`（逻辑槽位），它的 token id 按顺序存在 `token_pool[table_idx]` 里。但它的 KV 存到**哪个物理页**，由 `page_table[table_idx]` 逐 token 记录。

这一层间接为什么重要？因为 PagedAttention 里 KV 是**离散分页**的——一个请求的连续 token 可能散落在完全不相邻的物理页里。有了 `table_idx` 这个「逻辑视图」，调度器只需要关心「第 `table_idx` 个请求的第 `pos` 个 token」，具体 KV 在哪由 `page_table` 翻译。换页（evict/重分配）时只需改 `page_table`，逻辑层无感。

### 难点 2：为什么 `token_pool` 和 `page_table` 形状要一致？

因为两者都是「按逻辑位置索引」的：`token_pool[table_idx, pos]` 拿 token id，`page_table[table_idx, pos]` 拿物理 KV 位置，**同一个 `(table_idx, pos)` 坐标对应同一个逻辑 token 的「内容」和「KV 落点」**。

形状一致让 index 计算可以复用（Step 8 的 `_make_input_tuple` 里，`input_mapping` 既是取 token 的下标，也是取 `out_loc` 的下标）。

### 难点 3：`filter_reqs` 为什么是「重建」而不是「增量 add/remove」？

```python
self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}
```

用集合推导式**整体重建**，一行完成两件事：

1. `union(reqs)` 把刚 prefill 完成的请求加进来；
2. `if req.can_decode` 把「已经生成完的」请求剔除。

相比手写 `add` + `remove`，重建式更不容易漏（比如某个请求在 `remove_req` 时已经不在集合里，`discard` 静默忽略）。代价是每轮 O(n) 重建，但 decode 请求数通常不大，可接受。

### 难点 4：`abort_req` 的线性扫描

`DecodeManager.abort_req(uid)` 是 `for req in running_reqs` 线性找 `uid`。因为 `running_reqs` 是 `Set`（靠身份判等），没法用 `uid` 直接索引，只能扫。abort 是低频操作，线性扫描可接受。这是「用 Set 换 O(1) 身份去重，但牺牲按 uid 查找效率」的权衡。

---

## 四、注意事项

1. **`schedule_next_batch` 的 `sorted(key=uid)` 不可省略**：这是多卡顺序一致性的关键（Step 10），去掉后各 rank 的 decode batch 顺序可能因 `Set` 迭代顺序不同而错位。
2. **`token_pool` 必须初始化成 0**：`dummy_req`（Step 14/22 讲）会从 `token_pool` 取 `input_ids`，0 是合法 token id（通常是 padding/bos），避免读到未初始化垃圾值。
3. **`available_size` 就是 `len(_free_slots)`**：槽位池耗尽意味着 `max_running_req` 个请求都在跑，prefill 准入（Step 11 条件①）会拒绝新请求。
4. **`remove_req` 用 `discard` 不是 `remove`**：请求可能已经不在集合（比如已被 `filter_reqs` 剔出），`discard` 不抛异常，`remove` 会 `KeyError`。
5. **`inflight_tokens` 是「估算」不是「精确」**：`(page_size-1) * len(reqs)` 是页内碎片的上界，实际可能少一些，但用作准入预留是安全的（宁多勿少）。

---

## 五、反思题

1. 如果 `token_pool` 和 `page_table` 形状不一致（比如 token_pool 存 token 而 page_table 存 page 号），Step 8 的 `_make_input_tuple` 还能共用同一个 `input_mapping` 吗？
2. `filter_reqs` 用集合重建，如果改成手写 `add`/`remove`，哪种请求状态容易被漏删？（提示：`can_decode` 何时从 True 变 False）
3. `inflight_tokens` 里 `(page_size - 1) * len(reqs)` 的预留，当 `page_size = 1` 时是多少？为什么这时候不需要预留？
4. `TableManager.allocate` 是 `pop()`（从尾部取），如果改成 `pop(0)`（从头部取），对 `_free_slots` 的语义有影响吗？性能上呢？
5. 为什么 `DecodeManager.abort_req` 能线性扫，但 `PrefillManager.abort_req`（Step 11）也是线性扫 `pending_list`？两者谁更频繁、为什么都接受线性？

---

## 六、示意图

### 6.1 逻辑槽位与物理 KV 的解耦

```
  请求 req（table_idx = 3）
  token_pool[3]:   [tok0, tok1, tok2, tok3, ...]   ← token id 连续
                       │       │       │
  page_table[3]:  [ 12,   30,    7,    45,  ...]   ← 物理 KV 位置（散乱）
                       │       │       │
  物理 KV 页:      page12  page30  page7  page45   ← 完全不连续也没关系
```

### 6.2 一个请求从 prefill 到 decode 的归属流转

```
  UserMsg ──► PrefillManager.pending_list
                    │ prefill 完成（非 ChunkedReq）
                    ▼
              _forward 里 decode_manager.filter_reqs 加入 running_reqs
                    │
                    ▼
  DecodeManager.running_reqs（每轮 sorted by uid 打包 decode batch）
                    │ 生成完（finished）或 EOS
                    ▼
              _process_last_data 里 remove_req + free 资源
```

### 6.3 `inflight_tokens` 的页内碎片

```
  page_size = 4，一个请求 remain_len = 6
  ┌────┬────┬────┬────┐ ┌────┬────┬────┬────┐
  │ t0 │ t1 │ t2 │ t3 │ │ t4 │ t5 │ 空 │ 空 │   ← 最后一页 2 个空位
  └────┴────┴────┴────┘ └────┴────┴────┴────┘
   占 2 页 = 8 位置，但只用了 6 → 碎片 2（= page_size-1 的上界内）
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [scheduler/decode.py](python/minisgl/scheduler/decode.py) | decode 请求管理 | `DecodeManager`、`inflight_tokens` |
| [scheduler/table.py](python/minisgl/scheduler/table.py) | 逻辑槽位池 | `TableManager`、`token_pool` |

**下一步**：进入 Step 13（CacheManager 分页分配），看 `page_table` 里的「物理 KV 位置」是怎么被一页页分配和写进去的。
