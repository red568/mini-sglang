# Step 13：CacheManager 分页分配

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 13。
> 核心文件：[scheduler/cache.py](python/minisgl/scheduler/cache.py) 的 `CacheManager`。
>
> 这一 Step 回答：**KV cache 的「页」是怎么被一页页分出来、写进 `page_table` 的？请求算完后，它的前缀又是怎么被缓存、哪些页被释放的？**

---

## 一、这个 Step 要解决什么

Step 12 讲了 `page_table[table_idx, pos]` 存的是「物理 KV 位置」，但没说这些位置怎么来。`CacheManager` 就是那个**管物理页的人**：它手里攥着一堆空闲页，谁来要就发几页，请求完了再把页收回来（或者把前缀页留在 Radix Cache 里复用）。

这是 PagedAttention 的核心——**逻辑 token 位置 → 物理页**的映射在这里落地。

---

## 二、核心逻辑

### 2.1 `free_slots`：按页对齐的空闲页池

```python
def __init__(self, num_pages, page_size, page_table, type):
    self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
    self.prefix_cache = create_prefix_cache(device=device, type=type)
    self.num_pages = num_pages
    self.page_table = page_table
    self.page_size = page_size
```

注意 `free_slots` 存的是**页的起始 token 位置**（已经乘了 `page_size`）。例如 `page_size=2`、`num_pages=4` 时，`free_slots = [0, 2, 4, 6]`，每两个位置代表一页。所以 `free_slots` 里存的是「token 偏移」，一个元素 = 一页的起点。

### 2.2 `match_req`：查前缀（注意那个 -1）

```python
def match_req(self, req: PendingReq) -> MatchResult:
    return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
```

为什么是 `[: input_len - 1]`？因为 prompt 的**最后一个 token 的 KV 还没算出来**，它不可能已经被缓存。前缀缓存只能命中「已经算过 KV 的那部分」，也就是除了最后一个 token 之外的前 `input_len - 1` 个。这是前缀复用的边界。

### 2.3 `allocate_paged`：给一批请求分页

```python
def allocate_paged(self, reqs):
    needed_pages = 0
    allocation_info = []
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)   # 已经占到的页（向上取整）
        last_page  = div_ceil(req.device_len, self.page_size)   # 这次需要到第几页
        if last_page > first_page:                              # 需要新页
            needed_pages += last_page - first_page
            allocation_info.append((req.table_idx, first_page, last_page))
    if needed_pages > 0:
        allocated = self._page_to_token(self._allocate(needed_pages))  # 发页 → 展开成 token 位置
        _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
```

`div_ceil(a, b) = (a + b - 1) // b`。`cached_len` 是已缓存的 token 数，`device_len` 是这次 forward 后要到达的 token 数，两者换算成「页」的差就是**这次需要新分配的页数**。

`_write_page_table` 把分配的物理页号，按 `(table_idx, position)` 逐个写进 `page_table`。

### 2.4 `_page_to_token` 与 `_allocate`

```python
def _page_to_token(self, pages):
    if self.page_size == 1:
        return pages
    offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
    return (pages.unsqueeze(1) + offsets).flatten()   # 页号 → 页内每个 token 位置

def _allocate(self, needed_pages):
    if needed_pages > (free_pages := len(self.free_slots)):
        evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
        self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
    allocated = self.free_slots[:needed_pages]
    self.free_slots = self.free_slots[needed_pages:]
    return allocated
```

- `_allocate`：空闲页不够就先 `evict`（从 Radix Cache 里逐出没被引用的前缀页），凑够了再取走前 `needed_pages` 页。
- `_page_to_token`：把「页号」展开成「页内所有 token 位置」。`page_size=2` 时，页 `[0, 2]` → token `[0,1,2,3]`。

### 2.5 `cache_req`：算完后缓存前缀 + 释放

```python
def cache_req(self, req, *, finished):
    insert_ids = req.input_ids[: req.cached_len]
    page_indices = self.page_table[req.table_idx, : req.cached_len]
    old_handle = req.cache_handle
    cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
    self.unlock(old_handle)
    self._free(page_indices[old_handle.cached_len : cached_len])   # 这段已在缓存，释放
    if finished:
        self._free(page_indices[new_handle.cached_len :])          # 尾部页全部释放
    else:
        req.cache_handle = new_handle                               # 保留尾部，更新句柄
        self.lock(new_handle)
```

源码里那段长注释画了两块区域的边界，是本函数最核心的难点（见难点 2）。

### 2.6 `lazy_free_region`：把多次释放合并成一次

```python
@contextmanager
def lazy_free_region(self):
    def lazy_free(indices): lazy_free_list.append(indices[:: self.page_size])
    lazy_free_list = []
    try:
        self._free = lazy_free          # 临时替换 _free
        yield
    finally:
        del self._free
        self.free_slots = torch.cat([self.free_slots] + lazy_free_list)  # 一次性合并
```

`_process_last_data`（Step 9）里 `with self.cache_manager.lazy_free_region():` 包住整个循环。循环里每次 `_free` 其实只是把页号塞进 `lazy_free_list`，最后 `finally` 里**一次 `torch.cat`** 把所有页号合并进 `free_slots`，避免循环里反复 `cat`（每次 cat 都拷贝整个张量）。

---

## 三、难点解析

### 难点 1：`free_slots` 存「页起点」而不是「页号」的设计

`free_slots` 直接存乘了 `page_size` 的 token 偏移（`[0, 2, 4, 6]`），好处是 `_page_to_token` 和 `_free` 里可以**统一按 token 偏移**运算，不用反复换算。

代价是**要时刻记得「这里存的是页、值是 token 偏移」**。`_free` 里 `indices[:: self.page_size]` 就是从「token 位置」取「每页的起点」——因为一页内所有 token 位置连续，`::page_size` 正好隔页取一个起点。

### 难点 2：`cache_req` 的「合法缓存区 vs 已分配缓存区」（最烧脑）

那段注释画了三个 `cached_len` 相关的边界，用代码里的变量翻译：

- `req.cached_len`：这个请求**当前**缓存到的长度（prefill 完成后 = device_len）。
- `old_handle.cached_len`：这个请求 prefill **之前**就已经在缓存里的前缀长度。
- `cached_len`（`insert_prefix` 的返回值）：插入后，**新 handle 覆盖到的**长度。

关键逻辑：`insert_prefix` 会把 `[0, req.cached_len)` 这段 token 插入 Radix 树，但它可能发现「这段里有一部分已经被别人缓存了」（`old_handle.cached_len` 之后、`cached_len` 之前的区域），于是：

```python
self._free(page_indices[old_handle.cached_len : cached_len])  # ① 已被别人缓存的部分 → 释放
```

这段 token 的 KV 页在「别人」名下，这个请求分配了但没用上（因为复用了别人的），所以释放掉，避免重复占用。

然后看尾部：

```python
if finished:
    self._free(page_indices[new_handle.cached_len :])   # ② 请求结束 → 尾部全释放
else:
    req.cache_handle = new_handle                        # ③ 没结束 → 保留尾部，交给下轮
    self.lock(new_handle)
```

`new_handle.cached_len` 之后是「没能插进前缀缓存的尾部」（比如最后一个 token 的 KV），如果请求结束就释放，否则保留（下一个 decode token 还会用到）。

### 难点 3：为什么 `insert_prefix` 之后要 `_free` 两处？

总结上面：第一处 `_free` 释放「复用别人 KV 而没花自己页」的部分；第二处 `_free` 释放「请求结束后的尾部」。两处释放的是**不同区间**，都指向同一个目标——**不让页被重复占用导致泄漏**。

### 难点 4：`lazy_free_region` 用「替换方法」而不是「攒 list 传参」

为什么不是显式 `lazy_free(list)` 传参，而是 `self._free = lazy_free` 临时替换？

因为 `_free` 在 `cache_req` 里被**多处调用**（上面那两处），如果改成传参，`cache_req` 的签名和所有调用点都要改。用 contextmanager 临时替换 `self._free`，让 `cache_req` 内部的 `self._free(...)` 调用**不知不觉地**变成延迟释放，零侵入。

---

## 四、注意事项

1. **`match_req` 的 `-1` 不能丢**：丢了这个请求的最后一个 token 会被误当缓存前缀，`cached_len` 会多算 1，导致 `extend_len` 少算 1 个 token（最后那个 token 其实没算过）。
2. **`free_slots` 全程按「页」对齐**：`page_size > 1` 时 `free_slots` 一定是 `page_size` 的倍数，`check_integrity` 里 `assert all(free_slots % page_size == 0)` 专门校验这个不变量。
3. **`_allocate` 的 evict 是「尽量逐出」**：`evict((needed - free) * page_size)` 逐出足够 token 数，但逐出的页号要 `evicted[::page_size]` 取「页起点」才能并入 `free_slots`。
4. **`cache_req` 只在非 chunk 的 prefill 完成后、以及 finished 时调用**：见 Step 9 的 `_process_last_data` 里 `elif batch.is_prefill: cache_req(finished=False)` 和 `if finished: ... _free_req_resources → cache_req(finished=True)`。
5. **`available_size` 是 `evictable_size + free_slots*page_size`**：可逐出的 Radix 缓存 + 空闲页，两者加起来才是「能分出去的空间」。

---

## 五、反思题

1. `page_size` 从 1 变成 2 时，`_page_to_token` 的返回值长度会发生什么变化？`allocate_paged` 分配同样 token 数时页数会怎样？
2. `cache_req` 里第一处 `_free(page_indices[old_handle.cached_len : cached_len])` 释放的到底是什么？（提示：这段 token 的 KV 属于谁？）
3. 如果去掉 `lazy_free_region`，直接把 `_free` 恢复成普通 `torch.cat`，`_process_last_data` 循环里每处理一个请求就 `cat` 一次，性能会差在哪？
4. `match_req` 为什么匹配 `input_len - 1` 而不是 `input_len`？用「最后一个 token 的 KV 还没算」解释。
5. `check_integrity` 里 `free_pages + cache_pages == num_pages` 校验的是什么不变量？什么操作会破坏它？

---

## 六、示意图

### 6.1 页分配：从「页号」到「token 位置」

```
  page_size = 2, num_pages = 4
  free_slots = [0, 2, 4, 6]          ← 页起点（token 偏移）

  _allocate(2) 取走 [0, 2] → free_slots = [4, 6]
  _page_to_token([0, 2]) → [0, 1, 2, 3]   ← 两页展开成 4 个 token 位置
```

### 6.2 `cache_req` 的两处 `_free`

```
  位置:  0 ── old_handle.cached_len ── cached_len ── new_handle.cached_len ── req.cached_len
        │                            │              │                        │
        │◄── 之前就缓存的 ────────────►│              │                        │
        │              ◄── 别人缓存、释放 ──►│           │                        │
        │                              ◄── 新插入前缀 ──►│◄── 尾部（finished 才释放）──►│
```

### 6.3 `allocate_paged` 全流程

```
  reqs（每个 req 有 cached_len / device_len）
        │
        ▼
  算 first_page = div_ceil(cached_len, page_size)
      last_page  = div_ceil(device_len, page_size)
        │ last_page > first_page ?
        ▼
  needed_pages = Σ(last_page - first_page)
        │
        ▼
  _allocate(needed_pages)：不够先 evict
        │
        ▼
  _page_to_token：页号 → token 位置
        │
        ▼
  _write_page_table：page_table[table_idx, pos] = 物理位置
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [scheduler/cache.py](python/minisgl/scheduler/cache.py) | 分页分配 + 前缀缓存 | `CacheManager`、`allocate_paged`、`cache_req`、`lazy_free_region` |

**下一步**：进入 Step 14（Engine 初始化），看 `Engine` 怎么把模型、KV cache、页表、后端、CUDA graph 全部组装起来。
