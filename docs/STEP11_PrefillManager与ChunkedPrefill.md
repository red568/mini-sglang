# Step 11：PrefillManager 与 Chunked Prefill

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 11。
> 核心文件：[scheduler/prefill.py](python/minisgl/scheduler/prefill.py) 的 `PrefillManager`、`PrefillAdder`、`ChunkedReq`。
>
> 这一 Step 回答：**一个新请求怎么被准入、分配资源、变成 `Req` 进入 prefill？一个超长 prompt 又怎么被切成多块分批算？**

---

## 一、这个 Step 要解决什么

Step 8 里 `UserMsg` 进来只是被 `add_one_req` 塞进了 `pending_list`，从「待处理」到「真的开始算」中间还隔着一道**准入控制**——这就是 `PrefillAdder`。

核心问题有两个：

1. **准入**：显存/槽位/算力都有限，一次能放多少请求进去 prefill？放不下怎么办？
2. **分块**：一个超长 prompt（比如 100K token）不能一次性 prefill 完（显存装不下、还会饿死 decode），怎么切成几块？

---

## 二、核心逻辑

### 2.1 三个准入条件（`PrefillAdder._try_allocate_one`）

```python
def _try_allocate_one(self, req: PendingReq):
    if self.table_manager.available_size == 0:          # 条件①：还有逻辑槽位吗
        return None

    handle = self.cache_manager.match_req(req).cuda_handle
    cached_len = handle.cached_len                       # 前缀复用：匹配到的长度
    extend_len = req.input_len - cached_len              # 本次要新算的长度
    estimated_len = extend_len + req.output_len          # 这个请求总共要占多少

    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return None                                      # 条件②：KV cache 还有空间吗

    self.cache_manager.lock(handle)
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return self.cache_manager.unlock(handle)         # 锁后再查一次（防竞态）

    table_idx = self.table_manager.allocate()            # 分配逻辑槽位
    if cached_len > 0:                                   # 有前缀复用 → 拷已缓存的 token/page
        device_ids = self.table_manager.token_pool[table_idx][:cached_len]
        page_entry = self.table_manager.page_table[table_idx][:cached_len]
        device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
        page_entry.copy_(handle.get_matched_indices())

    return handle, table_idx
```

三条准入线：

| 条件 | 检查什么 | 放不下就 |
|---|---|---|
| ① `table_manager.available_size == 0` | 逻辑槽位（`table_idx`）用完了 | 返回 None |
| ② `estimated_len + reserved_size > available_size` | KV cache 剩余空间不够 | 返回 None |
| ③ 锁后再查一次 ② | 加锁期间空间被抢了 | unlock 返回 None |

`reserved_size = decode_manager.inflight_tokens`，意思是「正在 decode 的请求将来还会吃掉这么多 KV 空间」，prefill 准入时**给它们预留**，不能把空间全占光。

### 2.2 分块逻辑（`_add_one_req`）

```python
def _add_one_req(self, pending_req, cache_handle, table_idx, cached_len):
    remain_len = pending_req.input_len - cached_len       # 还要算多少
    chunk_size = min(self.token_budget, remain_len)       # 本轮最多算 chunk_size
    is_chunked = chunk_size < remain_len                  # 算不完 → 分块
    CLS = ChunkedReq if is_chunked else Req
    self.token_budget -= chunk_size
    self.reserved_size += remain_len + pending_req.output_len
    _slice = slice(cached_len, cached_len + chunk_size)
    device_ids = self.table_manager.token_pool[table_idx, _slice]
    device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
    return CLS(input_ids=pending_req.input_ids[: cached_len + chunk_size],
               table_idx=table_idx, cached_len=cached_len, ...)
```

`token_budget = prefill_budget`（= `max_extend_tokens`）是「本轮 prefill 最多新算多少个 token」的算力预算。`chunk_size = min(budget, remain_len)`：预算够就整段算（`Req`），不够就只算前 `budget` 个（`ChunkedReq`）。

### 2.3 `try_add_one`：分块的续算

```python
def try_add_one(self, pending_req):
    if self.token_budget <= 0:
        return None                                      # 预算用完，本轮不再加

    if chunked_req := pending_req.chunked_req:           # 是上一轮切过的 → 继续算下一块
        return self._add_one_req(pending_req,
            cache_handle=chunked_req.cache_handle,       # 复用槽位和 handle
            table_idx=chunked_req.table_idx,
            cached_len=chunked_req.cached_len)           # 从上次断点继续

    if resource := self._try_allocate_one(pending_req):  # 全新请求 → 先准入再分块
        cache_handle, table_idx = resource
        return self._add_one_req(pending_req, cache_handle, table_idx, cache_handle.cached_len)
    return None
```

分块后的续算不重新分配资源，直接复用上一块的 `cache_handle` 和 `table_idx`，从 `cached_len` 断点接着算。

### 2.4 `ChunkedReq`：不能采样的半成品

```python
class ChunkedReq(Req):
    def append_host(self, next_token): raise NotImplementedError("ChunkedReq should not be sampled")
    @property
    def can_decode(self): return False   # 避免被加进 decode manager
```

一块 prefill 算完只产生了中间 KV，**还没到采样点**（prompt 都没读完），所以 chunk 请求不能采样、不能进 decode。这两个 override 就是「封印」它，让它在 Step 8 的 `_process_last_data` 里被 `continue` 跳过、在 Step 12 的 `filter_reqs` 里被剔出 decode 集合。

### 2.5 `schedule_next_batch`：pending_list 的重组

```python
def schedule_next_batch(self, prefill_budget):
    if len(self.pending_list) == 0:
        return None
    adder = PrefillAdder(token_budget=prefill_budget,
                         reserved_size=self.decode_manager.inflight_tokens, ...)
    reqs, chunked_list = [], []
    for pending_req in self.pending_list:
        if req := adder.try_add_one(pending_req):
            pending_req.chunked_req = None
            if isinstance(req, ChunkedReq):
                pending_req.chunked_req = req
                chunked_list.append(pending_req)         # 还没算完的，留到下一轮
            reqs.append(req)
        else:
            break                                        # 加不动了，停下
    if len(reqs) == 0:
        return None
    self.pending_list = chunked_list + self.pending_list[len(reqs):]
    return Batch(reqs=reqs, phase="prefill")
```

注意最后的 `pending_list = chunked_list + pending_list[len(reqs):]`：**没算完的 chunk 请求排在最前**，下一轮优先续算（保证长 prompt 能连续推进，而不是被新请求插队饿死）。

---

## 三、难点解析

### 难点 1：为什么「锁后再查一次」？

`_try_allocate_one` 里先查一次 `available_size`，`lock(handle)` 后又查一次。因为这个函数在一个循环里被反复调用，**两次检查之间没有全局锁**——其他 prefill 请求可能已经把空间占走了。

`lock(handle)` 锁的是「前缀缓存句柄」，防止这段已匹配的前缀在分配期间被驱逐。锁完之后再查一次，如果空间确实不够了，`unlock` 回滚。这是典型的「check-lock-recheck」防竞态模式。

### 难点 2：`cached_len > 0` 时要拷两样东西

前缀复用命中后（`cached_len > 0`），要把「别人已经算好的那部分」搬到自己名下：

- `token_pool[table_idx][:cached_len]` 拷 **token id**（账本对齐）；
- `page_table[table_idx][:cached_len]` 拷 **KV 物理页索引**（`handle.get_matched_indices()`），让这个请求的 page_table 指向那段已存在的 KV。

这样这个请求 forward 时 `extend_len = device_len - cached_len` 只算新部分，旧部分直接读缓存——这就是 Step 6 讲 `cached_len` 的用武之地。

### 难点 3：`token_budget` 和 `reserved_size` 是两个不同的预算

- `token_budget`：**算力预算**（一轮 prefill 最多新算多少 token），每加一个请求就扣 `chunk_size`，扣完这轮就不再加。控制的是「这一轮 GPU 要算多少」，防止 prefill 独占算力饿死 decode。
- `reserved_size`：**空间预算**（留给 in-flight decode 的 KV 空间），初始 = `decode_manager.inflight_tokens`，每加一个请求就累加它将来要占的空间。控制的是「未来会不会爆显存」。

两者都叫「预算」，但一个管时间、一个管空间。

### 难点 4：`pending_list` 为什么要把 `chunked_list` 放最前？

设想一个 100K 的长 prompt 被切成 10 块，如果每轮都重新从 `pending_list` 头部开始、新请求插进来，这个长 prompt 可能被无限插队，永远算不完。把 `chunked_list`（本轮没算完的）提到队首，下一轮 `schedule_next_batch` 第一个就处理它，保证长 prompt 的连续性。

---

## 四、注意事项

1. **`match_req` 匹配的是 `input_ids[: input_len - 1]`**：最后一个 token 的 KV 还没算出来，不能作为缓存前缀（详见 Step 13 的 `match_req`）。
2. **`ChunkedReq` 继承 `Req` 但改了 `can_decode` 和 `append_host`**：任何把它当正常请求用的地方（采样、拼 host、进 decode）都会被拦下。
3. **`estimated_len = extend_len + output_len` 是「这个请求总共还要占多少」**：`extend_len`（新算的）+ `output_len`（将来 decode 生成的），两者都要预留 KV 空间。
4. **准入失败的请求不丢**：`schedule_next_batch` 里 `break` 后，未处理的留在 `pending_list`，下一轮再试。
5. **`pending_req.chunked_req = None` 在成功后清空**：一个请求一旦整段算完（`Req` 而非 `ChunkedReq`），就不再需要 chunk 标记。

---

## 五、反思题

1. 三个准入条件（槽位、空间、预算）分别防的是什么问题？如果只留「空间」一个条件，会发生什么？
2. `chunk_size = min(token_budget, remain_len)` 里 `token_budget` 是从哪来的？它和 `--max-prefill-length` 参数是什么关系？
3. 为什么 `ChunkedReq` 要用 `append_host` 抛异常 + `can_decode` 返回 False **两处**封印？只改一处会怎样？
4. `reserved_size` 初始值是 `decode_manager.inflight_tokens`，为什么 prefill 准入要给 decode 预留空间？（提示：decode 的请求每生成一个 token 都要占新 KV）
5. 一个 100 个 token 的 prompt、`token_budget=30`、无前缀复用，会被切成几块？每块的 `cached_len` / `device_len` 分别是多少？

---

## 六、示意图

### 6.1 长 prompt 的分块过程

```
  prompt: [0,1,2,3,4,5,6,7,8,9]   （10 个 token），token_budget=6

  第 1 轮: chunk_size=min(6,10)=6 → ChunkedReq(cached=0, device=6)
           算 [0,6)，剩余 [6,10) 还在 pending_list

  第 2 轮: 续算 chunk_size=min(6,4)=4 → Req(cached=6, device=10)
           算 [6,10)，整段完成 → 进入 decode
```

### 6.2 `try_add_one` 的两条路径

```
  pending_req
     │
     ├─ 有 chunked_req?（上一轮切过）──► 复用 handle/table_idx，从 cached_len 续算
     │
     └─ 无 ──► _try_allocate_one（三条件准入）
                    │ 通过 ──► _add_one_req（chunk_size 分块）
                    │ 失败 ──► None（留待下一轮）
```

### 6.3 准入的「空间」检查

```
  cache.available_size（剩余 KV 空间）
  │
  ├─ estimated_len（这个请求 extend+output 要占）
  ├─ reserved_size（in-flight decode 将来要占）
  │
  若 estimated_len + reserved_size > available_size → 拒绝
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [scheduler/prefill.py](python/minisgl/scheduler/prefill.py) | prefill 调度 + 分块 | `PrefillManager`、`PrefillAdder`、`ChunkedReq` |
| [scheduler/utils.py](python/minisgl/scheduler/utils.py) | 待处理请求 | `PendingReq` |

**下一步**：进入 Step 12（DecodeManager 与 TableManager），看正在生成的请求怎么被管理、`table_idx` 逻辑槽位和物理 KV 页怎么解耦。
