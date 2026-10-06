# Step 23：采样

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 23。
> 核心文件：[engine/sample.py](python/minisgl/engine/sample.py) 的 `Sampler`。
>
> 这一 Step 回答：**模型吐出的 logits 是怎么变成下一个 token 的？「全 greedy」的 batch 为什么有特判？batch 里不同请求不同采样参数是怎么处理的？**

---

## 一、这个 Step 要解决什么

`forward_batch` 拿到 `logits` 后（Step 14），最后一步是 `self.sampler.sample(logits, args)`。这一步把 `[batch, vocab_size]` 的 logits 变成 `[batch]` 的 token id。

难点不在「怎么采样」（`argmax` 或 `softmax`+随机），而在「**batch 里每个请求的采样参数不同**」——有的请求贪心、有的要 temperature=0.7、有的要 top_p。这些逐请求的参数怎么高效地一起算。

---

## 二、核心逻辑

### 2.1 `BatchSamplingArgs`：逐请求的采样参数

```python
@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
```

三个字段都是**张量**（不是标量），长度 = batch 里的请求数。每个请求一行，对应它自己的采样参数。这就是「batch 内不同请求不同采样」的实现方式。

### 2.2 `prepare`：把参数收集成张量 + greedy 特判

```python
def prepare(self, batch):
    params = [r.sampling_params for r in batch.reqs]
    if all(p.is_greedy for p in params):          # ① 全 greedy 特判
        return BatchSamplingArgs(temperatures=None)

    MIN_P = MIN_T = 1e-6
    ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
    top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
    top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
    temperatures = make_device_tensor(ts, torch.float32, self.device)
    top_k, top_p = None, None
    if any(k != self.vocab_size for k in top_ks):
        top_k = make_device_tensor(top_ks, torch.int32, self.device)
    if any(p < 1.0 for p in top_ps):
        top_p = make_device_tensor(top_ps, torch.float32, self.device)
    return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)
```

两个关键：

- **① 全 greedy 特判**：如果 batch 里所有请求都是贪心（`temperature <= 0` 或 `top_k == 1`，Step 6 的 `is_greedy`），直接返回 `temperatures=None`——**跳过所有 softmax/top_k/top_p 的准备**，后面走 `argmax`。
- **② 逐请求张量**：非 greedy 时，把每个请求的 temperature/top_k/top_p 收集成张量。`top_ks` 里 `top_k < 1`（即 `-1` 表示禁用）时替换成 `vocab_size`（等于「不过滤」）；`top_ps` 里 `top_p >= 1` 时也视为「不过滤」。

### 2.3 `sample`：greedy 走 argmax，随机走 flashinfer

```python
def sample(self, logits, args):
    if args.temperatures is None:        # ① 全 greedy
        return torch.argmax(logits, dim=-1)
    return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
```

- **greedy**：`torch.argmax` 直接取概率最大的 token，**不做 softmax**（argmax 不需要归一化），最快。
- **随机**：调 `sample_impl`，先 softmax 再按 top_k/top_p 采样。

### 2.4 `sample_impl`：softmax + top_k/top_p 组合

```python
def sample_impl(logits, temperatures, top_k, top_p):
    import flashinfer.sampling as sampling
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)
    if top_p is None:
        return sampling.top_k_sampling_from_probs(probs, top_k)
    if top_k is None:
        return sampling.top_p_sampling_from_probs(probs, top_p)
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)
```

四个分支对应四种组合：无过滤、只 top_k、只 top_p、top_k+top_p。都调 flashinfer 的采样 kernel（GPU 上高效采样）。

---

## 三、难点解析

### 难点 1：为什么「全 greedy」要特判？

三个理由：

1. **性能**：greedy 用 `argmax`，不用 softmax（softmax 有指数运算，慢），也不用 flashinfer 采样 kernel。这是 decode 最常见路径（默认 `temperature=0`），特判让它最快。
2. **避免构造无用张量**：全 greedy 时 `top_k`/`top_p` 都是「不过滤」，构造这些张量纯浪费。
3. **避免边界值问题**：greedy 的 `temperature=0` 会让 softmax 除零，特判直接绕开。

`is_greedy` 的定义（Step 6）：`(temperature <= 0 or top_k == 1) and top_p == 1.0`。注意 `top_k == 1` 也算 greedy（只能选 top1 = argmax）。

### 难点 2：`temperatures` 是逐请求张量，softmax 怎么处理？

flashinfer 的 `sampling.softmax(logits, temperatures)` 接受**逐元素**的 temperature（每个请求一个温度），内部对第 i 个请求用 `temperatures[i]` 除它的 logits。

这就是为什么 `prepare` 要把 `temperature` 收集成张量——batch 里请求 A 是 0.7、请求 B 是 0（greedy），各自用各自的温度。flashinfer 的 kernel 天然支持这种 per-element 参数。

### 难点 3：`top_k = -1` 和 `top_p = 1.0` 的「禁用」语义

`SamplingParams` 里 `top_k = -1`、`top_p = 1.0` 表示「不启用这个过滤」（Step 6 的默认值）。`prepare` 里把它们规范化：

- `top_k < 1`（即 `-1`）→ 替换成 `vocab_size`（相当于「保留所有 token」，即不过滤）；
- `top_p >= 1.0` → 视为不过滤。

然后 `if any(k != vocab_size)` 判断「是否真的有人用了 top_k」，没有就 `top_k = None`，`sample_impl` 走「无 top_k」分支。

### 难点 4：`MIN_P = MIN_T = 1e-6` 的边界处理

```python
ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
```

- `temperature` 夹到 `>= 1e-6`：避免除零（`temperature=0` 时 logits/temperature 会爆）。
- `top_p` 夹到 `[1e-6, 1.0]`：避免 0 或 >1 的非法值。

这些是防御性的边界钳制，保证喂给 flashinfer 的值合法。

### 难点 5：`enable_pdl=is_sm90_supported()` 是什么？

```python
probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
```

`enable_pdl`（Power Distribution Linearization? 实际是 flashinfer 的一个优化开关）在 SM90（Hopper）及以上硬件启用，用更快的 softmax 路径。`is_sm90_supported()` 检测硬件是否支持，不支持的硬件（如 A100）关闭。

---

## 四、注意事项

1. **`logits.float()` 强制转 float32**：模型 logits 可能是 bf16/fp16，采样前转 float32 保证数值精度（softmax 对低精度敏感）。
2. **`make_device_tensor` 用 `pin_memory` + `to(device, non_blocking)`**：CPU 上的参数张量异步拷到 GPU，减少阻塞（Step 9 的 overlap 思想）。
3. **`sample` 返回的是 `[batch]` 的 token id**：不是 one-hot 或概率分布，就是最终要生成的 token。
4. **greedy 的 `argmax(dim=-1)` 沿 vocab 维**：`logits` 形状 `[batch, vocab]`，`dim=-1` 就是每个请求在 vocab 上取最大。
5. **`top_k`/`top_p` 的 `None` 传播**：`prepare` 里 `if any(...)` 才构造，`sample_impl` 里 `if top_k is None` 判断走哪个分支，两者要一致。

---

## 五、反思题

1. 为什么 greedy 用 `torch.argmax` 而不是先 softmax 再取最大？softmax 是单调变换，对 argmax 有影响吗？
2. `prepare` 里 `top_k < 1` 替换成 `vocab_size` 的语义是什么？为什么 `-1` 和 `vocab_size` 都表示「不过滤」？
3. 如果 batch 里一半 greedy、一半 temperature=0.7，`temperatures` 张量长什么样？greedy 那几行是什么值？
4. `enable_pdl=is_sm90_supported()` 这个开关在什么硬件上打开？为什么要在非 SM90 上关闭？
5. 采样结果 `next_token` 在 Step 9 的 `_process_last_data` 里被拿去做了什么？（提示：`append_host`、EOS 判断）

---

## 六、示意图

### 6.1 采样两条路径

```
  logits [batch, vocab]
        │
        ├─ args.temperatures is None（全 greedy）
        │     └─ torch.argmax(dim=-1)  ──► next_token
        │
        └─ 随机采样
              ├─ softmax(logits, temperatures)
              ├─ top_k 过滤（可选）
              ├─ top_p 过滤（可选）
              └─ sampling_from_probs ──► next_token
```

### 6.2 `prepare` 的逐请求参数

```
  batch.reqs:  [A(temperature=0), B(temperature=0.7, top_p=0.9), C(temperature=0)]
        │
        ▼
  temperatures = [0.0, 0.7, 0.0]      （A/C 是 greedy，B 是随机）
  top_p        = None 或 [1.0, 0.9, 1.0]（取决于是否有人用 top_p）
```

### 6.3 logits → token 的完整链路

```
  model.forward ──► logits ──► sampler.sample ──► next_tokens_gpu
                                                    │ .to("cpu", non_blocking)
                                                    ▼
                                              next_tokens_cpu（Step 9 处理）
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [engine/sample.py](python/minisgl/engine/sample.py) | 采样 | `Sampler`、`sample_impl`、`BatchSamplingArgs` |
| [core.py](python/minisgl/core.py) | 采样参数定义 | `SamplingParams`、`is_greedy` |

**下一步**：进入 Step 24（完整生命周期串讲），把 Step 0.3 的 8 步主线串成一条线，看全貌。
