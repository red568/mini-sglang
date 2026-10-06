# Step 24：完整生命周期串讲 + 进阶地图

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 24。
> 这是全系列最后一篇：把 Step 1~23 连成一条线，再看四个可深入的进阶方向。
>
> 这一 Step 回答：**一个请求从 HTTP 进来、到 SSE 吐字出去，中间到底穿过了哪些进程、哪些函数、哪些数据结构？读完主干后，想再往上走，该往哪四个方向钻？**

---

## 一、这个 Step 要解决什么

前面 23 个 Step 是「横切面」——每个 Step 聚焦一个模块。但真正读懂一个推理引擎，要能**纵向串起来**：给我一个请求，我能一口气讲出它走过的每一步、每一步在哪个进程、调用哪个函数、传什么消息、改什么数据结构。

本 Step 做两件事：

1. **串讲**：对着 Step 0.3 的 8 步主线，把 23 个 Step 全部挂上去，验证「主线是否打通」。
2. **进阶地图**：主干走完后，四个可深入的硬核方向（MoE / 自定义 kernel / 通信 / 离线接口）。

---

## 二、串讲：一个请求的完整生命周期

> 对应 Step 0.3 的时序图。下面每一步标注「进程 → 函数 → 消息/数据结构 → 对应 Step」。

### ① HTTP 请求进 API Server

- **进程**：主进程（API Server，FastAPI / uvicorn）
- **函数**：[api_server.py](python/minisgl/server/api_server.py) 的 `v1_completions`（`/v1/chat/completions` 端点）
- **做了什么**：解析请求体 → 构造 `SamplingParams`（Step 23 的逐请求采样参数源头）→ `new_user` 分配 `uid` → 调 `wait_for_ack` 建一个异步生成器，等着逐 token 吐结果。
- **对应 Step**：Step 4（前端）、Step 6（`SamplingParams`）

### ② API Server → tokenizer：`TokenizeMsg`

- **进程**：主进程 → tokenizer 子进程
- **函数**：`new_user` → `send_one`
- **消息**：`TokenizeMsg`（原始文本 + sampling_params + uid）
- **通道**：ZMQ `minisgl_4`（共享模式下 = `minisgl_1`），tokenizer bind / API Server connect（Step 0.2 的速查表）。
- **对应 Step**：Step 3（消息定义）、Step 4（发送端）

### ③ tokenizer → rank0：`UserMsg`

- **进程**：tokenizer 子进程 → Scheduler rank 0
- **函数**：`TokenizeManager.tokenize`（Step 5）→ `send_backend.put`
- **消息**：`UserMsg`（`input_ids` 是 **CPU 上的 int32 张量** + sampling_params + uid）
- **关键动作**：`apply_chat_template`（是否套模板）→ `encode` 转 int32 → 通过两段式序列化（`serialize_type` → msgpack，Step 3）跨进程传。
- **对应 Step**：Step 5（tokenize）、Step 3（序列化）

### ④ rank0 → 其他 rank：PUB/SUB 广播（仅多卡）

- **进程**：Scheduler rank 0 → rank 1..N
- **函数**：[scheduler/io.py](python/minisgl/scheduler/io.py) 的 `_recv_msg_multi_rank0`
- **关键动作**：rank0 收到 `UserMsg` 后 `put_raw` 广播给所有 rank，再用 `torch.distributed.broadcast` 同步「消息条数」——保证所有 rank 看到**同样的请求顺序**（对应 commit `9a91cfa` 修复的 decode 顺序问题）。
- **对应 Step**：Step 10（I/O 与广播）

### ⑤ 各 rank 本地：调度 + 前向 + 采样（最重的一步）

这一步是核心，拆成四段：

**5a. 调度（Step 8/9/11/12/13）**

- `normal_loop` / `overlap_loop` 的主循环四动作：`receive_msg` → `_schedule_next_batch` → `_forward` → `_process_last_data`。
- `_schedule_next_batch` 内部：
  - **PrefillManager**（Step 11）：三个准入条件（table 有槽位、cache 有空间、token_budget 够）→ 长 prompt 切成 `ChunkedReq`。
  - **DecodeManager**（Step 12）：从 `running_reqs` 里按 uid 排序挑 decode 请求，reserve `inflight_tokens`。
  - **TableManager**（Step 12）：分配 `table_idx` 逻辑槽位。
  - **CacheManager**（Step 13）：`allocate_paged` 写 page_table，`cache_req` 做前缀复用（命中 Radix Cache 时释放多占的页）。
- 组装出 `Batch`（Step 7）：`reqs`/`phase`/`input_ids`/`positions`/`out_loc` 等字段在这一步被填上（`_prepare_batch`，Step 8）。

**5b. 模型前向（Step 14/15/16/17/18）**

- `forward_batch`（Step 14）：`can_use_cuda_graph` 决定走 CUDA graph replay 还是正常 forward。
- 模型 `forward()` **不带参数**（Step 7 的全局 ctx，Step 15 的 `BaseLLMModel`）：`embed_tokens → N 层 decoder → norm → lm_head`。
- 每层 `LlamaDecoderLayer`（Step 15）：`input_layernorm → self_attn → post_attention_layernorm → mlp`，残差流传递。
- 注意力层（Step 18）：`qkv.split → RoPE → attn_backend.forward`。
- 权重加载/张量并行（Step 16/17）：`BaseOP` 的 `__dict__` 递归；`LinearRowParallel` 的 all-reduce。

**5c. KV Cache 读写（Step 19/20/21）**

- 注意力后端 `forward`（Step 21）：先 `store_kv`（Step 19）把本层 K/V 写进 `_kv_buffer`，再从缓存读完整 KV 算注意力；prefill 走 FlashAttention、decode 走 FlashInfer（`HybridBackend` 分发）。
- Radix Cache（Step 20）：`insert_prefix` 缓存新前缀、`match_prefix` 复用旧前缀、`evict` 按 LRU 驱逐。

**5d. 采样（Step 23）**

- `Sampler.sample`：全 greedy 走 `torch.argmax`，随机走 flashinfer 的 softmax + top_k/top_p。
- 产出 `next_tokens_gpu` → `.to("cpu", non_blocking)` 异步拷回（Step 14）。

**CUDA Graph 的加速贯穿 5b~5d（Step 22）**：decode 阶段整个 forward 被「录」成一张图，`replay` 一次重放。

### ⑥ rank0 → detokenizer：`DetokenizeMsg`

- **进程**：Scheduler rank 0 → detokenizer
- **函数**：`_process_last_data`（Step 9）→ `send_result`
- **关键动作**：拿到 `next_tokens_cpu` → `append_host`（拼进 Req 的 `input_ids`，Step 6）→ 判断 EOS/`max_tokens` 决定 `finished` → 构造 `DetokenizeMsg`（含 `next_token`、`finished`）→ `lazy_free_region` 释放已完成的 KV 页。
- **对应 Step**：Step 9（overlap）、Step 6（`append_host`）

### ⑦ detokenizer → API Server：`UserReply`

- **进程**：detokenizer → 主进程
- **函数**：`DetokenizeManager.detokenize`（Step 5）→ `send_frontend.put`
- **关键动作**：用 `read_offset` / `surr_offset` / `sent_offset` 三个偏移量做**流式增量 detokenize**（按完整词 / 中文字符边界吐字），拼出 `UserReply`。
- **对应 Step**：Step 5（detokenize 的难点核心）

### ⑧ API Server → 用户：SSE 流式返回

- **进程**：主进程 → 客户端
- **函数**：`stream_chat_completions`
- **关键动作**：`listen` 后台任务收 `UserReply`，`wait_for_ack` 生成器逐 token 产出，拼成 SSE chunk 发回。
- **对应 Step**：Step 4（前端）

> **串讲自检**：如果能不看笔记，把上面 8 步的「进程 → 函数 → 消息 → 数据结构」一口气讲下来，说明主线已打通。哪一步卡壳，就回对应的 Step 重读。

---

## 三、进阶地图（四个可深入方向）

主干走完，剩下的硬核方向都在「和 GPU / 通信 / 接口打交道」上。按兴趣选一条往下钻。

### 方向 1：MoE（专家混合）

- **入口**：[models/qwen3_moe.py](python/minisgl/models/qwen3_moe.py) 的 `Qwen3MoeForCausalLM`——和 Llama 的差异只在 MLP 换成 MoE。
- **结构**：[models/utils.py](python/minisgl/models/utils.py) 的 `MoEMLP`：`gate`（`LinearReplicated` 算出每个 token 对每个专家的路由分）→ `experts`（`MoELayer`）。
- **调度后端**：[layers/moe.py](python/minisgl/layers/moe.py) 的 `MoELayer.forward`——它自己**不实现计算**，而是把 `hidden_states` / `w1` / `w2` / `router_logits` 甩给 `ctx.moe_backend.forward`（和注意力层「甩给 attn_backend」是同一个模式，Step 18/21）。
- **后端实现**：[moe/base.py](python/minisgl/moe/base.py)（接口 `BaseMoeBackend`）→ [moe/fused.py](python/minisgl/moe/fused.py) 的 `FusedMoe`：
  1. `fused_topk`：对每个 token 取 top-k 专家，softmax 归一化权重；
  2. `moe_align_block_size`：把 token 按专家排序、padding 到 block size 对齐；
  3. `fused_experts_impl`：两次 `fused_moe_kernel_triton`（gate_up 投影 + down 投影）+ `moe_sum_reduce_triton`（把多个专家的结果加权求和）。
- **真正的 kernel**：[kernel/triton/fused_moe.py](python/minisgl/kernel/triton/fused_moe.py)（Triton 写的 kernel）+ [kernel/moe_impl.py](python/minisgl/kernel/moe_impl.py)（Python 封装）。

**要搞懂的关键点**：MoE 的核心不是「怎么算」，而是「**每个 token 只激活 top-k 个专家**」带来的稀疏计算，以及怎么把「token → 专家」的分配做成 GPU 友好的连续访存（`moe_align_block_size` 的排序 + padding）。

### 方向 2：自定义 CUDA kernel + JIT 编译

项目没有把 CUDA kernel 预编译成 `.so`，而是**运行时 JIT**。核心在 [kernel/utils.py](python/minisgl/kernel/utils.py)：

- `load_aot`：编译 `kernel/csrc/src/` 下的 `.cpp` / `.cu` 文件，通过 `tvm_ffi.cpp.load` 加载；
- `load_jit`：`load_inline` 直接内联源码（`#include "xxx.cu"`）+ 包装 `TVM_FFI_DLL_EXPORT_TYPED_FUNC` 导出。
- 两个典型例子：
  - [kernel/radix.py](python/minisgl/kernel/radix.py) 的 `fast_compare_key`：C++ 写的最长前缀比较（Step 20 的 `_tree_walk` 靠它找分叉点），纯 Python 写这个会慢几十倍。
  - [kernel/store.py](python/minisgl/kernel/store.py) 的 `store_cache`：按 `element_size` 模板参数 JIT 出 `StoreKernel<N>::run`，把 K/V 按 `out_loc` 写进缓存池（Step 19）。

**要搞懂的关键点**：`functools.cache` 保证同一个 kernel 只编译一次；`make_cpp_args` 把 Python 的 dtype/形状转成 C++ 模板参数，做到「按形状特化 kernel」。这套「JIT + 模板特化」是 vLLM/SGLang 的通行做法。

### 方向 3：通信（TP 的底层）

- [distributed/impl.py](python/minisgl/distributed/impl.py)：`DistributedCommunicator` 维护一个 `plugins` 列表（插件栈），`all_reduce` / `all_gather` 总是调**最后一个**插件。
  - `TorchDistributedImpl`：默认用 `torch.distributed`；
  - `PyNCCLDistributedImpl`：换成项目自己的 `PyNCCLCommunicator`。
  - `enable_pynccl_distributed`：把 PyNCCL 插件 append 到栈顶，之后所有 all-reduce 走 PyNCCL。
- [kernel/pynccl.py](python/minisgl/kernel/pynccl.py)：`init_pynccl` 里 rank0 用 `create_nccl_uid()` 生成 NCCL 唯一 ID，通过 `broadcast_object_list` 分发给其他 rank，再 `tvm_ffi.register_object` 包一个 `NCCLWrapper` 的 FFI 对象。
- 这套「插件栈」设计让通信后端**可插拔**，Step 17 的 `LinearRowParallel.all_reduce` 就是调 `DistributedCommunicator().all_reduce`，底层换插件不影响上层。

**要搞懂的关键点**：为什么要有自己的 PyNCCL 而不直接用 `torch.distributed`——为了和 CUDA graph 配合（能在 graph 捕获时固定通信算子），以及更细的显存控制（`max_size_bytes` / `PYNCCL_MAX_BUFFER_SIZE`）。

### 方向 4：离线 Python 接口（`LLM`）

[llm/llm.py](python/minisgl/llm/llm.py) 的 `LLM(Scheduler)`：

- 它**直接继承 Scheduler**，把「多进程 + ZMQ」这套东西**替换成进程内函数调用**：`offline_receive_msg` 模拟「从 tokenizer 收 UserMsg」（自己 tokenize + 造 UserMsg），`offline_send_result` 模拟「发 DetokenizeMsg 给 detokenizer」（自己收集 token）。
- `generate(prompts, sampling_params)`：把 pending 请求喂进去 → `run_forever()`（复用整个调度器主循环）→ `RequestAllFinished` 异常退出 → 收结果。
- 这是「在线服务」和「离线 batch 推理」**共用同一套调度器**的典型设计——`offline_mode=True` 时，I/O 层被换成内存内的模拟。

**要搞懂的关键点**：`offline_receive_msg` / `offline_send_result` 这两个「假 I/O」是怎么插进 `normal_loop` 的 `receive_msg` / `send_result` 位置的（通过 override 基类方法，多态替换）。

---

## 四、难点解析

### 难点 1：为什么串讲要「纵向」而不是「横向」？

读一个模块（横向）只能知道「它在干什么」，读一条请求路径（纵向）才能知道「它**为什么**被这样设计」。比如：

- 你横向读懂了 `Req` 的三个长度（Step 6），但只有纵向串起来，才能理解「`cached_len` 是 Radix Cache 命中时（Step 20）填的、`device_len` 是每次 forward 后（Step 9）递增的、`extend_len` 决定 prefill 算多长（Step 21）」。
- 横向读懂了 `Context` 是全局单例（Step 7），但只有纵向看到「模型 `forward()` 不带参数 → 靠 `get_global_ctx().batch`」，才知道这个设计是为了**让 forward 能被 CUDA graph 录制**（Step 22 里 capture 时 `with get_global_ctx().forward_batch(batch)`）。

**所以串讲不是「复习」，是「把 23 个 Step 的知识点用一条请求的流动重新组织一遍」。**

### 难点 2：⑤ 是「重活」，但它内部是层层委托

第 ⑤ 步（调度+前向+采样）占了大半代码，但它的核心结构是**三层委托**：

```
Scheduler._forward
  └─ Engine.forward_batch          （Step 14，组装 + 采样）
       └─ model.forward()           （Step 15，残差流）
            └─ AttentionLayer       （Step 18，切分+RoPE）
                 └─ attn_backend    （Step 21，真正算注意力）
            └─ MoELayer（若 MoE）    （进阶，切路由）
                 └─ moe_backend     （进阶，真正算专家）
```

理解这个「每层只做一点点、把重活往下甩」的委托链，是读懂整个 engine 的关键。注意力层（Step 18）和 MoE 层（进阶）是**同一个设计模式**：层只负责「切分 + 准备参数」，kernel 级的重活全交给可插拔 backend。

### 难点 3：五个「异步边界」把系统串成流水线

请求穿过整个系统时，有五个地方发生了「异步」（不是同步等着）：

1. ②③ 之间：API Server 发完 `TokenizeMsg` 不等，靠 `ack_map` 异步收（Step 4）；
2. ⑤ 内部：`overlap_loop` 把 CPU 处理藏进 GPU 前向的阴影（Step 9）；
3. ⑤ 尾部：`next_tokens_gpu.to("cpu", non_blocking)` 异步拷回（Step 14）；
4. ⑤ 内部 KV：`store_kv` 用 `out_loc` 的 gather/scatter（Step 19）；
5. ⑦⑧ 之间：detokenize 是流式的，按词边界增量吐（Step 5）。

这五个异步点是性能的关键——**主线不是「同步函数调用链」，而是「消息 + 异步回调 + 双 stream 重叠」**。

### 难点 4：进阶方向的「统一模式」——backend 接口 + JIT kernel

四个进阶方向看似不相关，但底层有统一的模式：

- **可插拔 backend**：注意力（Step 21）和 MoE（方向 1）都是「接口 + 多实现」，通过 `ctx.attn_backend` / `ctx.moe_backend` 二选一；
- **JIT kernel**：所有自定义算子（`fast_compare_key`、`store_cache`、MoE triton）都走 `load_aot` / `load_jit`（方向 2）；
- **插件栈**：通信的 `plugins` 列表（方向 3）也是「可插拔」思路的变体。

看懂一个，就懂了所有——**这个项目的「架构基因」就是「接口抽象 + 可插拔实现 + JIT 特化」**。

---

## 五、注意事项

1. **串讲时每个 Step 的「对应 Step」不是硬绑定**：有些功能跨多个 Step（比如 KV cache 读写同时涉及 Step 19/20/21），别把 Step 当「只有一处」的标签。
2. **`overlap_loop` 和 `normal_loop` 的 ⑤⑥ 顺序相反**（Step 9）：串讲时别默认「先处理结果再算下一批」，overlap 模式下是「先算下一批、再处理上一批结果」。
3. **只有 rank0 对外通信**（Step 10）：串讲第 ③④⑥⑦ 步时，要清楚「④ 广播后每个 rank 都跑 ⑤，但只有 rank0 跑 ⑥」。
4. **MoE 的 `all_reduce` 在 `MoELayer.forward` 末尾**（进阶方向 1）：`tp_size > 1` 时才做，和 `LinearRowParallel` 的 all-reduce 一样是「按行切需要归约」。
5. **`load_jit` 和 `load_aot` 的区别**（进阶方向 2）：`load_jit` 是 `load_inline`（内联源码 + 包装导出），`load_aot` 是 `load`（编译 `.cpp/.cu` 文件），别混。
6. **离线接口 `LLM` 仍要 `run_forever`**（进阶方向 4）：它没绕开调度器主循环，只是替换了 I/O 层。

---

## 六、反思题

1. 不看笔记，从「用户 POST」到「SSE 返回」，默写 8 步的「进程 → 函数 → 消息 → 数据结构」。哪一步卡壳，回对应 Step 重读。
2. 为什么说「⑤ 是层层委托」？把 `Scheduler._forward → Engine.forward_batch → model.forward → AttentionLayer → attn_backend` 这条链每层「做了什么、没做什么」说清楚。
3. 五个异步边界分别在哪？如果把它们都改成「同步等待」，吞吐会怎么变？（提示：哪个是最大瓶颈？）
4. MoE 和注意力层在「委托给 backend」这件事上是同一个模式。两者的 backend 接口分别定义在哪个文件、有哪些方法？
5. `LLM.generate` 怎么复用 `Scheduler.run_forever` 的？`offline_receive_msg` 和 `offline_send_result` 分别顶替了 `normal_loop` 里的哪两个动作？

---

## 七、示意图

### 7.1 完整生命周期（8 步 + 对应 Step）

```
 用户 ──①POST──► API Server ──②TokenizeMsg──► tokenizer ──③UserMsg──► rank0 ──④广播──► rank1..N
   ▲              │(Step4)                       │(Step5,3)             │(Step10)
   │              │                              │                     ▼
   │              │                              │               ┌─────────────┐
   │              │                              │               │ ⑤ 调度+前向+采样 │
   │              │                              │               │ (Step8,9,11,12,13│
   │              │                              │               │  14,15,16,17,18 │
   │              │                              │               │  19,20,21,22,23)│
   │              │                              │               └──────┬──────┘
   │              │                              │                      │⑥DetokenizeMsg
   │              │                              │◄─────────────────────┘(Step9)
   │              │◄────────⑦UserReply───────────┤ (detokenize, Step5)
   │◄─────⑧SSE────┤ (stream_chat_completions, Step4)
```

### 7.2 ⑤ 内部的委托链（三层）

```
 Scheduler._schedule_next_batch ──► 组装 Batch（Step 8/11/12/13）
        │
        ▼
 Scheduler._forward ──► Engine.forward_batch（Step 14）
        │                    ├─ can_use_cuda_graph? ──► replay（Step 22）
        │                    └─ model.forward()（无参数，靠全局 ctx）
        ▼
 model.forward ──► embed → N×decoder → norm → lm_head（Step 15）
        │
        ▼
 AttentionLayer.forward ──► qkv.split → RoPE → attn_backend.forward（Step 18）
        │                                          │
        │                                          ├─ store_kv（Step 19）
        │                                          ├─ 读 KV 算注意力（Step 21）
        │                                          └─ Radix 复用（Step 20）
        ▼
 Sampler.sample ──► argmax / flashinfer（Step 23）
```

### 7.3 进阶地图四方向

```
                    ┌─ 方向1 MoE：models/qwen3_moe.py → layers/moe.py → moe/fused.py → kernel/triton/fused_moe.py
 主干（Step 1~24）─┼─ 方向2 kernel：kernel/utils.py(load_aot/load_jit) → kernel/radix.py / store.py
                    ├─ 方向3 通信：distributed/impl.py(插件栈) → kernel/pynccl.py(NCCLWrapper)
                    └─ 方向4 离线：llm/llm.py(LLM 继承 Scheduler，override I/O)
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [models/qwen3_moe.py](python/minisgl/models/qwen3_moe.py) | MoE 模型 | `Qwen3MoeForCausalLM`、`Qwen3DecoderLayer` |
| [models/utils.py](python/minisgl/models/utils.py) | MoE/Gated MLP | `MoEMLP`、`GatedMLP` |
| [layers/moe.py](python/minisgl/layers/moe.py) | MoE 层（委托后端） | `MoELayer` |
| [moe/base.py](python/minisgl/moe/base.py) | MoE 后端接口 | `BaseMoeBackend` |
| [moe/fused.py](python/minisgl/moe/fused.py) | Fused MoE 实现 | `FusedMoe`、`fused_topk`、`moe_align_block_size` |
| [kernel/triton/fused_moe.py](python/minisgl/kernel/triton/fused_moe.py) | Triton MoE kernel | `fused_moe_kernel_triton` |
| [kernel/utils.py](python/minisgl/kernel/utils.py) | JIT/AOT 编译 | `load_aot`、`load_jit`、`KernelConfig` |
| [kernel/radix.py](python/minisgl/kernel/radix.py) | 前缀比较 kernel | `fast_compare_key` |
| [kernel/store.py](python/minisgl/kernel/store.py) | KV 写 kernel | `store_cache` |
| [distributed/impl.py](python/minisgl/distributed/impl.py) | 通信插件栈 | `DistributedCommunicator`、`PyNCCLDistributedImpl` |
| [kernel/pynccl.py](python/minisgl/kernel/pynccl.py) | PyNCCL 封装 | `init_pynccl`、`PyNCCLCommunicator` |
| [llm/llm.py](python/minisgl/llm/llm.py) | 离线接口 | `LLM`、`offline_receive_msg`、`generate` |

**全系列完**。至此 Step 1~24 全部覆盖。建议：先按 Step 0 跑通 `--dummy-weight --shell`，再对着本系列把主线串一遍，最后从「二次开发实战」表里挑一个 L1~L2 的项目动手改。
