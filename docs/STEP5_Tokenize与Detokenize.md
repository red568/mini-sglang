# Step 5：Tokenize 与 Detokenize（文本 ⇄ token）

> 对应 [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) 的 Step 5。
> 核心文件：[tokenizer/tokenize.py](python/minisgl/tokenizer/tokenize.py)、[tokenizer/detokenize.py](python/minisgl/tokenizer/detokenize.py)、[tokenizer/server.py](python/minisgl/tokenizer/server.py)。
>
> 这一 Step 回答：**用户打的字是怎么变成数字（token）的，模型吐出的数字又是怎么流式变回字的？**

---

## 一、这个 Step 要解决什么

tokenizer/detokenizer 进程是「文本世界」和「数字世界」的翻译官。它同时干两件事：

- **Tokenize**：`TokenizeMsg`（文本）→ `UserMsg`（`input_ids`）。
- **Detokenize**：`DetokenizeMsg`（一个 token）→ `UserReply`（一段增量文本）。

难点几乎全在 detokenize 的「流式增量解码」上。

---

## 二、核心逻辑

### 2.1 Tokenize（相对简单）

[tokenize.py](python/minisgl/tokenizer/tokenize.py) 的 `TokenizeManager.tokenize`：

```python
if isinstance(msg.text, list):   # 是 messages 列表（chat 模式）
    prompt = self.tokenizer.apply_chat_template(
        msg.text, tokenize=False, add_generation_prompt=True)
else:                            # 是纯字符串
    prompt = msg.text
input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
```

- `apply_chat_template` 把 `[{role, content}, ...]` 拼成带特殊 token（`<|im_start|>` 等）的 prompt 字符串。
- `encode` 把字符串转成 token id 张量，`view(-1).to(torch.int32)` 压成 1D int32。

### 2.2 Detokenize（核心难点）

[detokenize.py](python/minisgl/tokenizer/detokenize.py) 的 `DetokenizeManager.detokenize` 为**每个 uid 维护一个 `DecodeStatus`**：

```python
@dataclass
class DecodeStatus:
    decoded_ids: List[int]   # 已累积的 token id
    decoded_str: str         # 已累积的文本
    read_offset: int         # 已读到的 token 数
    surr_offset: int         # 已稳定的 token 数（不含可能变化的后缀）
    sent_offset: int         # 已发送给用户的字符数
```

核心思路：**token 边界不等于字符边界**。一个新 token 可能只补全了某个字/词的「后半截」，如果立刻把整段解码结果发出去，末尾那半个字下次可能变样。所以要「攒着不稳定的尾巴，只发稳定部分」。

`find_printable_text` 处理了三种「可打印边界」：

- 结尾是换行 → 整段都发；
- 结尾是中文字符（CJK）→ 整段都发（中文没有「半个字」的中间态问题，但代理对要小心）；
- 否则 → 只发到最后一个空格为止（避免把不完整的英文单词发出去）。

### 2.3 tokenizer/detokenizer 进程主循环

[tokenizer/server.py](python/minisgl/tokenizer/server.py) 的 `tokenize_worker`：

```python
while True:
    pending_msg = _unwrap_msg(recv_listener.get())      # 拉一批
    while len(pending_msg) < local_bs and not recv_listener.empty():
        pending_msg.extend(_unwrap_msg(recv_listener.get()))  # 攒到 local_bs

    detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
    tokenize_msg   = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
    abort_msg      = [m for m in pending_msg if isinstance(m, AbortMsg)]
    # 分别处理，各自 push 回去
```

一个进程同时 pull 三类消息，用 `isinstance` 分拣后各走各的处理。

---

## 三、难点解析

### 难点 1：`DecodeStatus` 的三个 offset 到底在管什么？

这是整个 Step 的题眼。逐字段理解：

- `read_offset`：已经「读过」多少个 token 用于解码。
- `surr_offset`：其中多少个 token 是「稳定」的（不会因为下一个 token 而改变解码结果）。
- `sent_offset`：已经「发给用户」多少个字符。

`read_ids = decoded_ids[surr_offset:]` 拿「从上次稳定点之后的所有 id」去解码；`surr_texts = decoded_ids[surr_offset:read_offset]` 拿「上一轮已解码但未稳定的部分」，用来对齐新解码结果（因为新 token 可能改变前一个词的拼写）。

### 难点 2：为什么要分 `read_ids` 和 `surr_ids` 两段去 `batch_decode`？

tokenizer 的 `decode` 是「无状态」的，每次给你一段 token 从头解码。但流式场景里，上一段 token 的「尾巴」可能因为新 token 的加入而改变。所以：

- 把「稳定部分 + 新 token」一起解码，得到最新文本；
- 再单独解码「稳定部分」作为参照；
- `new_text = read_str[len(surr_str):]` 用「参照文本的长度」把新解码结果里「之前已经发过」的部分切掉，只留下真正新增的字符。

### 难点 3：一个进程怎么同时干 tokenize 和 detokenize 两件事？

关键在 `tokenize_worker` 的 `recv_listener` 收到的是**混合消息流**（Step 0 里讲过：默认共享模式下 `minisgl_1` 一个地址同时收 `TokenizeMsg` 和 `DetokenizeMsg`）。于是主循环用 `isinstance` 把一批消息分成三组，分别交给 `TokenizeManager` / `DetokenizeManager` 处理，再分别 `put` 到不同的下游队列。

### 难点 4：`local_bs` 攒批的意义

主循环 `while len(pending_msg) < local_bs and not empty()` 会尽量攒到 `local_bs` 条再一起处理，减少 `batch_decode` 的调用次数、提高吞吐。默认 `local_bs=1`（不攒批），是个可调的性能旋钮。

---

## 四、注意事项

1. **EOS token 不加入解码**：`if not (msg.finished and msg.next_token == eos_token_id): s.decoded_ids.append(...)`，避免把结束符也 decode 成奇怪字符。
2. **`find_printable_text` 处理 `�`（替换字符）**：当解码出 `�`（UTF-8 代理对的中间态）时，说明 token 边界切在了多字节字符中间，这一小段要等下一个 token 再一起解码。
3. **`apply_chat_template` 只在 `text` 是 list 时用**：纯字符串 prompt 直接 encode，不加 chat 模板。
4. **每条消息处理完 `msg.finished` 时删掉 `decode_map[uid]`**，避免状态泄漏。

---

## 五、反思题

1. 为什么不能「每个 token 独立 decode 再拼接」？用一个英文单词被拆成多个 token 的例子说明。
2. `surr_offset` 和 `read_offset` 的差代表什么？为什么要把这段单独 `batch_decode` 一次？
3. `find_printable_text` 里，中文为什么可以「整段都发」，而英文要「发到最后一个空格」？
4. 如果 `tokenize_worker` 里忘了用 `isinstance` 分拣，直接把所有消息当 `DetokenizeMsg` 处理，会发生什么？
5. `local_bs` 调大有什么收益和代价？

---

## 六、示意图

### 6.1 增量 Detokenize 的状态流转

```
时间轴（每个 token 进来一次）
        token #1      token #2      token #3      token #4 ...
decoded_ids: [a]      [a,b]         [a,b,c]        [a,b,c,d]
                │             │              │
解码结果:     "hel"        "hello"        "hello "       "hello w"
                │             │              │
稳定部分(surr): ""           "hello"        "hello "       ...
已发送(sent):   ""           "hello"        "hello "       ...
                │             │              │
                └── 只发稳定部分，不稳定尾巴攒着等下一 token
```

### 6.2 tokenize_worker 主循环

```
        recv_listener.get()  ← 混合消息（TokenizeMsg / DetokenizeMsg / AbortMsg）
                │
                ▼
        _unwrap_msg 展开 Batch*Msg
                │
                ▼
      isinstance 分拣成三组
     ┌──────────┼──────────┐
     ▼          ▼          ▼
 tokenize    detokenize   abort
     │          │          │
     ▼          ▼          ▼
 UserMsg     UserReply   AbortBackendMsg
 (发 backend) (发 frontend) (发 backend)
```

---

## 附：本 Step 涉及文件清单

| 文件 | 职责 | 关键符号 |
|---|---|---|
| [tokenizer/tokenize.py](python/minisgl/tokenizer/tokenize.py) | 文本→token | `TokenizeManager.tokenize` |
| [tokenizer/detokenize.py](python/minisgl/tokenizer/detokenize.py) | token→增量文本 | `DetokenizeManager.detokenize`、`DecodeStatus` |
| [tokenizer/server.py](python/minisgl/tokenizer/server.py) | 进程主循环 | `tokenize_worker` |

**下一步**：进入 Step 6（核心数据结构 `Req`），看请求进入 Scheduler 后，它的状态是怎么被三个长度字段精确描述的。
