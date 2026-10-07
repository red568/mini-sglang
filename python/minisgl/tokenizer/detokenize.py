from dataclasses import dataclass
from typing import Dict, List

from minisgl.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# Borrowed from sglang
"""
这段代码是流式推理的增量解码器(incremental detokenizer),
核心要解决的问题是:模型每次吐出一个 token,但单个 token 可能只对应半个多字节字符(比如半个 emoji),
直接解码会产生 � 乱码。所以需要"攒够"再输出,同时每个请求(uid)要记住自己的状态。
"""

def _is_chinese_char(cp: int):
    """判断码点是否是 CJK 汉字。因为中文没有空格分词,不需要等空格边界,可以直接输出。"""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False

"""
返回当前文本中"能安全输出"的最长前缀(避免输出半个词):

以 \n 结尾 → 整个输出(换行是天然边界)
最后一个字符是汉字 → 整个输出(汉字之间不需要空格)
倒数第二个字符是汉字 → 输出到倒数第二个(最后那个拉丁字符可能是半个词,先留着)
其他情况 → 截断到最后一个空格之后(不输出半个英文单词)
"""
def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    decoded_ids: List[int] #到目前为止收到的所有 token id
    decoded_str: str    #已经确认稳定并累积的文本
    read_offset: int  # length of read ids
    surr_offset: int  # length of surr ids
    sent_offset: int  # length of sent out string


class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs
