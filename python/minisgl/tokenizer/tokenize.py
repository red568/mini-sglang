from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization

        # 这里的逻辑是：如果 msg.text 是一个 list，说明它是一个聊天消息的列表，需要先应用聊天模板（apply_chat_template）把它们拼接成一个完整的 prompt，然后再进行编码。
        # 如果 msg.text 是一个字符串，就直接使用它作为 prompt。
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results
