from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]

"""
Scheduler → detokenizer 方向，是 decode 阶段的输入。
关键在 next_token 是单个 int（不是序列）
——这体现了流式解码：Scheduler 每生成一个新 token 就发一条 DetokenizeMsg。
"""
@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool

"""
API Server → tokenizer， 用户请求进来，先变成它，再 tokenize 成 input_ids 转成 UserMsg 发给 Scheduler
union类型，对应两种输入方式:
    str → 纯文本 prompt；
    List[Dict[str, str]] → chat 消息列表（如 [{"role":"user","content":"..."}]）
"""
@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]] 
    sampling_params: SamplingParams #SamplingParams采样相关参数：temperature,top k等


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
