from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass

"""
API Server ──TokenizeMsg(text)──► tokenizer ──UserMsg(input_ids)──► Scheduler
将经过tokenize的input_ids发给Scheduler，Scheduler再发给模型进行推理。
"""
@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor,经过tokenize后的输入id 
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
