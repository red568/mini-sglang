from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    # 为什么这里是静态方法：因为要作为回调函数传给别的函数，不能是绑定了self的实例方法
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]

"""
API Server ──UserReply(incremental_output)──► Client
将解码得到的增量文本（一段文本）返回给API Server，再由API Server发送给客户端。
"""
@dataclass
class UserReply(BaseFrontendMsg):
    uid: int  # 请求的唯一标识。
    incremental_output: str #  这次 decode 出来的增量文本，不是全文。这一个 token 触发的、重新解码累积序列后新出现的文本」，不是「单个 token 的字面解码结果」，并且可能为空
    finished: bool #生成是否结束的终止信号
