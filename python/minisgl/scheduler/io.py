"""SchedulerIOMixin：调度器与 tokenizer 之间的消息通信层。

根据 TP 规模选择不同的通信拓扑：
- 单 rank：直接通过 pull/push 队列收发。
- 多 rank：rank0 从 tokenizer 拉取，再通过 pub/sub 广播给其余 rank；
  回复时只有 rank0 将结果 push 回 tokenizer，其余 rank 不动作。
此外支持 offline 模式（本地直接驱动，无 tokenizer 链路）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        # 离线模式：直接替换收发函数为本地实现，提前返回
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        # 主 rank（rank0）创建与 tokenizer 通信的两条链路：拉取请求 + 推送结果
        if tp_info.is_primary():
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                # 多 rank 时 rank0 额外负责广播原始消息给其它 rank
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # 非主 rank 通过订阅队列接收 rank0 广播
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        """在 CPU 侧进程组上做 barrier，使所有 rank 达到同步点。"""
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """单 rank：阻塞拉取一条，再尽量排空队列中剩余消息。"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # 空闲时调用 run_when_idle 执行后台任务（如完整性检查）
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 rank 的 rank0：拉取消息并以原始字节广播给其它 rank，再本地解码。"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        # 先广播待处理消息条数，保证各 rank 同步消费相同数量的消息
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 rank 的非主 rank：订阅 rank0 广播，并按约定的条数消费。"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        # 从 broadcast 结果获取 rank0 广播的消息条数
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        """rank0 将结果回传 tokenizer：单条直接发，多条打包成 BatchTokenizerMsg。"""
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
