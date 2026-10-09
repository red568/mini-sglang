"""distributed 子包：张量并行（TP）的进程信息与通信后端。

- `info.py`：定义 `DistributedInfo`（rank/size）及全局 TP 信息的读写。
- `impl.py`：定义统一的集合通信接口，并提供 torch.distributed 与 PyNCCL
  两种实现，通过插件栈切换。
"""

from .impl import DistributedCommunicator, destroy_distributed, enable_pynccl_distributed
from .info import DistributedInfo, get_tp_info, set_tp_info, try_get_tp_info

__all__ = [
    "DistributedInfo",
    "get_tp_info",
    "set_tp_info",
    "enable_pynccl_distributed",
    "DistributedCommunicator",
    "try_get_tp_info",
    "destroy_distributed",
]
