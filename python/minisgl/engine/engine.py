"""Engine：推理引擎主体。

职责：初始化分布式通信、加载模型、按显存规划 KV Cache、建立页表、
初始化 attention/MoE 后端与采样器，并对 CUDA Graph 进行捕获；对外提供
`forward_batch` 完成一批请求的前向计算与采样。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    """一次 forward 的输出：GPU/CPU 上的 next token 与拷贝完成事件。"""

    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        # 设置 TP 信息，供后续按 rank 切分模型/KV
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        # 全局上下文：保存 page_size、KV cache、页表、后端等，供模型层访问
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # 初始化跨 rank 通信（gloo/nccl），并同步获取最小空闲显存
        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        # 先在 meta 设备上建图（不占显存），再 load 权重
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        # 根据显存余量计算可分配的页数
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        # 页表：每行对应一个请求，存 token 级别的物理位置；+1 行给 dummy 请求
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        # dummy 请求用于 padding 与 CUDA Graph 捕获
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """初始化进程组：单卡或用 pynccl 时用 gloo；否则用 nccl + 额外 gloo CPU 组。"""
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            # 预分配 pynccl 通信缓冲：按单步最大传输量估算
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            # 额外的 gloo CPU 组用于同步（如显存统计）
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """加载权重：dummy 模式下随机生成，否则从磁盘加载并转 dtype。"""
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """根据模型加载前后的显存差，估算可为 KV Cache 分配的页数。"""
        new_free_memory = self._sync_get_memory()[1]
        # 每页 KV 占用的字节数 = 2(K+V) * head_dim * 本地 KV 头数 * page_size * dtype * 层数
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            # 模型占用 = 加载前后显存差；可用显存 = 比例 * 初始空闲 - 模型占用
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks.

        各 rank 同步取空闲显存的最小值/最大值，并检测跨 rank 是否失衡。
        """
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        # 用 (free, -free) 一次 all_reduce MIN 同时得到最小与最大
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        # 跨 rank 显存差距过大时报警（> 2GiB）
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """执行一批请求的前向计算与采样，返回 GPU/CPU 上的 next token。"""
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            # decode 阶段且批大小在捕获范围内时用 CUDA Graph 重放
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        # 每个请求完成一步，更新其内部状态
        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 异步拷回 CPU，并用 event 记录完成点供上层同步
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """销毁 CUDA Graph 并关闭分布式资源。"""
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    """向上对齐到 32（页表行长度对齐，便于硬件优化）。"""
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    """按硬件能力/模型类型自动调整配置（attention/moe 后端、页大小等）。"""

    def override(attr: str, value: Any):  # this is dangerous, use with caution
        # frozen dataclass 下强制改字段
        object.__setattr__(config, attr, value)

    # 按 GPU 架构自动选 attention 后端
    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    # TRTLLM 后端对页大小有约束，必要时强制为 64
    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    # MoE 模型自动选 fused 后端
    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
