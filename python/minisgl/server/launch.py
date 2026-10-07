from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs

"""
scheduler 进程的启动函数，负责启动一个 minisgl.scheduler.Scheduler 对象，并进入调度循环。
每个rank自己启动一个进程，管理自己的资源（含模型、KV cache、GPU 资源）
"""
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    # sys.argv[1:]可以获取命令行参数（list格式）。如 ["--model-path", "meta-llama/Llama-3.1-8B", "--tp-size", "2"]
    # run_shell 是一个布尔标志，用来决定这个程序是作为一个常驻的 HTTP API 服务器运行，还是作为一个交互式的终端 shell 运行。它是这个项目里两种「启动模式」的开关。
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    # __name__ 是 Python 的内置变量，保存当前模块的名字
    logger = init_logger(__name__, "initializer")

    
    
    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        # spawn使得子进程更干净，避免cuda上下文也被复制
        # force=True 的原因：这个函数 start_subprocess 可能被多次调用，或者项目其它地方（比如某些库）已经设过 start method。用 force=True 确保这里一定生效，不会被之前的设置拦住。
        mp.set_start_method("spawn", force=True)


        # tp的数量：一共要启动多少个 TP 进程，一个gpu一个进程
        world_size = server_args.tp_info.size
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            new_args = replace(  # dataclasses.replace() 基于原配置生成一个新副本，只替换 tp_info 一个字段，其余字段照抄。配合循环，给每个子进程「定制」自己的 rank
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()



        # 启动 m个tokenizers的进程和1个 detokenizer进程
        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        # m个tokenizers的进程, m可为0
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack，只有rank=0的scheduler会发送ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    # 为什么start_subprocess要传入api server中进行回调，因为zmq启动需要顺序，需要先binding
    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
