"""The two-rank share protocol returns the list rank 0 sent, at every length."""

import socket

import pytest
import torch
import torch.distributed as dist

LENGTHS = [0, 1, 3, 254, 255, 256, 600, 5000]


def _worker(rank: int, port: int) -> None:
    from tensorfold.families.qwen3_5.cuda.decode_tp import _share

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        for n in LENGTHS:
            values = [(-1) ** i * (i * 7919 % 248320) for i in range(n)]
            got = _share(values if rank == 0 else None, rank, torch.device("cpu"))
            assert got == values, (rank, n)
    finally:
        dist.destroy_process_group()


def test_share_round_trips_every_length():
    pytest.importorskip("triton")
    from torch.multiprocessing import spawn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    spawn(_worker, args=(port,), nprocs=2)
