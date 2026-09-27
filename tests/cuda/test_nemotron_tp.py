"""Two ranks on one GPU over gloo: drafts equal serial, both ranks agree, and a split MTP head changes drafts only."""

import os
import socket

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, out, split: bool, ids: bool):
    import sys

    import torch.distributed as dist

    sys.path.insert(0, os.path.dirname(__file__))
    from nemotron_fakes import tiny_weights

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.nemotron_h.cuda.decode import draft_decode, prefill, serial_decode
    from tensorfold.families.nemotron_h.cuda.mtp import MTPHead
    from tensorfold.families.nemotron_h.cuda.tp import TPEngine, split_weights

    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)

    def gather(local):
        cpu = local.contiguous().cpu()
        parts = [torch.empty_like(cpu) for _ in range(2)]
        dist.all_gather(parts, cpu)
        return torch.cat(parts).to(local.device)

    w = tiny_weights(5, heads=32, kv_heads=2)
    eng = TPEngine(split_weights(w, rank), gather, max_len=1024, graphs=False)
    mtp = MTPHead(eng, split=split, draft_ids=list(range(0, 512, 2)) if ids else None)      # even ids only
    prompt = [(37 * i + 11) % 500 + 1 for i in range(21)]
    results = {}
    even = True
    for name, sampling in (("keyed", Sampling(1234, 1.0, 20, 0.95)), ("greedy", None)):
        pre = prefill(eng, mtp, prompt, sampling)
        serial = serial_decode(eng, pre, 40, sampling)
        drafted = []
        for d in (1, 3):
            drafted.append(draft_decode(eng, mtp, pre, 40, sampling, drafts=d, copy=False).tokens)
            even = even and all(t % 2 == 0 for t in mtp.drafts())
        results[name] = (serial.tokens, drafted)
    out.put((rank, results, even))
    dist.barrier()
    dist.destroy_process_group()


def _run_pair(split: bool = False, ids: bool = False):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, out, split, ids)) for r in (0, 1)]
    for p in procs:
        p.start()
    items = [out.get(timeout=600) for _ in procs]
    got = {rank: results for rank, results, _ in items}
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    if ids:
        assert all(even for _, _, even in items)
    assert got[0] == got[1]                                   # both ranks decoded the same tokens
    for name, (serial, drafted) in got[0].items():
        for tokens in drafted:
            assert tokens == serial, name
    return got[0]


def test_two_rank_drafted_equals_serial():
    whole = _run_pair()
    split = _run_pair(split=True)                             # the MTP head split over the ranks: other drafts,
    split_ids = _run_pair(split=True, ids=True)               # same verified tokens
    for name in whole:
        assert split[name][0] == whole[name][0] and split_ids[name][0] == whole[name][0]
