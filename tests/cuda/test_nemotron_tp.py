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


def _worker(rank: int, port: int, out, split: bool, ids: bool, shaped: bool = False):
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
    samplings = (("keyed", Sampling(1234, 1.0, 20, 0.95)), ("min_p", Sampling(1234, 1.0, 20, 0.95, 0.1)),
                 ("nucleus", Sampling(1234, 1.0, 0, 0.9, 0.02)), ("top_k 40", Sampling(1234, 1.0, 40, 0.95)),
                 ("greedy", None))
    from tensorfold.families.nemotron_h.cuda import sampler as S

    g = torch.Generator(device="cuda").manual_seed(9)
    whole = torch.randn(6, 512, generator=g, device="cuda") * 3                # both ranks: the same rows
    meta = torch.tensor([50, 0, 0, 0], dtype=torch.int32, device="cuda")
    for name, sampling in samplings:                  # a rank's shard draws what one rank draws from the whole row
        eng.set_sampling(sampling)
        one, two = (torch.zeros(6, dtype=torch.int32, device="cuda") for _ in range(2))
        S.sample(whole, meta, eng.params, one)
        eng._sample_shards(whole[:, rank * 256:(rank + 1) * 256].contiguous(), meta, two)
        results[name + "-draw"] = (one.tolist(), [two.tolist()])
    for name, sampling in samplings:
        pre = prefill(eng, mtp, prompt, sampling, keep_at=len(prompt) - 1)
        serial = serial_decode(eng, pre, 40, sampling)
        kept = pre.kept
        resumed = prefill(eng, mtp, prompt, sampling,
                          resume=(kept["engine"], kept["mtp"], len(prompt) - 1, kept["tail"]))
        results[name + "-resumed"] = (serial.tokens, [serial_decode(eng, resumed, 40, sampling).tokens])
        drafted = []
        for d in (1, 3):
            drafted.append(draft_decode(eng, mtp, pre, 40, sampling, drafts=d, copy=False).tokens)
            even = even and all(t % 2 == 0 for t in mtp.drafts())
        results[name] = (serial.tokens, drafted)
        if shaped:                            # each rank masks its own head columns by the same grammar state
            from toy_grammar import toy

            grammars, compiled = toy(512)
            c = grammars.constraint(compiled)
            serial = serial_decode(eng, prefill(eng, mtp, prompt, sampling, constraint=c), 40, sampling, constraint=c)
            drafted = []
            for d in (1, 3):
                c = grammars.constraint(compiled)
                drafted.append(draft_decode(eng, mtp, prefill(eng, mtp, prompt, sampling, constraint=c), 40, sampling,
                                            drafts=d, copy=False, constraint=c).tokens)
            results[name + "-grammar"] = (serial.tokens, drafted)
    out.put((rank, results, even))
    dist.barrier()
    dist.destroy_process_group()


def _run_pair(split: bool = False, ids: bool = False, shaped: bool = False):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, out, split, ids, shaped)) for r in (0, 1)]
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


def test_two_rank_grammar_drafted_equals_serial():
    """A grammar on two ranks: each masks its half of the head, both follow the same tokens, drafted == serial."""

    pytest.importorskip("xgrammar")
    split = _run_pair(split=True, shaped=True)
    for name in ("keyed", "greedy"):
        assert split[name + "-grammar"][0] != split[name][0]              # the grammar changed the reply
