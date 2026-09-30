"""Concurrent CUDA engines pass the one startup admission before loading, and two ranks agree on their streams."""

import importlib
from types import SimpleNamespace

import pytest

from tests.test_cuda_capacity import HEAD, Loaded, checkpoint, construct, fake_runtime, small_config  # noqa: F401
from tests.test_cuda_geometry import allocations, bytes_in  # noqa: F401

WEIGHTS = HEAD


def start(family, path, requested, explicit, world, streams, rank=0):
    obj, _ = construct(family, path, requested, explicit, world, rank)
    kw = dict(tp=world, rank=rank, master="example", context_explicit=explicit, streams=streams)
    if family == "linear":
        return obj, lambda: obj.__init__(path, None, context=requested, **kw)
    return obj, lambda: obj.__init__(path, max_len=requested, depth=3, **kw)


@pytest.mark.torch
@pytest.mark.parametrize("family,world", [("linear", 1), ("linear", 2), ("indexed", 1)])
def test_parallel_startup_is_admitted_before_any_load(tmp_path, fake_runtime, family, world):  # noqa: F811
    checkpoint(tmp_path, small_config(), WEIGHTS)
    calls, _ = fake_runtime
    obj, go = start(family, tmp_path, None, None, world, 4)
    with pytest.raises(Loaded):
        go()
    assert len(calls) == 1
    plan = obj.capacity_plan
    assert plan is not None and 0 < plan["context_window"] <= 65536
    assert plan["serving_peak_bytes_estimate"] <= plan["budget_bytes"]


@pytest.mark.torch
@pytest.mark.parametrize("family,world", [("linear", 1), ("linear", 2), ("indexed", 1)])
def test_parallel_window_that_cannot_fit_every_stream_is_refused_before_loading(tmp_path, monkeypatch, fake_runtime,
                                                                                family, world):  # noqa: F811
    from tensorfold.cuda.geometry import indexed_stream_geometry, stream_geometry
    checkpoint(tmp_path, small_config(), WEIGHTS)
    calls, capacity = fake_runtime
    # one GPU: the window is what one stream reaches beside the others' first rows; two ranks: every stream's
    four = (stream_geometry(small_config(), world, 4, 8, first=256 if world == 1 else None) if family == "linear"
            else indexed_stream_geometry(small_config(), 4, 4, 8, mtp=True))
    budget = four.needed(12000) + 32768                   # the streams fit 12,000 tokens, not 60,000
    monkeypatch.setattr(capacity, "available_bytes", lambda t: budget)
    _, go = start(family, tmp_path, 60000, True, world, 4)
    with pytest.raises(ValueError, match="largest fitting"):
        go()
    assert not calls
    obj, go = start(family, tmp_path, None, None, world, 4)
    with pytest.raises(Loaded):
        go()
    assert 12000 <= obj.capacity_plan["context_window"] < 60000
    _, fits = start(family, tmp_path, 12000, True, world, 4)
    with pytest.raises(Loaded):
        fits()


def test_stream_geometry_counts_every_stream_and_kept_prompt_end():
    from tensorfold.cuda.geometry import draft_geometry, indexed_stream_geometry, stream_geometry
    text = small_config()
    for make in (lambda n, keep: stream_geometry(text, 1, n, keep),
                 lambda n, keep: indexed_stream_geometry(text, n, 4, keep, mtp=True)):
        grow = [make(n, 8).bytes_at(65536) - make(n, 8).bytes_at(32768) for n in (2, 4, 8)]
        assert grow[0] < grow[1] < grow[2]
        assert make(4, 8).bytes_at(32768) > make(4, 0).bytes_at(32768)
    draft = dict(num_hidden_layers=2, num_key_value_heads=2, head_dim=64, hidden_size=512, intermediate_size=1024,
                 sliding_window=2048)
    single = draft_geometry(draft, 1, 12, bounded=True)
    assert draft_geometry(draft, 1, 12, bounded=True, streams=1, kept=0).bytes_at(9000) == single.bytes_at(9000)
    assert draft_geometry(draft, 1, 12, bounded=True, streams=4, kept=9).bytes_at(9000) > 4 * single.bytes_at(9000) // 2


@pytest.mark.torch
@pytest.mark.parametrize("streams", [2, 5])
@pytest.mark.parametrize("kv_dtype,bits", [("bf16", 16), ("int8", 8), ("int4", 4)])
def test_flash_parallel_decoder_allocations_are_budgeted(monkeypatch, allocations, streams, kv_dtype,
                                                         bits):  # noqa: F811
    arrays, fake = allocations
    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    for mod in (state, state.gdn_mod, state.attn_mod, state.moe_mod, state.kvcache):
        monkeypatch.setattr(mod, "torch", fake)
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    text = {"hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "num_hidden_layers": 4, "layer_types": ["linear_attention", "full_attention"] * 2,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 1024, "hc_count": 4}
    cfg = SimpleNamespace(hidden=512, streams=4, conv_kernel=4, conv_dim=1024, nk=2, nv=4, dk=128, dv=128,
                          kv_heads=2, head_dim=64, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3,
                          ple_layers=[], heads=8, index_heads=4, index_budget=2048, low=320, experts=8, top_k=2,
                          moe_width=512, shared_width=512, heads_per_ngram=8, ple_dim=512, eos=(0,))
    weights = SimpleNamespace(cfg=cfg, device="cpu", layers=[SimpleNamespace(index=i, linear=i % 2 == 0)
                                                             for i in range(4)],
                              mtp=SimpleNamespace(), meta={"world": 1}, head=SimpleNamespace(n=1024), comm=None,
                              draft_ids=None)
    slots, depth, keep = 65536, 3, 8
    dec = multi.MultiDecoder(weights, slots=streams, capacity=slots, depth=depth, keep=keep,
                             kv_dtype=kv_dtype)                                          # no late memory query
    from tensorfold.cuda.geometry import indexed_stream_geometry, kv_bytes
    one = dec.free[0]
    snapshot = bytes_in([one.rec]) // 2 + bytes_in([one.conv, one.ple_tail])
    first = [t for t in arrays if t.shape[:2] == (multi.FIRST, cfg.kv_heads)]     # K and V: two layers and the MTP's
    assert len(dec.free) == streams and all(st.kv_dtype == kv_dtype for st in dec.free)
    assert bytes_in(first) == streams * 3 * 2 * multi.FIRST * cfg.kv_heads * kv_bytes(cfg.head_dim, bits)
    # one stream grown to the window beside the others' first rows
    used = bytes_in(arrays) - one.cache_bytes(multi.FIRST) + one.cache_bytes(slots) + (min(keep, streams) + 1) * snapshot
    assert used <= indexed_stream_geometry(text, streams, depth + 1, keep, mtp=True, kv_bits=bits).bytes_at(slots)


def handshake(monkeypatch, path, rank, **kw):
    """The settings row a rank sends in the two-rank handshake (the constructor stops right after sending it)."""
    import torch.distributed as dist
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    class Sent(Exception):
        pass

    rows = []

    def gather(recv, send):
        rows.append(send.clone())
        raise Sent

    monkeypatch.setattr(dist, "all_gather_into_tensor", gather)
    obj = Qwen27Engine.__new__(Qwen27Engine)
    with pytest.raises(Sent):
        obj.__init__(path, None, tp=2, rank=rank, master="example", **kw)
    return rows[0]


@pytest.mark.torch
@pytest.mark.parametrize("peer", [dict(streams=4), dict(streams=2, context=8192, context_explicit=True),
                                  dict(streams=2, keep=6)])
def test_two_ranks_with_different_streams_or_context_refuse_to_start(tmp_path, monkeypatch, fake_runtime,
                                                                     peer):  # noqa: F811
    import torch
    import torch.distributed as dist
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    checkpoint(tmp_path, small_config(), WEIGHTS)
    calls, _ = fake_runtime
    theirs = handshake(monkeypatch, tmp_path, 1, **peer)
    mine = dict(streams=2, context=None, context_explicit=None)

    def gather(recv, send):
        other = theirs if send.numel() == theirs.numel() else send
        recv.view(-1).copy_(torch.cat([send.view(-1), other.view(-1)]))

    monkeypatch.setattr(dist, "all_gather_into_tensor", gather)
    obj = Qwen27Engine.__new__(Qwen27Engine)
    with pytest.raises(RuntimeError, match="different settings"):
        obj.__init__(tmp_path, None, tp=2, rank=0, master="example", **mine)
    assert not calls
