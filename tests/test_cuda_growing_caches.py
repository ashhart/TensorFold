"""Concurrent Flash Next slots grow with their streams: a copy keeps every committed row, the gate decides who grows."""

import importlib
from types import SimpleNamespace

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch


def weights():
    cfg = SimpleNamespace(hidden=512, streams=4, conv_kernel=4, conv_dim=1024, nk=2, nv=4, dk=128, dv=128,
                          kv_heads=2, head_dim=64, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3,
                          ple_layers=[], eos=(0,))
    layers = [SimpleNamespace(index=i, linear=i % 2 == 0) for i in range(4)]
    return SimpleNamespace(cfg=cfg, device="cpu", layers=layers, mtp=SimpleNamespace())


def filled(state_mod, torch, kv_dtype: str, rows: int = 256):
    st = state_mod.State(weights(), rows, 4, kv_dtype, limit=65536)
    for t in [kv.k for kv in st.kc] + [kv.v for kv in st.kc] + st.ikc + st.pooled + \
            [st.mtp_kc.k, st.mtp_kc.v, st.mtp_ikc, st.mtp_pooled]:
        t.copy_((torch.rand(t.shape) * 100).to(t.dtype))
    st.set_pos(200)
    st.set_mtp_len(203)
    return st


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
def test_a_state_grows_by_steps_and_keeps_every_committed_row(allocations, kv_dtype):  # noqa: F811
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    st = filled(state, torch, kv_dtype)
    before = {"k": st.kc[0].k[:200].clone(), "ks": st.kc[1].ks[:200].clone(), "ikc": st.ikc[1][:200].clone(),
              "pooled": st.pooled[0][:50].clone(), "mtp": st.mtp_kc.v[:203].clone(),
              "mtp_pooled": st.mtp_pooled[:51].clone()}
    assert st.ensure(250) == 0 and st.version == 0                     # inside the first rows: nothing moves
    added = st.ensure(300)
    assert st.capacity == 8192 and st.version == 1 and added == st.cache_bytes(8192) - st.cache_bytes(256)
    assert torch.equal(st.kc[0].k[:200], before["k"]) and torch.equal(st.kc[1].ks[:200], before["ks"])
    assert torch.equal(st.ikc[1][:200], before["ikc"]) and torch.equal(st.pooled[0][:50], before["pooled"])
    assert torch.equal(st.mtp_kc.v[:203], before["mtp"]) and torch.equal(st.mtp_pooled[:51], before["mtp_pooled"])
    grown = [kv.k for kv in st.kc] + [kv.v for kv in st.kc] + st.ikc + st.pooled + [st.mtp_kc.k, st.mtp_kc.v,
                                                                                    st.mtp_ikc, st.mtp_pooled]
    scales = [t for kv in [*st.kc, st.mtp_kc] for t in (kv.ks, kv.vs)] if kv_dtype != "bf16" else []
    assert st.cache_bytes() == sum(t.numel() * t.element_size() for t in grown + scales)
    assert st.ensure(60000) and st.capacity == 65536                   # the last step stops at the window
    with pytest.raises(ValueError, match="window"):
        st.ensure(65537)


def decoder(multi, state, torch, room: int):
    """A decoder over real CPU states and a gate of ``room`` bytes (no rounds run here)."""

    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.w, dec.depth, dec.streams, dec.free, dec.kept, dec.keep = weights(), 3, {}, [], [], 8
    dec.filling, dec.fills, dec.held = [], {}, {}
    dec.solo, dec.solo_on, dec.planning = None, False, False       # no graph slot, one GPU
    dec.memory_gate = importlib.import_module("tensorfold.cuda.memory_gate").MemoryGate(room, reserve=0)
    return dec


def stream(multi, state, torch, sid: int, pos: int):
    st = state.State(weights(), 256, 4, "bf16", limit=65536)
    st.set_pos(pos)
    st.set_mtp_len(pos)
    return SimpleNamespace(sid=sid, st=st, drafts=[], done=False, waiting=False, out=[1] * 5, error=None)


def test_the_newest_waits_for_room_then_the_newest_ends_when_even_the_oldest_cannot_grow(allocations):  # noqa: F811
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    probe = state.State(weights(), 256, 4, "bf16", limit=65536)
    one = probe.cache_bytes(8192) - probe.cache_bytes(256) + probe.layer_bytes(8192)
    dec = decoder(multi, state, torch, room=one)                        # room for one stream's first step
    old, new = stream(multi, state, torch, 0, 254), stream(multi, state, torch, 1, 254)
    dec.streams = {0: old, 1: new}
    assert dec._make_room() == [] and not old.waiting and new.waiting   # the oldest grows, the newest waits
    assert old.st.capacity == 8192 and new.st.capacity == 256
    old.st.set_pos(8190)                                                 # the oldest needs its next step now
    ended = dec._make_room()
    assert ended == [new] and new.done and "ran out of memory" in str(new.error)
    assert dec.memory_gate.ends == 1 and 1 not in dec.streams and new.st in dec.free
    assert not old.waiting and old.st.capacity == 16384                 # alone, it grows: startup fits one window


def test_kept_prompt_ends_go_before_a_stream_waits(allocations):  # noqa: F811
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    probe = state.State(weights(), 256, 4, "bf16", limit=65536)
    step = probe.cache_bytes(8192) - probe.cache_bytes(256) + probe.layer_bytes(8192)
    dec = decoder(multi, state, torch, room=step)
    kept = state.State(weights(), 256, 4, "bf16", limit=65536)
    kept.ensure(300)                                                   # an idle kept prompt end holding one step
    dec.memory_gate.take(kept.cache_bytes() - kept.cache_bytes(256))
    dec.kept = [([1, 2, 3], kept, {}, None)]
    live = stream(multi, state, torch, 0, 254)
    dec.streams = {0: live}
    assert dec._make_room() == [] and not live.waiting and live.st.capacity == 8192
    assert dec.kept == [] and kept.capacity == 256 and kept in dec.free


def test_a_request_waits_while_a_stream_waits_and_starts_alone_regardless(allocations):  # noqa: F811
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    gate = importlib.import_module("tensorfold.cuda.memory_gate")
    dec = decoder(multi, state, torch, room=0)                          # no room at all
    dec.capacity = 65536
    old = stream(multi, state, torch, 0, 254)
    old.waiting = True
    dec.streams = {0: old}
    with pytest.raises(gate.NoRoom, match="wait"):
        dec.admit(SimpleNamespace(prompt=[1] * 300, count=10, draft=False))
    assert dec.free == []                                              # nothing taken
    dec.streams = {}
    st = state.State(weights(), 256, 4, "bf16", limit=65536)
    assert dec._grow(st, 300, alone=True) and st.capacity == 8192       # alone: startup fitted one whole window
    assert not dec._grow(state.State(weights(), 256, 4, "bf16", limit=65536), 300)


def test_the_lone_streams_graph_slot_keeps_its_rows_until_memory_is_short(allocations):  # noqa: F811
    """Every resize of the graph slot recaptures its graphs: it keeps its rows when idle and grows by doubling."""
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    probe = state.State(weights(), 256, 4, "bf16", limit=65536)
    one = probe.cache_bytes(8192) - probe.cache_bytes(256) + probe.layer_bytes(8192)
    dec = decoder(multi, state, torch, room=one)                        # room for one slot's first step
    solo = state.State(weights(), 256, 4, "bf16", limit=65536)
    dec.solo = SimpleNamespace(st=solo)
    resized = []
    dec._state_changed = lambda st: resized.append(st) if st is solo else None    # drops the slot's graphs
    assert dec._grow(solo, 300, alone=True) and solo.capacity == 8192 and resized == [solo]
    dec._shrink(solo)                                                   # a request ended: the rows stay
    assert solo.capacity == 8192 and solo.pos == 0 and resized == [solo]
    other = stream(multi, state, torch, 0, 254)
    dec.streams = {0: other}
    assert dec._grow(other.st, 300)                                     # another stream needs the memory
    assert other.st.capacity == 8192 and solo.capacity == 256 and resized == [solo, solo]
    dec._shrink(other.st)                                               # any other slot shrinks as before
    assert other.st.capacity == 256

    big = decoder(multi, state, torch, room=1 << 40)
    solo = state.State(weights(), 256, 4, "bf16", limit=65536)
    big.solo, big._state_changed = SimpleNamespace(st=solo), (lambda st: None)
    assert big._grow(solo, 300, alone=True) and solo.capacity == 8192
    assert big._grow(solo, 8193, alone=True) and solo.capacity == 16384
    assert big._grow(solo, 16385, alone=True) and solo.capacity == 32768      # not 24576: doubling
    assert big._grow(solo, 40000, alone=True) and solo.capacity == 65536      # and never past the window
    big._shrink(solo, release=True)                                     # memory is short: it gives them back
    assert solo.capacity == 256


def test_the_graph_slot_keeps_its_rows_on_two_ranks_too(allocations):  # noqa: F811
    """The graph slot is recognised through its Shadow, and keeping its rows records no resize for rank 1."""
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    plan = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi_plan")
    dec = decoder(multi, state, torch, room=1 << 40)
    solo = state.State(weights(), 256, 4, "bf16", limit=65536)
    other = state.State(weights(), 256, 4, "bf16", limit=65536)
    dec._state_changed = lambda st: None
    dec.solo = SimpleNamespace(st=solo)
    assert dec._grow(solo, 300, alone=True) and solo.capacity == 8192
    actions = []
    shadow, other_shadow = plan.Shadow(solo, 0, actions), plan.Shadow(other, 1, actions)
    dec.solo, dec.planning = SimpleNamespace(st=shadow), True
    assert dec._is_solo(shadow) and dec._is_solo(solo) and not dec._is_solo(other_shadow)
    dec._shrink(shadow)                                                 # a request ended on two ranks
    assert actions == [["reset", 0]] and shadow.capacity == 8192
    dec._shrink(shadow, release=True)                                   # memory is short: rank 1 shrinks it too
    assert actions[-1] == ["resize", 0, 256]


def _mover(multi, state, torch, solo):
    dec = decoder(multi, state, torch, room=1 << 40)
    dec.solo = SimpleNamespace(st=solo)
    dropped = []
    dec._state_changed = lambda st: dropped.append(st) if st is solo else None       # the slot's graphs dropped
    return dec, dropped


def test_a_lone_stream_moves_into_a_graph_slot_with_more_rows(allocations):  # noqa: F811
    """The graph slot keeps its rows, so a lone stream from a smaller slot lands in its first rows, unresized."""
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    solo = state.State(weights(), 8192, 4, "bf16", limit=65536)
    old = filled(state, torch, "bf16")                                   # 256 rows, 200 committed
    k, mtp = old.kc[0].k[:200].clone(), old.mtp_kc.v[:203].clone()
    dec, dropped = _mover(multi, state, torch, solo)
    s = SimpleNamespace(sid=0, st=old)
    dec._move_to_solo(s)
    assert s.st is solo and dec.solo.st is solo and solo.capacity == 8192 and not dropped
    assert solo.pos == 200 and torch.equal(solo.kc[0].k[:200], k) and torch.equal(solo.mtp_kc.v[:203], mtp)
    assert old in dec.free


def test_a_different_lone_request_takes_the_graph_slot_and_the_kept_prefix_moves_out(allocations):  # noqa: F811
    """A kept end moves to spare rows so the next request takes the graph slot without dropping graphs or keeps."""
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    solo = filled(state, torch, "bf16", rows=8192)                     # holds a kept prompt end
    kept_k = solo.kc[0].k[:200].clone()
    spare = state.State(weights(), 256, 4, "bf16", limit=65536)
    old = filled(state, torch, "bf16")
    k = old.kc[0].k[:200].clone()
    dec, dropped = _mover(multi, state, torch, solo)
    snap = {"pos": 200}
    dec.kept, dec.free = [([1, 2, 3], solo, snap, None)], [spare]
    s = SimpleNamespace(sid=0, st=old)
    dec._move_to_solo(s)
    assert dec.solo.st is solo and s.st is solo and not dropped and solo.capacity == 8192
    assert torch.equal(solo.kc[0].k[:200], k)
    assert dec.kept == [([1, 2, 3], spare, snap, None)] and spare.capacity == 8192
    assert torch.equal(spare.kc[0].k[:200], kept_k) and spare not in dec.free


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("source_rows,target_rows", [(8192, 32768), (24576, 16384)])
def test_lone_migration_copies_into_retained_or_doubled_graph_capacity(allocations, kv_dtype, source_rows, target_rows):
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    old = filled(state, torch, kv_dtype, source_rows)
    for cache in old.kc + [old.mtp_kc]:
        cache.ks.fill_(1.25)
        cache.vs.fill_(2.5)
    old.conv.fill_(3)
    old.rec.fill_(4)
    old.cur = [1] * len(old.cur)
    target = state.State(weights(), target_rows, 4, kv_dtype, limit=65536)
    dec = decoder(multi, state, torch, room=1 << 40)
    dec.solo, dec.free = SimpleNamespace(st=target), [target]
    changed = []
    dec._state_changed = lambda st: changed.append(st)
    live = SimpleNamespace(sid=0, st=old)
    dec.streams = {0: live}
    before = old.clone()
    dec._move_to_solo(live)
    assert live.st is target and target.capacity == 32768
    assert target.pos == before.pos and target.mtp_len == before.mtp_len and target.cur == before.cur
    assert torch.equal(target.rec, before.rec) and torch.equal(target.conv, before.conv)
    for dst, src in zip(target.kc + [target.mtp_kc], before.kc + [before.mtp_kc]):
        for name in ("k", "v", "ks", "vs"):
            a, b = getattr(dst, name), getattr(src, name)
            assert torch.equal(a[:len(b)].view(torch.uint8), b.view(torch.uint8))
    for a, b in zip(target.ikc + target.pooled + [target.mtp_ikc, target.mtp_pooled],
                    before.ikc + before.pooled + [before.mtp_ikc, before.mtp_pooled]):
        assert torch.equal(a[:len(b)].view(torch.uint8), b.view(torch.uint8))
    assert old in dec.free and old.capacity == 256
    assert (target in changed) == (target_rows < source_rows)
