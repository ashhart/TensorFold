"""GLM-5.3-Flash's streamed experts: the slot kernels' bits, the checkpoint's expert names, and whole forwards."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import glm5_fakes  # noqa: E402
from tensorfold.families import glm5_next  # noqa: E402
from tensorfold.families.glm5_next import config, stream, weights  # noqa: E402
from tensorfold.kernels.glm.flash.v1 import moe as MK  # noqa: E402
from tensorfold.kernels.glm.flash.v1 import stream_moe  # noqa: E402
from tensorfold.streaming.pool import Slots, expert_nbytes, sources  # noqa: E402


def metal():
    try:
        return mx.metal.is_available()
    except Exception:  # noqa: BLE001 - no Metal: the kernels do not apply
        return False


needs_metal = pytest.mark.skipif(not metal(), reason="needs a Metal GPU")


def same(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def pooled(block, slots, at):
    """A pool of ``slots`` slots holding expert e of ``block`` at slot at[e] (the rest zero)."""

    pool = {}
    for name, q in (("gate", block.gate), ("up", block.up), ("down", block.down)):
        parts = []
        for a in q.arrays():
            full = mx.zeros((slots, *a.shape[1:]), dtype=a.dtype)
            full[mx.array(at)] = a
            parts.append(full)
        pool[name] = Slots(*parts)
    return pool


def fake_streamer(block, slots, at, layer):
    """A streamer whose host has already answered: the slot table filled, hold passes the token through."""

    experts = int(block.gate.weight.shape[0])
    table = np.zeros((layer + 1, experts), dtype=np.int32)
    table[layer] = at
    s = SimpleNamespace(pool=pooled(block, slots, at), slot_of=mx.array(table.reshape(-1)),
                        layer_ids=[mx.array([i] + [0] * 7, dtype=mx.int32) for i in range(layer + 1)],
                        box=mx.zeros((max(16 * block.cfg.num_experts_per_tok, experts),), dtype=mx.uint32))
    s.hold = lambda token, kind, at_layer, rows: token
    s.window_views = lambda: {p: Slots(*(getattr(v, f)[:experts] for f in ("weight", "scales", "biases")))
                              for p, v in s.pool.items()}
    mx.eval(s.slot_of, s.box, *s.layer_ids, *[a for v in s.pool.values() for a in (v.weight, v.scales, v.biases)])
    return s


@needs_metal
def test_decode_windows_through_slots_are_the_resident_fused_block():
    from test_glm5_row_kernels import _moe

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        block = _moe()
        assert block.fused_ok and MK.SPLIT_SHARED
        experts = int(block.gate.weight.shape[0])
        at = np.random.default_rng(3).permutation(3 * experts)[:experts].astype(np.int32)
        streamer = fake_streamer(block, 3 * experts, at, layer=2)
        x = (0.5 * mx.random.normal((16, 512))).astype(mx.bfloat16)
        for rows in (1, 2, 5, 16):
            want = MK.moe_rows(block, x[:rows])
            got = stream_moe.moe_rows(block, x[:rows], streamer, 2)
            mx.eval(want, got)
            assert same(got, want), rows
        picks = np.array(streamer.box)[:16 * block.cfg.num_experts_per_tok]
        logits = MK.router_rows(x.astype(mx.float32), block)
        idx, _ = block.route(logits)
        assert sorted(picks.reshape(16, -1)[3]) == sorted(np.array(idx)[3])
    finally:
        mx.set_default_device(previous)


@needs_metal
def test_prompt_chunks_through_the_window_are_the_resident_block():
    from test_glm5_row_kernels import _moe

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        block = _moe()
        experts = int(block.gate.weight.shape[0])
        x = (0.5 * mx.random.normal((40, 512))).astype(mx.bfloat16)
        want = block(x, False)
        block.streamer = fake_streamer(block, 2 * experts, np.arange(experts, dtype=np.int32), layer=0)
        block.stream_layer = 0
        got = block(x, False)
        mx.eval(want, got)
        assert same(got, want)
        flags = np.array(block.streamer.box)[:experts]
        used = set(np.array(block.select(x)[0]).reshape(-1).tolist())
        assert {e for e in range(experts) if flags[e]} == used
    finally:
        mx.set_default_device(previous)


def test_the_checkpoints_decoder_experts_are_named_and_counted(tmp_path):
    folder = glm5_fakes.write_checkpoint(tmp_path / "glm", seed=1)
    text = glm5_fakes.TEXT
    names = stream.expert_names(folder, text["num_hidden_layers"])
    sparse = [i for i, kind in enumerate(text["mlp_layer_types"]) if kind == "sparse"]
    assert {key[0] for key in names} == set(sparse)                      # not the MTP layer's
    assert len(names) == len(sparse) * 3 * 3 * text["n_routed_experts"]
    found, shapes = sources(folder, names)
    total = sum(src.nbytes for src in found.values())
    assert glm5_next.expert_bytes(folder) == total
    assert shapes[("down", "weight")][0] == (glm5_fakes.D, text["moe_intermediate_size"] * 4 // 32)


@needs_metal
def test_streamed_experts_give_the_resident_forward(tmp_path, monkeypatch):
    from tensorfold.streaming import build

    try:
        build.load()
    except RuntimeError as error:                                        # no cmake or nanobind here
        pytest.skip(str(error))
    monkeypatch.setattr(glm5_fakes, "D", 512)
    monkeypatch.setitem(glm5_fakes.TEXT, "hidden_size", 512)
    monkeypatch.setitem(glm5_fakes.TEXT, "moe_intermediate_size", 512)
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        folder = glm5_fakes.write_checkpoint(tmp_path / "glm", seed=2)
        resident = weights.load_backbone(folder)
        streamed = weights.load_backbone(folder, stream=True)
        per = expert_nbytes(sources(folder, stream.expert_names(folder, 6))[0], 1)
        streamer = stream.attach(streamed, folder, (2 * 8 + 4) * per / 2**30)     # a layer's window, 12 LRU slots
        prompt = mx.array(np.random.default_rng(5).integers(0, glm5_fakes.VOCAB, 40), dtype=mx.uint32)
        caches = [resident.make_cache(), streamed.make_cache()]
        outs = [m.hidden(prompt, c) for m, c in zip((resident, streamed), caches)]
        assert same(outs[1], outs[0])
        for rows in (1, 1, 4, 16):
            step = mx.array(np.random.default_rng(rows).integers(0, glm5_fakes.VOCAB, rows), dtype=mx.uint32)
            outs = [m.hidden(step, c) for m, c in zip((resident, streamed), caches)]
            assert same(outs[1], outs[0]), rows
        assert streamer.misses > 0 and streamer.error is None
        streamer.close()
    finally:
        mx.set_default_device(previous)
