"""DeepSeek-V4.1-Flash's per-rank state and scratch: shapes on the meta device, byte totals, snapshot row views."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import buffers, weights

CFG = Config.read(Path(__file__).parent / "fixtures" / "deepseek_v41")
CAPACITIES = (4096, 65536, 1 << 20)
BF, F32, I32, U8 = torch.bfloat16, torch.float32, torch.int32, torch.uint8


def _walk(obj) -> int:
    """Every tensor reachable from ``obj``'s attributes, counted once, independently of the module's own walk."""

    seen, total, stack = set(), 0, [obj]
    while stack:
        v = stack.pop()
        if isinstance(v, torch.Tensor):
            if id(v) not in seen:
                seen.add(id(v))
                total += v.numel() * v.element_size()
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif hasattr(v, "__dict__") and type(v).__module__.startswith("tensorfold."):
            stack.extend(vars(v).values())
    return total


@pytest.mark.parametrize("capacity", CAPACITIES)
def test_state_shapes(capacity):
    st = buffers.State(CFG, capacity, "meta")
    assert st.rings.shape == (43, 128, 512) and st.rings.dtype == BF
    assert {k: tuple(v.shape) for k, v in st.comp.items()} == {
        2: (capacity // 2 + 1, 512), 8: (capacity // 2 + 1, 512), 14: (capacity // 2 + 1, 512), 20: (capacity, 512)}
    assert {k: tuple(v.shape) for k, v in st.index_k.items()} == {
        2: (capacity // 2 + 1, 128), 8: (capacity // 2 + 1, 128), 14: (capacity // 2 + 1, 128), 20: (capacity, 128)}
    assert all(v.dtype == BF for v in [*st.comp.values(), *st.index_k.values()])
    assert st.pooled == (2, 8, 14)
    assert st.tail.shape == (3, 2, 512) and st.tail.dtype == F32
    assert st.tail_valid.shape == (3,) and st.tail_valid.dtype == I32
    assert st.pos_dev.shape == (1,) and st.pos_dev.dtype == I32
    fixed = 43 * 128 * 512 * 2 + 3 * 2 * 512 * 4 + 3 * 4 + 4 + 3 * 1280      # rings, tails, flags, pos, entry rounding
    assert st.nbytes() == _walk(st) == 3200 * capacity + fixed


def test_state_position_and_reset():
    st = buffers.State(CFG, 64, "cpu")
    st.set_pos(37)
    assert st.pos == 37 and int(st.pos_dev) == 37
    st.tail_valid.fill_(1)
    st.history = [5, 6, 7]
    st.reset()
    assert st.pos == 0 and int(st.pos_dev) == 0 and not st.tail_valid.any() and st.history == []


@pytest.mark.parametrize("n", [0, 1, 2, 7, 127, 128, 129, 2047, 4095])
def test_row_views(n):
    st = buffers.State(CFG, 4096, "meta")
    views = st.row_views(n)
    assert [tuple(v.shape) for v in views] == [(n // 2 + 1, 512), (n // 2 + 1, 128)] * 3 + [(n + 1, 512),
                                                                                         (n + 1, 128)]
    size = sum(v.numel() * v.element_size() for v in views)
    assert size == 3 * (n // 2 + 1) * 1280 + (n + 1) * 1280
    assert 0 < size - 3200 * n <= 5 * 1280                      # entry rounding: at most one entry a cache, plus one
    assert all(v._base is not None for v in views)              # views of the live caches, not copies


def test_row_views_stop_at_the_cache():
    st = buffers.State(CFG, 4096, "meta")
    assert [v.shape[0] for v in st.row_views(4096)] == [2049, 2049] * 3 + [4096, 4096]


def _expected(rows: int, capacity: int, prefill: bool) -> dict:
    """The buffer shapes for the pack's config on two ranks."""

    taps = min(rows, 128)
    scored = min(rows, buffers.score_rows(capacity)) if prefill else rows
    out = {
        "ids": ((rows,), I32), "X": ((rows, 4, 5120), BF), "pre_in": ((rows, 4), F32),
        "hcpart": ((rows, 16, 32), F32), "pre_a": ((rows, 4), F32), "post_a": ((rows, 4), F32),
        "comb_a": ((rows, 16), F32), "pre_f": ((rows, 4), F32), "post_f": ((rows, 4), F32),
        "comb_f": ((rows, 16), F32), "xn": ((rows, 5120), BF),
        "qakv": ((rows, 1792), BF), "qr": ((rows, 1280), BF), "q": ((rows, 32, 512), BF), "o": ((rows, 32, 512), BF),
        "u": ((rows, 4096), BF), "kvw": ((43, rows, 512), BF),
        "cmp": ((3, rows, 2, 512), F32), "lat": ((rows, 512), BF), "kI": ((rows, 128), BF),
        "epos": ((rows,), I32), "qI": ((rows, 32, 128), BF),
        "wI": ((rows, 32), F32), "scores": ((scored, capacity), F32), "cand": ((rows, 2048), I32),
        "cand_n": ((rows,), I32), "lists": ((rows, 512), I32), "list_n": ((rows,), I32),
        "mlog": ((rows, 384), F32), "pick": ((rows, 6), I32), "wts": ((rows, 6), F32),
        "sgu": ((rows, 2304), BF), "sact": ((rows, 1152), BF), "sd": ((rows, 5120), F32),
        "part": ((rows, 5120), F32), "gath": ((2, rows, 5120), F32),
        "eraw": ((rows, 2, 12, 264), U8), "eloc": ((rows, 2, 3072), BF), "egat": ((2, rows, 2, 3072), BF),
        "eng": ((rows, 2, 6144), BF), "ekv": ((rows, 12800), BF), "ekv_gat": ((2, rows, 12800), BF),
        "taps": ((taps, 3, 5120), BF), "hidden": ((rows, 5120), BF), "fnormed": ((rows, 5120), BF),
        "logits": ((1 if prefill else rows, 64640), F32), "mx": ((taps, 5120), BF), "mx_gat": ((2, taps, 2560), BF),
    }
    if not prefill:
        out.update({
            "attn_part": ((rows, 32, 5, 514), F32),
            "xd": ((5, 4, 5120), BF), "bkv": ((5, 512), BF), "dmlog": ((5, 128), F32), "dpick": ((5, 4), I32),
            "dwts": ((5, 4), F32), "dact": ((5, 4, 1152), BF), "dey": ((5, 4, 5120), F32),
            "dlog": ((5, 64640), F32), "me": ((5, 256), BF), "conf": ((5,), F32),
        })
    return out


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("rows,prefill", [(6, False), (2048, True)])
def test_buffer_shapes(rows, prefill, capacity):
    b = buffers.Buffers(CFG, 2, rows, capacity, prefill, device="meta")
    for name, (shape, dtype) in _expected(rows, capacity, prefill).items():
        t = getattr(b, name)
        assert (tuple(t.shape), t.dtype) == (shape, dtype), name
    assert b.exl3.rows == (1024 if prefill else rows) and b.exl3.slots == 6       # no shared-expert slot (D10)
    assert b.exl3.y.shape == (b.exl3.rows * 6, 5120)
    if prefill:
        assert b.attn_part is None and not hasattr(b, "dplan")
        assert b.scores.numel() * 4 == min(1 << 30, 2048 * capacity * 4)
    else:
        assert (b.dplan.rows, b.dplan.slots, b.dplan.experts) == (5, 4, 129)
    assert b.ids_host is None and b.staged is None and b.eraw_host == [] and b.eraw_done == []   # meta: no pinning


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("rows,prefill", [(6, False), (2048, True)])
def test_bytes_is_the_allocation(rows, prefill, capacity):
    b = buffers.Buffers(CFG, 2, rows, capacity, prefill, device="meta")
    assert buffers.bytes(CFG, 2, rows, capacity, prefill) == b.nbytes() == _walk(b)


def test_prompt_byte_figures():
    b = buffers.Buffers(CFG, 2, 2048, 1 << 20, True, device="meta")
    assert b.kvw.numel() * 2 == 43 * 2048 * 1024                     # 90 MB
    assert 490e6 < _walk(b.exl3) < 500e6                             # 1024 rows x 6 slots, no sentinel (D10)
    assert b.scores.numel() * 4 == 1 << 30


def test_score_rows():
    assert buffers.score_rows(1 << 20) == 256
    assert buffers.score_rows(65536) == 2048
    assert buffers.score_rows(3_000_000) == 80
    assert buffers.score_rows(1 << 30) == 16
    assert all(buffers.score_rows(e) % 16 == 0 for e in (1, 1000, 123_457, 999_999))


def test_one_rank_holds_whole_tensors():
    b = buffers.Buffers(CFG, 1, 6, 4096, device="meta")
    assert b.q.shape == (6, 64, 512) and b.logits.shape == (6, 129280) and b.eraw.shape == (6, 2, 24, 264)
    assert b.gath.shape == (1, 6, 5120) and b.sact.shape == (6, 2304)


def test_reduced_layer_set():
    cfg = Config.read(Path(__file__).parent / "fixtures" / "deepseek_v41", layers=8)
    st = buffers.State(cfg, 4096, "meta")
    assert st.rings.shape[0] == 11 and list(st.comp) == [2] and st.pooled == (2,)
    b = buffers.Buffers(cfg, 2, 6, 4096, device="meta")
    assert b.kvw.shape == (11, 6, 512) and b.cmp.shape == (1, 6, 2, 512) and b.scores.shape == (6, 2049)
    assert b.eraw.shape == (6, 1, 12, 264)


def test_weights_fields():
    assert [f.name for f in dataclasses.fields(weights.Weights)] == [
        "cfg", "rank", "world", "comm", "device", "vocab_offset", "embed", "layers", "norm", "head", "dspark",
        "engram", "rope"]
    assert issubclass(weights.StageW, weights.LayerW)
