"""Qwen3.8-27B's prompt path on an EXL3 pack (folded weights): a prompt of several chunks run layer by layer gives the
bits of the same prompt run chunk by chunk, states, kept states and drafter taps included, and prompts filled
together give each one's bits alone. Synthetic layers: EXL3 trellis projections (random words, mul1), plain bf16
GDN gates and embedding, as a real pack stores them."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.exl3 import prefill as exl3_prefill  # noqa: E402
from tensorfold.cuda.exl3.linear import Exl3Linear  # noqa: E402
from tensorfold.families.qwen3_5.cuda import prefill  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import State  # noqa: E402
from tensorfold.families.qwen3_5.cuda.prefill import Piece, prefill_batch, prefill_state  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Exl3, Layer, Plain, Weights  # noqa: E402

V = 256


def _model(layers: int = 2) -> Weights:
    """GDN and attention layers in turn (``layers`` of them, the pair repeated), every projection an EXL3 group."""

    gen = torch.Generator().manual_seed(17)
    ws = exl3_prefill.Workspace(fold=True)

    def ex(n, k, bits=4):
        words = torch.randint(0, 1 << 16, (k // 16, n // 16, 16 * bits), generator=gen, dtype=torch.int32)
        suh = (torch.randn(k, generator=gen) * 0.05).half()
        svh = (torch.randn(n, generator=gen) * 0.05).half()
        return Exl3(Exl3Linear.from_tensors(words.to(torch.int16), suh, svh, "mul1", device="cuda"), workspace=ws)

    def plain(n, k):
        return Plain((torch.randn(n, k, generator=gen) * 0.05).to(torch.bfloat16).cuda())

    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    conv = (torch.randn(384, 4, generator=gen) * 0.1).to(torch.bfloat16).cuda()
    gdn = GDN(ex(384, 128), ex(128, 128), plain(1, 128), plain(1, 128), ex(128, 128), conv,
              torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda"), norm)
    attn = Attention(ex(2 * 2 * 128, 128), ex(128, 128), ex(128, 128), ex(128, 2 * 128), norm, norm)
    pair = [Layer(True, norm, norm, gdn, None, ex(128, 128), ex(128, 128), ex(128, 128)),
            Layer(False, norm, norm, None, attn, ex(128, 128), ex(128, 128), ex(128, 128))]
    config = Config(hidden=128, intermediate=128, layers=layers, heads=2, kv_heads=1, head_dim=128, vocab=V,
                    k_heads=1, v_heads=1, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    return Weights(config, plain(V, 128), (pair * (layers // 2))[:layers], norm, ex(V, 128, 6),
                   torch.ones(16, device="cuda"), quant="exl3")


@pytest.fixture(scope="module")
def w():
    return _model()


def _prompt(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (n,), generator=g).tolist()


def _same(a: State, b: State) -> None:
    assert a.pos == b.pos
    for x, y in zip(a.rec, b.rec):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.conv, b.conv):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.kv, b.kv):
        if x is not None:
            assert torch.equal(x[0][:a.pos], y[0][:b.pos]) and torch.equal(x[1][:a.pos], y[1][:b.pos])


class _Recorder:
    """A stand-in drafter: what it was given, in order (rows between snapshots, skips)."""

    window = 40

    def __init__(self) -> None:
        self.log: list = []

    def skip(self, n: int) -> None:
        self.log.append(("skip", n))

    def add_taps(self, taps: torch.Tensor) -> None:
        self.log.append(("taps", taps.clone()))

    def snapshot(self):
        self.log.append(("snap",))
        return sum(1 for e in self.log if e[0] == "snap")

    def segments(self) -> list:
        """Taps joined between snapshots, so the two orders may hand them over in pieces of any size."""

        out, rows = [], []
        for e in self.log:
            if e[0] == "taps":
                rows.append(e[1])
                continue
            if rows:
                out.append(("taps", torch.cat(rows)))
                rows = []
            out.append(e)
        if rows:
            out.append(("taps", torch.cat(rows)))
        return out


def _run(monkeypatch, w, prompt, base, keep_at, layer_major, draft=None):
    monkeypatch.setattr(prefill, "LAYER_MAJOR", layer_major)
    st = State(w)
    if base:
        prefill_state(w, prompt[:base], st, size=16)
    out = prefill_state(w, prompt, st, size=16, keep_at=keep_at, draft=draft)
    return out, st


# chunks of 16 over 70 rows from 0 start at 0, 14, 28, 42, 56; from 9 at 9, 21, 33, 45, 57
@pytest.mark.parametrize("base,keep_at", [(0, None), (0, 0), (0, 20), (0, 28), (0, 69), (0, 70), (9, None), (9, 9),
                                          (9, 33), (9, 50)])
def test_layer_major_prompts_equal_chunk_by_chunk(monkeypatch, w, base, keep_at):
    prompt = _prompt(70)
    lm, st_lm = _run(monkeypatch, w, prompt, base, keep_at, True)
    cm, st_cm = _run(monkeypatch, w, prompt, base, keep_at, False)
    _same(st_lm, st_cm)
    if keep_at is None:
        assert torch.equal(lm, cm)
        return
    assert torch.equal(lm[0], cm[0])
    _same(lm[1][0], cm[1][0])
    assert lm[1][0].pos == keep_at


@pytest.mark.parametrize("keep_at", [None, 30, 47, 70])
def test_layer_major_feeds_the_drafter_as_chunk_by_chunk(monkeypatch, keep_at):
    """64 layers (the drafter taps five of them): the same taps, skips and snapshot points in both orders."""

    w = _model(64)
    prompt = _prompt(70, seed=1)
    lm_draft, cm_draft = _Recorder(), _Recorder()
    lm, st_lm = _run(monkeypatch, w, prompt, 0, keep_at, True, lm_draft)
    cm, st_cm = _run(monkeypatch, w, prompt, 0, keep_at, False, cm_draft)
    _same(st_lm, st_cm)
    assert torch.equal(lm if keep_at is None else lm[0], cm if keep_at is None else cm[0])
    a, b = lm_draft.segments(), cm_draft.segments()
    assert [e[0] for e in a] == [e[0] for e in b]
    for x, y in zip(a, b):
        assert x[0] != "taps" or torch.equal(x[1], y[1])
        assert x[0] != "skip" or x[1] == y[1]


def test_layer_major_holds_no_weights_after_the_prompt(monkeypatch, w):
    _run(monkeypatch, w, _prompt(70), 0, None, True)
    assert exl3_prefill.SCOPE is None
    held = [x for layer in w.layers for x in vars(layer).values() if isinstance(x, Exl3)]
    assert held and all(getattr(x.layer, "_prompt_w", None) is None for x in held)


def test_prompts_filled_together_equal_each_alone(w):
    """prefill_batch on the folded EXL3 path: each piece's last row, state and kept state are its prefill alone."""

    prompts = [_prompt(5, 2), _prompt(40, 3), _prompt(33, 4), _prompt(70, 5)]
    starts = [0, 0, 12, 0]
    keeps = [None, 17, 20, None]
    alone = []
    for prompt, start, keep in zip(prompts, starts, keeps):
        st = State(w)
        if start:
            prefill_state(w, prompt[:start], st)
        alone.append((prefill_state(w, prompt, st, keep_at=keep), st))
    sts = []
    for prompt, start in zip(prompts, starts):
        st = State(w)
        if start:
            prefill_state(w, prompt[:start], st)
        sts.append(st)
    outs = prefill_batch(w, [Piece(p, st, keep) for p, st, keep in zip(prompts, sts, keeps)])
    for (normed, kept, _), st, keep, (ref, ref_st) in zip(outs, sts, keeps, alone):
        _same(st, ref_st)
        if keep is None:
            assert torch.equal(normed, ref)
        else:
            assert torch.equal(normed, ref[0])
            _same(kept[0], ref[1][0])
