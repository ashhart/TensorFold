"""The fp32 reference's weights and drivers: EXL3 experts decoded on the GPU against the numpy format decoder, Engram
rows and ids, chained layers against the model, and DSpark resumed against fresh.

The tiny synthetic checkpoint runs wherever a GPU does; the pack's checks need ``TF_DSV41_MODEL``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

import dsv41_tiny
from dsv41_ref_weights import EXL3_BOUND, HAD, RefWeights
from dsv41_reference import FP32, MIRROR, State, hc_pre, rms_norm

from tensorfold.cuda.exl3 import experts as exl3
from tensorfold.cuda.exl3 import format as fmt

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the EXL3 decode runs on CUDA")
MODEL = os.environ.get("TF_DSV41_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TF_DSV41_MODEL to the checkpoint")
FIXTURES = Path(__file__).parent.parent / "fixtures" / "deepseek_v41"
SMOKE = "<｜begin▁of▁sentence｜><｜User｜>What is 17*19? Return only the integer.<｜Assistant｜></think>"

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture


def _check_exl3(w: RefWeights, name: str) -> None:
    """The kernels' W_q equals the numpy decoder's bits; W is within EXL3_BOUND of the float64 dequantization."""

    trellis, suh, svh = (w.pack.tensor(f"{name}.{p}") for p in ("trellis", "suh", "svh"))
    wq = fmt.unpack(trellis.numpy(), 2, "mcg")
    assert np.array_equal(exl3.dequant(trellis.cuda(), "mcg").cpu().numpy().view(np.int16), wq.view(np.int16)), name
    exact = torch.from_numpy(fmt.dequantize(trellis.numpy(), suh.numpy(), svh.numpy(), 2, "mcg"))
    k, n = exact.shape
    block = torch.from_numpy(np.abs(wq.astype(np.float64))).view(k // HAD, HAD, n // HAD, HAD).sum((1, 3))
    span = block.repeat_interleave(HAD, 0).repeat_interleave(HAD, 1) / HAD
    bound = EXL3_BOUND * suh.double().abs()[:, None] * svh.double().abs()[None, :] * span
    got = w.exl3(name).cpu().double()
    assert (got - exact).abs().le(bound).all(), f"{name}: worst {((got - exact).abs() / bound).max():.3f} of the bound"


def _check_engram_rows(w: RefWeights, layer: int, ids: np.ndarray) -> None:
    """Rows equal model.py's lookup of the stored bytes: bf16(float(e4m3) * 2^(e - 127)) per 32 bytes."""

    got = w.engram_rows(layer, torch.from_numpy(ids))
    for i, row in enumerate(ids.reshape(-1).tolist()):
        v = w.pack.tensor(f"layers.{layer}.engram.embed.weight", slice(row, row + 1)).float()
        s = w.pack.tensor(f"layers.{layer}.engram.embed.scale", slice(row, row + 1)).float()
        want = (v.unflatten(-1, (-1, 32)) * s[..., None]).flatten(-2).bfloat16()
        assert torch.equal(got.reshape(-1, got.shape[-1])[i].view(torch.int16), want[0].view(torch.int16)), row


# -- the tiny checkpoint ------------------------------------------------------------------------------------------

def test_tiny_exl3_experts_dequantize_within_the_bound(tiny_dir):
    w = RefWeights(tiny_dir)
    for e in (0, 7):
        for proj in ("w1", "w2", "w3"):
            _check_exl3(w, f"layers.3.ffn.experts.{e}.{proj}")
    w1, w2, w3 = w.expert("layers.3", 7)
    assert w1.shape == w3.shape == (256, 512) and w2.shape == (512, 256)
    assert torch.equal(w2, w.exl3("layers.3.ffn.experts.7.w2").cpu().T)


def test_tiny_expert_cache_keeps_the_most_recent(tiny_dir):
    w = RefWeights(tiny_dir, experts=2)
    first = w.expert("layers.0", 0)
    w.expert("layers.0", 1)
    assert w.expert("layers.0", 0) is first             # a hit refreshes it
    w.expert("layers.0", 2)                             # drops expert 1
    assert list(w._lru) == [("layers.0", 0), ("layers.0", 2)]
    mtp = w.expert("mtp.0", 3)                          # DSpark experts are MXFP4
    assert torch.equal(mtp[1], w.pack.dense_fp32("mtp.0.ffn.experts.3.w2"))


def test_tiny_refuses_layers_it_does_not_keep_and_whole_tables(tiny_dir):
    w = RefWeights(tiny_dir, layers=4)
    assert w("layers.3.attn_norm.weight").dtype == torch.float32
    for name in ("layers.4.attn_norm.weight", "layers.1.engram.embed.weight", "layers.0.ffn.experts.0.w1.weight"):
        with pytest.raises(KeyError):
            w(name)


def test_tiny_engram_rows_are_the_stored_bytes(tiny_dir):
    w = RefWeights(tiny_dir)
    ids = w.hasher.ids([], list(range(20, 32)))
    assert ids.shape == (12, 2, 24)
    for i, layer in enumerate(w.cfg.engram_layer_ids):
        _check_engram_rows(w, layer, ids[:, i])


@pytest.mark.parametrize("mode", [FP32, MIRROR], ids=["fp32", "mirror"])
def test_tiny_chained_layers_equal_the_model(tiny_dir, mode):
    w = RefWeights(tiny_dir)
    c, chunks = w.cfg, [list(range(20, 61)), list(range(100, 109))]
    whole = State()
    for chunk in chunks:
        logits, _, whole = w.model(chunk, mode, whole, all_logits=True)
    chained = State()
    for chunk in chunks:
        X = w("embed.weight")[torch.tensor(chunk)].unsqueeze(1).repeat(1, c.hc_mult, 1)
        pre = torch.zeros(len(chunk), c.hc_mult)
        pre[:, 0] = 1.0
        for layer in range(c.num_hidden_layers):
            X, pre = w.layer(layer, X, pre, chained, mode, tokens=chunk)
        chained.pos += len(chunk)
        chained.tokens += chunk
    x = rms_norm(hc_pre(X, pre, mode), w("norm.weight"), c.rms_norm_eps, mode)
    assert torch.isfinite(logits).all() and logits.shape == (len(chunks[-1]), c.vocab_size)
    assert torch.equal(x @ w("lm_head.weight").T, logits)


def test_tiny_dspark_resumed_equals_fresh(tiny_dir):
    w = RefWeights(tiny_dir)
    c = w.cfg
    logits, taps, _ = w.model(list(range(20, 60)), all_logits=True)
    y = int(logits[-1].argmax())
    drafts, block, conf, _ = w.dspark(taps, y, 39)
    assert len(drafts) == c.dspark_block_size and block.shape == (c.dspark_block_size, c.vocab_size)
    assert torch.isfinite(block).all() and torch.isfinite(conf).all()
    state = State(pos=25)
    w.dspark(taps[:25], 0, 24, state=state)
    state.pos = 40
    again, block2, conf2, _ = w.dspark(taps[25:], y, 39, state=state)
    assert again == drafts and torch.equal(block2, block) and torch.equal(conf2, conf)
    with pytest.raises(ValueError, match="committed"):
        w.dspark(taps, y, 38, state=State(pos=40))


# -- the pack -----------------------------------------------------------------------------------------------------

@needs_model
@pytest.mark.parametrize("expert", [0, 1, 383])
def test_pack_exl3_experts_dequantize_within_the_bound(expert):
    w = RefWeights(MODEL, layers=4)
    for proj in ("w1", "w2", "w3"):
        _check_exl3(w, f"layers.0.ffn.experts.{expert}.{proj}")


@needs_model
def test_pack_engram_ids_equal_the_fixture_and_rows_the_stored_bytes():
    data = json.loads((FIXTURES / "engram_ids.json").read_text())
    w = RefWeights(MODEL)
    ids = w.hasher.ids([], data["raw_ids"])
    assert np.array_equal(ids, np.array(data["expected"], dtype=np.int64))
    for i, layer in enumerate(w.cfg.engram_layer_ids):
        _check_engram_rows(w, layer, ids[-2:, i])


@needs_model
def test_pack_reduced_model_agrees_fp32_and_mirror_on_the_smoke_prompt():
    from tokenizers import Tokenizer

    ids = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json")).encode(SMOKE, add_special_tokens=False).ids
    w = RefWeights(MODEL, layers=4, experts=16)
    fp32, _, _ = w.model(ids, FP32, all_logits=True)
    mirror, _, _ = w.model(ids, MIRROR, all_logits=True)
    assert fp32.shape == mirror.shape == (len(ids), w.cfg.vocab_size)
    assert torch.isfinite(fp32).all() and torch.isfinite(mirror).all()
    assert torch.equal(fp32.argmax(-1), mirror.argmax(-1))
