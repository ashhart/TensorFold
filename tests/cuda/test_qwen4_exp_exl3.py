"""Flash Next's EXL3 path on CUDA (``families/qwen4_exp/cuda/exl3*.py``).

Always: the unquantized fp16 matmul against an fp64 reference and its row invariance (a row alone, in any window,
and through the K-split partials), and the n-gram row decoder against a reference of ExLlamaV3's row codec
(``ngram_codec.dequant_rows``) on random packed rows at every width.

With ``TENSORFOLD_EXL3_FLASHNEXT=<an EXL3 Flash Next checkpoint>`` also: the pack's centred norms detected as
gamma - 1, n-gram rows bit-equal to ExLlamaV3's own ``ngram_dequant`` (needs ``exllamav3`` importable), a real
layer's routed experts (every width the layer mixes) against an fp64 reference from ``format.dequantize``, and a
model cut to ``TENSORFOLD_EXL3_FLASHNEXT_LAYERS`` layers (default 4, plus the head and the MTP head): forward
windows give one-row steps' logits bit for bit, and MTP-drafted decoding emits serial decoding's tokens.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import exl3, exl3_mm, exl3_pack  # noqa: E402

DEV = "cuda"
MODEL = os.environ.get("TENSORFOLD_EXL3_FLASHNEXT", "")
LAYERS = int(os.environ.get("TENSORFOLD_EXL3_FLASHNEXT_LAYERS", "4"))
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                 reason="set TENSORFOLD_EXL3_FLASHNEXT to an EXL3 Flash Next checkpoint")
WINDOWS = (1, 2, 3, 16, 17, 64, 128)


# -- the fp16 matmul --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(324, 10240), (10240, 320), (96, 2560), (2560, 2560), (10240, 2560)])
def test_f16_matches_fp64_and_is_row_invariant(n, k):
    g = torch.Generator().manual_seed(n + k)
    sc = exl3_mm.Scratch(11)
    w = (torch.randn(n, k, generator=g) * 0.02).half()
    lin = exl3_mm.f16(sc, [w], DEV)
    sc.part = torch.empty((lin.sk * exl3_mm.ROWS * n,), dtype=torch.float32, device=DEV)
    x = torch.randn(exl3_mm.ROWS, k, generator=g).to(torch.bfloat16).to(DEV)
    full = torch.empty((exl3_mm.ROWS, n), dtype=torch.bfloat16, device=DEV)
    lin(x, full)
    ref = x.double() @ w.double().to(DEV).t()
    assert ((full.double() - ref).norm() / ref.norm()).item() < 1e-2
    for rows in WINDOWS:
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=DEV)
        lin(x[:rows].contiguous(), out)
        assert torch.equal(out, full[:rows]), rows
        one = torch.empty((1, n), dtype=torch.bfloat16, device=DEV)
        lin(x[rows - 1:rows].contiguous(), one)
        assert torch.equal(one, full[rows - 1:rows]), rows
    alone = lin.partials(x[3:4]).clone()
    assert torch.equal(lin.partials(x[:5])[:, 3], alone[:, 0])


# -- the n-gram rows ----------------------------------------------------------------------------------------------
def _codec_reference(packed: np.ndarray, bits: int, bias: np.ndarray, heads: int) -> np.ndarray:
    """ExLlamaV3's ngram_codec.dequant_rows (unpack the ring, mul1 codebook, * scale + head bias), rounded to fp16."""

    n = packed.shape[0]
    scales = packed[:, 0].copy().view(np.float16).astype(np.float32)
    words = packed[:, 1:].view(np.uint16).astype(np.int64)
    stream = ((words[..., None] >> np.arange(16)) & 1).reshape(n, 160 * bits)
    i = np.arange(160)[:, None]
    m = np.arange(16)[None, :]
    src = ((i - m // bits) % 160) * bits + m % bits
    states = (stream[:, src] << m).sum(axis=-1)
    prod = (states * 0x83DCD12D) & 0xFFFFFFFF
    hs = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    cb = ((1024 + hs).astype(np.float32) * np.float32(exl3_mm.K_INV) + np.float32(exl3_mm.K_BIAS)).astype(np.float16)
    head = np.arange(n) % heads
    return (cb.astype(np.float32) * scales[:, None] + bias[head].astype(np.float32)).astype(np.float16)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_ple_rows_match_the_row_codec(bits):
    rng = np.random.default_rng(bits)
    heads, rows = 16, 9
    words = 1 + 160 * bits // 16
    packed = rng.integers(-(2**15), 2**15, size=(rows * heads, words), dtype=np.int64).astype(np.int16)
    packed[:, 0] = (rng.random(rows * heads).astype(np.float16) * np.float16(0.05) + np.float16(0.001)).view(np.int16)
    bias = (rng.standard_normal((heads, 160)) * 0.01).astype(np.float16)
    out = torch.empty((rows, heads * 160), dtype=torch.float16, device=DEV)
    exl3_mm.ple_rows(rows, torch.from_numpy(packed).to(DEV), torch.from_numpy(bias).to(DEV), heads, 160, bits, out)
    ref = _codec_reference(packed, bits, bias, heads).reshape(rows, heads * 160)
    got = out.cpu().numpy()
    assert np.array_equal(got.view(np.int16), ref.view(np.int16)), np.abs(got.astype(np.float32) - ref).max()


# -- a real checkpoint -----------------------------------------------------------------------------------------
@needs_model
def test_pack_norms_are_centred_and_rows_equal_exllamav3():
    pk = exl3_pack.Pack(MODEL)
    names = [f"model.language_model.layers.{i}.attn_hyper_connection.hc_norm.weight" for i in range(8)]
    assert exl3.centred_offset(pk, names) == 1.0
    ext = pytest.importorskip("exllamav3.ext").exllamav3_ext
    base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    t = exl3_pack.NgramTable(pk, base, 128, DEV)
    rng = np.random.default_rng(1)
    ids = np.stack([rng.integers(int(t.head_offsets[h]), int(t.head_offsets[h] + t.head_sizes[h]), size=(64,))
                    for h in range(16)], axis=1).reshape(-1)
    rows = torch.from_numpy(t.gather(ids)).to(DEV)
    heads = torch.from_numpy(np.tile(np.arange(16), 64)).to(torch.int32).to(DEV)
    ref = torch.empty((len(ids), 160), dtype=torch.half, device=DEV)
    ext.ngram_dequant(rows, t.bits, heads, t.head_bias, ref)
    out = torch.empty((64, 16 * 160), dtype=torch.half, device=DEV)
    exl3_mm.ple_rows(64, rows, t.head_bias, 16, 160, t.bits, out)
    assert torch.equal(out.view(-1, 160), ref)


@needs_model
def test_real_routed_experts_match_fp64_and_are_row_invariant():
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.cuda.exl3.experts import Scratch, routed

    pk = exl3_pack.Pack(MODEL)
    name = "model.language_model.layers.5.mlp"
    ex = exl3.expert_table(pk, name + ".experts", 512, name + ".shared_expert", DEV)
    s = Scratch(ex, exl3_mm.MOE_WINDOW, 11, device=DEV)
    g = torch.Generator().manual_seed(2)
    rows = 64
    x = (torch.randn(rows, 2560, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    pick = torch.stack([torch.randperm(512, generator=g)[:10] for _ in range(rows)]).to(torch.int32)
    pick = torch.cat([pick, torch.full((rows, 1), 512, dtype=torch.int32)], 1).to(DEV).contiguous()
    y = routed(x, pick, None, ex, s, None, rows).view(rows, 11, 2560).clone()
    for n in (1, 2, 3, 16, 17):
        got = routed(x[:n].contiguous(), pick[:n].contiguous(), None, ex, s, None, n).view(n, 11, 2560)
        assert torch.equal(got, y[:n]), n
        one = routed(x[n - 1:n].contiguous(), pick[n - 1:n].contiguous(), None, ex, s, None, 1).view(1, 11, 2560)
        assert torch.equal(one, y[n - 1:n]), n
    for r in range(2):
        for k in range(11):
            e = int(pick[r, k])
            nm = f"{name}.experts.{e}" if e < 512 else f"{name}.shared_expert"

            def wmat(p):
                t = pk.get(f"{nm}.{p}.trellis")
                return fmt.dequantize(t.numpy(), pk.get(f"{nm}.{p}.suh").numpy(), pk.get(f"{nm}.{p}.svh").numpy(),
                                      fmt.bits_of(t.shape), pk.codebook(f"{nm}.{p}"))

            xr = x[r].double().cpu().numpy()
            gt, up = xr @ wmat("gate_proj"), xr @ wmat("up_proj")
            ref = (gt / (1 + np.exp(-gt)) * up) @ wmat("down_proj")
            got = y[r, k].double().cpu().numpy()
            assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 2e-2, (r, k, e)


@pytest.fixture(scope="module")
def cut_model():
    from tensorfold.families.qwen4_exp.cuda import weights as W

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = LAYERS
        c.ple_layers = [i for i in c.ple_layers if i < LAYERS]
        return c

    W.Config.read = staticmethod(cut)
    try:
        w = exl3.load(MODEL, DEV, mtp=True, draft_vocab="default")
    finally:
        W.Config.read = real
    yield w
    del w
    torch.cuda.empty_cache()


@needs_model
def test_cut_model_windows_equal_one_row_steps(cut_model):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward

    w = cut_model
    e = Engine(w, capacity=512, max_rows=max(WINDOWS), prefill_rows=128, graphs=False)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, w.cfg.vocab, size=150)]
    e.reset()
    ref = []
    for t in toks:
        ref.append(forward(w, e.st, e.buf, [t])[:1].clone())
        commit(w, e.st, e.buf, 1, 1)
    ref = torch.cat(ref)
    for rows in WINDOWS:
        e.reset()
        got = []
        for at in range(0, len(toks), rows):
            chunk = toks[at:at + rows]
            got.append(forward(w, e.st, e.buf, chunk)[:len(chunk)].clone())
            commit(w, e.st, e.buf, len(chunk), len(chunk))
        assert torch.equal(torch.cat(got), ref), rows
    e.reset()
    forward(w, e.st, e.buf, toks[:10])
    commit(w, e.st, e.buf, 10, 4)
    assert torch.equal(forward(w, e.st, e.buf, toks[4:12])[:8], ref[4:12])


@needs_model
@pytest.mark.parametrize("graphs", [False, True])
def test_cut_model_mtp_drafts_emit_serial_tokens(cut_model, graphs):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    w = cut_model
    assert w.mtp is not None
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=graphs)
    prompt = [int(t) for t in np.random.default_rng(4).integers(0, w.cfg.vocab, size=37)]
    for sampling in (None, Sampling(seed=1234, top_k=20, top_p=0.95)):
        s = serial_decode(e, prefill(e, prompt, sampling, mtp=False), 48, sampling)
        d = mtp_decode(e, prefill(e, prompt, sampling, mtp=True), 48, sampling, depth=6)
        assert s.tokens == d.tokens, sampling


@needs_model
def test_cut_model_prompts_ignore_chunking_and_resume_as_fresh(cut_model):
    """The EXL3 prompt path: state and first token do not depend on chunk size; a kept prompt end resumes as fresh."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill

    w = cut_model
    prompt = [int(t) for t in np.random.default_rng(5).integers(0, w.cfg.vocab, size=150)]
    runs = []
    for rows in (150, 64, 17):
        e = Engine(w, capacity=512, max_rows=8, prefill_rows=rows, graphs=False)
        first = prefill(e, prompt, None)
        runs.append((first, e.st.snapshot(), e.last_streams.clone()))
    for first, snap, tail in runs[1:]:
        assert first == runs[0][0]
        assert torch.equal(tail, runs[0][2])
        for key in ("rec", "conv", "ple_tail"):
            assert torch.equal(snap[key], runs[0][1][key]), key
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=64, graphs=False)
    prefill(e, prompt[:90], None)
    kept = {"state": e.st.snapshot(), "tail": e.last_streams.clone()}
    first = prefill(e, prompt, None, resume=kept)
    assert first == runs[0][0]
    snap = e.st.snapshot()
    for key in ("rec", "conv", "ple_tail"):
        assert torch.equal(snap[key], runs[0][1][key]), key


@needs_model
def test_cut_model_ngram_read_ahead_changes_no_bits(cut_model, monkeypatch):
    """TF_NGRAM_AHEAD: a prompt longer than a chunk asks for its later chunks' n-gram pages ahead; its first token,
    state and last streams are those without it."""

    from tensorfold.families.qwen4_exp.cuda import decode
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill

    w = cut_model
    prompt = [int(t) for t in np.random.default_rng(6).integers(0, w.cfg.vocab, size=300)]
    p = next(lay.ple for lay in w.layers if lay.ple is not None)
    assert p.table.willneed(p.ngram.ids(p.ngram.initial_history(), np.asarray(prompt[:64]))) > 0
    runs = []
    for ahead in (False, True):
        monkeypatch.setattr(decode, "NGRAM_AHEAD", ahead)
        e = Engine(w, capacity=512, max_rows=8, prefill_rows=64, graphs=False)
        first = prefill(e, prompt, None)
        runs.append((first, e.st.snapshot(), e.last_streams.clone()))
    assert runs[0][0] == runs[1][0] and torch.equal(runs[0][2], runs[1][2])
    for key in ("rec", "conv", "ple_tail"):
        assert torch.equal(runs[0][1][key], runs[1][1][key]), key
