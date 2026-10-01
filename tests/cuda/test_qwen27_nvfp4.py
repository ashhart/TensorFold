"""Qwen3.8-27B on an NVFP4 checkpoint: verify rows equal serial steps, prompts ignore chunking, drafted equals serial.

The synthetic part runs anywhere CUDA is: a one-layer model with NVFP4 MLP and head, FP8 GDN projections and bf16
gates, as a ModelOpt export stores them. The checkpoint part needs ``TENSORFOLD_QWEN27_NVFP4=<dir>`` and
``TENSORFOLD_QWEN27_DRAFTER=<DFlash2 dir>``; skipped otherwise.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear
from tensorfold.families.qwen3_5.cuda.dflash2 import _sub_parts
from tensorfold.families.qwen3_5.cuda.forward import State, commit, tree_forward
from tensorfold.families.qwen3_5.cuda.nvfp4_load import Plain8
from tensorfold.families.qwen3_5.cuda.weights import GDN, Config, Layer, Plain, Weights

MODEL = os.environ.get("TENSORFOLD_QWEN27_NVFP4", "")
DRAFTER = os.environ.get("TENSORFOLD_QWEN27_DRAFTER", "")


def _fp4(n: int, k: int, gen: torch.Generator) -> Fp4Linear:
    packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
    scale = torch.randint(0x20, 0x40, (n, k // 16), generator=gen, dtype=torch.uint8)
    return Fp4Linear.from_checkpoint(packed.cuda(), scale.cuda(), 0.05)


def _fp8(n: int, k: int, gen: torch.Generator) -> Fp8Linear:
    w = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.uint8)
    w[(w & 0x7F) >= 0x60] = 0x30                            # finite, magnitudes below 2^5
    return Fp8Linear.from_checkpoint(w.cuda().view(torch.float8_e4m3fn), 0.004)


def _model() -> Weights:
    gen = torch.Generator().manual_seed(9)

    def gate(n, k):
        w = (torch.randn(n, k, generator=gen) * 0.05).to(torch.bfloat16).cuda()
        return Plain8(w, rows8=Fp8Linear.from_bf16(w))

    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    conv = (torch.randn(384, 4, generator=gen) * 0.1).to(torch.bfloat16).cuda()
    gdn = GDN(_fp8(384, 128, gen), _fp8(128, 128, gen), gate(1, 128), gate(1, 128), _fp8(128, 128, gen), conv,
              torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda"), norm)
    layer = Layer(True, norm, norm, gdn, None, _fp4(128, 128, gen), _fp4(128, 128, gen), _fp4(128, 128, gen))
    config = Config(hidden=128, intermediate=128, layers=1, heads=1, kv_heads=1, head_dim=128, vocab=256, k_heads=1,
                    v_heads=1, dk=128, dv=128, conv_kernel=4, interval=4, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    embed = Plain((torch.randn(256, 128, generator=gen) * 0.5).to(torch.bfloat16).cuda())
    return Weights(config, embed, [layer], norm, _fp4(256, 128, gen), torch.ones(16, device="cuda"), quant="nvfp4")


def test_window_rows_equal_serial_steps():
    """A GDN verify window on NVFP4 / FP8 / bf16 weights: each path's rows and the committed state equal serial steps."""

    w = _model()
    tokens = torch.tensor([7, 8, 9, 10], device="cuda", dtype=torch.int32)
    parents = [-1, 0, 0, 1]
    state = State(w)
    logits, record = tree_forward(w, tokens, parents, state)
    serial, st = {}, State(w)
    for row, tok in ((0, 7), (1, 8), (3, 10)):
        out, rec = tree_forward(w, torch.tensor([tok], device="cuda", dtype=torch.int32), [-1], st)
        serial[row] = out[0]
        commit(st, rec, [0])
    for row in (0, 1, 3):
        assert torch.equal(logits[row], serial[row]), f"row {row} differs"
    commit(state, record, [0, 1, 3])
    assert torch.equal(state.rec[0], st.rec[0]) and torch.equal(state.conv[0], st.conv[0])


def test_prompts_ignore_chunking_and_resume_as_fresh():
    """FP8 prompt rows on staged NVFP4, FP8 and gate copies: the same state in any chunks, and resumed == fresh."""

    from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

    w = _model()
    prompt = [3 + (7 * i) % 250 for i in range(67)]
    runs = []
    for size in (67, 16, 5):
        st = State(w)
        runs.append((prefill_state(w, prompt, st, size=size), st))
    for normed, st in runs[1:]:
        assert torch.equal(normed, runs[0][0])
        assert torch.equal(st.rec[0], runs[0][1].rec[0]) and torch.equal(st.conv[0], runs[0][1].conv[0])
    st = State(w)
    prefill_state(w, prompt[:30], st, size=16)
    normed = prefill_state(w, prompt, st, size=16)
    assert torch.equal(normed, runs[0][0]) and torch.equal(st.rec[0], runs[0][1].rec[0])


def test_drafter_head_parts_are_the_head_columns():
    """The drafter's head parts (tile views, no copy) give the head's span columns; bits match at the same K split."""

    gen = torch.Generator().manual_seed(4)
    head = _fp4(1024, 512, gen)
    spans = ((0, 384), (672, 1000))
    parts = _sub_parts(head, spans)
    assert [p.n for p, _, _ in parts] == [384, 384] and parts[0][0].words.data_ptr() == head.words.data_ptr()
    x = torch.randn(7, 512, generator=gen).to(torch.bfloat16).cuda()
    got = torch.cat([p(x)[:, lo:hi] for p, lo, hi in parts], dim=1)
    want = torch.cat([head(x)[:, a:b] for a, b in spans], dim=1)
    assert torch.allclose(got.float(), want.float(), rtol=1e-2, atol=1e-2)
    full = head.tiles(0, 16)
    assert torch.equal(full(x), head(x))


def test_gate_copy_tracks_the_bf16_product():
    """A bf16 gate's e4m3 prompt copy: within e4m3's rounding of the bf16 product on FP8 rows."""

    from tensorfold.cuda.kernels.qmm import quantize_rows

    gen = torch.Generator().manual_seed(5)
    w = (torch.randn(48, 5120, generator=gen) * 0.02).to(torch.bfloat16).cuda()
    gate = Plain8(w, rows8=Fp8Linear.from_bf16(w))
    x = torch.randn(300, 5120, generator=gen).to(torch.bfloat16).cuda()
    want = (x.float() @ w.float().t())
    got = gate.prefill8(quantize_rows(x)).float()
    assert float((got - want).norm() / want.norm()) < 0.04
    assert torch.equal(gate(x[:3]), gate(x)[:3])


@pytest.fixture(scope="module")
def engine():
    if not (MODEL and Path(MODEL).is_dir() and DRAFTER and Path(DRAFTER).is_dir()):
        pytest.skip("needs TENSORFOLD_QWEN27_NVFP4 and TENSORFOLD_QWEN27_DRAFTER")
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    return Qwen27Engine(Path(MODEL), Path(DRAFTER), max_rows=12)


def _ids(engine, prompt, sampling, draft, tokens=48):
    out: list[int] = []
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new) and False, draft=draft)
    return out, stats


@pytest.mark.parametrize("seed", [None, 1234, 1237], ids=["greedy", "seed1234", "seed1237"])
def test_checkpoint_drafted_equals_serial(engine, seed):
    """The real checkpoint with DFlash2: drafted replies are serial's token ids (SHA-256 of the ids compared)."""

    from tokenizers import Tokenizer

    from tensorfold.engine.exact_sampling import Sampling

    tok = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    prompt = tok.encode("Write a short Python function that computes the Fibonacci sequence and explain it.",
                        add_special_tokens=False).ids
    sampling = None if seed is None else Sampling(seed, 1.0, 20, 0.95)
    serial, s_stats = _ids(engine, prompt, sampling, draft=False)
    drafted, d_stats = _ids(engine, prompt, sampling, draft=True)
    digest = [hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() for ids in (serial, drafted)]
    print(seed, tok.decode(serial)[:200].replace("\n", " "), s_stats, d_stats)
    assert len(serial) == 48 and digest[0] == digest[1]
    assert d_stats["rounds"] < s_stats["rounds"]                     # drafting did accept tokens


def test_an_nvfp4_checkpoint_refuses_two_ranks_and_vision(tmp_path, monkeypatch):
    """Two ranks and image input are tested on the MLX checkpoint only: an NVFP4 one stops at startup by name."""

    import json

    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    monkeypatch.setattr("torch.distributed.init_process_group", lambda *a, **k: pytest.fail("the ranks started"))

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "quantization_config": {
        "quant_method": "modelopt", "quant_algo": "NVFP4"}}))
    with pytest.raises(ValueError, match="one GPU"):
        Qwen27Engine(tmp_path, None, tp=2, master="127.0.0.1")
    with pytest.raises(ValueError, match="vision"):
        Qwen27Engine(tmp_path, None, vision=True)
