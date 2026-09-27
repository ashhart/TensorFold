"""Qwen3.8-27B on an EXL3 checkpoint: the verify window's rows are independent, and drafted replies equal serial.

The synthetic part runs anywhere CUDA is: a one-layer model whose projections are EXL3 trellis layers (random
words, the mul1 codebook) and whose in_proj_a/b and embedding are plain bf16, as a real pack stores them.

The checkpoint part needs ``TENSORFOLD_QWEN27_EXL3=<pack dir>`` and ``TENSORFOLD_QWEN27_DRAFTER=<DFlash2 dir>``
(turboderp/Qwen3.8-27B-exl3 at any width, z-lab/Qwen3.8-27B-DFlash2); skipped otherwise.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.exl3.linear import Exl3Linear
from tensorfold.families.qwen3_5.cuda.decode import draft_decode, serial_decode
from tensorfold.families.qwen3_5.cuda.dflash2 import _exl3_sub_head
from tensorfold.families.qwen3_5.cuda.exl3_load import PLANS
from tensorfold.families.qwen3_5.cuda.forward import State, commit, tree_forward
from tensorfold.families.qwen3_5.cuda.weights import GDN, Config, Exl3, Layer, Plain, Weights

MODEL = os.environ.get("TENSORFOLD_QWEN27_EXL3", "")
DRAFTER = os.environ.get("TENSORFOLD_QWEN27_DRAFTER", "")


def _trellis_layer(n: int, k: int, bits: int, gen: torch.Generator, split=None) -> Exl3Linear:
    words = torch.randint(0, 1 << 16, (k // 16, n // 16, 16 * bits), generator=gen, dtype=torch.int32)
    suh = (torch.randn(k, generator=gen) * 0.05).half()
    svh = (torch.randn(n, generator=gen) * 0.05).half()
    layer = Exl3Linear.from_tensors(words.to(torch.int16), suh, svh, "mul1", device="cuda")
    if split is not None:
        layer.split = split
    return layer


def _model() -> Weights:
    gen = torch.Generator().manual_seed(9)

    def ex(n, k):
        return Exl3(_trellis_layer(n, k, 3, gen))

    def plain(n, k):
        return Plain((torch.randn(n, k, generator=gen) * 0.05).to(torch.bfloat16).cuda())

    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    conv = (torch.randn(384, 4, generator=gen) * 0.1).to(torch.bfloat16).cuda()
    gdn = GDN(ex(384, 128), ex(128, 128), plain(1, 128), plain(1, 128), ex(128, 128), conv,
              torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda"), norm)
    layer = Layer(True, norm, norm, gdn, None, ex(128, 128), ex(128, 128), ex(128, 128))
    config = Config(hidden=128, intermediate=128, layers=1, heads=1, kv_heads=1, head_dim=128, vocab=256, k_heads=1,
                    v_heads=1, dk=128, dv=128, conv_kernel=4, interval=4, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    return Weights(config, plain(256, 128), [layer], norm, ex(256, 128), torch.ones(16, device="cuda"))


@pytest.mark.parametrize("shape", sorted(PLANS), ids=lambda s: f"{s[0]:g}b-{s[1]}x{s[2]}")
def test_tuned_plans_keep_rows_independent(shape):
    """Every tuned (K splits, warps) of the 27B's projections: a row's output is the same bits alone or among 16."""

    bits, k, n = shape
    gen = torch.Generator().manual_seed(k + n)
    layer = _trellis_layer(n, k, int(bits), gen, split=PLANS[shape])
    x = torch.randn(16, k, generator=gen).to(torch.bfloat16).cuda()
    full = layer(x)
    for rows in (1, 2, 5, 12):
        assert torch.equal(layer(x[:rows]), full[:rows]), rows
    for r in (3, 15):
        assert torch.equal(layer(x[r:r + 1]), full[r:r + 1]), r


def test_window_rows_equal_serial_steps():
    """A GDN verify window on EXL3 + plain weights: each path's rows and the committed state equal serial steps."""

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


def test_drafted_decode_keeps_serial_tokens():
    """draft_decode with a drafter that is right three times then wrong gives serial_decode's tokens."""

    import dataclasses

    import numpy as np

    base = _model()
    # the drafter's taps come from five of 64 layers: the one layer, 64 times
    w = Weights(dataclasses.replace(base.config, layers=64), base.embed, [base.layers[0]] * 64, base.norm,
                base.head, base.inv_freq)
    ref = serial_decode(w, State(w), 5, 24, None, stop_eos=False).tokens

    class Drafter:
        last_candidates = None

        def propose_tree(self, pending, context_length, max_nodes, sampling):
            k = context_length
            right = [ref[k + j] if k + j < len(ref) else 1 for j in range(4)]
            self.last_candidates = np.array([[t] * 16 for t in right])
            guesses = right[:3] + [(right[3] + 1) % 256, (right[0] + 1) % 256]
            return guesses[:max_nodes], [-1, 0, 1, 2, -1][:max_nodes]

        def add_taps(self, taps):
            pass

        def in_vocab(self, token):
            return True

    drafted = draft_decode(w, State(w), [], 5, 24, None, Drafter(), max_rows=8, allow_copy=False, stop_eos=False)
    assert drafted.tokens == ref and drafted.rounds < len(ref) - 1


def test_drafter_head_is_the_target_head_sliced():
    """The drafter's EXL3 sub-head (whole 128-column strips, same split) gives the target head's logits bit for bit."""

    gen = torch.Generator().manual_seed(4)
    head = _trellis_layer(1024, 256, 6, gen)
    spans = ((0, 300), (700, 1000))
    sub, cols = _exl3_sub_head(head, spans)
    assert sub.n == 128 * 6 and sub.split == head.split           # blocks 0-2 and 5-7
    x = torch.randn(7, 256, generator=gen).to(torch.bfloat16).cuda()
    want = torch.cat([head(x)[:, a:b] for a, b in spans], dim=1)
    assert torch.equal(sub(x).index_select(1, cols), want)


@pytest.fixture(scope="module")
def engine():
    if not (MODEL and Path(MODEL).is_dir() and DRAFTER and Path(DRAFTER).is_dir()):
        pytest.skip("needs TENSORFOLD_QWEN27_EXL3 and TENSORFOLD_QWEN27_DRAFTER")
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    return Qwen27Engine(Path(MODEL), Path(DRAFTER), max_rows=12)


def _ids(engine, prompt, sampling, draft, tokens=48):
    out: list[int] = []
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new) and False, draft=draft)
    return out, stats


@pytest.mark.parametrize("seed", [None, 1234, 1237], ids=["greedy", "seed1234", "seed1237"])
def test_checkpoint_drafted_equals_serial(engine, seed):
    """The real pack with DFlash2: drafted replies are serial's token ids (SHA-256 of the ids compared)."""

    from tokenizers import Tokenizer

    from tensorfold.engine.exact_sampling import Sampling

    tok = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    prompt = tok.encode("Write a short Python function that computes the Fibonacci sequence and explain it.",
                        add_special_tokens=False).ids
    sampling = None if seed is None else Sampling(seed, 1.0, 20, 0.95)
    serial, s_stats = _ids(engine, prompt, sampling, draft=False)
    drafted, d_stats = _ids(engine, prompt, sampling, draft=True)
    digest = [hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() for ids in (serial, drafted)]
    assert len(serial) == 48 and digest[0] == digest[1]
    assert d_stats["rounds"] < s_stats["rounds"]                     # drafting did accept tokens
