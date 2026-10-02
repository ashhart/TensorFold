"""Flash Next's EXL3 path with --parallel: a decode window and a prompt pass share each layer's expert launch.

Needs ``TENSORFOLD_EXL3_FLASHNEXT=<an EXL3 Flash Next checkpoint>`` (a model cut to a few layers, as in
test_qwen4_exp_exl3.py): one forward for a window and a pass gives both the rows they get apart, the window's slots in
fp32 and the pass's in bf16; and prompts filling inside the rounds leave every stream its solo tokens.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

MODEL = os.environ.get("TENSORFOLD_EXL3_FLASHNEXT", "")
LAYERS = int(os.environ.get("TENSORFOLD_EXL3_FLASHNEXT_LAYERS", "4"))
pytestmark = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                reason="set TENSORFOLD_EXL3_FLASHNEXT to an EXL3 Flash Next checkpoint")


@pytest.fixture(scope="module")
def cut_model():
    from tensorfold.families.qwen4_exp.cuda import exl3
    from tensorfold.families.qwen4_exp.cuda import weights as W

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = LAYERS
        c.ple_layers = [i for i in c.ple_layers if i < LAYERS]
        return c

    W.Config.read = staticmethod(cut)
    try:
        w = exl3.load(MODEL, "cuda", mtp=True, draft_vocab="default")
    finally:
        W.Config.read = real
    yield w
    del w
    torch.cuda.empty_cache()


def _tokens(seed: int, n: int, vocab: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(0, vocab, size=n)]


@pytest.mark.parametrize("pass_rows", [40, 300])
def test_exl3_window_and_pass_share_a_launch_and_keep_their_bits(cut_model, pass_rows, monkeypatch):
    """compute_mixed on an EXL3 pack == the window and the pass apart (logits, streams, the pass's heads), across a
    routed window boundary too."""

    from tensorfold.families.qwen4_exp.cuda import exl3_pack
    from tensorfold.families.qwen4_exp.cuda import forward as fwd
    from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, forward, stage
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = cut_model
    V = w.cfg.vocab
    if pass_rows > 256:
        monkeypatch.setattr(exl3_pack, "MOE_WINDOW", 256)   # the window's rows straddle two routed windows
    db, pb = Buffers(w, 16, 1024, moe_prefill=True), Buffers(w, pass_rows + 16, 1024, prefill=True)
    chains = [_tokens(1, 5, V), _tokens(2, 3, V)]
    pieces = [_tokens(3, pass_rows - 7, V), _tokens(4, 7, V)]
    dstates = []
    for i in range(2):
        st = State(w, 1024, 16)
        prompt = _tokens(10 + i, 9, V)
        forward(w, st, db, prompt)
        commit(w, st, db, len(prompt), len(prompt))
        dstates.append(st)
    pstates = [State(w, 1024, 16) for _ in pieces]

    def run(mixed: bool):
        d, p = [st.clone() for st in dstates], [st.clone() for st in pstates]
        if mixed:                                    # both staged first, as a round stages them
            segs, psegs = stage(w, db, list(zip(d, chains))), stage(w, pb, list(zip(p, pieces)))
            ends = [a1 - 1 for _, _, a1 in psegs]
            lg, heads = fwd.compute_mixed(w, segs, db, psegs, pb, ends=ends)
        else:                                        # each staged right before its own forward
            segs = stage(w, db, list(zip(d, chains)))
            lg = compute(w, segs, db).clone()
            psegs = stage(w, pb, list(zip(p, pieces)))
            ends = [a1 - 1 for _, _, a1 in psegs]
            heads = compute(w, psegs, pb, logits=True, ends=ends)
        rd, rp = segs[-1][2], psegs[-1][2]
        return lg[:rd].clone(), db.streams[:rd].clone(), heads[:len(ends)].clone(), pb.streams[:rp].clone()

    apart = run(False)
    together = run(True)
    assert all(torch.equal(x, y) for x, y in zip(together, apart))


@pytest.mark.parametrize("converge", [False, True])
def test_exl3_prompts_fill_inside_rounds_and_streams_keep_their_tokens(cut_model, converge, monkeypatch):
    """--parallel on an EXL3 pack converges (a pass shares the round's forward); every stream emits its solo run."""

    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda import multi
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    if not converge:
        monkeypatch.setattr(multi, "converges", lambda w: False)

    w = cut_model
    V = w.cfg.vocab
    prompts = [_tokens(20, 9, V), _tokens(21, 150, V), _tokens(22, 5, V)]
    samplings = [Sampling(seed=3, top_k=20, top_p=0.95), None, None]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=False)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 16, sampling).tokens)
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, prefill_rows=64)
    assert dec.converged == converge
    first = Stream(prompts[0], 16, samplings[0])
    dec.admit(first)
    dec.finish(dec.round())
    rest = [Stream(p, 16, s) for p, s in zip(prompts[1:], samplings[1:])]
    for s in rest:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in [first, *rest]] == refs
