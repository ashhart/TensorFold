"""Sequence-parallel prompt chunks (fused.compute_prompt_sp, TF_GLM53_PROMPT_SP=1) on real weights (TF_GLM53_CKPT),
four ranks as threads of one GPU, layers 0-3:

1. close to the exact reference (fused.compute, rank-order reduce) past index_topk, with the same greedy pick, the
   same bits on every rank, and (exact reduce) the same bits run to run;
2. the engine's reply with SP prompt chunks matches the default path's first token (MTP drafts on).
"""

from __future__ import annotations

import os

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from threadcomm import run_ranks  # noqa: E402
from test_glm_moe_dsa_fused import _fused, _tokens  # noqa: E402


def _chunks(w, st, b0, b1, pos1, cs, toks, rows, sp):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    hid = []
    for a in range(0, len(toks), rows):
        e = min(a + rows, len(toks))
        R = e - a
        T = e if e > w.cfg.index_topk else None
        b0.ids[:R].copy_(toks[a:e])
        st.pos.fill_(a)
        pos1.fill_(a + R // 2)
        last = "last" if e == len(toks) else "none"
        if sp:
            fused.compute_prompt_sp(w, st, b0, b1, R, T, pos1, cs, logits=last)
        else:
            fused.compute(w, st, b0, R, T, logits=last)
        hid.append(b0.hidden[:R].clone())
    return torch.cat(hid), b0.logits[:1].clone()


@pytest.mark.parametrize("reduce", ["rs", "ring"])
def test_sp_chunks_close_to_reference(monkeypatch, reduce):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    toks = _tokens(2600, seed=3).cuda()
    rows = 1024

    def run(rank, comm):
        w, st = _fused(rank, 4, comm, 4096)
        b0 = fused.Buffers(w, rows, 4096)
        b1 = fused.Buffers(w, rows - rows // 2, 4096)
        pos1 = torch.zeros((1,), dtype=torch.int32, device="cuda")
        cs = torch.cuda.Stream()
        fused.PREFILL_REDUCE = "rs"
        ref = _chunks(w, st, b0, b1, pos1, cs, toks, rows, False)
        comm.barrier()
        fused.PREFILL_REDUCE = reduce
        comm.barrier()
        sp = _chunks(w, st, b0, b1, pos1, cs, toks, rows, True)
        again = _chunks(w, st, b0, b1, pos1, cs, toks, rows, True)
        return ref, sp, again

    monkeypatch.setattr(fused, "PREFILL_REDUCE", "rs")
    res = run_ranks(run, 4)
    for ref, sp, again in res:
        rel = float((sp[0].float() - ref[0].float()).norm() / ref[0].float().norm())
        assert rel < 1e-2, rel
        lrel = float((sp[1] - ref[1]).norm() / ref[1].norm())
        assert lrel < 2e-2, lrel
        assert int(sp[1].argmax()) == int(ref[1].argmax())
        if reduce == "rs":
            assert torch.equal(sp[0], again[0]) and torch.equal(sp[1], again[1])
    for r in range(1, 4):
        assert torch.equal(res[r][1][0], res[0][1][0]), f"rank {r} differs"


def _reply(rank, comm, prompt):
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    e = Glm53Engine(CKPT, rank=rank, master="", port=0, context=1024, comm=comm, layers=4, mtp_drafts=2)
    if rank:
        e.follow(requests=1)
        return None
    got = []
    e.generate(prompt, 4, None, got.extend, stop_eos=False)
    return got


def test_engine_reply_with_sp_prompt(monkeypatch):
    from tensorfold.families.glm_moe_dsa.cuda import engine, fused

    monkeypatch.setattr(engine, "GRAPHS", False)          # rank threads must not capture graphs concurrently

    prompt = [int(t) for t in _tokens(700, seed=21)]
    base = run_ranks(lambda r, c: _reply(r, c, prompt), 4)[0]
    torch.cuda.empty_cache()
    calls = []
    real = fused.compute_prompt_sp
    monkeypatch.setattr(fused, "PROMPT_SP", True)
    monkeypatch.setattr(fused, "compute_prompt_sp", lambda *a, **k: (calls.append(1), real(*a, **k)))
    sp = run_ranks(lambda r, c: _reply(r, c, prompt), 4)[0]
    assert calls, "the SP prompt path did not run"
    assert sp[0] == base[0], (base, sp)
