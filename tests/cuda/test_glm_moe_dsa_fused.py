"""Full GLM-5.3's fused path (M3-M5) on real weights (TF_GLM53_CKPT), four ranks as threads of one GPU, layers 0-3
(three dense MLP layers, then the first MoE layer):

1. the fused forward agrees with the M2 reference path (torch ops) on a prompt;
2. a verify window of 3 rows gives each row the bits of 3 serial one-row steps (a prompt chunk's rows: close);
3. past index_topk the token-level selection agrees with the reference indexer, and the window rows stay exact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from threadcomm import run_ranks  # noqa: E402

LAYERS = (0, 1, 2, 3)


def _parts(rank, world):
    from tensorfold.families.glm_moe_dsa.config import Config
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_layer

    cfg = Config.from_dict(json.loads((Path(CKPT) / "config.json").read_text()))
    r = RankReader(CKPT, rank, world)
    return cfg, r, [r.get(t, "cuda") for t in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")], \
        [load_layer(r, cfg, i) for i in LAYERS]


def _fused(rank, world, comm, capacity):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    cfg, _, (embed, norm, head), layers = _parts(rank, world)
    w = fused.Weights(cfg, rank, world, comm, embed, norm, head, layers, None)
    st = fused.State(w, capacity)
    return w, st


def _tokens(n, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 150000, (n,), generator=g)


def _prefill(w, st, b, toks, a, e):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    R = e - a
    b.ids[:R].copy_(toks[a:e].cuda())
    st.pos.fill_(a)
    fused.compute(w, st, b, R, fused.bucket(e, w.cfg.index_topk) and e, logits="none")
    return b.hidden[:R].clone()


def _window(w, st, b, toks, a, e):
    """Rows a..e-1 as one decode window (graph-sized key range)."""
    from tensorfold.families.glm_moe_dsa.cuda import fused

    R = e - a
    b.ids[:R].copy_(toks[a:e].cuda())
    st.pos.fill_(a)
    fused.compute(w, st, b, R, fused.bucket(e, w.cfg.index_topk), logits="all")
    return b.hidden[:R].clone(), b.logits[:R].clone()


def test_fused_matches_reference():
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.model import RankModel

    T = 24
    toks = _tokens(T)

    def run(rank, comm):
        cfg, r, (embed, norm, head), layers = _parts(rank, 4)
        ref = RankModel(cfg, rank, 4, comm, embed=embed, final_norm=norm, lm_head=head, layers=layers)
        caches = [torch.zeros((T, cfg.latent_width), dtype=torch.bfloat16, device="cuda") for _ in layers]
        ic = {i: torch.zeros((T, cfg.index_head_dim), dtype=torch.bfloat16, device="cuda")
              for i in LAYERS if cfg.full_indexer(i)}
        want = ref.forward(toks.cuda(), torch.arange(T, device="cuda"), caches, ic).float()
        w, st = _fused(rank, 4, comm, 64)
        b = fused.Buffers(w, 32, 64)
        got = _prefill(w, st, b, toks, 0, T).float()
        return want, got

    for want, got in run_ranks(run, 4):
        rel = float((got - want).norm() / want.norm())
        assert rel < 2e-2, rel


def test_verify_window_rows_equal_serial_steps():
    from tensorfold.families.glm_moe_dsa.cuda import fused

    toks = _tokens(40, seed=7)

    def run(rank, comm):
        w, st = _fused(rank, 4, comm, 64)
        big = fused.Buffers(w, 64, 64)
        small = fused.Buffers(w, 4, 64)
        _prefill(w, st, big, toks, 0, 30)
        win_h, win_l = _window(w, st, small, toks, 30, 33)
        ser = [_window(w, st, small, toks, i, i + 1) for i in range(30, 33)]
        chunk = _prefill(w, st, big, toks, 0, 33)[30:33]          # the same rows inside a prompt chunk
        return win_h, win_l, ser, chunk

    for win_h, win_l, ser, chunk in run_ranks(run, 4):
        for i, (h, lg) in enumerate(ser):
            assert torch.equal(win_h[i:i + 1], h), f"row {i}: window hidden != serial"
            assert torch.equal(win_l[i:i + 1], lg), f"row {i}: window logits != serial"
        rel = float((chunk.float() - win_h.float()).norm() / win_h.float().norm())
        assert rel < 1e-2, rel                           # prompt chunks have their own tilings: close, not equal


def test_sparse_rows_past_index_topk():
    """Rows past index_topk attend their selected keys: exact across window sizes, close to the reference path."""
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.model import RankModel

    toks = _tokens(2100, seed=11)

    def run(rank, comm):
        w, st = _fused(rank, 4, comm, 4096)
        big = fused.Buffers(w, 128, 4096)
        small = fused.Buffers(w, 4, 4096)
        for a in range(0, 2096, 128):
            _prefill(w, st, big, toks, a, min(a + 128, 2096))
        win_h, _ = _window(w, st, small, toks, 2096, 2099)
        ser = [_window(w, st, small, toks, i, i + 1)[0] for i in range(2096, 2099)]
        cfg, r, (embed, norm, head), layers = _parts(rank, 4)
        ref = RankModel(cfg, rank, 4, comm, embed=embed, final_norm=norm, lm_head=head, layers=layers)
        n = 2099
        caches = [torch.zeros((n, cfg.latent_width), dtype=torch.bfloat16, device="cuda") for _ in layers]
        ic = {i: torch.zeros((n, cfg.index_head_dim), dtype=torch.bfloat16, device="cuda")
              for i in LAYERS if cfg.full_indexer(i)}
        want = ref.forward(toks[:n].cuda(), torch.arange(n, device="cuda"), caches, ic)[-3:].float()
        return win_h, ser, want

    for win_h, ser, want in run_ranks(run, 4):
        for i, h in enumerate(ser):
            assert torch.equal(win_h[i:i + 1], h), f"sparse row {i}: window != serial"
        rel = float((win_h.float() - want).norm() / want.norm())
        assert rel < 3e-2, rel


def test_runner_drafted_equals_serial_with_mtp():
    """The whole runner (prompt chunks, the MTP prompt pass, merged-refresh drafting) on layers 0-3 + the MTP layer:
    drafted and serial greedy replies are identical, for every MTP input variant."""
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner
    from tensorfold.families.glm_moe_dsa.cuda.weights import load_mtp

    prompt = _tokens(150, seed=13).tolist()

    def run(rank, comm):
        cfg, r, (embed, norm, head), layers = _parts(rank, 4)
        w = fused.Weights(cfg, rank, 4, comm, embed, norm, head, layers, load_mtp(r, cfg))
        out = {}
        for mode in [m for m in fused.MTP_MODES if m not in ("dflash", "auto")]:
            for k in (0, 2):
                run_ = Runner(w, 512, 2, graphs=False)
                st = run_.generate(prompt, 24, None, lambda t: False, lambda new: None, k, mode)
                out[(mode, k)] = (st["out"], st["accept"])
        return out

    for out in run_ranks(run, 4):
        for mode in [m for m in fused.MTP_MODES if m not in ("dflash", "auto")]:
            assert out[(mode, 2)][0] == out[(mode, 0)][0], f"{mode}: drafted != serial"


def test_wide_prompt_chunk_matches_reference():
    """A 300-row prompt chunk (prompt GEMM for the EXL3 linears, exact reduce-scatter, row-blocked absorb/expand)
    agrees with the reference path, and decode windows after it stay exact."""
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.model import RankModel

    T = 300
    toks = _tokens(T + 3, seed=17)

    def run(rank, comm):
        cfg, r, (embed, norm, head), layers = _parts(rank, 4)
        ref = RankModel(cfg, rank, 4, comm, embed=embed, final_norm=norm, lm_head=head, layers=layers)
        caches = [torch.zeros((T, cfg.latent_width), dtype=torch.bfloat16, device="cuda") for _ in layers]
        ic = {i: torch.zeros((T, cfg.index_head_dim), dtype=torch.bfloat16, device="cuda")
              for i in LAYERS if cfg.full_indexer(i)}
        want = ref.forward(toks[:T].cuda(), torch.arange(T, device="cuda"), caches, ic).float()
        w, st = _fused(rank, 4, comm, 512)
        big = fused.Buffers(w, 512, 512)
        small = fused.Buffers(w, 4, 512)
        got = _prefill(w, st, big, toks, 0, T).float()
        win, _ = _window(w, st, small, toks, T, T + 3)
        ser = [_window(w, st, small, toks, i, i + 1)[0] for i in range(T, T + 3)]
        return want, got, win, ser

    for want, got, win, ser in run_ranks(run, 4):
        rel = float((got - want).norm() / want.norm())
        assert rel < 2e-2, rel
        for i, h in enumerate(ser):
            assert torch.equal(win[i:i + 1], h), f"row {i}: window != serial after a wide prompt chunk"


DFLASH = os.environ.get("TF_GLM53_DFLASH_TEST", "")


@pytest.mark.skipif(not DFLASH, reason="set TF_GLM53_DFLASH_TEST to a GLM-5.3 DFlash2 checkpoint")
def test_runner_dflash_drafted_equals_serial():
    """DFlash2 rounds (drafter proposes up to 7, target verifies the window, kept taps extend the drafter) give the
    serial reply. Layers 0-3 only, so the taps past layer 3 stay zero: drafts are poor, the exactness is the point."""
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.dflash import GlmDrafter
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    prompt = _tokens(150, seed=19).tolist()
    cfg_d = json.loads((Path(DFLASH) / "config.json").read_text())

    def run(rank, comm):
        cfg, r, (embed, norm, head), layers = _parts(rank, 4)
        from tensorfold.families.glm_moe_dsa.cuda.weights import load_mtp

        w = fused.Weights(cfg, rank, 4, comm, embed, norm, head, layers, load_mtp(r, cfg))
        w.tap_slot = {int(i): s for s, i in enumerate(cfg_d["dflash_config"]["target_layer_ids"])}
        run_ = Runner(w, 512, 2, graphs=False)
        run_.drafter = GlmDrafter(DFLASH, w, capacity=512)
        a = run_.generate(prompt, 40, None, lambda t: False, lambda new: None, 2, "dflash")
        b = run_.generate(prompt, 40, None, lambda t: False, lambda new: None, 0, "normed/normed")
        c = run_.generate(prompt, 40, None, lambda t: False, lambda new: None, 2, "auto")
        return a["out"], b["out"], c["out"], c["arms"]

    for a, b, c, arms in run_ranks(run, 4):
        assert a == b, "DFlash2-drafted reply != serial reply"
        assert c == b, f"auto (arms {arms}) reply != serial reply"
        assert arms["m"] and arms["f"], arms                 # both arms ran


def _fused_dcp(rank, comm, capacity, dcp):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    cfg, _, (embed, norm, head), layers = _parts(rank, 4)
    w = fused.Weights(cfg, rank, 4, comm, embed, norm, head, layers, None)
    w.dcp = dcp
    return w, fused.State(w, capacity)


def test_dcp_matches_replicated_and_windows_stay_exact():
    """Decode context parallelism (KV positions interleaved over 4 ranks, all-gathered queries, rank-order merge of
    log-sum-exp partials, distributed top-2048): close to the replicated cache past index_topk, and a 3-row verify
    window still has 3 serial steps' bits."""
    from tensorfold.families.glm_moe_dsa.cuda import fused

    toks = _tokens(2110, seed=23)

    def run(rank, comm):
        out = {}
        for dcp in (1, 4):
            w, st = _fused_dcp(rank, comm, 4096, dcp)
            big = fused.Buffers(w, 128, 4096)
            small = fused.Buffers(w, 4, 4096)
            for a in range(0, 2100, 128):
                _prefill(w, st, big, toks, a, min(a + 128, 2100))
            win, _ = _window(w, st, small, toks, 2100, 2103)
            ser = [_window(w, st, small, toks, i, i + 1)[0] for i in range(2100, 2103)]
            out[dcp] = (win, ser)
            del w, st, big, small
            torch.cuda.empty_cache()
        return out

    for out in run_ranks(run, 4):
        win4, ser4 = out[4]
        for i, h in enumerate(ser4):
            assert torch.equal(win4[i:i + 1], h), f"DCP row {i}: window != serial"
        rel = float((win4.float() - out[1][0].float()).norm() / out[1][0].float().norm())
        assert rel < 2e-2, rel


@pytest.mark.skipif(not DFLASH, reason="set TF_GLM53_DFLASH_TEST to a GLM-5.3 DFlash2 checkpoint")
def test_capture_writes_trainer_sequences(tmp_path, monkeypatch):
    """TF_GLM53_CAPTURE_DIR: rank 0 writes dflash2_train.py's sequence layout - one row per committed position, ids =
    prompt + reply, fp8 taps that round-trip to the live ones (the drafter's ring buffer runs underneath)."""
    from safetensors import safe_open

    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.dflash import GlmDrafter
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    monkeypatch.setenv("TF_GLM53_CAPTURE_DIR", str(tmp_path))
    prompt = _tokens(150, seed=29).tolist()
    cfg_d = json.loads((Path(DFLASH) / "config.json").read_text())

    def run(rank, comm):
        cfg, r, (embed, norm, head), layers = _parts(rank, 4)
        w = fused.Weights(cfg, rank, 4, comm, embed, norm, head, layers, None)
        w.tap_slot = {int(i): s for s, i in enumerate(cfg_d["dflash_config"]["target_layer_ids"])}
        run_ = Runner(w, 512, 0, graphs=False)
        run_.drafter = GlmDrafter(DFLASH, w, capacity=512)
        st = run_.generate(prompt, 80, None, lambda t: False, lambda new: None, 0, "dflash")
        fn = run_.cap.finish(prompt + st["out"], {"temp": 0}) if run_.cap is not None else None
        return fn, st["out"]

    res = run_ranks(run, 4)
    fn, out = res[0]
    assert fn is not None and all(r[0] is None for r in res[1:]), "rank 0 alone writes"
    f = safe_open(fn, framework="pt")
    ids = f.get_tensor("ids").tolist()
    T = len(ids)
    assert T == len(prompt) + len(out) - 1, (T, len(prompt), len(out))      # the last token has no taps yet
    assert ids == (prompt + out)[:T]
    assert f.get_tensor("aux_fp8").shape == (T, 6 * 6144) and f.get_tensor("aux_scale").shape == (T, 6)
    assert (f.get_tensor("topk_ids") == -1).all()                            # greedy: no sampled rows
