"""Prompt upmix matches the released bf16-dequantized MMA path, including intermediate projection bytes."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA kernels require an NVIDIA GPU", allow_module_level=True)
if torch.cuda.get_device_capability() != (12, 1):
    pytest.skip("the experimental prompt layout targets GB10", allow_module_level=True)
pytest.importorskip("triton.experimental.gluon")

from test_flashnext_kernels import _mlx_weights
from tensorfold.families.qwen4_exp.cuda import glue, hc_upmix, qmm


@pytest.mark.parametrize("tile", [(16, 32, 4), (32, 32, 4), (32, 64, 4), (32, 64, 8),
                                  (64, 64, 8), (32, 128, 8), (64, 128, 8)])
@pytest.mark.parametrize("rows,dims", [(1, 512), (17, 2560), (128, 2560), (513, 2560)])
def test_prompt_upmix_matches_projection_mix_and_chunk_bytes(rows, dims, tile):
    streams, low = 4, 320
    raw = _mlx_weights(streams * dims, low, 419)
    tiled, frag = qmm.make_q4(*raw, "tiled"), qmm.make_q4(*raw, "frag")
    generator = torch.Generator(device="cuda").manual_seed(51 + rows)
    act = torch.randn((rows, low), generator=generator, device="cuda").to(torch.bfloat16)
    norm = torch.randn((rows, streams * dims), generator=generator, device="cuda").to(torch.bfloat16)
    reference_up = qmm.prefill_matmul(act, frag)
    reference = torch.empty((rows, dims), dtype=torch.bfloat16, device="cuda")
    reference_xs = torch.empty((rows, dims // 32), dtype=torch.float32, device="cuda")
    glue.hc_mix(reference_up, norm, reference, reference_xs, streams)
    up, mixed, xs = torch.empty_like(reference_up), torch.empty_like(reference), torch.empty_like(reference_xs)
    hc_upmix.prefill_upmix(act, tiled, norm, mixed, xs, streams, up=up, tile=tile)
    for old, new in zip((reference_up, reference, reference_xs), (up, mixed, xs)):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
    cuts = sorted({0, min(rows, 1), min(rows, 7), min(rows, 33), rows})
    for a, b in zip(cuts, cuts[1:]):
        hc_upmix.prefill_upmix(act[a:b], tiled, norm[a:b], mixed[a:b], xs[a:b], streams, tile=tile)
    assert torch.equal(reference.view(torch.uint8), mixed.view(torch.uint8))
    assert torch.equal(reference_xs.view(torch.uint8), xs.view(torch.uint8))


@pytest.mark.parametrize("unequal", [False, True])
def test_upmix_first_use_checks_both_kernel_variants_and_caches_fallback(monkeypatch, capsys, unequal):
    from types import SimpleNamespace
    from tensorfold.families.qwen4_exp.cuda import forward, hc_check

    device = torch.device("cuda", torch.cuda.current_device())
    hc_check._checked.clear()
    original, calls = hc_upmix.prefill_upmix, []
    def candidate(*args, **kwargs):
        calls.append(True)
        original(*args, **kwargs)
        if unequal:
            args[3].view(torch.int16)[0, 0].bitwise_xor_(1)
    monkeypatch.setattr(hc_upmix, "prefill_upmix", candidate)
    rng = torch.cuda.get_rng_state(device)
    try:
        got = hc_check.fuser(device, upmix=True)
        assert got is (None if unequal else candidate)
        assert len(calls) == (1 if unequal else 4)
        assert torch.equal(rng, torch.cuda.get_rng_state(device))
        assert hc_check.fuser(device, upmix=True) is got
        assert len(capsys.readouterr().out.splitlines()) == 1
        if unequal:
            monkeypatch.setattr(hc_upmix, "prefill_upmix", lambda *a, **k: pytest.fail("disabled upmix ran"))
            rows = 512
            normed = torch.ones((rows, 10240), dtype=torch.bfloat16, device=device)
            b = SimpleNamespace(prefill=True, normed=normed, act=normed[:, :320].contiguous(),
                                xs_act=torch.empty((rows, 10), device=device), up=torch.empty_like(normed),
                                mixed=torch.empty((rows, 2560), dtype=torch.bfloat16, device=device),
                                xs_mixed=torch.empty((rows, 80), device=device))
            up = object.__new__(qmm.Q4)
            up.n = 10240
            hc = SimpleNamespace(up=up, prefill_up=up)
            used = []
            monkeypatch.setattr(forward, "_down_act", lambda *a: None)
            monkeypatch.setattr(forward, "_mm", lambda x, q, xs, out, b: (used.append(True), out.zero_())[1])
            forward._readout_plain(hc, b, normed, rows, 1e-6, 4, 320, None, normed=True)
            assert used == [True] and torch.equal(b.mixed, torch.full_like(b.mixed, 0.5))
            assert torch.equal(b.xs_mixed, torch.full_like(b.xs_mixed, 16))
            assert not capsys.readouterr().out
    finally:
        hc_check._checked.clear()
