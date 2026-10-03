"""Fused prompt HC steps match release kernel bytes and each row's solo arithmetic."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA kernels require an NVIDIA GPU", allow_module_level=True)
if torch.cuda.get_device_capability() != (12, 1):
    pytest.skip("the guarded HC layout targets GB10", allow_module_level=True)
pytest.importorskip("triton.experimental.gluon")

from tensorfold.families.qwen4_exp.cuda import glue, hc_fused


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("rows,dims", [(1, 512), (17, 2560), (128, 2560), (513, 2560)])
def test_write_norm_matches_release_and_prompt_chunks(mode, rows, dims):
    streams, slots, world = 4, 9, 3
    gen = torch.Generator(device="cuda").manual_seed(903 + rows + mode)
    def random(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", generator=gen).to(dtype)
    source = random((rows, streams * dims))
    scale = random((streams * dims,))
    inject = random((rows, streams))
    branch = random((world, rows, dims), torch.float32) if mode in (3, 4) else random((rows, dims))
    y = random((rows, slots, dims))
    wts = random((rows, slots), torch.float32)
    def buffers():
        return (source.clone(), torch.empty((rows, dims // 256, streams), device="cuda"),
                torch.empty_like(source), torch.empty((rows, streams * dims // 32), device="cuda"))
    old, new, chunks = buffers(), buffers(), buffers()
    h, pss, normed, xs = old
    glue.hc_writeback(h, h, pss, streams, mode, branch=branch, inject=inject, y=y, wts=wts)
    glue.hc_normed(h, pss, scale, normed, xs, streams, 1e-6)
    h, pss, normed, xs = new
    hc_fused.write_norm(h, pss, scale, normed, xs, streams, 1e-6, mode, branch, inject, y, wts)
    cuts = sorted({0, min(1, rows), min(7, rows), min(17, rows), rows})
    h, pss, normed, xs = chunks
    for a, b in zip(cuts, cuts[1:]):
        br = branch[:, a:b].contiguous() if mode in (3, 4) else branch[a:b]
        hc_fused.write_norm(h[a:b], pss[a:b], scale, normed[a:b], xs[a:b], streams, 1e-6, mode,
                            br, inject[a:b], y[a:b], wts[a:b])
    for index, (reference, fused, chunked) in enumerate(zip(old, new, chunks)):
        assert torch.equal(reference.view(torch.uint8), fused.view(torch.uint8)), (mode, rows, index)
        assert torch.equal(fused.view(torch.uint8), chunked.view(torch.uint8)), (mode, rows, index)


@pytest.mark.parametrize("unequal", [False, True])
def test_first_use_checks_real_bytes_and_falls_back_on_a_forced_difference(monkeypatch, capsys, unequal):
    from types import SimpleNamespace
    from tensorfold.families.qwen4_exp.cuda import forward, hc_check, qmm

    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("the guarded fusion targets GB10")
    device = torch.device("cuda", torch.cuda.current_device())
    hc_check._checked.clear()
    original, calls = hc_fused.write_norm, []
    def candidate(*args, **kwargs):
        calls.append(True)
        original(*args, **kwargs)
        if unequal:
            args[3].view(torch.int16)[0, 0].bitwise_xor_(1)
    monkeypatch.setattr(hc_fused, "write_norm", candidate)
    state = torch.cuda.get_rng_state(device)
    try:
        got = hc_check.fuser(device)
        assert got is (None if unequal else candidate)
        assert len(calls) == (1 if unequal else 5)
        assert torch.equal(state, torch.cuda.get_rng_state(device))
        assert hc_check.fuser(torch.device("cuda")) is got
        assert len(capsys.readouterr().out.splitlines()) == 1
        if unequal:
            monkeypatch.setattr(hc_fused, "write_norm", lambda *a, **k: pytest.fail("disabled fusion ran"))
            h = torch.ones((17, 4 * 2560), dtype=torch.bfloat16, device=device)
            b = SimpleNamespace(prefill=True, pss=torch.empty((17, 10, 4), device=device))
            hc = SimpleNamespace(down=object.__new__(qmm.Q4), inject=False)
            readouts = []
            monkeypatch.setattr(forward, "_readout", lambda *a, **k: readouts.append(True))
            forward.hc_block(hc, b, 17, 1e-6, 4, 320, 0, None, torch.empty(17, 4, device=device), h)
            assert readouts == [True] and torch.equal(b.pss, torch.full_like(b.pss, 256))
            assert not capsys.readouterr().out
    finally:
        hc_check._checked.clear()
