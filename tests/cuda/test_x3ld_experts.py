"""x3ld (TF_EXPERT_LOADS) routed experts give bit-identical results to upstream's grouped kernel (GPU)."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tensorfold.cuda.exl3 import experts as ex3  # noqa: E402
from tensorfold.cuda.exl3 import x3ld  # noqa: E402


def layer(E, D, I, k2s, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def trellis(k, n, k2):
        return torch.randint(-32768, 32767, (k // 16, n // 16, 8 * k2), generator=g, device="cuda",
                             dtype=torch.int32).to(torch.int16)

    def sign(n):
        return (torch.randint(0, 2, (n,), generator=g, device="cuda") * 2 - 1).half()

    def scale(n):
        return (torch.rand((n,), generator=g, device="cuda") * 0.02 + 0.005).half()

    gate = [(trellis(D, I, k2s[e]), sign(D), scale(I)) for e in range(E)]
    up = [(trellis(D, I, k2s[e]), sign(D), scale(I)) for e in range(E)]
    down = [(trellis(I, D, k2s[e]), sign(I), scale(D)) for e in range(E)]
    return ex3.prepare(gate, up, down, ex3.CB_MUL1)


@pytest.mark.parametrize("k2s,I", [("mixed", 1024), ("mixed", 2048), ("uniform8", 1024)])
@pytest.mark.parametrize("cfg", [(8, 1), (8, 2), (4, 2)])
def test_x3ld_equals_grouped(k2s, I, cfg, monkeypatch):
    E, D, slots = 24, 4096, 6
    widths = [4 + 2 * (e % 4) for e in range(E)] if k2s == "mixed" else [8] * E
    ex = layer(E, D, I, widths, I + len(k2s))
    g = torch.Generator(device="cuda").manual_seed(5)
    taken = []
    real = x3ld.grouped
    monkeypatch.setattr(x3ld, "grouped", lambda *a, **k: taken.append(real(*a, **k)) or taken[-1])
    for R in (1, 2, 5, 16, 32):
        x = torch.randn((R, D), generator=g, device="cuda").to(torch.bfloat16)
        pick = torch.stack([torch.randperm(E, generator=g, device="cuda")[:slots] for _ in range(R)]).int()
        wts = torch.rand((R, slots), generator=g, device="cuda")
        outs = []
        for on in (False, True):
            monkeypatch.setitem(x3ld.CFG, "on", on)
            monkeypatch.setitem(x3ld.CFG, "gu", cfg)
            monkeypatch.setitem(x3ld.CFG, "dn", cfg)
            s = ex3.Scratch(ex, 32, slots)
            outs.append(ex3.routed(x, pick, wts, ex, s, None, R).clone())
        assert torch.equal(outs[0], outs[1]), (R, k2s, I, cfg)
        assert taken[-2:] == [True, True]                # both launches of the "on" run went through x3ld
        assert torch.isfinite(outs[0]).all()
