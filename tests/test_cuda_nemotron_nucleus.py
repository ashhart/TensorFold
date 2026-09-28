"""Nemotron's CUDA sampler draws top_k off (nucleus only) over the whole vocabulary with the keyed rule."""

import importlib
import math
import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling


@pytest.fixture
def sampler(monkeypatch):
    lang = ModuleType("triton.language")
    lang.constexpr = object
    triton = ModuleType("triton")
    triton.language = lang
    triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", lang)
    monkeypatch.delitem(sys.modules, "tensorfold.families.nemotron_h.cuda.sampler", raising=False)
    yield importlib.import_module("tensorfold.families.nemotron_h.cuda.sampler")
    sys.modules.pop("tensorfold.families.nemotron_h.cuda.sampler", None)


def reference(row, pos, seed, temp, top_p, mod):
    """The keyed rule in Python integers: rank by (value desc, id asc), cut at top_p, Gumbel-max by the hash."""
    mask = (1 << 64) - 1

    def mix(x):
        x ^= x >> 30
        x = (x * mod.M1) & mask
        x ^= x >> 27
        x = (x * mod.M2) & mask
        return x ^ (x >> 31)

    scaled = [v / temp for v in row]
    order = sorted(range(len(row)), key=lambda i: (-scaled[i], i))
    top = scaled[order[0]]
    p = [math.exp(scaled[i] - top) for i in order]
    total, run, limit = sum(p), 0.0, 0
    for q in p:
        run += q / total
        limit += run < top_p
    limit += 1
    best = None
    for rank, i in enumerate(order[:limit]):
        x = mix((seed + mod.C1) & mask)
        x = mix(x ^ ((pos * mod.C2) & mask))
        x = mix(x ^ i)
        u = (x >> 11) * 2.0 ** -53 + 2.0 ** -54
        score = scaled[i] - math.log(-math.log(u))
        if best is None or score > best[0]:
            best = (score, i)
    return best[1]


def test_top_k_off_samples_the_whole_vocabulary_nucleus(sampler):
    torch.manual_seed(0)
    logits = (torch.randn(3, 700) * 2).to(torch.bfloat16)          # more ids than the kernel's 256 candidates
    params = sampler.Params("cpu")
    params.set(Sampling(1234, 0.8, 0, 0.9))
    meta = torch.tensor([17, 0, 0, 0], dtype=torch.int32)
    out = torch.zeros(3, dtype=torch.int32)
    sampler.keyed(logits, meta, params, out)
    rows = logits.double().tolist()
    want = [reference(rows[r], 17 + r + 1, 1234, 0.8, 0.9, sampler) for r in range(3)]
    assert out.tolist() == want
