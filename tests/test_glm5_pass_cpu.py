"""GLM-5.3's prompt pass on the CPU (tiny checkpoint): every chunk keeps its own one-chunk forward's rows and states."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).parent))

from glm5_fakes import TEXT, write_checkpoint  # noqa: E402

from tensorfold.families.glm5_next import weights  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return write_checkpoint(tmp_path_factory.mktemp("glm5pass"))
    finally:
        mx.set_default_device(previous)


def _tokens(n, seed):
    return [int(t) for t in np.random.default_rng(seed).integers(6, TEXT["vocab_size"], size=n)]


def _cache_arrays(cache):
    out = []
    for c in cache:
        for name in sorted(vars(c)):
            value = getattr(c, name)
            if isinstance(value, mx.array):
                out.append((name, value))
            elif isinstance(value, (list, tuple)):
                out += [(f"{name}{i}", v) for i, v in enumerate(value) if isinstance(v, mx.array)]
    return out


@pytest.mark.parametrize("sizes", [(40, 40, 40), (64, 17, 90, 33), (128, 128)])
def test_a_pass_gives_each_chunk_its_own_forwards_bits(checkpoint, sizes):
    """Several prompt chunks in one pass: the same rows and cache states as the chunks one forward at a time."""

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = weights.load_backbone(checkpoint)
        ids = _tokens(sum(sizes), seed=9)
        one = model.make_cache()
        rows, at = [], 0
        for n in sizes:
            rows.append(model.hidden(mx.array([ids[at:at + n]], dtype=mx.uint32), one)[0])
            at += n
        solo = mx.concatenate(rows)
        passed = model.make_cache()
        both = model.hidden_pass(mx.array([ids], dtype=mx.uint32), passed, sizes)[0]
        mx.eval(solo, both)
        assert bool(mx.array_equal(solo, both).item())
        a, b = _cache_arrays(one), _cache_arrays(passed)
        assert [n for n, _ in a] == [n for n, _ in b]
        assert all(bool(mx.array_equal(x, y).item()) for (_, x), (_, y) in zip(a, b))
    finally:
        mx.set_default_device(previous)
