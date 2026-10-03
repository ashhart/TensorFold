"""A prompt pass of several chunks on the GPU gives every chunk the rows and cache states of its own one-chunk
forward (Flash Next, GLM-5.3, Nemotron; on M1-M4 the routed experts share one aligned gather)."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")
sys.path.insert(0, str(Path(__file__).parent))

SIZES = [(96, 96, 96), (128, 17, 70, 200), (64, 300)]


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def _states(cache):
    out = []
    for c in cache:
        state = getattr(c, "state", None)
        if isinstance(state, (list, tuple)):
            out += [a for a in state if isinstance(a, mx.array)]
        else:
            for name in sorted(vars(c)):
                value = getattr(c, name)
                if isinstance(value, mx.array):
                    out.append(value)
                elif isinstance(value, (list, tuple)):
                    out += [v for v in value if isinstance(v, mx.array)]
    return out


def _check(one_chunk, one_pass, make_cache, sizes):
    solo_cache, pass_cache = make_cache(), make_cache()
    rows, at = [], 0
    for n in sizes:
        rows.append(one_chunk(at, n, solo_cache))
        at += n
    solo = mx.concatenate(rows, axis=-2)
    both = one_pass(pass_cache)
    mx.eval(solo, both, *_states(solo_cache), *_states(pass_cache))
    assert _same(solo, both)
    a, b = _states(solo_cache), _states(pass_cache)
    assert len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("sizes", SIZES)
def test_flash_next_pass(sizes):
    from test_qwen4_exp_prefill import tiny_model

    model = tiny_model(seed=4)
    ids = np.random.default_rng(6).integers(6, 97, size=(1, sum(sizes)))
    _check(lambda a, n, cache: model.hidden(ids[:, a:a + n], cache)[0],
           lambda cache: model.hidden_pass(ids, cache, sizes)[0], model.make_cache, sizes)


@pytest.mark.parametrize("sizes", SIZES)
def test_glm_pass(tmp_path, sizes):
    from glm5_fakes import TEXT, write_checkpoint

    from tensorfold.families.glm5_next import weights

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        path = write_checkpoint(tmp_path / "glm5")
    finally:
        mx.set_default_device(previous)
    model = weights.load_backbone(path)
    ids = [int(t) for t in np.random.default_rng(9).integers(6, TEXT["vocab_size"], size=sum(sizes))]
    _check(lambda a, n, cache: model.hidden(mx.array([ids[a:a + n]], dtype=mx.uint32), cache)[0],
           lambda cache: model.hidden_pass(mx.array([ids], dtype=mx.uint32), cache, sizes)[0], model.make_cache,
           sizes)


@pytest.mark.parametrize("sizes", SIZES)
def test_nemotron_pass(sizes):
    from test_nemotron_pass import _tiny

    from tensorfold.families.nemotron_h import prompt_pass

    model = _tiny()
    backbone = model.backbone
    ids = mx.random.randint(0, 512, (1, sum(sizes)), key=mx.random.key(5))
    _check(lambda a, n, cache: backbone(ids[:, a:a + n], cache=cache)[0],
           lambda cache: prompt_pass.hidden(backbone, ids, cache, tuple(sizes))[0], model.make_cache, sizes)
