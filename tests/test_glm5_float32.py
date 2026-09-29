"""GLM-5.3-Flash with float32 activations (config tensorfold_activation_dtype): both paths, exact windows, drafts."""

from __future__ import annotations

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import write_checkpoint  # noqa: E402
from test_glm5_next_family import _run_engine, tokens  # noqa: E402
from tensorfold.families.glm5_next import config as C  # noqa: E402
from tensorfold.families.glm5_next import mtp as glm_mtp  # noqa: E402
from tensorfold.families.glm5_next import weights  # noqa: E402
from tensorfold.families.glm5_next.runtime import GLMFlash  # noqa: E402
from tensorfold.kernels.glm.flash.v1 import sparse_attention  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_and_bf16_after():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)
    weights.set_activation({})                                          # later tests see the default again


@pytest.fixture(scope="module")
def checkpoint32(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        folder = write_checkpoint(tmp_path_factory.mktemp("glm5f32"))
    finally:
        mx.set_default_device(previous)
    config = json.loads((folder / "config.json").read_text())
    config["tensorfold_activation_dtype"] = "float32"
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def test_float32_activations_and_caches(checkpoint32):
    model = weights.load_backbone(checkpoint32)
    assert C.act() == mx.float32 and not sparse_attention.metal()
    cache = model.make_cache()
    hidden = model.hidden(mx.array([tokens(40)]), cache)
    assert hidden.dtype == mx.float32
    assert cache[3].keys.dtype == mx.float32 and cache[3].pool.dtype == mx.float32
    assert model.head(hidden).dtype == mx.float32


def test_the_default_stays_bf16_and_restores_the_kernels(checkpoint32):
    weights.load_backbone(checkpoint32)
    weights.set_activation({})
    assert C.act() == mx.bfloat16
    assert sparse_attention.metal is weights._METAL[sparse_attention.__name__]   # the module's own function again


def test_unknown_activation_dtype_is_refused():
    with pytest.raises(ValueError):
        weights.set_activation({"tensorfold_activation_dtype": "float16"})


@pytest.mark.parametrize("length", [9, 40])
def test_float32_prefill_agrees_with_decode(checkpoint32, length):
    model = weights.load_backbone(checkpoint32)
    ids = tokens(length)
    a = model.head(model.hidden(mx.array([ids]), model.make_cache()))[0, -1]
    step = model.make_cache()
    for t in ids:
        b = model.head(model.hidden(mx.array([[t]]), step))[0, -1]
    a, b = np.array(a), np.array(b)
    assert int(a.argmax()) == int(b.argmax())
    assert np.max(np.abs(a - b)) < 1e-3 * np.max(np.abs(b)) + 1e-3


def test_float32_rows_are_exact_and_drafts_change_speed_only(checkpoint32):
    model = weights.load_backbone(checkpoint32)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = tokens(21, seed=4)
    engine_a, a = _run_engine(runtime, prompt, 24)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 24)
    assert engine_a.drafted > 0 and a.emitted == b.emitted


def test_on_metal_float32_rows_are_exact_and_drafts_change_speed_only(checkpoint32):
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    mx.set_default_device(mx.gpu)
    model = weights.load_backbone(checkpoint32)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = tokens(30, seed=6)
    engine_a, a = _run_engine(runtime, prompt, 20)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 20)
    assert engine_a.drafted > 0 and a.emitted == b.emitted
