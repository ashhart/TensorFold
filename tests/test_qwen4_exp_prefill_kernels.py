"""Flash Next's prompt attention through the decode's selection and attention kernels (GPU): close to the
reference forward's, and a prompt resumed from a grid checkpoint still equals the same prompt fed fresh."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from test_qwen4_exp_family import TEXT, _flat  # noqa: E402

from tensorfold.families.qwen4_exp import model as q4  # noqa: E402


def gpu_tiny(seed: int = 0) -> q4.Qwen4Exp:
    """The family tests' tiny config with DeltaNet heads of 32 dims (MLX's recurrence kernel needs 32), in bf16
    (the kernels read bf16 queries, keys and values)."""

    mx.random.seed(seed)
    text = dict(TEXT, linear_key_head_dim=32, linear_value_head_dim=32)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": text}))
    params = []
    for name, value in _flat(model.parameters()):
        if name.endswith("norm.weight") or "layernorm" in name or "hc_norm" in name or name.endswith("norm_key.weight"):
            value = 0.1 * mx.random.normal(value.shape)
        params.append((name, value))
    model.load_weights(params)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _copy(cache: q4.AttentionCache) -> q4.AttentionCache:
    out = q4.AttentionCache()
    if cache.keys is not None:
        out.keys, out.values, out.index_keys = mx.array(cache.keys), mx.array(cache.values), mx.array(cache.index_keys)
        out.pooled = None if cache.pooled is None else mx.array(cache.pooled)
        out.offset = cache.offset
    return out


def test_selected_keys_match_the_reference_attention(monkeypatch):
    """One attention layer, chunk after chunk of the same inputs: the kernel path and the reference give each row
    the same keys and outputs within rounding (dense rows, rows past the budget, and chunks mixing both)."""

    attn = next(layer.self_attn for layer in gpu_tiny().layers if not layer.is_linear)
    monkeypatch.setattr(q4.prefill, "DENSE_KEYS", 0)       # the kernels from the first chunk past the budget
    rng = np.random.default_rng(13)
    cache = q4.AttentionCache()
    for chunk in range(6):
        x = mx.array(rng.normal(size=(1, 8, 64)).astype(np.float32)).astype(mx.bfloat16)
        attn.__dict__.pop("kernel_select", None)
        reference = attn(x, _copy(cache)).astype(mx.float32)
        attn.__dict__["kernel_select"] = True
        kernels = attn(x, _copy(cache)).astype(mx.float32)
        scale = float(mx.abs(reference).max())
        assert float(mx.abs(kernels - reference).max()) <= 1e-2 * scale, chunk
        attn.__dict__.pop("kernel_select")
        attn(x, cache)
        mx.eval(*cache.state)


@pytest.mark.parametrize("dense_keys", [0, 20])
def test_resumed_prompt_equals_fresh_through_the_kernels(monkeypatch, dense_keys):
    """Resumed equals fresh with the dense-or-kernels switch inside the prompt (DENSE_KEYS 20) and without."""

    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    monkeypatch.setattr(q4.prefill, "DENSE_KEYS", dense_keys)
    model = gpu_tiny(1)
    q4.select_by_kernels(model.layers)
    flash = FlashNext(model, None, drafts=0)
    prompt = [int(t) for t in np.random.default_rng(12).integers(6, 97, size=44)]

    def run(ids, **kw):
        engine = LaneEngine(flash)
        engine.prefill_align = 8
        stream = LaneStream(stream_id="x", prompt_ids=list(ids), max_new_tokens=6)
        engine.add_stream(stream, **kw)
        while engine.active_count:
            engine.step()
        return stream

    first = run(prompt[:36], checkpoints_at=(34,))
    tokens, cache = first.history_checkpoints[0]
    assert tokens == prompt[:32]
    fresh = run(prompt)
    resumed = run(prompt, cache=LaneEngine.copy_single_cache(cache), cached_tokens=32)
    assert resumed.emitted == fresh.emitted

