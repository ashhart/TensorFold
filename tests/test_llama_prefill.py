"""Stock Llama prefill must ignore other models' process-wide quantized routing."""

from __future__ import annotations

import json

import pytest
from test_llama_family import TINY, mx, nn, same, tiny, tokens


@pytest.mark.parametrize("installed", ["before_load", "after_load"])
@pytest.mark.parametrize("quantized", [False, True])
def test_prefill_stays_native_across_global_quantized_routing(tmp_path, monkeypatch, installed, quantized):
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from tensorfold import families
    from tensorfold.families.llama import load
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    original = tiny(quantized=quantized)
    config = dict(TINY)
    if quantized:
        config["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine",
                                  "lm_head": False, "model.embed_tokens": False}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(original.parameters())))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f"token{i}": i for i in range(64)}, unk_token="token0")), eos_token="token1")
    tokenizer.save_pretrained(tmp_path)
    ids = mx.array([tokens(23)], dtype=mx.uint32)
    native_cache = original.make_cache()
    native_hidden = original.model(ids, native_cache)
    native_logits = original.lm_head(native_hidden)
    mx.eval(native_hidden, native_logits)

    # install() changes these globals; restore their exact previous values, not uninstall()'s defaults.
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", nn.QuantizedLinear.__call__)
    for name in ("_ORIG", "enabled", "max_rows"):
        monkeypatch.setattr(lane_qmm, name, getattr(lane_qmm, name))
    routed = []

    def alternate_matmul(x, weight, sbt, **kwargs):
        # A distinct, executable route on pre-M5 Metal; installing the real wrapper requires no tensor units.
        routed.append(int(x.size // x.shape[-1]))
        y = mx.quantized_matmul(x, weight, scales=sbt[:, :, 0].T, biases=sbt[:, :, 1].T,
                                transpose=True, group_size=kwargs["group"], bits=8)
        return y + mx.array(0.5, dtype=y.dtype)

    monkeypatch.setattr(lane_qmm, "lane_matmul", alternate_matmul)
    if installed == "before_load":
        lane_qmm.install()
    model, _ = load(tmp_path)
    descriptor = families.families()["llama"]
    key = families.kernel_version(descriptor, model)
    if installed == "after_load":
        before = model.prefill(ids, model.make_cache())
        mx.eval(before)
        assert same(before, native_hidden)
        lane_qmm.install()
    assert nn.QuantizedLinear.__call__ is lane_qmm._call
    routed.clear()
    cache = model.make_cache()
    hidden = model.prefill(ids, cache)
    logits = model.inner(ids, model.make_cache())
    mx.eval(hidden, logits)
    assert same(hidden, native_hidden)
    assert same(logits, native_logits)
    assert model._same_cache(cache, native_cache)
    assert routed == []
    assert families.kernel_version(descriptor, model) == key
    if quantized:
        assert isinstance(model.core.layers[0].self_attn.q_proj, nn.QuantizedLinear)
        # Isolation applies only to this family. Other models still use the installed wrapper.
        assert not same(original.model(ids, original.make_cache()), native_hidden)
        assert routed
    else:
        assert model.backend is None
        assert same(original.model(ids, original.make_cache()), native_hidden)
        assert routed == []
