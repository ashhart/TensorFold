"""Qwen3.6 MoE on unquantized bf16 weights: the row decoder reads plain linears, windows keep one-row bits."""

import json
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_forward, row_matmul  # noqa: E402


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _config(tmp_path, quantization=None):
    config = {"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text"}}
    if quantization is not None:
        config["quantization"] = quantization
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture(scope="module")
def tiny():
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    # Qwen3.6-35B-A3B's layer pattern and expert routing on a small residual, left unquantized (bf16)
    args = TextModelArgs(model_type="qwen3_5_moe_text", hidden_size=1024, intermediate_size=512, num_hidden_layers=4,
                         num_attention_heads=16, num_key_value_heads=2, head_dim=256, rms_norm_eps=1e-6,
                         vocab_size=512, linear_num_value_heads=32, linear_num_key_heads=16,
                         linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                         full_attention_interval=4, tie_word_embeddings=False, max_position_embeddings=4096,
                         num_experts=16, num_experts_per_tok=8, shared_expert_intermediate_size=512,
                         moe_intermediate_size=512, norm_topk_prob=True)
    mx.random.seed(21)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    exact_attention.install()
    from tensorfold.families.qwen3_5 import install_row_decoder

    assert install_row_decoder(model) and row_matmul.plain_fits(model)
    assert row_matmul.BACKEND.name == "plain"
    return model


def _run(model, tokens, cache, start):
    parents = [-1] + list(range(len(tokens) - 1))
    logits, record = row_forward.forward(model.model, model.lm_head, tokens, parents, cache, start, pipeline_layers=2)
    row_forward.commit(cache, record, list(range(len(tokens))), len(tokens), start)
    mx.eval(logits, *[a for c in cache for a in c.state if a is not None])
    return logits


@pytest.mark.parametrize("mode", ["batched", "rows"])
def test_bf16_windows_reproduce_one_row_steps(tiny, mode, monkeypatch):
    monkeypatch.setattr(row_forward, "MOE_ROWS", mode)
    mx.random.seed(5)
    prompt = [int(t) for t in mx.random.randint(0, 512, (21,)).tolist()]
    tokens = [int(t) for t in mx.random.randint(0, 512, (row_matmul.WINDOW_ROWS,)).tolist()]
    base = LaneEngine.copy_single_cache(tiny.make_cache())
    for begin in range(0, len(prompt), row_matmul.WINDOW_ROWS):
        _run(tiny, prompt[begin:begin + row_matmul.WINDOW_ROWS], base, begin)
    start = len(prompt)
    serial_cache = LaneEngine.copy_single_cache(base)
    serial = [_run(tiny, [t], serial_cache, start + i)[0, -1] for i, t in enumerate(tokens)]
    for width in range(2, len(tokens) + 1):
        window = _run(tiny, tokens[:width], LaneEngine.copy_single_cache(base), start)
        for i in range(width):
            assert _same(window[0, i], serial[i]), f"row {i} of a {width}-row window differs from its one-row step"


def test_bf16_streams_in_one_forward_equal_each_alone(tiny):
    ok, failures = row_forward.check_streams(tiny.model, tiny.lm_head, tiny.make_cache, LaneEngine.copy_single_cache,
                                             mixes=[(1, 1), (1, 4), (8, 8), (3, 1, 8, 5), (16, 2)])
    assert ok, failures


def test_plain_projections_reproduce_their_modules_bits(tiny):
    mx.random.seed(9)
    attn = tiny.model.layers[3].self_attn                      # layer 3 is this pattern's attention layer
    for name, module in attn.named_modules():
        if isinstance(module, nn.Linear):
            x = mx.random.normal(shape=(1, 3, int(module["weight"].shape[1]))).astype(mx.bfloat16)
            assert _same(row_matmul.project(module, x), module(x)), name
    shared = tiny.model.layers[0].mlp.shared_expert
    x = mx.random.normal(shape=(1, 2, 1024)).astype(mx.bfloat16)
    assert _same(row_matmul.project(shared.gate_proj, x), shared.gate_proj(x))
    assert _same(row_matmul.logits(tiny.lm_head, x), tiny.lm_head(x))


def test_a_partly_quantized_model_is_refused_not_read_as_plain(tiny):
    from mlx_lm.models.qwen3_5 import TextModel

    from tensorfold.families.qwen3_5 import install_row_decoder

    cloned = TextModel(tiny.args)
    cloned.update(tiny.parameters())
    nn.quantize(cloned.model.layers[3].self_attn, group_size=64, bits=4)
    assert install_row_decoder(cloned) is False


def test_check_admits_unquantized_on_a_mac_and_refuses_it_on_linux(tmp_path, monkeypatch):
    from tensorfold import families
    from tensorfold.families import qwen3_5_moe

    assert families.detect(_config(tmp_path)).module == "tensorfold.families.qwen3_5_moe"
    monkeypatch.setattr(sys, "platform", "darwin")
    qwen3_5_moe.check(_config(tmp_path))                              # no quantization block: bf16 weights
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError, match="groups of 64"):
        qwen3_5_moe.check(_config(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="groups of 64"):
        qwen3_5_moe.check(_config(tmp_path, quantization={"bits": 8, "group_size": 64, "mode": "affine"}))


def test_the_mac_lane_loads_bf16_through_the_row_decoder(monkeypatch):
    """load() installs the row decoder for bf16 weights as it does for MLX 4-bit ones."""

    from tensorfold.families import qwen3_5, qwen3_5_moe

    calls = {"rows": []}
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path: (object(), "tok"))
    monkeypatch.setattr(qwen3_5, "lane_family",
                        lambda model, **kw: calls["rows"].append(model) or model)
    family, tokenizer = qwen3_5_moe.load(Path("unused"))
    assert calls["rows"] == [family] and tokenizer == "tok"
