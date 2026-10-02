"""A checkpoint stored in float16 (oQ packs: fp16 scales, biases and norms) runs on the row decoder once its float16
arrays are bfloat16, as a bf16 checkpoint does: prompts through mlx_lm, then windows that equal one-row steps."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten  # noqa: E402

from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.families.qwen3_5 import bf16_floats  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_forward, row_matmul  # noqa: E402

PROMPT, WINDOW = [3, 8, 11, 4, 6, 1, 12, 10], [5, 9, 13, 2, 7]


def _fp16_model():
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    args = TextModelArgs(model_type="qwen3_5_text", hidden_size=1024, intermediate_size=17408, num_hidden_layers=4,
                         num_attention_heads=24, num_key_value_heads=4, head_dim=256, rms_norm_eps=1e-6,
                         vocab_size=512, linear_num_value_heads=48, linear_num_key_heads=16,
                         linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                         full_attention_interval=4, tie_word_embeddings=False, max_position_embeddings=4096)
    mx.random.seed(21)
    model = TextModel(args)
    model.set_dtype(mx.float16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


def _float_dtypes(model):
    return {v.dtype for _, v in tree_flatten(model.parameters()) if v.dtype != mx.uint32}


@pytest.fixture(scope="module")
def model():
    exact_attention.install()
    model = _fp16_model()
    assert mx.float16 in _float_dtypes(model)
    converted = bf16_floats(model)
    assert converted > 0 and mx.float16 not in _float_dtypes(model)
    row_matmul.install(model)
    return model


def _prefilled(model):
    cache = model.make_cache()
    mx.eval(model.model(mx.array([PROMPT]), cache=cache))
    return cache


def test_prompts_keep_bf16_rows_and_caches(model):
    cache = _prefilled(model)
    kv = next(c for c in cache if hasattr(c, "keys"))
    assert kv.keys.dtype == mx.bfloat16 and kv.values.dtype == mx.bfloat16


def test_a_window_equals_its_one_row_steps(model):
    base = _prefilled(model)
    chain = list(range(-1, len(WINDOW) - 1))
    window, _ = row_forward.forward(model.model, model.lm_head, WINDOW, chain, LaneEngine.copy_single_cache(base),
                                    len(PROMPT))
    step_cache, steps = LaneEngine.copy_single_cache(base), []
    for i, token in enumerate(WINDOW):
        logits, record = row_forward.forward(model.model, model.lm_head, [token], [-1], step_cache, len(PROMPT) + i)
        row_forward.commit(step_cache, record, [0], 1, len(PROMPT) + i)
        steps.append(logits[0, 0])
    mx.eval(window, *steps)
    assert window.dtype == mx.bfloat16
    assert all(bool(mx.array_equal(window[0, i], steps[i]).item()) for i in range(len(WINDOW)))


def test_a_bf16_checkpoint_is_left_as_it_is():
    from mlx_lm.models.qwen3_5 import TextModel

    model = _fp16_model()
    model.set_dtype(mx.bfloat16)
    before = {k: v for k, v in tree_flatten(model.parameters())}
    assert bf16_floats(model) == 0
    assert all(v is before[k] for k, v in tree_flatten(model.parameters()))
    assert isinstance(model, TextModel)
