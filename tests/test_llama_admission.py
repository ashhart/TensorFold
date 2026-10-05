"""Llama runtime refusal, qualified row limits, and context-safe server admission."""

from __future__ import annotations

import pytest
from test_llama_family import TINY, LaneEngine, family, mx, tiny, tokens


@pytest.mark.parametrize("kind", ["head", "embedding", "half", "fourbit"])
def test_runtime_refuses_unsupported_weight_layout_before_probing(kind):
    stock = tiny()
    if kind == "head":
        stock.lm_head = stock.lm_head.to_quantized(bits=8, group_size=64)
    elif kind == "embedding":
        stock.model.embed_tokens = stock.model.embed_tokens.to_quantized(bits=8, group_size=64)
    elif kind == "half":
        stock.set_dtype(mx.float16)
    else:
        stock.model.layers[0].self_attn.q_proj = stock.model.layers[0].self_attn.q_proj.to_quantized(bits=4, group_size=64)
    with pytest.raises(ValueError):
        family(stock)


@pytest.mark.parametrize("name,part,dtype,shape", [
    ("lm_head", "weight", mx.uint32, None),
    ("model.embed_tokens", "weight", mx.uint32, None),
    ("model.norm", "weight", mx.uint32, None),
    ("model.layers.0.input_layernorm", "weight", mx.uint32, None),
    ("model.layers.0.self_attn.o_proj", "bias", mx.uint32, None),
    ("lm_head", "weight", mx.bfloat16, (63, 128)),
    ("model.embed_tokens", "weight", mx.bfloat16, (64, 127)),
    ("model.norm", "weight", mx.bfloat16, (127,)),
    ("model.layers.0.self_attn.o_proj", "weight", mx.bfloat16, (128, 64)),
    ("model.layers.0.self_attn.o_proj", "bias", mx.bfloat16, (1, 128)),
])
def test_runtime_dense_parameters_are_checked_before_prepare(monkeypatch, name, part, dtype, shape):
    from tensorfold.families.llama.family import LlamaFamily
    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    stock = tiny(quantized=True)
    # Leave one dense decoder projection, but keep other packed projections to exercise the prepare boundary.
    stock.model.layers[0].self_attn.o_proj = tiny().model.layers[0].self_attn.o_proj
    module = dict(stock.named_modules())[name]
    module[part] = mx.zeros(shape or module[part].shape, dtype=dtype)

    def never(*args, **kwargs):
        pytest.fail("invalid weights reached preparation or a model probe")

    monkeypatch.setattr(row_matmul, "simd_qmm_backend", never)
    monkeypatch.setattr(LlamaFamily, "check_windows", never)
    with pytest.raises(ValueError, match="bf16|shape"):
        family(stock)


@pytest.mark.parametrize("part,dtype,shape", [
    ("weight", mx.bfloat16, (128, 32)),
    ("weight", mx.uint32, (128,)),
    ("weight", mx.uint32, (128, 0)),
    ("weight", mx.uint32, (128, 16)),
    ("weight", mx.uint32, (120, 32)),
    ("scales", mx.bfloat16, (128,)),
    ("scales", mx.bfloat16, (128, 1)),
    ("biases", mx.bfloat16, (128, 1)),
    ("biases", mx.uint32, (128, 2)),
    ("bias", mx.bfloat16, (1, 128)),
    ("bias", mx.bfloat16, (127,)),
    ("scales", None, None),
    ("biases", None, None),
    ("weight", None, None),
])
def test_runtime_packed_geometry_is_checked_before_prepare(monkeypatch, part, dtype, shape):
    from tensorfold.families.llama.family import LlamaFamily
    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    stock = tiny(quantized=True)
    projection = stock.model.layers[0].self_attn.q_proj
    if dtype is None:
        projection.pop(part)
    else:
        projection[part] = mx.zeros(shape, dtype=dtype)

    def never(*args, **kwargs):
        pytest.fail("malformed packed buffers reached preparation or a model probe")

    monkeypatch.setattr(row_matmul, "simd_qmm_backend", never)
    monkeypatch.setattr(LlamaFamily, "check_windows", never)
    with pytest.raises(ValueError, match="bf16|shape|U32|packed values"):
        family(stock)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("bias", [None, [0]])
def test_runtime_nonarray_model_bias_is_refused_before_prepare(monkeypatch, quantized, bias):
    from tensorfold.families.llama.family import LlamaFamily
    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    stock = tiny(quantized=quantized)
    stock.model.layers[0].self_attn.q_proj.bias = bias

    def never(*args, **kwargs):
        pytest.fail("invalid model bias reached preparation or a model probe")

    monkeypatch.setattr(row_matmul, "simd_qmm_backend", never)
    monkeypatch.setattr(LlamaFamily, "check_windows", never)
    with pytest.raises(ValueError, match="bias"):
        family(stock)


def test_bad_cache_and_input_are_refused_without_advancing_state():
    model = family(tiny())
    cache = model.make_cache()
    for ids in (mx.array([[64]]), mx.array([[-1]]), mx.array([[1.5]])):
        with pytest.raises(ValueError, match="token"):
            model.hidden(ids, cache)
        assert [c.offset for c in cache] == [0, 0]
    with pytest.raises(ValueError, match="cache"):
        model.hidden(mx.array([[3]]), cache[:1])
    with pytest.raises(ValueError, match="independent"):
        model.hidden_rows([[3], [4]], [cache, cache])
    mx.eval(model.hidden(mx.array([[3, 4]]), cache))
    with pytest.raises(ValueError, match="prefix"):
        model.keep_rows(cache, 2, [1])
    assert [c.offset for c in cache] == [2, 2]


def test_startup_checks_shared_bits_and_measures_shared_costs():
    model = family(tiny(quantized=True))
    assert model.streams_exact is True
    assert model.check_streams()
    assert set(model.shared_costs) == set(range(2, 17))
    assert all(cost > 0 for cost in model.shared_costs.values())
    assert not model._last


@pytest.mark.parametrize("width", [1, 2, 3, 4, 8])
def test_shared_admission_never_exceeds_qualified_total(monkeypatch, width):
    from tensorfold.families.llama.family import LlamaFamily

    totals = []
    original = LlamaFamily.hidden_rows

    def counted(self, windows, caches, parents=None):
        if len(windows) > 1:
            totals.append(sum(len(w) for w in windows))
        return original(self, windows, caches, parents)

    monkeypatch.setattr(LlamaFamily, "hidden_rows", counted)
    model = family(tiny(quantized=True), widest=width)
    assert model.batch_rows == width
    assert set(model.shared_costs) == set(range(2, width + 1))
    assert totals == list(range(2, width + 1))
    with pytest.raises(ValueError, match="width"):
        model.hidden_rows([[3]] * (width + 1), [model.make_cache() for _ in range(width + 1)])
    engine = LaneEngine(model)
    assert engine.batch_rows == width


def test_failed_shared_probe_closes_direct_shared_admission(monkeypatch):
    from tensorfold.families.llama.family import LlamaFamily

    original = LlamaFamily.hidden_rows

    def wrong_shared(self, windows, caches, parents=None):
        out = original(self, windows, caches, parents)
        return out + mx.array(1, dtype=out.dtype) if len(windows) > 1 else out

    monkeypatch.setattr(LlamaFamily, "hidden_rows", wrong_shared)
    model = family(tiny())
    assert not model.streams_exact
    assert not LaneEngine(model).family_streams
    with pytest.raises(ValueError, match="width"):
        model.hidden_rows([[3], [4]], [model.make_cache(), model.make_cache()])


def test_shared_cache_alias_and_invalid_keep_are_atomic():
    model = family(tiny())
    caches = [model.make_cache() for _ in range(2)]
    with pytest.raises(ValueError, match="independent"):
        model.hidden_rows([[3], [4]], [caches[0], list(caches[0])])
    with pytest.raises(ValueError, match="token"):
        model.hidden(mx.array([[3], [4]]), caches[0])
    mx.eval(model.hidden_rows([[3, 4], [5, 6]], caches))
    with pytest.raises(ValueError, match="prefix"):
        model.keep_rows_streams(caches, [2, 2], [1, [1]])
    assert [c.offset for cache in caches for c in cache] == [2] * 4
    model.keep_rows_streams(caches, [2, 2], [1, 1])


def test_context_limit_and_cache_adoption():
    model = family(tiny())
    cache = model.make_cache()
    cache[1].offset = 1
    with pytest.raises(ValueError, match="cache"):
        model.adopt_cache(cache)
    cache[1].offset = 0
    with pytest.raises(ValueError, match="context"):
        model.prefill(mx.array([tokens(1025)]), cache)
    assert all(c.offset == 0 for c in cache)


@pytest.mark.parametrize("bad", ["missing", "dtype", "shape", "length"])
def test_malformed_cache_is_refused_instead_of_inventing_prefix(bad):
    model = family(tiny())
    cache = model.make_cache()
    for c in cache:
        c.offset = 5
        if bad != "missing":
            c.keys = mx.zeros((1, 1, 4 if bad == "length" else 8, 63 if bad == "shape" else 64),
                              dtype=mx.float16 if bad == "dtype" else mx.bfloat16)
            c.values = mx.zeros_like(c.keys)
    with pytest.raises(ValueError, match="cache"):
        model.adopt_cache(cache)
    with pytest.raises(ValueError, match="cache"):
        model.hidden(mx.array([[3]]), cache)


@pytest.mark.parametrize("context", [576, 2048, 4096])
def test_server_admission_uses_native_context_safe_grid(context):
    from functools import partial

    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from tensorfold.engine.prefill_plan import PrefillPlan, message_markers
    from tensorfold.families.llama import engine_settings
    from tensorfold.server.app import ChatApp

    config = dict(TINY, num_hidden_layers=1, vocab_size=8192, max_position_embeddings=context)
    model = family(tiny(quantized=True, config=config))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f"token{i}": i for i in range(8192)}, unk_token="token0")), eos_token="token1")

    class MarkedTokenizer(TokenizerWrapper):
        message_markers = ((3,), (3, 4))

    tokenizer = MarkedTokenizer(tokenizer)
    settings = engine_settings(model)
    grid = settings.pop("prefill_steps", (LaneEngine.prefill_step,))[0]
    openers, assistant = message_markers(tokenizer)
    plan = PrefillPlan(grid, openers, assistant=assistant)
    assert grid >= plan.min_chunk == 256
    assert plan.openers and plan.assistant
    app = ChatApp(model, tokenizer, served_name="llama-admission-fixture", lanes=2,
                  engine_factory=partial(LaneEngine, prefill_plan=plan),
                  memory_budget_bytes=2 * 1024**3, memory_overhead_bytes=0,
                  memory_fraction=0.7, context_window=context, default_max_tokens=16,
                  checkpoint_slots=0, **settings)
    try:
        measured = app.scheduler.admission.memory
        assert measured.chunk == grid
        assert measured.long_tokens == grid + 64
        assert 2 * grid + 64 <= context
        assert measured.per_token > 0
        assert app.prompt_memory.profile.bytes_per_token > 0
        assert app.prompt_memory.workspace_profiled
    finally:
        app.close()
