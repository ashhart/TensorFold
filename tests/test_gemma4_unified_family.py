"""Gemma 4 unified (dense 12B layout) as a lane family on a tiny random checkpoint (Metal)."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.models.gemma4_text")
if not mx.metal.is_available():
    pytest.skip("the Gemma decode kernels are Metal kernels", allow_module_level=True)

from gemma4_unified_tiny import FOUR_BIT, TINY, tiny_assistant, tiny_text, tokens  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.families.gemma4_unified.assistant import Assistant  # noqa: E402
from tensorfold.families.gemma4_unified.model import Gemma4Unified  # noqa: E402
from tensorfold.kernels.gemma.dense.v1.matmul import tensor_units  # noqa: E402

LANE = pytest.param("lane", marks=pytest.mark.skipif(not tensor_units(), reason="needs tensor units"))
BACKENDS = ["rows", LANE, pytest.param("qkv=lane,gate_up=lane", marks=LANE.marks)]
copy = LaneEngine.copy_single_cache


@pytest.fixture(scope="module")
def text():
    return tiny_text()


def family(text, backend: str, width: int = 16, drafter=None) -> Gemma4Unified:
    model = Gemma4Unified(text, backend=backend, check=False, drafter=drafter)
    model.exact_width = width
    return model


def prefilled(model, n: int, seed: int = 3) -> list:
    cache = model.make_cache()
    mx.eval(model.prefill(mx.array([tokens(n, seed)], dtype=mx.uint32), cache))
    return cache


def steps(model, cache: list, window: list[int]) -> list:
    out = []
    for token in window:
        logits = model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache))
        mx.eval(logits)
        out.append(logits[0, -1])
    return out


def same(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def test_the_tiny_checkpoint_has_mixed_widths(text):
    model = family(text, "rows")
    widths = [[s.bits for s in p.stacks] for p in model.decode.qkv]
    assert widths[1] == [8, 4] and widths[0] == [8]          # q|k at 8 bits, v at 4: two runs, one output
    assert {s.bits for s in model.decode.down[2].stacks} == {4}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("prompt", [20, 150])
def test_windows_give_every_row_its_serial_bits(text, backend, prompt):
    model = family(text, backend)
    base = prefilled(model, prompt)
    window = tokens(14, seed=5)
    serial = steps(model, copy(base), window)
    for width in range(2, len(window) + 1):
        logits = model.head(model.hidden(mx.array([window[:width]], dtype=mx.uint32), copy(base)))
        mx.eval(logits)
        assert all(same(logits[0, i], serial[i]) for i in range(width)), width


@pytest.mark.parametrize("backend", ["rows", LANE])
def test_decode_rows_follow_mlx_lms_forward(text, backend):
    """The kernels' arithmetic is not MLX's, but the logits are the same model's: top tokens agree."""

    model = family(text, backend)
    prompt, follow = tokens(40, seed=8), tokens(12, seed=9)
    cache = prefilled(model, 40, seed=8)
    ours = model.head(model.hidden(mx.array([follow], dtype=mx.uint32), cache))[0].astype(mx.float32)
    ref = text(mx.array([prompt + follow]))[0, 40:].astype(mx.float32)
    agree = (mx.argmax(ours, -1) == mx.argmax(ref, -1)).astype(mx.float32).mean().item()
    # random bf16 weights amplify rounding (the 12B agrees on every top token): bounds, not equality
    assert agree >= 0.75
    assert mx.mean(mx.abs(ours - ref)).item() < 0.15 * mx.std(ref).item()


@pytest.mark.parametrize("backend", ["rows", LANE])
def test_rollback_then_steps_equal_serial_decoding(text, backend):
    model = family(text, backend)
    base = prefilled(model, 131)                      # the window's rows straddle the ring's end
    window = tokens(12, seed=5)
    serial = steps(model, copy(base), window)
    work = copy(base)
    mx.eval(model.hidden(mx.array([window[:3] + tokens(6, seed=9)], dtype=mx.uint32), work))
    model.keep_rows(work, 9, 3)
    assert [c.offset for c in work] == [131 + 3] * len(work)
    assert all(same(a, b) for a, b in zip(steps(model, work, window[3:]), serial[3:]))


@pytest.mark.parametrize("backend", ["rows", LANE])
def test_a_shared_forward_gives_each_stream_its_own_bits(text, backend):
    model = family(text, backend)
    assert model.check_streams(None)
    bases = [prefilled(model, n, seed=n) for n in (20, 133, 140)]
    windows = [tokens(n, seed=11 + n) for n in (3, 5, 1)]
    alone = []
    for base, window in zip(bases, windows):
        logits = model.head(model.hidden(mx.array([window], dtype=mx.uint32), copy(base)))
        mx.eval(logits)
        alone.append(logits[0])
    joint = model.head(model.hidden_rows(windows, [copy(b) for b in bases]))[0]
    mx.eval(joint)
    at = 0
    for window, own in zip(windows, alone):
        assert same(joint[at:at + len(window)], own)
        at += len(window)


class Oracle(Assistant):
    """The real assistant forward, then a known reply's tokens at the asked positions (wrong from ``good`` on)."""

    def __init__(self, inner: Assistant, sequence: list[int], good=(3, 0, 5, 1, 9)) -> None:
        self.__dict__.update(inner.__dict__)
        self.sequence, self.good, self.calls, self.checked = list(sequence), good, 0, 0

    def draft(self, layers, hidden, token, row_position, position, sampling, count):
        drafted = super().draft(layers, hidden, token, row_position, position, sampling, count)
        mx.eval(drafted)
        assert drafted.shape == (count,)
        # the last kept row sits two before the first draft; the pending token right before it
        assert row_position == position - 2 and token == self.sequence[position - 1]
        self.checked += 1
        good = self.good[self.calls % len(self.good)]
        self.calls += 1
        out = [self.sequence[position + s] if position + s < len(self.sequence) else 0 for s in range(count)]
        return mx.array([t if s < good else (t + 1 + s) % TINY["vocab_size"] for s, t in enumerate(out)],
                        dtype=mx.uint32)


def _reply(model, prompt, n, *, sampling=None, drafts=False, cache=None, cached=0, engine=None):
    engine = engine or LaneEngine(model, max_rows=16, max_draft=15)
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, sampling=sampling, drafts=drafts)
    engine.add_stream(stream, cache=cache, cached_tokens=cached)
    engine.run()
    return list(stream.emitted), engine


@pytest.mark.parametrize("backend", ["rows", LANE])
@pytest.mark.parametrize("sampled", [False, True])
def test_mtp_drafted_replies_equal_serial_ones(text, backend, sampled):
    plain = family(text, backend)
    prompt = tokens(140, seed=21)
    sampling = Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95) if sampled else None
    serial, _ = _reply(plain, prompt, 40, sampling=sampling)
    oracle = Oracle(tiny_assistant(), prompt + serial)
    model = family(text, backend, drafter=oracle)
    assert model.mtp is oracle and model.drafts == oracle.default_drafts
    drafted, engine = _reply(model, prompt, 40, sampling=sampling, drafts=True)
    assert drafted == serial
    assert oracle.checked > 0 and engine.drafted > 0 and 0 < engine.accepted < engine.drafted
    # the real (random) assistant: whatever it drafts, the reply is the serial one
    model = family(text, backend, drafter=tiny_assistant())
    assert _reply(model, prompt, 40, sampling=sampling, drafts=True)[0] == serial


@pytest.mark.parametrize("backend", ["rows", LANE])
def test_concurrent_streams_with_mtp_drafts_emit_what_they_emit_alone(text, backend):
    plain = family(text, backend)
    prompts = [tokens(n, seed=30 + n) for n in (25, 140, 60, 9)]
    samplings = [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95), None,
                 Sampling(seed=8, temperature=0.8, top_k=40, top_p=0.9)]
    alone = [_reply(plain, p, 20, sampling=s)[0] for p, s in zip(prompts, samplings)]

    class ByPrompt(Oracle):                                  # each stream's own known reply, keyed by its prompt
        def draft(self, layers, hidden, token, row_position, position, sampling, count):
            match = [p + a for p, a in zip(prompts, alone) if position - 1 < len(p + a) and
                     (p + a)[position - 1] == token]
            self.sequence = match[0] if match else [0] * (position + count + 1)
            return super().draft(layers, hidden, token, row_position, position, sampling, count) if match else \
                Assistant.draft(self, layers, hidden, token, row_position, position, sampling, count)

    model = family(text, backend, drafter=ByPrompt(tiny_assistant(), []))
    engine = LaneEngine(model, max_rows=16, max_draft=15)
    streams = []
    for i, (p, s) in enumerate(zip(prompts, samplings)):
        stream = LaneStream(stream_id=f"s{i}", prompt_ids=list(p), max_new_tokens=20, sampling=s, drafts=i % 2 == 0)
        engine.add_stream(stream)
        streams.append(stream)
    engine.run()
    assert [list(s.emitted) for s in streams] == alone
    assert any(r.streams > 1 for r in engine.round_stats) and engine.accepted > 0


def test_the_assistant_reads_only_committed_keys(text):
    """Keys of rows a round rejected (still in the buffers before keep_rows) never reach a draft."""

    assistant = tiny_assistant()
    model = family(text, "rows", drafter=assistant)
    base = prefilled(model, 30)
    window = tokens(5, seed=12)
    trimmed, untrimmed = copy(base), copy(base)
    for cache in (trimmed, untrimmed):
        mx.eval(model.hidden(mx.array([window], dtype=mx.uint32), cache))
    model.keep_rows(trimmed, 5, 2)
    hidden = model._hidden[:, 1:2]
    a = assistant.draft(model._layers(trimmed), hidden, 7, 31, 33, None, 4)
    b = assistant.draft(model._layers(untrimmed), hidden, 7, 31, 33, None, 4)
    assert same(a, b) and a.shape == (4,)


def test_a_prompt_resumed_from_a_grid_checkpoint_equals_a_fresh_one(text, monkeypatch, tmp_path):
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    monkeypatch.setattr(LaneEngine, "prefill_plan", PrefillPlan(64))
    model = family(text, "rows", drafter=tiny_assistant())
    first = tokens(150, seed=40)
    reply, _ = _reply(model, first, 12, drafts=True)
    engine = LaneEngine(model, max_rows=16, max_draft=15, retain_finished_caches=True)
    stream = LaneStream(stream_id="a", prompt_ids=list(first), max_new_tokens=12, drafts=True)
    engine.add_stream(stream, checkpoints_at=(len(first),))
    engine.run()
    (at, checkpoint), = [(len(t), c) for t, c in stream.history_checkpoints]
    follow = first + reply + tokens(20, seed=41)
    fresh, _ = _reply(model, follow, 16, drafts=True)
    resumed, _ = _reply(model, follow, 16, cache=copy(checkpoint), cached=at, drafts=True)
    assert resumed == fresh
    path = save_snapshot(tmp_path, "gemma-unified-test", first[:at], checkpoint)
    loaded_tokens, loaded = load_snapshot(path, "gemma-unified-test")
    assert loaded_tokens == first[:at] and len(loaded) == len(TINY["layer_types"])     # the assistant slot not stored
    stored, _ = _reply(model, follow, 16, cache=model.adopt_cache(loaded), cached=at, drafts=True)
    assert stored == fresh


def _write_config(tmp_path, text_config, quantization):
    config = {"model_type": "gemma4_unified", "architectures": ["Gemma4UnifiedForConditionalGeneration"],
              "text_config": dict(text_config, model_type="gemma4_unified_text"), "quantization": quantization}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_the_family_is_found_by_model_type():
    from tensorfold.families import families

    found = families()
    assert found["gemma4_unified"].module == "tensorfold.families.gemma4_unified"
    assert found["gemma4_unified_text"].module == "tensorfold.families.gemma4_unified"
    assert found["gemma4"].module == "tensorfold.families.gemma4"


@pytest.mark.parametrize("change, quant, words", [
    ({}, {"bits": 8, "group_size": 64}, None),
    ({}, {"bits": 4, "group_size": 32}, None),
    ({}, {"bits": 6, "group_size": 64, "model.language_model.layers.0.mlp.down_proj": {"bits": 8, "group_size": 64}},
     None),
    ({"enable_moe_block": True}, {"bits": 8, "group_size": 64}, "MoE"),
    ({"hidden_size_per_layer_input": 256}, {"bits": 8, "group_size": 64}, "per-layer inputs"),
    ({"num_kv_shared_layers": 2}, {"bits": 8, "group_size": 64}, "shared-KV"),
    ({}, {"bits": 8, "group_size": 32}, "groups of 64"),
    ({}, {"bits": 4, "group_size": 128}, "groups of 64"),
    ({}, {"bits": 8, "group_size": 64, "model.language_model.layers.0.mlp.down_proj": {"bits": 8, "group_size": 128}},
     "down_proj"),
    ({}, {"bits": 4, "group_size": 32, "mode": "mxfp4"}, "affine"),
])
def test_check_reads_config_alone(tmp_path, change, quant, words):
    from tensorfold.families import gemma4_unified

    model_dir = _write_config(tmp_path, dict(TINY, **change), quant)
    if words is None:
        gemma4_unified.check(model_dir)
    else:
        with pytest.raises(ValueError, match=words):
            gemma4_unified.check(model_dir)


def test_the_assistant_loads_from_a_checkpoint_directory(tmp_path):
    """Assistant.load: the HF/mlx-vlm layout (model.*, pre/post_projection, 4-bit) into the drafter's modules."""

    from mlx.utils import tree_flatten

    from gemma4_unified_tiny import ASSISTANT

    made = tiny_assistant(seed=4)
    weights = {f"model.{k.removeprefix('model.')}" if k.startswith("model.") else k: v
               for k, v in tree_flatten(made.text.parameters())}
    weights.update({f"pre_projection.{k}": v for k, v in tree_flatten(made.pre.parameters())})
    weights.update({f"post_projection.{k}": v for k, v in tree_flatten(made.post.parameters())})
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(ASSISTANT))
    loaded = Assistant.load(tmp_path)
    x = (mx.random.normal((1, 1, 2 * TINY["hidden_size"])) * 0.5).astype(mx.bfloat16)
    assert same(loaded.pre(x), made.pre(x))
    assert loaded.kinds == ASSISTANT["text_config"]["layer_types"]
    with pytest.raises(ValueError, match="ordered"):
        (tmp_path / "config.json").write_text(json.dumps(dict(ASSISTANT, use_ordered_embeddings=True)))
        Assistant.load(tmp_path)


def test_four_bit_modules_are_listed(text):
    paths = {p for p, m in text.named_modules() if getattr(m, "bits", None) == 4}
    assert all(any(p.endswith(f) for p in paths) for f in FOUR_BIT)


def test_batched_drafts_match_each_streams_own(text):
    """One assistant forward a step for several streams (keys padded and masked) drafts what each drafts alone."""

    from gemma4_unified_tiny import tiny_assistant as make

    assistant = make()
    model = family(text, "rows", drafter=assistant)
    caches, slots = [], []
    for n, seed in ((20, 1), (140, 2), (7, 3)):
        cache = prefilled(model, n, seed=seed)
        mx.eval(model.hidden(mx.array([tokens(3, seed=seed + 50)], dtype=mx.uint32), cache))
        model.speculate(cache, [9, 10, 11], n, None, start=0)
        caches.append(cache)
        slots.append(cache[-1])
    positions = [s.position + 2 for s in slots]
    alone = [assistant.draft(model._layers(c), s.hidden, s.token, s.position, p, None, 5)
             for c, s, p in zip(caches, slots, positions)]
    joint = assistant.draft_many([model._layers(c) for c in caches], slots, positions, [None] * 3, [5, 3, 4])
    for a, j, n in zip(alone, joint, (5, 3, 4)):
        assert j.tolist() == a[:n].tolist()
