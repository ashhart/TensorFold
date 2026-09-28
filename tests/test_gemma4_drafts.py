"""Gemma 4 with a DFlash draft model (tiny, random) on the lanes (Metal): drafted replies equal serial ones."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.models.gemma4_text")
if not mx.metal.is_available():
    pytest.skip("the Gemma decode kernels are Metal kernels", allow_module_level=True)

from gemma4_tiny import TINY, tiny_text, tokens  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from tensorfold.drafters.dflash_drafter import DFlashDrafter, DFlashProposer, _vendor  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.families.gemma4.model import Gemma4  # noqa: E402

DRAFT = {"architectures": ["DFlashDraftModel"], "block_size": 8, "hidden_size": TINY["hidden_size"], "head_dim": 64,
         "intermediate_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2,
         "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 16, "rms_norm_eps": 1e-6,
         "rope_theta": 10000.0, "max_position_embeddings": 4096, "vocab_size": TINY["vocab_size"],
         "num_target_layers": TINY["num_hidden_layers"], "final_logit_softcapping": 30.0,
         "dflash_config": {"mask_token_id": 4, "target_layer_ids": [0, 2]}}


@pytest.fixture(scope="module")
def drafted(tmp_path_factory):
    """The tiny target and a random DFlash model saved as a checkpoint, loaded the way ``--drafter`` loads one."""

    folder = tmp_path_factory.mktemp("dflash")
    (folder / "config.json").write_text(json.dumps(DRAFT))
    vendor = _vendor()
    mx.random.seed(11)
    config = vendor.DFlashConfig(
        hidden_size=DRAFT["hidden_size"], num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, intermediate_size=256, vocab_size=DRAFT["vocab_size"], rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=4096, block_size=8, target_layer_ids=(0, 2), num_target_layers=4, mask_token_id=4,
        layer_types=tuple(DRAFT["layer_types"]), sliding_window=16, final_logit_softcapping=30.0)
    model = vendor.DFlashDraftModel(config)
    mx.eval(model.parameters())
    mx.save_safetensors(str(folder / "model.safetensors"), dict(tree_flatten(model.parameters())))
    text = tiny_text()
    return text, DFlashDrafter(text, str(folder), bits=0)


def reply(model: Gemma4, prompt: list[int], n: int, *, sampling=None, drafts: bool) -> tuple[list[int], LaneEngine]:
    engine = LaneEngine(model, max_rows=16, max_draft=15)
    stream = LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=n, sampling=sampling, drafts=drafts)
    engine.add_stream(stream)
    engine.run()
    return list(stream.emitted), engine


def family(drafted) -> Gemma4:
    text, drafter = drafted
    model = Gemma4(text, backend="rows", check=False, drafter=drafter)
    model.exact_width = 16
    return model


@pytest.mark.parametrize("sampled", [False, True])
def test_dflash_drafted_replies_equal_serial_ones(drafted, sampled):
    model = family(drafted)
    prompt = tokens(40, seed=41)
    sampling = Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95) if sampled else None
    serial, _ = reply(model, prompt, 32, sampling=sampling, drafts=False)
    got, engine = reply(model, prompt, 32, sampling=sampling, drafts=True)
    assert got == serial
    assert engine.drafted > 0


@pytest.mark.parametrize("sampled", [False, True])
def test_drafts_that_land_are_kept_and_the_reply_is_unchanged(drafted, sampled, monkeypatch):
    """The drafter's forward runs every round; its drafts become the serial reply's, broken at varying depths."""

    model = family(drafted)
    prompt = tokens(40, seed=42)
    sampling = Sampling(seed=6, temperature=1.0, top_k=20, top_p=0.95) if sampled else None
    serial, _ = reply(model, prompt, 40, sampling=sampling, drafts=False)
    expected, calls, real = prompt + serial, [0], DFlashProposer.propose

    def propose(self, context, max_draft):
        out = real(self, context, max_draft)
        good = (6, 2, 0, 9, 3)[calls[0] % 5]
        calls[0] += 1
        want = expected[len(context):len(context) + len(out)]
        return [t if j < good else (t + 1 + j) % TINY["vocab_size"] for j, t in enumerate(want)]

    monkeypatch.setattr(DFlashProposer, "propose", propose)
    got, engine = reply(model, prompt, 40, sampling=sampling, drafts=True)
    assert got == serial
    assert 0 < engine.accepted < engine.drafted and max(r.rows for r in engine.round_stats) > 2


def test_streams_sharing_rounds_with_drafts_emit_what_they_emit_alone(drafted):
    model = family(drafted)
    prompts = [tokens(n, seed=50 + n) for n in (25, 60, 9)]
    samplings = [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95), None]
    alone = [reply(model, p, 20, sampling=s, drafts=False)[0] for p, s in zip(prompts, samplings)]
    engine = LaneEngine(model, max_rows=16, max_draft=15)
    streams = []
    for i, (p, s) in enumerate(zip(prompts, samplings)):
        stream = LaneStream(stream_id=f"s{i}", prompt_ids=list(p), max_new_tokens=20, sampling=s, drafts=True)
        engine.add_stream(stream)
        streams.append(stream)
    engine.run()
    assert [list(s.emitted) for s in streams] == alone
    assert any(r.streams > 1 for r in engine.round_stats)


def test_long_and_resumed_prompts_draft_past_the_drafters_window(drafted, monkeypatch):
    """A prompt past the drafter's window starts its full-attention context late; resumed prompts start late too."""

    from tensorfold.engine.prefill_plan import PrefillPlan

    monkeypatch.setattr(LaneEngine, "prefill_plan", PrefillPlan(64))
    model = family(drafted)
    prompt = tokens(300, seed=43)
    serial, _ = reply(model, prompt, 24, drafts=False)
    got, engine = reply(model, prompt, 24, drafts=True)
    assert got == serial and engine.drafted > 0
    at = 256
    cache = LaneEngine(model, max_rows=16, max_draft=15).prefill_prefix(prompt[:at])
    engine = LaneEngine(model, max_rows=16, max_draft=15)
    stream = LaneStream(stream_id="r", prompt_ids=list(prompt), max_new_tokens=24, drafts=True)
    engine.add_stream(stream, cache=model.adopt_cache(cache[:len(model.text.layers)]), cached_tokens=at)
    engine.run()
    assert list(stream.emitted) == serial and stream.cached_tokens == at
