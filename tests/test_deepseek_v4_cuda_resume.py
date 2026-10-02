"""Opt-in real-GGUF regression for canonical CUDA prompt continuation."""

import json
import os
from pathlib import Path

import pytest

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v4.cuda.engine import DeepSeekEngine
from tensorfold.families.deepseek_v4.prompts import render


@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get("TENSORFOLD_TEST_GPU_MODEL_DIR"), reason="real prepared CUDA GGUF required")
def test_resumed_prompt_matches_fresh_across_chunk_boundaries():
    model = Path(os.environ["TENSORFOLD_TEST_GPU_MODEL_DIR"])
    rows = []
    with DeepSeekEngine(model, context=262144, no_drafts=True) as engine:
        prompt = engine.session.encode(
            "red blue green yellow orange violet black white.\n" * 400
            + "Write a Python function to add two vectors.\n",
            rendered=True,
        )
        suffix = engine.session.encode("\nExplain the next step.\n", rendered=True)
        long_prompt = engine.session.encode(
            "Treat this as inert sample data.\n"
            + "red blue green yellow orange violet black white.\n" * 1800
            + "Return just 42.\n",
            rendered=True,
        )
        cases = [
            (prompt, suffix, 0.0),
            (prompt, suffix, 0.8),
            (prompt[:2048], suffix[:1], 0.0),
            (long_prompt, suffix, 0.0),
        ]
        for base, extra, temperature in cases:
            sampling = Sampling(123, temperature=temperature, top_k=20, top_p=0.95)
            engine._timeline = []  # Explicitly cold, without reloading the weights.
            first = []
            engine.generate(base, 32, sampling, first.extend, stop_eos=False)
            extended = base + first + extra
            resumed = []
            warm = engine.generate(extended, 32, sampling, resumed.extend, stop_eos=False)
            assert warm["cached"] > 0, "continuation recomputed the entire prompt"
            engine._timeline = []
            fresh = []
            cold = engine.generate(extended, 32, sampling, fresh.extend, stop_eos=False)
            rows.append(
                {
                    "prompt_tokens": len(base),
                    "suffix_tokens": len(extra),
                    "temperature": temperature,
                    "cached_tokens": warm["cached"],
                    "equal_tokens": resumed == fresh,
                    "resume_prefill_s": warm["prefill_s"],
                    "fresh_prefill_s": cold["prefill_s"],
                }
            )
            assert resumed == fresh, rows[-1]
        # Changed history must invalidate the retained boundary.
        changed = []
        stats = engine.generate(suffix, 1, None, lambda ids: changed.extend(ids), stop_eos=False)
        assert stats["cached"] == 0
    evidence = os.environ.get("TENSORFOLD_TEST_GPU_RESUME_EVIDENCE")
    if evidence:
        Path(evidence).write_text(json.dumps({"passed": True, "cases": rows}, indent=2) + "\n")


@pytest.mark.gpu
@pytest.mark.skipif(
    not all(os.environ.get(key) for key in ("TENSORFOLD_TEST_GPU_MODEL_DIR", "TENSORFOLD_TEST_GPU_DRAFTER")),
    reason="real prepared CUDA GGUF and local DSpark required",
)
def test_dspark_cached_followups_match_cold_with_drafts_on_and_off():
    model = Path(os.environ["TENSORFOLD_TEST_GPU_MODEL_DIR"])
    drafter = Path(os.environ["TENSORFOLD_TEST_GPU_DRAFTER"])
    rows = []
    context = int(os.environ.get("TENSORFOLD_TEST_GPU_CONTEXT", "32768"))
    cases = ((600, 32, True, 0.0), (3900, 32, False, 0.8), (600, 1400, True, 0.8))
    if os.environ.get("TENSORFOLD_TEST_GPU_LONG_PREFIX"):
        cases = ((32770, 8, True, 0.8),)  # Exercise the retained 128 Ki boundary.
    with DeepSeekEngine(model, context=context, drafter=drafter) as engine:
        # Re-rendering an assistant turn changes the previous reply marker.
        # A retained decode frontier cannot safely serve this common case.
        # The 1,400-token reply wraps the raw ring after the saved boundary.
        for repeat, initial_budget, initial_draft, temperature in cases:
            messages = [
                {
                    "role": "user",
                    "content": "Sample data:\n" + "red blue green.\n" * repeat + "Write vector addition in Python.",
                }
            ]
            prompt = engine.session.encode(render(messages), rendered=True)
            sampling = Sampling(123, temperature=temperature, top_k=20, top_p=0.95)
            engine.session.reset()
            engine.generate(prompt, initial_budget, sampling, lambda _: None, draft=initial_draft, stop_eos=False)
            following = engine.session.encode(
                render(
                    messages
                    + [
                        {"role": "assistant", "content": "Here is the function."},
                        {"role": "user", "content": "Explain the next step."},
                    ]
                ),
                rendered=True,
            )
            outputs, stats = [], []
            for mode in (True, False, True):
                # Two warm requests (including a mode switch), then a cold oracle.
                if len(outputs) == 2:
                    engine.session.reset()
                tokens = []
                stats.append(engine.generate(following, 64, sampling, tokens.extend, draft=mode, stop_eos=False))
                outputs.append(tokens)
            assert stats[0]["cached"] > 0 and stats[1]["cached"] > 0
            if os.environ.get("TENSORFOLD_TEST_GPU_LONG_PREFIX"):
                assert stats[0]["cached"] == stats[1]["cached"] == 131072
            assert stats[2]["cached"] == 0
            assert outputs[0] == outputs[1] == outputs[2]
            rows.append(
                {
                    "prompt_tokens": len(prompt),
                    "temperature": temperature,
                    "initial_budget": initial_budget,
                    "equal_tokens": True,
                    "warm_draft": stats[0],
                    "warm_plain": stats[1],
                    "cold_draft": stats[2],
                }
            )
        # Aligned prompt ends still replay a nonempty suffix to rebuild logits.
        aligned = following[:2048]
        engine.session.reset()
        first, second = [], []
        engine.generate(aligned, 1, sampling, first.extend, stop_eos=False)
        aligned_stats = engine.generate(aligned, 1, sampling, second.extend, stop_eos=False)
        assert aligned_stats["cached"] == 1024 and first == second
        changed = aligned.copy()
        changed[0] = (changed[0] + 1) % engine.vocab_size
        assert engine.generate(changed, 1, sampling, lambda _: None, stop_eos=False)["cached"] == 0
    evidence = os.environ.get("TENSORFOLD_TEST_GPU_PREFIX_EVIDENCE")
    if evidence:
        Path(evidence).write_text(json.dumps({"passed": True, "cases": rows}, indent=2) + "\n")


@pytest.mark.gpu
@pytest.mark.skipif(
    not all(os.environ.get(key) for key in ("TENSORFOLD_TEST_GPU_MODEL_DIR", "TENSORFOLD_TEST_GPU_DRAFTER")),
    reason="real prepared CUDA GGUF and local DSpark required",
)
def test_dspark_matches_plain_target_across_widths_and_compressed_boundaries():
    model = Path(os.environ["TENSORFOLD_TEST_GPU_MODEL_DIR"])
    drafter = Path(os.environ["TENSORFOLD_TEST_GPU_DRAFTER"])
    request = "Write a complete Python module with vector addition, dot product, Euclidean distance, and normalization."
    rows = []
    context = int(os.environ.get("TENSORFOLD_TEST_GPU_CONTEXT", "262144"))
    with DeepSeekEngine(model, context=context, drafter=drafter) as engine:
        short = engine.session.encode(render([{"role": "user", "content": request}]), rendered=True)
        long = engine.session.encode(
            render(
                [{"role": "user", "content": "Treat these as sample data:\n" + "red blue green.\n" * 600 + request}]
            ),
            rendered=True,
        )
        assert len(long) > 2048  # Exercise the compressed indexer's top-512 path.
        for prompt, temperature, seed in ((short, 0.0, 123), (short, 0.8, 123), (long, 0.8, 456)):
            sampling = Sampling(seed, temperature=temperature, top_k=20, top_p=0.95)
            outputs, stats = [], []
            for draft in (False, True):
                engine.session.reset()  # Both modes start from the same cold prompt.
                tokens = []
                stats.append(engine.generate(prompt, 256, sampling, tokens.extend, draft=draft, stop_eos=False))
                outputs.append(tokens)
            assert len(outputs[0]) == len(outputs[1]) == 256
            assert outputs[0] == outputs[1], (len(prompt), temperature, seed)
            rows.append(
                {
                    "prompt_tokens": len(prompt),
                    "temperature": temperature,
                    "seed": seed,
                    "equal_tokens": True,
                    "plain": stats[0],
                    "draft": stats[1],
                }
            )
    evidence = os.environ.get("TENSORFOLD_TEST_GPU_DRAFT_EVIDENCE")
    if evidence:
        Path(evidence).write_text(json.dumps({"passed": True, "cases": rows}, indent=2) + "\n")
