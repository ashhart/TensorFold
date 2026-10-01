"""Opt-in real-GGUF regression for canonical CUDA prompt continuation."""

import json
import os
from pathlib import Path

import pytest

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v4.cuda.engine import DeepSeekEngine


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
