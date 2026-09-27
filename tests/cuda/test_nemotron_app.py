"""The server's Nemotron engine on the real checkpoint: drafts equal ``draft=False``, resumes equal fresh prompts."""

import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

MODEL = Path(os.environ.get("TF_NEMOTRON_MODEL", "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit"))
pytestmark = pytest.mark.skipif(not (MODEL / "config.json").is_file(), reason="real checkpoint not mounted")

RAW = "Write a short Python function that computes the Fibonacci sequence and explain it."


@pytest.fixture(scope="module")
def engine():
    from tensorfold.families.nemotron_h.cuda.app import NemotronEngine

    return NemotronEngine(MODEL, context=2032, context_explicit=True)     # 2,048 cache rows


def _ids(text: str) -> list[int]:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(MODEL / "tokenizer.json")).encode(text).ids


@pytest.mark.parametrize("greedy", [False, True])
def test_drafted_request_equals_serial(engine, greedy):
    from tensorfold.engine.exact_sampling import Sampling

    prompt = _ids(RAW)
    sampling = None if greedy else Sampling(1234, 1.0, 20, 0.95)
    drafted: list[int] = []
    stats = engine.generate(prompt, 48, sampling, lambda new: drafted.extend(new))
    serial: list[int] = []
    engine.generate(prompt, 48, sampling, lambda new: serial.extend(new), draft=False)
    assert drafted == serial
    assert stats["accepted"] > 0 and stats["min_rows"] >= 2


def test_resumed_prompts_equal_fresh(engine):
    from tensorfold.engine.exact_sampling import Sampling

    sampling = Sampling(7, 1.0, 20, 0.95)
    first = _ids(RAW)
    reply: list[int] = []
    engine.generate(first, 32, sampling, lambda new: reply.extend(new))
    for prompt in (first + _ids(" Also give its complexity."), first + reply + _ids(" Now in C.")):
        resumed: list[int] = []
        stats = engine.generate(prompt, 32, sampling, lambda new: resumed.extend(new))
        assert stats["cached"] == len(first)                  # prompt ends only: a reply prefills again
        engine.cache = []                      # fresh: nothing to resume from
        fresh: list[int] = []
        engine.generate(prompt, 32, sampling, lambda new: fresh.extend(new))
        assert resumed == fresh
        engine.generate(first, 32, sampling, lambda new: None)            # keep the first prompt again


def test_app_answers_completion_and_chat(engine):
    from tensorfold.cuda.server import App

    app = App(engine, MODEL, "nemotron-cuda")
    for chat, body in ((False, {"prompt": RAW, "max_tokens": 24, "temperature": 0}),
                       (True, {"messages": [{"role": "user", "content": "Name three prime numbers."}],
                               "max_tokens": 24, "temperature": 1.0, "seed": 7,
                               "chat_template_kwargs": {"enable_thinking": False}})):
        deltas: list[dict] = []
        res = app.run(body, chat, lambda d: deltas.append(d) or True)
        assert 1 <= res["completion_tokens"] <= 24
        assert res["finish"] in ("stop", "length")
