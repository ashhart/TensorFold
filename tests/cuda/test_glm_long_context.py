"""GLM-5.3-Flash's engine past the dense limit (2,051 tokens) on the tiny synthetic checkpoint of
test_glm_engine.py: every decoded row attends to its DSA-selected tokens through the latent cache. Drafted replies
equal serial ones, steps replayed as sparse CUDA graphs give the tokens eager steps give, and a prompt resumed
from a kept state equals a fresh prefill, with long prefill chunks."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

from test_glm_engine import _checkpoint, _forget, _generate, _TwoCopies  # noqa: E402  (pytest puts tests/cuda on sys.path)

PROMPT = 2100              # past the dense limit: the first reply token is already a sparse row
CONTEXT = 2600


@pytest.fixture(scope="module")
def engine_long(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_long")
    _checkpoint(path)
    saved = os.environ.get("TF_GLM_PREFILL_ROWS")
    os.environ["TF_GLM_PREFILL_ROWS"] = "256"            # long chunks, past the 128-row matmul block
    try:
        return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=CONTEXT)
    finally:
        if saved is None:
            os.environ.pop("TF_GLM_PREFILL_ROWS", None)
        else:
            os.environ["TF_GLM_PREFILL_ROWS"] = saved


def _prompt(seed=11, n=PROMPT):
    return list(np.random.default_rng(seed).integers(0, 1000, size=n))


def test_sparse_graphs_are_used_past_the_dense_limit(engine_long):
    g = engine_long.e.graphs
    assert g is not None and g.sparse and g.sparse_mtp, "no sparse CUDA graphs captured for a long-context engine"
    before = dict(engine_long.e.replays)
    _generate(engine_long, _prompt(seed=16), None, tokens=32)
    used = {k: engine_long.e.replays[k] - before[k] for k in before}
    assert used["sparse"] > 0 and used["sparse_mtp"] > 0, used
    assert used["main"] == 0, used                  # every step of this reply is past the dense limit


@pytest.mark.parametrize("sampling", [Sampling(99, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_long_context_drafted_equals_serial(engine_long, sampling):
    prompt = _prompt()
    serial, stats = _generate(engine_long, prompt, sampling, draft=False, tokens=32)
    assert len(serial) == 32 and stats["drafts"] is False
    for policy in (None, "2", "c3:0.35"):
        drafted, _ = _generate(engine_long, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy


@pytest.mark.parametrize("sampling", [Sampling(5, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_sparse_graphs_equal_eager_steps(engine_long, sampling):
    prompt = _prompt(seed=12)
    with_graphs, _ = _generate(engine_long, prompt, sampling, tokens=32)
    graphs, engine_long.e.graphs = engine_long.e.graphs, None
    try:
        eager, _ = _generate(engine_long, prompt, sampling, tokens=32)
    finally:
        engine_long.e.graphs = graphs
    assert with_graphs == eager


def test_long_prompt_resumes_like_a_fresh_prefill(engine_long):
    sampling = Sampling(21, 1.0, 20, 0.95)
    first = _prompt(seed=13, n=2080)                    # crosses the dense limit inside a 256-row chunk
    reply, _ = _generate(engine_long, first, sampling, tokens=16)
    follow = first + reply + _prompt(seed=14, n=40)
    warm, stats = _generate(engine_long, follow, sampling, tokens=16)
    assert stats["cached"] == len(first)                # resumed at the prompt's end; the reply prefills again
    _forget(engine_long)                                # every kept state goes: the next prefill is fresh
    cold, stats = _generate(engine_long, follow, sampling, tokens=16)
    assert stats["cached"] == 0 and warm == cold


@pytest.mark.parametrize("n", [300, 2080], ids=["short", "long"])
def test_switching_conversations_resumes_each_like_a_fresh_prefill(engine_long, n):
    """Conversation A, then B (which takes the live caches), then A again: A resumes from its prompt's saved rows
    and replies exactly as a fresh prefill of the same prompt does; then B resumes too."""
    sampling = Sampling(31, 1.0, 20, 0.95)
    a = _prompt(seed=40, n=n)
    b = _prompt(seed=41, n=n + 17)
    reply_a, _ = _generate(engine_long, a, sampling, tokens=12)
    reply_b, _ = _generate(engine_long, b, sampling, tokens=12)
    next_a = a + reply_a + _prompt(seed=42, n=9)
    warm_a, stats = _generate(engine_long, next_a, sampling, tokens=12)
    assert stats["cached"] == len(a), stats                              # resumed from A's saved rows
    next_b = b + reply_b + _prompt(seed=43, n=9)
    warm_b, stats = _generate(engine_long, next_b, sampling, tokens=12)
    assert stats["cached"] == len(b), stats
    serial_a, _ = _generate(engine_long, next_a, sampling, draft=False, tokens=12)   # fresh prefill, no cache
    serial_b, _ = _generate(engine_long, next_b, sampling, draft=False, tokens=12)
    assert warm_a == serial_a and warm_b == serial_b


def test_kept_entries_stay_within_the_memory_budget(engine_long):
    """Everything the kept conversations hold (KDA states and saved attention rows) stays within the budget,
    oldest entries dropped first, and the newest conversation still resumes."""
    from tensorfold.families.glm5_next.cuda.decode import snapshot_bytes

    sampling = Sampling(51, 1.0, 20, 0.95)
    _forget(engine_long)
    saved = engine_long.cache_bytes
    try:
        first = _prompt(seed=60, n=300)
        _generate(engine_long, first, sampling, tokens=8)
        state = snapshot_bytes(engine_long.cache[-1])
        engine_long.cache_bytes = 5 * state                  # a few entries' states, and not much room for rows
        last = None
        for seed in range(61, 67):
            last = _prompt(seed=seed, n=300 + seed)
            reply, _ = _generate(engine_long, last, sampling, tokens=8)
            assert engine_long._held_bytes() <= engine_long.cache_bytes
        warm, stats = _generate(engine_long, last + reply + _prompt(seed=70, n=5), sampling, tokens=8)
        assert stats["cached"] >= len(last)
        assert len(engine_long.cache) >= 1
    finally:
        engine_long.cache_bytes = saved
        _forget(engine_long)
