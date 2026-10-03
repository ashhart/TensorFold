"""A GLM reply stops on both ranks after the round its caller asks (the client left, a stop string), on the tiny
synthetic checkpoint: before, the pair decoded on to max_tokens with nobody reading. The engine's state after a
stopped reply serves the next request as after a finished one."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _forget, _generate, _TwoCopies  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_stop")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies())


@pytest.mark.parametrize("policy", [None, "2", "serial"])
@pytest.mark.parametrize("sampling", [Sampling(41, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_a_reply_stops_after_the_round_its_caller_asks(engine, sampling, policy):
    prompt = [int(t) for t in np.random.default_rng(21).integers(0, 900, size=60)]
    draft = policy != "serial"
    policy = None if policy == "serial" else policy
    _forget(engine)
    want, _ = _generate(engine, prompt, sampling, draft=draft, policy=policy, tokens=200)
    for at in (1, 5):                                  # right after prefill's token, and a few rounds in
        _forget(engine)
        out: list[list[int]] = []
        engine.request.policy, engine.request.stop_eos = policy, False

        def take(new):
            out.append(list(new))
            return len(out) >= at

        engine.generate(list(prompt), 200, sampling, take, draft=draft)
        assert len(out) == at, (at, len(out))          # no round after the ask
        got = [t for r in out for t in r]
        assert got == want[:len(got)] and len(got) < len(want)
        again, _ = _generate(engine, prompt, sampling, draft=draft, policy=policy, tokens=200)
        assert again == want                           # a stopped reply leaves a usable engine
    _forget(engine)
