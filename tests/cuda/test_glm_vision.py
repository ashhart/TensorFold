"""GLM-5.3-Flash image prompts on the CUDA engine, on the tiny synthetic checkpoint (one GPU as rank 0 of two).

Image rows whose features are the replaced tokens' own embeddings must leave the text prompt's state, MTP cache
and reply bit for bit: the rows reach the main model and the MTP head, chunk boundaries included, and nothing else
changes. An image prompt never resumes or keeps a prompt state."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import D, _checkpoint, _forget, _generate, _state, _TwoCopies  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.vision.qwen_cuda import EncodedVision  # noqa: E402

IMAGE = 999                                   # a placeholder id no prompt below otherwise holds
ROWS = tuple(range(20, 45)) + tuple(range(60, 63))      # two images across 16-row prefill chunks


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_vision")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies())


def _prompts():
    text = [int(t) for t in np.random.default_rng(9).integers(0, 900, size=100)]
    image = [IMAGE if i in ROWS else t for i, t in enumerate(text)]
    return text, image


def _features(engine, tokens) -> torch.Tensor:
    """The embedding rows of ``tokens``: what the tower would have to return to stand in for them."""

    from tensorfold.families.glm5_next.cuda import glue

    ids = torch.tensor(tokens, dtype=torch.int32, device="cuda")
    out = torch.empty((len(tokens), D), dtype=torch.bfloat16, device="cuda")
    return glue.embed(ids, engine.w.embed, D, 1, out)


def test_image_rows_reach_the_model_and_the_mtp_head(engine):
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    text, image = _prompts()
    payload = EncodedVision(ROWS, _features(engine, [text[i] for i in ROWS]), None, 0)
    ref = Engine(engine.w, capacity=2560, max_rows=8, prefill_rows=16)
    first = prefill(ref, text, None)
    want = [t.clone() for t in _state(ref)]
    e = Engine(engine.w, capacity=2560, max_rows=8, prefill_rows=16)
    assert prefill(e, image, None, vision=payload) == first
    assert all(torch.equal(a, b) for a, b in zip(_state(e), want))
    prefill(e, image, None)                                       # the placeholders' own embeddings differ
    assert not all(torch.equal(a, b) for a, b in zip(_state(e), want))
    with pytest.raises(ValueError):
        prefill(e, image, None, vision=payload, keep_at=50, keep=lambda snap: None)


@pytest.mark.parametrize("sampling", [Sampling(77, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_an_image_request_replies_as_its_text_twin_and_keeps_no_state(engine, sampling):
    text, image = _prompts()
    features = _features(engine, [text[i] for i in ROWS])
    encoded = []
    engine.vision = SimpleNamespace(encode=lambda prepared, prompt: encoded.append(prompt) or
                                    EncodedVision(ROWS, features, None, 0))
    engine.image_token = IMAGE
    try:
        _forget(engine)
        want, _ = _generate(engine, text, sampling)
        kept = [list(c.ids) for c in engine.cache]
        assert kept                                               # a text prompt is kept for its resend
        out: list[int] = []
        engine.request.policy, engine.request.stop_eos = None, False
        stats = engine.generate(list(image), 24, sampling, out.extend, vision=object())
        assert out == want and stats["cached"] == 0 and encoded == [image]
        assert [list(c.ids) for c in engine.cache] == kept        # no image prompt state is kept
        again, stats = _generate(engine, text, sampling)           # the text prompt still resumes, same reply
        assert again == want and stats["cached"] > 0
    finally:
        engine.vision = engine.image_token = None
        _forget(engine)


LONG = 2_400                                  # past the 2,051-token dense limit
LONG_ROWS = tuple(range(1_990, 2_110)) + tuple(range(2_300, 2_320))   # across a 256-row chunk and the dense limit


@pytest.fixture(scope="module")
def engine_long(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_vision_long")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=2_600, prefill_rows=256)


@pytest.mark.parametrize("sampling", [Sampling(31, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_a_long_image_prompt_replies_as_its_text_twin(engine_long, sampling):
    text = [int(t) for t in np.random.default_rng(13).integers(0, 900, size=LONG)]
    image = [IMAGE if i in LONG_ROWS else t for i, t in enumerate(text)]
    features = _features(engine_long, [text[i] for i in LONG_ROWS])
    engine_long.vision = SimpleNamespace(encode=lambda prepared, prompt: EncodedVision(LONG_ROWS, features, None, 0))
    engine_long.image_token = IMAGE
    try:
        _forget(engine_long)
        want, _ = _generate(engine_long, text, sampling, draft=False)
        _forget(engine_long)
        out: list[int] = []
        engine_long.request.policy, engine_long.request.stop_eos = None, False
        stats = engine_long.generate(list(image), 24, sampling, out.extend, vision=object())
        assert out == want and stats["cached"] == 0
    finally:
        engine_long.vision = engine_long.image_token = None
        _forget(engine_long)


def test_a_server_without_the_tower_refuses_images(engine):
    with pytest.raises(ValueError, match="--vision"):
        engine.generate([1, 2, 3], 4, None, lambda new: None, vision=object())
