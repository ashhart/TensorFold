"""Kolibri 1's CUDA forward on a tiny checkpoint: the fp32 reference's logits, chunk- and batch-independent rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sys.path.insert(0, str(Path(__file__).parent))

from kolibri1_reference import Kolibri  # noqa: E402
from kolibri1_tiny import write  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return write(tmp_path_factory.mktemp("kolibri1-tiny"))


@pytest.fixture(scope="module")
def weights(tiny):
    from tensorfold.families.kolibri1.cuda.weights import load

    return load(tiny)


def tokens(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 500, (n,), generator=g).tolist()


def cosines(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def test_the_forward_matches_the_fp32_reference(tiny, weights):
    from tensorfold.families.kolibri1.cuda.forward import Chain, Model

    ids = tokens(100)
    ref = Kolibri(tiny)
    want, _ = ref.forward(ids, 0)
    m = Model(weights, 256)
    got = m.forward([Chain(0, 0, ids)], prompt=True, rows=range(100))
    assert cosines(got, want).min() > 0.99
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.9
    step = m.step([7], [100], [0])                               # a decode row after the prompt
    nxt, _ = ref.forward([7], 100)
    assert cosines(step, nxt).min() > 0.99


def test_prompt_chunks_and_decode_rows_agree(weights):
    from tensorfold.families.kolibri1.cuda.forward import Chain, Model

    ids = tokens(90, seed=1)
    whole = Model(weights, 256).forward([Chain(0, 0, ids)], prompt=True, rows=range(90))
    last = Model(weights, 256).prefill(ids, chunk=32)            # three chunks: same rows, same bits
    assert torch.equal(last[0], whole[-1])
    c = Model(weights, 256)
    c.prefill(ids[:60])
    decoded = torch.cat([c.step([t], [60 + i], [0]) for i, t in enumerate(ids[60:])])
    cos = cosines(decoded, whole[60:])                            # decode rows: the full layers' other kernel
    # a near-tie between two of 8 random experts can flip on the last bit (one row of this seed)
    assert (cos > 0.999).float().mean() >= 0.9 and cos.min() > 0.9, cos


def test_streams_decoded_together_equal_each_alone(weights):
    from tensorfold.families.kolibri1.cuda.forward import Model

    prompts = [tokens(40 + 13 * k, seed=10 + k) for k in range(3)]
    together = Model(weights, 256, slots=3)
    alone = [Model(weights, 256) for _ in prompts]
    nxt = []
    for k, p in enumerate(prompts):
        a, b = together.prefill(p, slot=k), alone[k].prefill(p)
        assert torch.equal(a, b)
        nxt.append(int(a.argmax()))
    pos = [len(p) for p in prompts]
    for _ in range(6):
        both = together.step(nxt, pos, [0, 1, 2])
        for k in range(3):
            assert torch.equal(both[k], alone[k].step([nxt[k]], [pos[k]], [0])[0])
        nxt = [int(r.argmax()) for r in both]
        pos = [p + 1 for p in pos]


def test_a_prompt_longer_than_the_ring_matches_the_reference(tiny, weights, monkeypatch):
    from tensorfold.families.kolibri1.cuda import forward

    monkeypatch.setattr(forward, "PROMPT_CHUNK", 64)
    m = forward.Model(weights, 512)
    assert m.ring == 128                                        # 64 rows written, then 33 keys back
    ids = tokens(400, seed=3)
    got = m.prefill(ids)
    want, _ = Kolibri(tiny).forward(ids, 0)
    assert cosines(got[0], want[-1]) > 0.99
