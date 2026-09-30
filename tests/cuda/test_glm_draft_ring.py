"""DFlash2's block attention from the window's first tile, and its context ring (TF_GLM_DRAFT_RING), against the
loop over every context tile from 0 and the flat buffer, bit for bit: the attention kernel over contexts far past the
window (the ring wrapped many times, its other rows garbage), and the synthetic drafter of test_glm_engine end to end
(eager and CUDA graphs): the same candidates, logits and selector rows at every round over contexts longer than the
window, after a kept state is taken, the ring overwritten and the state restored.

Small: one-layer drafter, no engine (no admission), a few MiB of caches."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.geometry import draft_ring_rows  # noqa: E402
from test_glm_engine import DRAFT, D, V, _drafter  # noqa: E402


@pytest.mark.parametrize("window", [2047, 48])
@pytest.mark.parametrize("causal", [False, True])
def test_the_window_loop_and_the_ring_keep_the_full_loops_bits(window, causal):
    from tensorfold.families.glm5_next.cuda.dflash2 import _dattn_kernel

    gen = torch.Generator(device="cuda").manual_seed(window)
    H, KV, N, HD = 16, 4, 8, 128
    ring = draft_ring_rows(window, N)
    flat_cap = 13000
    kc = torch.randn((KV, flat_cap, HD), generator=gen, device="cuda").bfloat16()
    vc = torch.randn((KV, flat_cap, HD), generator=gen, device="cuda").bfloat16()
    q = torch.randn((H, N, HD), generator=gen, device="cuda").bfloat16()

    def run(k, v, cap, s, ring_mode, skip=True):
        out = torch.empty((N, H * HD), dtype=torch.bfloat16, device="cuda")
        _dattn_kernel[(KV,)](q, k, v, out, torch.tensor([s], device="cuda"), window, HD ** -0.5, N=N, G=H // KV,
                             NH=H, HD=HD, CAP=cap, BK=64, CAUSAL=causal, RING=ring_mode, SKIP=skip, num_warps=4)
        return out

    starts = sorted({0, 1, 7, 63, 64, 65, window - 1, window, window + 1, window + 63, window + 64, ring - N,
                     ring, ring + 1, 2 * ring + 5, 4095, 4096, 5000, 12345, flat_cap - N})
    for s in starts:
        want = run(kc, vc, flat_cap, s, False, skip=False)       # the old loop: every tile from 0
        assert torch.equal(run(kc, vc, flat_cap, s, False), want), s         # the loop from the window's tile
        rk = torch.randn((KV, ring, HD), generator=gen, device="cuda").bfloat16() * 100      # stale rows
        rv = torch.randn((KV, ring, HD), generator=gen, device="cuda").bfloat16() * 100
        pos = torch.arange(max(0, s + N - ring), s + N, device="cuda")
        rk[:, pos % ring], rv[:, pos % ring] = kc[:, pos], vc[:, pos]
        assert torch.equal(run(rk, rv, ring, s, True), want), s


def _weights():
    from tensorfold.families.glm5_next.cuda import qmm

    g = torch.Generator(device="cuda").manual_seed(5)
    embed = (torch.randn((V, D), generator=g, device="cuda") * 0.05).bfloat16()
    head = qmm.quantize4((torch.randn((V, D), generator=g, device="cuda") * 0.03).bfloat16())
    return SimpleNamespace(device=torch.device("cuda"), rank=0, world=1, comm=None, embed=embed, head=head,
                           draft_head=None, vocab_offset=0)


def _engine_stub():
    """What ``take_snapshot`` and ``restore`` touch of an engine besides the drafter."""

    st = SimpleNamespace(cur=[], rec=[torch.zeros(1, device="cuda")], conv=torch.zeros(1, device="cuda"), mtp_len=0,
                         mtp_drafted=0, set_pos=lambda n: None, set_mtp_len=lambda n: None)
    return SimpleNamespace(st=st)


@pytest.fixture(scope="module", params=[2048, 49], ids=["window2047", "window48"])
def pair(request, tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

    path = tmp_path_factory.mktemp(f"dring{request.param}")
    _drafter(path)
    (path / "config.json").write_text(json.dumps(dict(DRAFT, sliding_window=request.param)))
    w = _weights()
    flat = Drafter(path, w, capacity=12000, ring=False)
    ring = Drafter(path, w, capacity=12000, ring=True)
    assert flat.ring == 0 and ring.ring == draft_ring_rows(request.param - 1, flat.block) and ring.cap == ring.ring
    assert ring.nbytes() < flat.nbytes()
    yield flat, ring
    del flat, ring
    torch.cuda.empty_cache()


def _drive(pair, seed: int) -> int:
    """Prompt chunks and decode rounds far past the window on both drafters, a kept state taken, the context run on
    past it (the ring overwritten), restored and run on again; every round's candidates compared bit for bit."""

    from tensorfold.families.glm5_next.cuda import decode

    flat, ring = pair
    rng = np.random.default_rng(seed)
    width = flat.tap_in.shape[1]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    e = _engine_stub()
    rounds = 0

    def taps(n: int) -> torch.Tensor:
        return (torch.randn((n, width), generator=gen, device="cuda") * 0.5).bfloat16()

    def both_add(n: int) -> None:
        t = taps(n)
        flat.add_taps(t)
        ring.add_taps(t)
        assert flat.context_end == ring.context_end

    def compare(k: int) -> None:
        nonlocal rounds
        for _ in range(k):
            pending, depth = int(rng.integers(0, 1000)), int(rng.integers(1, flat.block))
            a, b = flat.candidates(pending, depth), ring.candidates(pending, depth)
            assert all(np.array_equal(x, y) for x, y in zip(a, b)), (flat.context_end, rounds)
            rounds += 1
            both_add(int(rng.integers(1, 9)))

    for d in (flat, ring):
        d.reset()
    for n in (300, 64, 1000, 17, 2100):                  # a prompt in chunks: 3,481 rows
        both_add(n)
    compare(40)
    n = flat.context_end
    snaps = [decode.take_snapshot(e, list(range(n)), None, mtp=False, drafter=d) for d in (flat, ring)]
    assert snaps[0].drafter_rows is None and snaps[1].drafter_rows is not None
    both_add(3000)                                       # another conversation, past the ring many times over
    compare(20)
    for d, s in zip((flat, ring), snaps):
        decode.restore(e, s, d)
    compare(60)                                          # resumed: the kept window came back
    for n in (64, 1500):                                 # a longer prompt resumed from there, then decode on
        both_add(n)
    compare(40)
    return rounds


def test_the_ring_drafts_the_flat_buffers_bits_eager(pair):
    assert _drive(pair, 1) == 160


def test_the_ring_drafts_the_flat_buffers_bits_in_cuda_graphs(pair):
    for d in pair:
        d.capture()
    try:
        assert _drive(pair, 2) == 160
    finally:
        for d in pair:
            d.block_graph, d.tap_graphs = None, {}
