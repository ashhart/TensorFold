"""Prompts prefilled together in one forward: each stream's states, last row and kept state equal its prefill alone,
for any mix of lengths, resumed starts and piece bounds, with bf16 prompts and --prefill-fp8 alike."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import prompt_precision  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import State  # noqa: E402
from tensorfold.families.qwen3_5.cuda.prefill import Piece, prefill_batch, prefill_state  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402

V = 256
VOLTA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7   # no FP8 prompts, no sm_80 kernels


@pytest.fixture(params=[False, True], ids=["bf16", "fp8"])
def fp8(request):
    if request.param and VOLTA:
        pytest.skip("FP8 prompts need sm_89")
    with prompt_precision.using(request.param):
        yield request.param


@pytest.fixture(scope="module")
def w():
    gen = torch.Generator(device="cuda").manual_seed(31)
    dev = "cuda"

    def qlinear(n, k):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen,
                              device=dev, dtype=torch.int64).to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 + 0.001).bfloat16()
        biases = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 - 0.0015).bfloat16()
        return QLinear(words, scales, biases)

    norm = torch.ones(128, device=dev, dtype=torch.bfloat16)

    def gdn():
        return GDN(qlinear(384, 128), qlinear(128, 128), qlinear(1, 128), qlinear(1, 128),
                   qlinear(128, 128), torch.randn(384, 4, generator=gen, device=dev).bfloat16() * 0.1,
                   torch.zeros(1, device=dev), torch.zeros(1, device=dev), norm)

    attn = Attention(qlinear(2 * 2 * 128, 128), qlinear(128, 128), qlinear(128, 128), qlinear(128, 2 * 128),
                     norm, norm)
    mlp = lambda: (qlinear(128, 128), qlinear(128, 128), qlinear(128, 128))  # noqa: E731
    layers = [Layer(True, norm, norm, gdn(), None, *mlp()), Layer(False, norm, norm, None, attn, *mlp()),
              Layer(True, norm, norm, gdn(), None, *mlp())]
    config = Config(hidden=128, intermediate=128, layers=3, heads=2, kv_heads=1,
                    head_dim=128, vocab=V, k_heads=1, v_heads=1, dk=128, dv=128,
                    conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    weights = Weights(config, qlinear(V, 128), layers, norm, qlinear(V, 128), torch.ones(16, device=dev))
    prepare(weights)
    return weights


def _prompt(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (n,), generator=g).tolist()


def _same(a, b):
    assert a.pos == b.pos
    for x, y in zip(a.rec, b.rec):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.conv, b.conv):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.kv, b.kv):
        if x is not None:
            assert torch.equal(x[0][:a.pos], y[0][:b.pos]) and torch.equal(x[1][:a.pos], y[1][:b.pos])


def _alone(w, prompt, start, stop, keep_at):
    """The piece prompt[start:stop] alone, from the state prompt[:start] alone left."""

    st = State(w)
    if start:
        prefill_state(w, prompt[:start], st)
    out = prefill_state(w, prompt[:stop], st, keep_at=keep_at)
    return (out, None) if keep_at is None else out, st


# (length, start, stop, keep_at): fresh and resumed starts, keeps at the start, inside, a token early and the end
CASES = [(1, 0, 1, 1), (5, 0, 5, 4), (37, 0, 37, None), (64, 0, 64, 63), (65, 30, 65, 64), (130, 0, 130, 130),
         (90, 0, 50, None), (90, 50, 90, 89), (200, 128, 200, 128), (17, 0, 17, 0), (300, 1, 300, 150)]


def test_pieces_together_equal_each_alone(w, fp8):
    prompts = [_prompt(n, seed) for seed, (n, *_) in enumerate(CASES)]
    alone = [_alone(w, p, start, stop, keep) for p, (_, start, stop, keep) in zip(prompts, CASES)]
    sts = []
    for p, (_, start, _, _) in zip(prompts, CASES):
        st = State(w)
        if start:
            prefill_state(w, p[:start], st)
        sts.append(st)
    got = prefill_batch(w, [Piece(p[:stop], st, keep) for p, st, (_, _, stop, keep) in zip(prompts, sts, CASES)])
    for ((normed, kept), st_alone), st, (normed_b, kept_b, snap), case in zip(alone, sts, got, CASES):
        _same(st, st_alone)
        assert torch.equal(normed_b, normed), case
        assert snap is None and (kept is None) == (kept_b is None), case
        if kept is not None:
            _same(kept_b[0], kept[0])
            assert kept_b[1] is None


@pytest.mark.parametrize("order", [0, 1, 2])
def test_any_mix_of_pieces_gives_the_same_bits(w, order):
    """Three prompts in pieces batched three ways (alone, two together, all together) end in the same states."""

    prompts = [_prompt(n, 40 + n) for n in (70, 9, 150)]
    plans = [[[(0, 70)], [(0, 9)], [(0, 150)]],
             [[(0, 33), (33, 70)], [(0, 9)], [(0, 100), (100, 150)]],
             [[(0, 1), (1, 69), (69, 70)], [(0, 8), (8, 9)], [(0, 64), (64, 65), (65, 150)]]][order]
    ref = []
    for p in prompts:
        st = State(w)
        ref.append((prefill_state(w, p, st), st))
    sts = [State(w) for _ in prompts]
    last = [None] * len(prompts)
    for step in range(max(len(pl) for pl in plans)):
        batch = [(j, pl[step]) for j, pl in enumerate(plans) if step < len(pl)]
        out = prefill_batch(w, [Piece(prompts[j][:b], sts[j]) for j, (_, b) in batch])
        for (j, _), (normed, _, _) in zip(batch, out):
            last[j] = normed
    for (normed, st_ref), st, normed_b in zip(ref, sts, last):
        _same(st, st_ref)
        assert torch.equal(normed_b, normed)
