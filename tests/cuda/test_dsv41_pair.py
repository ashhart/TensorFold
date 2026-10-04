"""Two thread ranks on one GPU gather like NCCL, and sampling over two vocabulary halves equals one full-vocabulary rank."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from dsv41_pair import run_pair

from tensorfold.cuda import comm as C
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.cuda import sample as S
from tensorfold.families.glm5_next.cuda import decode as glm_decode

VOCAB = 129280                     # the checkpoint's vocabulary: 64,640 head rows per rank
HALF = VOCAB // 2
POSITIONS = [517, 518, 519, 520, 521, 522]
SAMPLINGS = [None] + [Sampling(seed=seed, temperature=t, top_k=k, top_p=p)
                      for seed in (0, 7, 2 ** 61 + 3) for t in (0.0, 1.0) for k in (0, 20) for p in (0.95, 1.0)]


def _logits(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((len(POSITIONS), VOCAB), generator=g, device="cuda") * 4.0


def _halves(fn, logits: torch.Tensor, sampling):
    """fn on two ranks, each holding its vocabulary half -> both ranks' results."""

    def rank(r):
        def go(c):
            w = SimpleNamespace(comm=c, world=2, vocab_offset=r * HALF)
            return fn(w, logits[:, r * HALF:(r + 1) * HALF].contiguous(), POSITIONS, sampling)
        return go

    return run_pair(rank(0), rank(1))


def _whole(fn, logits: torch.Tensor, sampling):
    return fn(SimpleNamespace(comm=None, world=1, vocab_offset=0), logits, POSITIONS, sampling)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32, torch.uint8])
def test_all_gather_is_cat_in_rank_order(dtype):
    sends = [(torch.arange(37, device="cuda") + 100 * r).to(dtype) for r in range(2)]

    def rank(r):
        def go(c):
            got = torch.empty(2 * 37, dtype=dtype, device="cuda")
            C.fast_gather(c, sends[r], got)                 # PairComm has no fast path: its all_gather
            return got
        return go

    a, b = run_pair(rank(0), rank(1))
    want = torch.cat(sends)
    assert torch.equal(a, want) and torch.equal(b, want)


def test_run_pair_raises_the_failing_rank_and_frees_the_other():
    def gathers(c):
        x = torch.zeros(4, device="cuda")
        c.all_gather(x, torch.empty(8, device="cuda"))

    def fails(c):
        raise KeyError("rank 1")

    with pytest.raises(KeyError, match="rank 1"):
        run_pair(gathers, fails)
    with pytest.raises(KeyError, match="rank 1"):
        run_pair(fails, gathers)


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=str)
def test_target_rows_on_two_halves_equal_the_full_vocabulary(sampling):
    logits = _logits(sampling.seed if sampling else 1)
    want, host = _whole(S.target_rows, logits, sampling)
    (a, ta), (b, tb) = _halves(S.target_rows, logits, sampling)
    assert a == want and b == want
    assert min(host, ta, tb) >= 0.0


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=str)
def test_draft_rows_agree_on_both_ranks(sampling):
    logits = _logits(sampling.seed if sampling else 1)
    want, _ = _whole(S.draft_rows, logits, sampling)
    (a, _), (b, _) = _halves(S.draft_rows, logits, sampling)
    assert a == want and b == want
    if sampling is None or sampling.top_k:              # greedy and top_k > 0: a draft is the target's draw
        assert a == _whole(S.target_rows, logits, sampling)[0]


def test_top_k_off_drafts_never_read_whole_shards(monkeypatch):
    def whole_shards(*args, **kwargs):
        raise AssertionError("the nucleus path reads whole shards")

    monkeypatch.setattr(glm_decode, "nucleus_rows", whole_shards)
    logits = _logits(5)
    sampling = Sampling(seed=5, temperature=1.0, top_k=0, top_p=1.0)
    with pytest.raises(AssertionError, match="whole shards"):       # the target does take that path
        _whole(S.target_rows, logits, sampling)
    (a, _), (b, _) = _halves(S.draft_rows, logits, sampling)
    keyed = _whole(S.target_rows, logits, Sampling(seed=5, temperature=1.0, top_k=S.DRAFT_TOP_K, top_p=1.0))[0]
    assert a == b == keyed
