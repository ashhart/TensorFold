"""Flash Next forward on CUDA with small random weights (the real head sizes, two layers, 64 experts, an MTP
head): windows give serial steps' bits, commits of a window prefix continue like serial decoding, CUDA
graphs replay the eager bits, and MTP-drafted decoding emits serial decoding's tokens."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import (  # noqa: E402
    AttnW, Config, GDNW, HC, LayerW, MoEW, MTPW, Weights)

DEV = "cuda"
D, S, LOW, E, W, V = 1024, 4, 320, 64, 128, 4096


def _cfg() -> Config:
    return Config(hidden=D, layers=2, layer_types=["linear", "attention"], vocab=V, eps=1e-6, heads=24, kv_heads=2,
                  head_dim=256, rope_theta=1e7, rotary_dim=64, nk=16, nv=48, dk=128, dv=128, conv_kernel=4,
                  experts=E, top_k=10, moe_width=W, shared_width=W, streams=S, low=LOW, index_heads=4,
                  index_dim=128, index_budget=2048, index_ratio=4, ple_layers=[], ple_dim=D, ple_kernel=4,
                  ngram_size=3, heads_per_ngram=8, ngram_base=1000, ngram_divisor=128, ngram_shards=1, seed=1,
                  ple_eos=0, eos=(0,), group_size=32, bits=4)


class _Rand:
    def __init__(self, seed: int) -> None:
        self.g = torch.Generator(device=DEV).manual_seed(seed)

    def mlx(self, n: int, k: int, lead: tuple = (), scale: float = 0.02):
        words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=self.g, device=DEV,
                              dtype=torch.int64).to(torch.int32)
        s = (torch.rand((*lead, n, k // 32), generator=self.g, device=DEV) * scale / 8 + scale / 64).to(torch.bfloat16)
        b = (-(s.float() * 7.5)).to(torch.bfloat16)                  # centred: values in about [-scale, scale]
        return words, s, b

    def q4(self, n: int, k: int, scale: float = 0.02) -> qmm.Q4:
        return qmm.make_q4(*self.mlx(n, k, scale=scale))

    def norm(self, n: int) -> torch.Tensor:
        return (1 + 0.05 * torch.randn((n,), generator=self.g, device=DEV)).float()

    def hc(self, inject: bool) -> HC:
        down = qmm.stack_q4([self.mlx(LOW, S * D)] + ([self.mlx(S, S * D)] if inject else []))
        return HC(down, self.q4(S * D, LOW), self.norm(S * D), inject)

    def moe(self) -> MoEW:
        router = (torch.randn((E + 1, D), generator=self.g, device=DEV) * 0.05).to(torch.bfloat16)
        ex = qmm.make_experts(self.mlx(W, D, (E,)), self.mlx(W, D, (E,)), self.mlx(D, W, (E,)),
                              (self.mlx(W, D), self.mlx(W, D), self.mlx(D, W)))
        return MoEW(router, ex)

    def attention(self, c: Config) -> AttnW:
        n = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
        return AttnW(self.q4(n, D), self.norm(c.head_dim), self.norm(c.head_dim), self.norm(c.index_dim),
                     self.norm(c.index_dim), self.q4(D, c.heads * c.head_dim))

    def gdn(self, c: Config) -> GDNW:
        pw = c.conv_dim + c.nv * c.dv + 2 * c.nv
        conv = (torch.randn((c.conv_dim, 4), generator=self.g, device=DEV) * 0.3).to(torch.bfloat16)
        a_log = torch.randn((c.nv,), generator=self.g, device=DEV) * 0.5
        dt = torch.randn((c.nv,), generator=self.g, device=DEV) * 0.5
        return GDNW(self.q4(pw, D), conv, a_log, dt, self.norm(c.dv).to(torch.bfloat16), self.q4(D, c.nv * c.dv))


def _model(seed: int = 3) -> Weights:
    c = _cfg()
    r = _Rand(seed)
    layers = [LayerW(0, True, r.hc(True), r.hc(True), r.gdn(c), None, r.moe()),
              LayerW(1, False, r.hc(True), r.hc(True), None, r.attention(c), r.moe())]
    embed = r.mlx(V, D, scale=0.5)
    inv = (c.rope_theta ** (-torch.arange(0, 32, dtype=torch.float64) / 32)).float().to(DEV)
    w = Weights(c, embed, layers, r.hc(False), r.q4(V, D, scale=0.2), inv)
    w.mtp = MTPW(r.norm(D), r.norm(S * D), r.q4(D, D), r.q4(D, D),
                 LayerW(-1, False, r.hc(True), r.hc(True), None, r.attention(c), r.moe()), r.hc(False))
    return w


def test_windows_match_serial_steps_and_prefix_commits_continue():
    w = _model()
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12]
    forward(w, e.st, e.buf, prompt)
    commit(w, e.st, e.buf, len(prompt), len(prompt))
    nxt = [401, 33, 2048, 5, 77, 1500, 9, 10, 11]
    serial = e.st.clone()
    logits, streams = [], []
    for t in nxt:
        lg = forward(w, serial, e.buf, [t])
        logits.append(lg[0].clone())
        streams.append(e.buf.streams[0].clone())
        commit(w, serial, e.buf, 1, 1)
    for R in (2, 3, 4, 8):
        for keep in sorted({1, max(1, R // 2), R}):
            st = e.st.clone()
            lg = forward(w, st, e.buf, nxt[:R])
            for r in range(R):
                assert torch.equal(lg[r], logits[r]), (R, r)
                assert torch.equal(e.buf.streams[r], streams[r]), (R, r)
            commit(w, st, e.buf, R, keep)
            assert torch.equal(forward(w, st, e.buf, [nxt[keep]])[0], logits[keep]), (R, keep)


@pytest.mark.parametrize("sampling", [None, Sampling(seed=1234, top_k=20, top_p=0.95)])
def test_graphs_and_mtp_drafts_give_serial_tokens(sampling):
    w = _model()
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    eager = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    first = prefill(eager, prompt, sampling)
    ref = serial_decode(eager, first, 24, sampling).tokens
    graphs = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=True)
    assert prefill(graphs, prompt, sampling) == first
    assert serial_decode(graphs, first, 24, sampling).tokens == ref
    for depth in (1, 2, 3):
        for e in (eager, graphs):
            prefill(e, prompt, sampling)
            got = mtp_decode(e, first, 24, sampling, depth=depth, confidence=0.0)
            assert got.tokens == ref, (depth, e.graphs is not None)


@pytest.mark.parametrize("sampling", [None, Sampling(seed=99, top_k=20, top_p=0.95)])
def test_a_draft_vocabulary_changes_speed_only(sampling):
    """Drafts scored over a token subset (every other id) still give serial decoding's tokens."""

    w = _model()
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=True)
    first = prefill(e, prompt, sampling)
    ref = serial_decode(e, first, 24, sampling).tokens
    words, scales, biases = qmm.to_mlx(w.head)
    ids = torch.arange(0, V, 2, device=DEV)
    w.draft_ids = ids
    w.draft_head = qmm.make_q4(words[ids], scales[ids], biases[ids])
    e2 = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=True)
    for depth in (2, 4):
        prefill(e2, prompt, sampling)
        assert mtp_decode(e2, first, 24, sampling, depth=depth, confidence=0.0).tokens == ref


@pytest.mark.parametrize("sampling", [None, Sampling(seed=5, top_k=20, top_p=0.95)])
@pytest.mark.parametrize("vocab", [False, True])
def test_confidence_stopped_chains_give_serial_tokens(sampling, vocab):
    """Chains that end before a low-probability draft (the head's softmax at temperature 1) change speed only."""

    w = _model()
    if vocab:
        words, scales, biases = qmm.to_mlx(w.head)
        ids = torch.arange(1, V, 3, device=DEV)
        w.draft_ids = ids
        w.draft_head = qmm.make_q4(words[ids], scales[ids], biases[ids])
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=True)
    first = prefill(e, prompt, sampling)
    ref = serial_decode(e, first, 24, sampling).tokens
    drafted = {}
    for conf in (0.0, 0.0005, 0.002, 0.9):
        prefill(e, prompt, sampling)
        got = mtp_decode(e, first, 24, sampling, depth=5, confidence=conf)
        assert got.tokens == ref, conf
        drafted[conf] = got.drafted
    assert drafted[0.9] == 0 and drafted[0.0] > 0


def test_server_engine_streams_serial_tokens(tmp_path):
    """engine.FlashNextEngine on one GPU: the streamed tokens are serial decoding's, and a client that stops
    early stops the decode."""

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    from test_flashnext_tp import _checkpoint

    _checkpoint(tmp_path)
    eng = FlashNextEngine(tmp_path, depth=5, confidence=0.001, draft_vocab=None, max_len=512, prefetch=False)
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    for sampling in (None, Sampling(seed=7, top_k=20, top_p=0.95)):
        first = prefill(eng.e, prompt, sampling)
        ref = serial_decode(eng.e, first, 30, sampling).tokens
        got: list[int] = []
        stats = eng.generate(prompt, 30, sampling, lambda new: got.extend(new))
        eos = [i for i, t in enumerate(ref) if t in eng.eos]
        want = ref[:eos[0] + 1] if eos else ref
        assert got == want and "prefill_s" in stats, (got, want)
        seen: list[int] = []
        eng.generate(prompt, 30, sampling, lambda new: (seen.extend(new), len(seen) >= 3)[1])
        assert seen[:3] == want[:3] and len(seen) < len(want) + 1


@pytest.mark.parametrize("sampling", [None, Sampling(seed=11, top_k=20, top_p=0.95)])
def test_long_prefill_chunks_match_short_chunks(sampling):
    """A 64-row prefill chunk (the server's) gives the bits of 16-row chunks, and drafting after it emits serial
    tokens. The MTP head absorbs a chunk's rows through fc_hidden as rows x 4 streams, past one 128-row tile."""

    w = _model()
    prompt = [(37 * i + 11) % V for i in range(90)]
    short = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=True)
    long = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=True)
    first = prefill(short, prompt, sampling)
    ref = serial_decode(short, first, 20, sampling).tokens
    assert prefill(long, prompt, sampling) == first
    assert serial_decode(long, first, 20, sampling).tokens == ref
    prefill(long, prompt, sampling)
    assert mtp_decode(long, first, 20, sampling, depth=4, confidence=0.0).tokens == ref


@pytest.mark.parametrize("sampling", [None, Sampling(seed=21, top_k=20, top_p=0.95)])
def test_the_family_hook_serves_the_recipe(tmp_path, sampling):
    """``cuda_engine``, what ``tensorfold serve`` calls, builds the measured recipe (up to 6 drafts, the 30% stop,
    the packaged draft vocabulary, an 8,192-token context) and streams serial decoding's tokens; with drafts off
    it decodes one token a round and streams the same tokens."""

    from tensorfold.families.qwen4_exp import cuda_engine
    from tensorfold.families.qwen4_exp.cuda import CONFIDENCE, CONTEXT, DEPTH

    from test_flashnext_tp import _checkpoint

    assert (DEPTH, CONFIDENCE, CONTEXT) == (6, 0.3, 8192)
    _checkpoint(tmp_path)
    eng = cuda_engine(tmp_path)
    assert (eng.depth, eng.confidence, eng.max_len, eng.tp) == (6, 0.3, 8192, 1)
    assert eng.w.draft_ids is not None
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    first = prefill(eng.e, prompt, sampling)
    ref = serial_decode(eng.e, first, 30, sampling, stop_eos=True).tokens
    got: list[int] = []
    eng.generate(prompt, 30, sampling, lambda new: got.extend(new))
    assert got == ref
    serial = cuda_engine(tmp_path, no_drafts=True, context=1024)
    assert (serial.depth, serial.max_len) == (0, 1024) and serial.w.mtp is None
    plain: list[int] = []
    serial.generate(prompt, 30, sampling, lambda new: plain.extend(new))
    assert plain == ref
    with pytest.raises(ValueError):
        cuda_engine(tmp_path, drafter="some/draft-model")


@pytest.mark.parametrize("sampling", [None, Sampling(seed=31, top_k=20, top_p=0.95)])
def test_prefix_reuse_and_the_serial_switch(tmp_path, sampling):
    """A prompt that extends the last request's reply or prompt resumes from the kept state and decodes what a
    fresh prefill of it decodes; ``draft=False`` decodes the same tokens one a round and leaves the kept states."""

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    from test_flashnext_tp import _checkpoint

    _checkpoint(tmp_path)
    eng = FlashNextEngine(tmp_path, depth=4, confidence=0.001, draft_vocab=None, max_len=1024, prefetch=False)
    first = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]

    def ask(prompt, **kw):
        got: list[int] = []
        stats = eng.generate(prompt, 16, sampling, lambda new: got.extend(new), **kw)
        return got, stats

    reply, stats = ask(first)
    assert stats["cached"] == 0
    for extend in ("reply", "prompt"):
        if extend == "prompt":
            ask(first)                                           # the first request's states again
        prompt = first + (reply if extend == "reply" else []) + [401, 33, 2048]
        warm, warm_stats = ask(prompt)
        # the reply's kept state holds every token but the pending last one (all of them when the last round
        # kept one past the limit)
        want = len(first) + len(reply) - 1 if extend == "reply" else len(first)
        assert warm_stats["cached"] in (want, want + (extend == "reply")), (extend, warm_stats)
        serial, serial_stats = ask(prompt, draft=False)          # one token a round, a fresh prefill
        assert serial == warm and serial_stats["drafts"] is False and serial_stats["cached"] == 0
        again, again_stats = ask(prompt + [9])                   # the kept states survived the serial request
        assert again_stats["cached"] >= len(prompt)
        ask([1500, 9, 10])                                       # an unrelated prompt: nothing to resume from
        cold, cold_stats = ask(prompt)
        assert cold_stats["cached"] == 0 and cold == warm, extend
