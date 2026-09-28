"""ignore_eos on the 27B: replies decoded past end tokens equal serial decoding, alone and beside streams that stop."""

import dataclasses
import random
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.scheduler import Scheduler  # noqa: E402
from tensorfold.cuda.streams import PrefixCache, Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import draft_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE, Qwen27Engine  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import State  # noqa: E402
from tensorfold.families.qwen3_5.cuda.multi import TREE, MultiDecoder  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402

V = 256
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]


def _model():
    """A GDN and an attention layer as in test_qwen27_multi.py, with weights centred on zero so greedy paths vary."""

    gen = torch.Generator(device="cuda").manual_seed(11)
    dev = "cuda"

    def qlinear(n, k):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen,
                              device=dev, dtype=torch.int64).to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 + 0.001).bfloat16()
        noise = torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 - 0.0015
        biases = (noise - 7.5 * scales.float()).bfloat16()   # 4-bit codes average 7.5: the weights average 0
        return QLinear(words, scales, biases)

    norm = torch.ones(128, device=dev, dtype=torch.bfloat16)
    gdn = GDN(qlinear(384, 128), qlinear(128, 128), qlinear(1, 128), qlinear(1, 128),
              qlinear(128, 128), torch.randn(384, 4, generator=gen, device=dev).bfloat16() * 0.1,
              torch.zeros(1, device=dev), torch.zeros(1, device=dev), norm)
    attn = Attention(qlinear(2 * 2 * 128, 128), qlinear(128, 128), qlinear(128, 128), qlinear(128, 2 * 128),
                     norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, qlinear(128, 128), qlinear(128, 128), qlinear(128, 128)),
              Layer(False, norm, norm, None, attn, qlinear(128, 128), qlinear(128, 128), qlinear(128, 128))]
    config = Config(hidden=128, intermediate=128, layers=2, heads=2, kv_heads=1,
                    head_dim=128, vocab=V, k_heads=1, v_heads=1, dk=128, dv=128,
                    conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    return Weights(config, qlinear(V, 128), layers, norm, qlinear(V, 128), torch.ones(16, device=dev))


def _deep():
    """64 copies of the GDN layer, so a drafter's taps can be captured (as the decode trace test builds it)."""

    base = _model()
    return Weights(dataclasses.replace(base.config, layers=64), base.embed, [base.layers[0]] * 64, base.norm,
                   base.head, base.inv_freq)


def _with_eos(w, eos):
    """The same weights with other end tokens: stopping changes, the arithmetic does not."""

    return dataclasses.replace(w, config=dataclasses.replace(w.config, eos=tuple(eos)))


def _crossing(ref):
    """An end token for this path: one it first produces after its first token and before its last."""

    for lo in (4, 1):
        for token in ref[lo:-1]:
            if ref.index(token) >= lo:
                return token, ref.index(token)
    pytest.fail(f"the fixture's path {ref} has no token to use as an end token")


class _Truth:
    """Drafts the reference's next ``depth`` tokens as a chain, and a wrong sibling of the first."""

    def __init__(self, ref, depth):
        self.ref, self.depth = ref, depth

    def propose_tree(self, pending, context_length, max_nodes, sampling):
        k = context_length                          # the prompt is empty: the context is the reply so far
        chain = [self.ref[k + j] if k + j < len(self.ref) else 1 for j in range(self.depth)]
        guesses = chain + [(chain[0] + 1) % V]
        return guesses[:max_nodes], (list(range(-1, self.depth - 1)) + [-1])[:max_nodes]

    def add_taps(self, taps):
        pass


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_drafted_equals_serial_past_an_end_token(sampling):
    base = _deep()
    ref = serial_decode(base, State(base), 5, 40, sampling, stop_eos=False).tokens
    end, at = _crossing(ref)
    w = _with_eos(base, (end,))
    assert serial_decode(w, State(w), 5, 40, sampling).tokens == ref[:at + 1]      # the path crosses it
    assert serial_decode(w, State(w), 5, 40, sampling, stop_eos=False).tokens == ref
    inside = False
    for depth in (2, 3, 4, 5):                   # rounds of up to 3 to 6 tokens: some round holds the end token
        rounds: list[list[int]] = []
        drafted = draft_decode(w, State(w), [], 5, 40, sampling, _Truth(ref, depth), max_rows=8, allow_copy=False,
                               stop_eos=False, on_tokens=lambda new: rounds.append(list(new)) or False)
        assert drafted.tokens == ref and [5] + [t for r in rounds for t in r] == ref, depth
        assert max(drafted.widths) > 1
        inside |= any(end in r[:-1] for r in rounds)        # accepted as a draft, with tokens after it
        stopped = draft_decode(w, State(w), [], 5, 40, sampling, _Truth(ref, depth), max_rows=8, allow_copy=False)
        assert stopped.tokens == ref[:at + 1], depth
    assert inside


def _engine(w):
    """The 27B engine on one GPU without its loader: no drafter, copy proposals on."""

    eng = Qwen27Engine.__new__(Qwen27Engine)
    eng.torch, eng.tp, eng.rank, eng.max_rows, eng.allow_copy = torch, 1, 0, 12, True
    eng.w, eng.draft, eng.cache, eng.multi, eng.scheduler = w, None, PrefixCache(KEEP_ONE), None, None
    eng.points = None
    eng.context_window, eng.eos, eng.concurrent = 4096, tuple(w.config.eos), False
    return eng


def _reference(w, prompt, sampling, count, stop_eos=True):
    st, first = prefill(w, prompt, sampling)
    return serial_decode(w, st, first, count, sampling, stop_eos=stop_eos).tokens


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_engine_decodes_past_end_tokens_and_keeps_only_prompt_ends(sampling):
    base = _model()
    first, longer = [9, 10, 11, 12, 13], [9, 10, 11, 12, 13, 14, 15, 16]
    ends = [_crossing(_reference(base, p, sampling, 32, stop_eos=False))[0] for p in (first, longer)]
    w = _with_eos(base, ends)
    refs = {(tuple(p), stop): _reference(w, p, sampling, 32, stop_eos=stop)
            for p in (first, longer) for stop in (True, False)}
    assert all(len(refs[(tuple(p), True)]) < 32 == len(refs[(tuple(p), False)]) for p in (first, longer))
    eng = _engine(w)
    # drafted from a fresh prefill, then again from its own entry (one token early); the serial reference from
    # scratch; then a longer prompt from the first one's entry, and again from its own
    cached = {(True, False): 0, (True, True): len(first) - 1, (False, False): 0, (False, True): 0}
    for prompt, draft in ((first, True), (first, False), (longer, True)):
        for stop_eos in (False, True):
            got: list[int] = []
            stats = eng.generate(prompt, 32, sampling, lambda new: got.extend(new) or False, draft=draft,
                                 stop_eos=stop_eos)
            assert got == refs[(tuple(prompt), stop_eos)], (prompt, draft, stop_eos)
            want = cached[(draft, stop_eos)] if prompt is first else (len(first) - 1, len(longer) - 1)[stop_eos]
            assert stats["cached"] == want, (prompt, draft, stop_eos)
    # the kept states end one token before their prompts: no token past an end token (or any reply token) entered
    assert [(ids, st.pos) for ids, st, _ in eng.cache.entries] == [(first[:-1], len(first) - 1),
                                                                   (longer[:-1], len(longer) - 1)]


class _Oracle(MultiDecoder):
    """Trees mixing each prompt's true continuation with wrong tokens (as tests/cuda/test_qwen27_multi.py)."""

    def __init__(self, w, truth, seed=0, **kw):
        super().__init__(w, None, max_rows=6, **kw)
        self.truth, self.rng = truth, random.Random(seed)

    def _mode(self, s, copied):
        mode = super()._mode(s, copied)
        return TREE if s.draft and not copied.get(s.sid) else mode

    def _trees(self, plan, blocks):
        out = {}
        for sid, mode, _, _ in plan:
            if mode != TREE:
                continue
            s = self.streams[sid]
            ref, n = self.truth[tuple(s.prompt)], len(s.out)
            guesses, parents = [], []
            for d in range(self.rng.randint(1, 4)):
                guesses.append(ref[n + d] if n + d < len(ref) and self.rng.random() < 0.8 else self.rng.randrange(1, V))
                parents.append(d - 1)
            if self.rng.random() < 0.5:
                guesses.append(self.rng.randrange(1, V))
                parents.append(-1)
            out[sid] = (guesses, parents, [0.3 * (k + 1) for k in range(len(guesses))])
        return out


PROMPTS = [[5, 6, 7], [9, 10, 11, 12, 13], [3, 4]]
PAIRS = list(zip(PROMPTS, SAMPLINGS * 2))


def _setup(count):
    """Weights whose end tokens every prompt's path crosses; references that stop at them and that do not."""

    base = _model()
    full = {tuple(p): _reference(base, p, s, count, stop_eos=False) for p, s in PAIRS}
    w = _with_eos(base, [_crossing(ref)[0] for ref in full.values()])
    refs = {(tuple(p), stop): _reference(w, p, s, count, stop_eos=stop) for p, s in PAIRS for stop in (True, False)}
    for p in full:
        assert len(refs[(p, True)]) < count and refs[(p, False)] == full[p]
    return w, full, refs


@pytest.mark.parametrize("ignoring", [0, 1, 2])
def test_concurrent_streams_one_ignoring_eos_equal_their_serial_references(ignoring):
    w, full, refs = _setup(24)
    dec = _Oracle(w, full)
    streams = []
    for i, (prompt, sampling) in enumerate(PAIRS):
        got: list[int] = []
        s = Stream(prompt, 24, sampling, draft=i != 2, stop_eos=i != ignoring,
                   emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        streams.append((s, got))
    dec.finish([s for s, _ in streams if s.done])       # a first token that is an end token ends its stream here
    while dec.live():
        dec.finish(dec.round())
    for s, got in streams:
        assert got == refs[(tuple(s.prompt), s.stop_eos)] and s.out == got, (s.prompt, s.stop_eos)
        if s.draft and not s.stop_eos:                   # its drafted rounds kept runs through the end tokens
            assert s.rounds < len(got) - 1, (s.prompt, s.rounds)


def test_scheduler_serves_a_stream_ignoring_eos_beside_ones_that_stop():
    w, full, refs = _setup(20)
    sched = Scheduler(_Oracle(w, full, seed=3), max_streams=3)
    results: dict[int, list[int]] = {}

    def go(i, stop_eos):
        got: list[int] = []
        sched.submit(PROMPTS[i], 20, PAIRS[i][1], draft=True, emit=lambda new: got.extend(new) or False,
                     stop_eos=stop_eos)
        results[i] = got

    threads = [threading.Thread(target=go, args=(i, i != 1)) for i in range(len(PROMPTS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert results == {i: refs[(tuple(PROMPTS[i]), i != 1)] for i in range(len(PROMPTS))}
