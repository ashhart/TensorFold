"""Flash Next forward on CUDA with small random weights (the real head sizes, two layers, 64 experts, an MTP
head): windows give serial steps' bits, commits of a window prefix continue like serial decoding, CUDA
graphs replay the eager bits, and MTP-drafted decoding emits serial decoding's tokens."""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import (  # noqa: E402
    AttnW, Config, GDNW, HC, LayerW, MoEW, MTPW, PLEW, Weights)
from tensorfold.families.qwen4_exp.host_table import BF16Table, read_header  # noqa: E402

DEV = "cuda"
D, S, LOW, E, W, V = 1024, 4, 320, 64, 128, 4096


def _cfg(ple: bool = False) -> Config:
    return Config(hidden=D, layers=2, layer_types=["linear", "attention"], vocab=V, eps=1e-6, heads=24, kv_heads=2,
                  head_dim=256, rope_theta=1e7, rotary_dim=64, nk=16, nv=48, dk=128, dv=128, conv_kernel=4,
                  experts=E, top_k=10, moe_width=W, shared_width=W, streams=S, low=LOW, index_heads=4,
                  index_dim=128, index_budget=2048, index_ratio=4, ple_layers=[1] if ple else [], ple_dim=D,
                  ple_kernel=4, ngram_size=3, heads_per_ngram=8, ngram_base=1000, ngram_divisor=128,
                  ngram_shards=1, seed=1, ple_eos=0, eos=(0,), group_size=32, bits=4)


def _bf16_table(path: Path, rows: int, dims: int, seed: int = 7) -> BF16Table:
    """A one-shard n-gram table in the layout the published NVFP4 revision ships: plain bf16 rows."""

    g = torch.Generator().manual_seed(seed)
    values = ((torch.rand((rows, dims), generator=g) - 0.5) * 0.02).to(torch.bfloat16)
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    blob = json.dumps({name: {"dtype": "BF16", "shape": [rows, dims], "data_offsets": [0, rows * dims * 2]}})
    blob = blob.encode() + b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(values.view(torch.uint16).numpy().tobytes())
    return BF16Table([(path, read_header(path)[name])])


def _ple(c: Config, table: BF16Table, r: "_Rand") -> PLEW:
    """The PLE layer's faces over ``table``: the key/value and the norms the engine's gate and conv read."""

    conv = (torch.randn((S * D, c.ple_kernel), generator=r.g, device=DEV) * 0.3).to(torch.bfloat16)
    return PLEW(table, r.q4(S * D, c.ple_dim), r.q4(D, c.ple_dim), r.norm(S * D), r.norm(S * D), r.norm(S * D),
                conv, c.ngram(0))


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
        parts = [self.mlx(LOW, S * D)] + ([self.mlx(S, S * D)] if inject else [])
        up = self.mlx(S * D, LOW)
        return HC(qmm.stack_q4(parts, "tiled"), qmm.make_q4(*up, "tiled"), self.norm(S * D), inject,
                  qmm.stack_q4(parts, "frag"), qmm.make_q4(*up, "frag"))

    def moe(self) -> MoEW:
        router = (torch.randn((E + 1, D), generator=self.g, device=DEV) * 0.05).to(torch.bfloat16)
        def table(routed, shared):
            return tuple(torch.cat([a, b[None]]) for a, b in zip(routed, shared))

        ex = grouped.make([table(self.mlx(W, D, (E,)), self.mlx(W, D)), table(self.mlx(W, D, (E,)), self.mlx(W, D))],
                          table(self.mlx(D, W, (E,)), self.mlx(D, W)), 32)
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


def _model(seed: int = 3, ple: PLEW | None = None) -> Weights:
    c = _cfg(ple is not None)
    r = _Rand(seed)
    layers = [LayerW(0, True, r.hc(True), r.hc(True), r.gdn(c), None, r.moe()),
              LayerW(1, False, r.hc(True), r.hc(True), None, r.attention(c), r.moe())]
    if ple is not None:
        layers[1].ple = ple
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
    prefill(e, prompt, None)
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


def test_a_bf16_ngram_table_holds_the_same_window_contract():
    """The published revision's n-gram table is plain bf16 rows, no per-shard scales: with the PLE layer live,
    a window's logits still equal serial steps' bits, the rows gathered host-side and staged to the device."""

    c = _cfg(ple=True)
    r = _Rand(3)
    with tempfile.TemporaryDirectory() as tmp:
        table = _bf16_table(Path(tmp) / "shard_0.safetensors", c.ngram(0).rows, c.ngram(0).dims)
        assert table.rows == c.ngram(0).rows and table.width == c.ngram(0).dims
        w = _model(ple=_ple(c, table, r))
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
        prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12]
        prefill(e, prompt, None)
        nxt = [401, 33, 2048, 5, 77, 1500, 9, 10, 11]
        serial = e.st.clone()
        logits = []
        for t in nxt:
            logits.append(forward(w, serial, e.buf, [t])[0].clone())
            commit(w, serial, e.buf, 1, 1)
        for R in (2, 3, 4, 8):
            st = e.st.clone()
            lg = forward(w, st, e.buf, nxt[:R])
            for i in range(R):
                assert torch.equal(lg[i], logits[i]), (R, i)


@pytest.mark.parametrize("sampling", [None, Sampling(seed=1234, top_k=20, top_p=0.95),
                                      Sampling(seed=1234, top_k=20, top_p=0.95, min_p=0.1)])
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
    """Chains that end before a low-probability draft (the head's softmax at temperature 1) change speed only, and
    every round still verifies the pending token and at least one draft: no round decodes one token."""

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
    drafted, rounds = {}, {}
    for conf in (0.0, 0.0005, 0.002, 0.9):
        prefill(e, prompt, sampling)
        got = mtp_decode(e, first, 24, sampling, depth=5, confidence=conf)
        assert got.tokens == ref, conf
        assert min(got.widths) >= 2 and len(got.widths) == got.rounds, (conf, got.widths)
        drafted[conf], rounds[conf] = got.drafted, got.rounds
    assert drafted[0.9] == rounds[0.9] and drafted[0.0] > drafted[0.9]      # at 0.9: the first draft alone


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


def _rows(cache, n: int) -> list[torch.Tensor]:
    """A KV cache's first n positions: keys and values, and their scales when quantized."""

    return [t[:n] for t in ((cache.k, cache.v, cache.ks, cache.vs) if cache.quantized else (cache.k, cache.v))]


def _state(e: Engine) -> list[torch.Tensor]:
    """What a prefill leaves: the kept-state snapshot, the caches' rows below the position, the MTP cache's."""

    st = e.st
    snap = st.snapshot()
    out = [snap["rec"], snap["conv"], snap["ple_tail"], e.last_streams]
    out += [x for c in st.kc for x in _rows(c, st.pos)] + [k[:st.pos] for k in st.ikc] + _rows(st.mtp_kc, st.mtp_len)
    return out


def test_prefill_tracks_the_decode_path():
    """Prompt chunks and decode windows agree to bf16 rounding: DeltaNet states, attention keys, the last logits."""

    w = _model()
    prompt = [(29 * i + 3) % V for i in range(40)]
    pre = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    prefill(pre, prompt, None, mtp=False)                     # the head would reuse the logits buffer
    dec = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    dec.reset()
    for s0 in range(0, len(prompt), 8):
        chunk = prompt[s0:s0 + 8]
        logits = forward(w, dec.st, dec.buf, chunk)[len(chunk) - 1:len(chunk)].float()
        commit(w, dec.st, dec.buf, len(chunk), len(chunk))
    a, b = pre.st.snapshot()["rec"], dec.st.snapshot()["rec"]
    assert float((a - b).abs().max()) <= 2e-2 * float(b.abs().max())
    ka, kb = pre.st.kc[0].k[:len(prompt)].float(), dec.st.kc[0].k[:len(prompt)].float()
    assert float((ka - kb).abs().max()) <= 2e-2 * float(kb.abs().max())
    cos = torch.nn.functional.cosine_similarity(pre.pbuf.logits[:1].float(), logits, dim=1)
    assert float(cos) > 0.999, float(cos)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=11, top_k=20, top_p=0.95)])
def test_prefill_chunks_and_resumes_give_the_same_state(sampling, kv_dtype):
    """Any chunking, or a resume from another prompt's end, leaves the same state bit for bit; drafts stay serial."""

    w = _model()
    prompt = [(37 * i + 11) % V for i in range(300)]
    ref_e = Engine(w, capacity=1024, max_rows=8, prefill_rows=300, graphs=True, kv_dtype=kv_dtype)
    first = prefill(ref_e, prompt, sampling)
    want = _state(ref_e)
    ref = serial_decode(ref_e, first, 20, sampling).tokens
    for rows in (7, 16, 64):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=rows, graphs=True, kv_dtype=kv_dtype)
        assert prefill(e, prompt, sampling) == first, rows
        assert all(torch.equal(a, b) for a, b in zip(_state(e), want)), rows
        assert mtp_decode(e, first, 20, sampling, depth=4, confidence=0.0).tokens == ref, rows
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=True, kv_dtype=kv_dtype)
    prefill(e, prompt[:131], sampling)
    kept = {"state": e.st.snapshot(), "tail": e.last_streams.clone()}
    serial_decode(e, 5, 9, sampling)                             # a reply decodes past the kept prompt
    assert prefill(e, prompt, sampling, resume=kept) == first
    assert all(torch.equal(a, b) for a, b in zip(_state(e), want))
    assert mtp_decode(e, first, 20, sampling, depth=4, confidence=0.0).tokens == ref


@pytest.mark.parametrize("ple", [False, True])
@pytest.mark.parametrize("mtp", [True, False])
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=13, top_k=20, top_p=0.95)])
def test_a_point_kept_one_token_early_resumes_the_same_prompt_and_a_next_turn(tmp_path, sampling, kv_dtype, mtp,
                                                                             ple):
    """``keep_at`` keeps a fresh prefill of prompt[:keep_at]; the same prompt and a next turn resume from it as fresh."""

    if ple:
        c = _cfg(ple=True)
        table = _bf16_table(tmp_path / "shard_0.safetensors", c.ngram(0).rows, c.ngram(0).dims)
        w = _model(ple=_ple(c, table, _Rand(3)))
    else:
        w = _model()
    prompt = [(37 * i + 11) % V for i in range(300)]
    turn = prompt[:299] + [271, 77, 78]

    def engine():
        return Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=True, kv_dtype=kv_dtype)

    def decode(e, first):
        if mtp:
            return mtp_decode(e, first, 20, sampling, depth=4, confidence=0.0).tokens
        return serial_decode(e, first, 20, sampling).tokens

    def fresh(p):
        e = engine()
        first = prefill(e, p, sampling, mtp=mtp)
        return first, _state(e), decode(e, first)

    def same(e, first, want, ref, tag):
        assert all(torch.equal(a, b) for a, b in zip(_state(e), want)), tag
        assert decode(e, first) == ref, tag

    first, want, ref = fresh(prompt)
    turn_first, turn_want, turn_ref = fresh(turn)
    e = engine()
    for keep_at in (1, 64, 100, 299, 300):
        assert prefill(e, prompt, sampling, mtp=mtp, keep_at=keep_at) == first, keep_at
        kept = e.kept
        alone = engine()
        prefill(alone, prompt[:keep_at], sampling, mtp=mtp)
        snap = alone.st.snapshot()
        assert all(torch.equal(kept["state"][k], snap[k]) for k in ("rec", "conv", "ple_tail")), keep_at
        assert (kept["state"]["pos"], kept["state"]["mtp_len"]) == (snap["pos"], snap["mtp_len"]), keep_at
        assert (kept["state"]["ple_history"] is None) == (snap["ple_history"] is None), keep_at
        assert snap["ple_history"] is None or np.array_equal(kept["state"]["ple_history"], snap["ple_history"])
        assert torch.equal(kept["tail"], alone.last_streams) if mtp else kept["tail"] is None, keep_at
        same(e, first, want, ref, keep_at)                       # a reply decodes past the kept point
        if keep_at == len(prompt):                               # the whole prompt: nothing it resumes
            continue
        assert prefill(e, prompt, sampling, mtp=mtp, resume=kept, keep_at=299) == first, keep_at
        assert (e.kept is kept) == (keep_at == 299), keep_at     # the same prompt again keeps its own point
        same(e, first, want, ref, keep_at)
        assert prefill(e, turn, sampling, mtp=mtp, resume=kept, keep_at=len(turn) - 1) == turn_first, keep_at
        same(e, turn_first, turn_want, turn_ref, keep_at)


@pytest.mark.parametrize("sampling", [None, Sampling(seed=21, top_k=20, top_p=0.95)])
def test_the_family_hook_serves_the_recipe(tmp_path, sampling):
    """``cuda_engine``, what ``tensorfold serve`` calls, builds the measured recipe (up to 6 drafts, the 30% stop,
    the packaged draft vocabulary, an 8,192-token context) and streams serial decoding's tokens; with drafts off
    it decodes one token a round and streams the same tokens."""

    from tensorfold.families.qwen4_exp import cuda_engine
    from tensorfold.families.qwen4_exp.cuda import CONFIDENCE, CONTEXT, DEPTH

    from test_flashnext_tp import _checkpoint

    assert (DEPTH, CONFIDENCE, CONTEXT) == (6, 0.7, 8192)
    _checkpoint(tmp_path)
    eng = cuda_engine(tmp_path, context=8185)                  # the synthetic checkpoint names no native window
    assert (eng.depth, eng.confidence, eng.max_len, eng.tp) == (6, 0.7, 8192, 1)
    assert eng.w.draft_ids is not None
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    first = prefill(eng.e, prompt, sampling)
    ref = serial_decode(eng.e, first, 30, sampling, stop_eos=True).tokens
    got: list[int] = []
    stats = eng.generate(prompt, 30, sampling, lambda new: got.extend(new))
    assert got == ref and stats["min_rows"] >= 2
    serial = cuda_engine(tmp_path, no_drafts=True, context=1024)
    assert (serial.depth, serial.max_len) == (0, 1025) and serial.w.mtp is None    # the window and one row
    plain: list[int] = []
    serial.generate(prompt, 30, sampling, lambda new: plain.extend(new))
    assert plain == ref
    with pytest.raises(ValueError):
        cuda_engine(tmp_path, drafter="some/draft-model")


@pytest.mark.parametrize("sampling", [None, Sampling(seed=31, top_k=20, top_p=0.95)])
def test_prefix_reuse_and_the_serial_switch(tmp_path, sampling):
    """Extending or repeating a prompt resumes as a fresh prefill; ``draft=False`` decodes the same, kept states intact."""

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
        # kept one token before a prompt's end: the reply and the first prompt's last token prefill again
        assert warm_stats["cached"] == len(first) - 1, (extend, warm_stats)
        serial, serial_stats = ask(prompt, draft=False)          # one token a round, a fresh prefill
        assert serial == warm and serial_stats["drafts"] is False and serial_stats["cached"] == 0
        again, again_stats = ask(prompt + [9])                   # the kept states survived the serial request
        assert again_stats["cached"] == len(prompt) - 1
        same, same_stats = ask(prompt + [9])                     # the same prompt again: all but its last token kept
        assert same == again and same_stats["cached"] == len(prompt), (extend, same_stats)
        third, third_stats = ask(prompt + [9])                   # and a third time: every resend hits
        assert third == again and third_stats["cached"] == len(prompt), (extend, third_stats)
        ask([1500, 9, 10])                                       # an unrelated prompt: nothing to resume from
        cold, cold_stats = ask(prompt)
        assert cold_stats["cached"] == 0 and cold == warm, extend


def _prefill_state(e: Engine) -> dict:
    st = e.st
    n, m = st.pos, st.mtp_len
    out = {"streams": e.last_streams, "rec": st.rec[st.cur[0]], "conv": st.conv,
           "mtp_kc": st.mtp_kc.k[:m], "mtp_vc": st.mtp_kc.v[:m], "mtp_ikc": st.mtp_ikc[:m]}
    for i in range(len(st.kc)):
        out[f"kc{i}"], out[f"vc{i}"], out[f"ikc{i}"] = st.kc[i].k[:n], st.kc[i].v[:n], st.ikc[i][:n]
        out[f"pooled{i}"] = st.pooled[i][:n // 4]
    return {k: v.clone() for k, v in out.items()}


@pytest.mark.parametrize("sampling", [None, Sampling(seed=17, top_k=20, top_p=0.95)])
def test_prefill_head_on_the_final_chunk_keeps_every_bit(monkeypatch, sampling):
    """Past the sparse attention budget, any chunking with the head on the final chunk alone gives the one-chunk prefill's first token and whole state."""

    from tensorfold.families.qwen4_exp.cuda import decode

    w = _model(7)
    g = torch.Generator().manual_seed(8)
    prompt = torch.randint(1, V, (2600,), generator=g).tolist()
    one = Engine(w, capacity=4096, max_rows=8, prefill_rows=len(prompt))
    first = prefill(one, prompt, sampling)
    want = _prefill_state(one)
    del one
    heads = []
    real = decode.forward
    monkeypatch.setattr(decode, "forward", lambda *a, **kw: (heads.append(kw.get("logits", True)), real(*a, **kw))[1])
    for rows in (64, 256, 1024):
        heads.clear()
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=rows)
        assert prefill(e, prompt, sampling) == first
        got = _prefill_state(e)
        assert [k for k in want if not torch.equal(want[k], got[k])] == [], rows
        assert heads == [False] * (-(-len(prompt) // rows) - 1) + [True], rows
        del e


def test_capture_is_declined_when_the_experts_cannot_be_captured():
    """An engine asked for graphs must not build them over experts that declare themselves uncapturable: the
    NVFP4 route reads its plan on the host, which a capture rejects. Decoding still works, eagerly."""

    w = _model()
    for layer in w.layers:
        layer.moe.experts.capturable = False
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=True)
    assert e.graphs is None
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
    first = prefill(e, prompt, None)
    assert len(serial_decode(e, first, 8, None).tokens) == 8


def _set_end(eng, end: int) -> None:
    """The engine's end tokens become ``end`` alone: stopping changes, the arithmetic does not."""

    import dataclasses

    eng.w.cfg = dataclasses.replace(eng.w.cfg, eos=(end,))
    eng.eos = (end,)
    if getattr(eng, "multi", None) is not None:
        eng.multi.eos = (end,)


@pytest.mark.parametrize("parallel", [1, 2])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=23, top_k=20, top_p=0.95)])
def test_the_engine_decodes_past_end_tokens_with_ignore_eos(tmp_path, sampling, parallel):
    """``stop_eos=False`` (ignore_eos) runs a reply to max_tokens through end tokens, drafted as serial, alone or
    under ``--parallel`` beside a stream that stops at them (each equal to its solo run)."""

    import threading

    from tensorfold.families.qwen4_exp import cuda_engine

    from test_flashnext_tp import _checkpoint

    _checkpoint(tmp_path)
    eng = cuda_engine(tmp_path, context=1017, **({"parallel": parallel} if parallel > 1 else {}))
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]

    def ask(p=prompt, **kw):
        got: list[int] = []
        eng.generate(p, 32, sampling, lambda new: got.extend(new) or False, **kw)
        return got

    free = ask(draft=False, stop_eos=False)
    end = next(t for i, t in enumerate(free) if i >= 3 and free.index(t) == i and i < len(free) - 1)
    _set_end(eng, end)
    assert len(free) == 32 and ask(stop_eos=False) == free and ask(draft=False, stop_eos=False) == free
    assert ask() == free[:free.index(end) + 1] == ask(draft=False)
    if parallel > 1:                             # one stream through its end token beside one that stops at it
        other = prompt[:5] + [42]
        alone = [ask(stop_eos=False), ask(other)]
        together: list = [None, None]
        threads = [threading.Thread(target=lambda: together.__setitem__(0, ask(stop_eos=False))),
                   threading.Thread(target=lambda: together.__setitem__(1, ask(other)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert together == alone
