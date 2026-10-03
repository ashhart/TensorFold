"""Tree verify graphs give the eager forward's bits on every real row; commits, decodes and draft steps match too."""

from __future__ import annotations

import os
import socket

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] < 8:
    pytest.skip("tree graphs run on sm_80 and newer (the engine keeps sm_70 eager)", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.decode import clone_state, draft_decode, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import GDNRecord, State, commit, tree_forward  # noqa: E402
from tensorfold.families.qwen3_5.cuda.prefill import prefill_state  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import multi_tree_forward  # noqa: E402
from tensorfold.families.qwen3_5.cuda.tree_graphs import (MultiGraphs, TreeGraphs, rows_bucket, slot_class,  # noqa: E402
                                                          tree_decode)
from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, Plain, QLinear, Weights  # noqa: E402
from tensorfold.cuda.kernels import gdn as deltanet  # noqa: E402

CONFIG = Config(hidden=128, intermediate=256, layers=2, heads=2, kv_heads=2, head_dim=128, vocab=256, k_heads=2,
                v_heads=2, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32, rope_theta=10000000.0,
                eos=(0,))


def _inv_freq():
    half = CONFIG.rope_dims // 2
    return (CONFIG.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)).float().cuda()


def _mlx_model() -> Weights:
    gen = torch.Generator(device="cuda").manual_seed(21)

    def q(n, k):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda",
                              dtype=torch.int64).to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
        biases = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16()
        return QLinear(words, scales, biases)

    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    gdn = GDN(q(768, 128), q(256, 128), q(2, 128), q(2, 128), q(128, 256),
              (torch.randn(768, 4, generator=gen, device="cuda") * 0.1).bfloat16(),
              torch.zeros(2, device="cuda"), torch.zeros(2, device="cuda"), norm)
    attn = Attention(q(512, 128), q(256, 128), q(256, 128), q(128, 256), norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, q(256, 128), q(256, 128), q(128, 256)),
              Layer(False, norm, norm, None, attn, q(256, 128), q(256, 128), q(128, 256))]
    w = Weights(CONFIG, q(256, 128), layers, norm, q(256, 128), _inv_freq(), tap_layers=(0, 1))
    return w


def _nvfp4_model() -> Weights:
    """NVFP4 MLP, FP8 GDN and attention projections, bf16 gates, embedding and head."""

    from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear

    gen = torch.Generator().manual_seed(23)

    def fp4(n, k):
        packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).cuda()
        scale = torch.randint(0x20, 0x40, (n, k // 16), generator=gen, dtype=torch.uint8).cuda()
        return Fp4Linear.from_checkpoint(packed, scale, 0.02)

    def fp8(n, k):
        raw = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.uint8)
        raw[(raw & 0x7F) >= 0x60] = 0x30
        return Fp8Linear.from_checkpoint(raw.cuda().view(torch.float8_e4m3fn), 0.003)

    def bf16(n, k, scale=0.05):
        return Plain((torch.randn(n, k, generator=gen) * scale).to(torch.bfloat16).cuda())

    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    gdn = GDN(fp8(768, 128), fp8(256, 128), bf16(2, 128), bf16(2, 128), fp8(128, 256),
              (torch.randn(768, 4, generator=gen) * 0.1).to(torch.bfloat16).cuda(),
              torch.zeros(2, device="cuda"), torch.zeros(2, device="cuda"), norm)
    attn = Attention(fp8(512, 128), fp8(256, 128), fp8(256, 128), fp8(128, 256), norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, fp4(256, 128), fp4(256, 128), fp4(128, 256)),
              Layer(False, norm, norm, None, attn, fp4(256, 128), fp4(256, 128), fp4(128, 256))]
    w = Weights(CONFIG, bf16(256, 128, 0.5), layers, norm, bf16(256, 128), _inv_freq(), quant="nvfp4",
                tap_layers=(0, 1))
    w.precision = "full"
    return w


MODELS = {"mlx": _mlx_model, "nvfp4": _nvfp4_model}

# chains, branching trees of each GDN slot class, a star, one row, and widths that pad (3 -> 4, 13 -> 16, 30 -> 32)
SHAPES = [[-1], [-1, 0, 1], list(range(-1, 11)), [-1, 0, 0, 1, 1, 2], [-1, 0, 1, 1, 0, 4, 5, 2],
          [-1] + [0] * 12, [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], list(range(-1, 29)),
          [-1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7]]


def _tokens(n: int, seed: int) -> list[int]:
    return [3 + (seed * 31 + 17 * i) % 250 for i in range(n)]


def _same(eager, replay, n: int) -> None:
    le, re_, te = eager
    lg, rg, tg = replay
    assert lg.shape[0] == n and torch.equal(le, lg), "logits differ on real rows"
    assert torch.equal(te, tg[:n]), "taps differ on real rows"
    for a, b in zip(re_, rg):
        names = ("q", "k", "v", "g", "beta", "qkv") if isinstance(a, GDNRecord) else ("k", "v")
        for name in names:
            assert torch.equal(getattr(a, name), getattr(b, name)[:n]), f"record {name} differs on real rows"


@pytest.fixture(scope="module", params=sorted(MODELS))
def model(request):
    return MODELS[request.param]()


def test_slot_classes_cover_the_shapes():
    slots = {slot_class(deltanet.schedule(p)[1]) for p in SHAPES}
    assert {0, 2}.issubset(slots) and max(slots) >= 4
    assert [rows_bucket(n) for n in (1, 3, 4, 13, 30, 128)] == [4, 4, 4, 16, 32, 128]


@pytest.mark.parametrize("prompt_len", [9, 505])           # 505: the windows cross the first 512-key chunk
def test_replay_equals_eager_for_every_shape(model, prompt_len):
    w = model
    st = State(w)
    prefill_state(w, _tokens(prompt_len, 1), st)
    runner = TreeGraphs(w)
    view = runner.load(st)
    for rep in range(2):                                   # again, after every other shape replayed in between
        for k, parents in enumerate(SHAPES):
            toks = _tokens(len(parents), 7 * k + rep)
            eager = tree_forward(w, torch.tensor(toks, device="cuda", dtype=torch.int32), parents, st,
                                 capture_taps=True)
            _same(eager, runner.verify(view, toks, parents), len(parents))
    assert runner.captures == len({(rows_bucket(len(p)), slot_class(deltanet.schedule(p)[1])) for p in SHAPES})


def test_graph_commits_equal_eager_commits(model):
    """Graph verifies and commits equal eager ones round after round, across cache growth and a chunk boundary."""

    w = model
    st = State(w)
    prefill_state(w, _tokens(500, 2), st)
    ref = clone_state(st)
    ref.kv = [None if kv is None else (kv[0].clone(), kv[1].clone()) for kv in st.kv]
    runner = TreeGraphs(w)
    view = runner.load(st)
    plan = [([-1, 0, 1, 2], [0, 1, 2, 3]), ([-1, 0, 0, 1, 1, 2], [0, 2]), ([-1, 0, 0, 1, 1, 2], [0]),
            (list(range(-1, 11)), list(range(12))), ([-1, 0, 1, 1, 0, 4, 5, 2], [0, 4, 5, 6]),
            ([-1] + [0] * 12, [0, 7]), (list(range(-1, 29)), list(range(20)))] * 3
    for k, (parents, path) in enumerate(plan):
        toks = _tokens(len(parents), 100 + k)
        eager = tree_forward(w, torch.tensor(toks, device="cuda", dtype=torch.int32), parents, ref, capture_taps=True)
        replay = runner.verify(view, toks, parents)
        _same(eager, replay, len(parents))
        commit(ref, eager[1], path)
        runner.commit(view, replay[1], path)
        assert view.pos == ref.pos
        for a, b in zip(view.rec + view.conv, ref.rec + ref.conv):
            if a is not None:
                assert torch.equal(a, b), f"round {k}: GDN state differs after the commit"
        for a, b in zip(view.kv, ref.kv):
            if a is not None:
                assert torch.equal(a[0][:ref.pos], b[0][:ref.pos]) and torch.equal(a[1][:ref.pos], b[1][:ref.pos])
    assert ref.pos > 512 + 64                                  # the caches grew past their first allocation


class _Trees:
    """A drafter stand-in proposing deterministic chains and branching trees."""

    layers = 0

    def __init__(self) -> None:
        self.taps = 0

    def propose_tree(self, pending: int, length: int, max_nodes: int, sampling):
        n = min(max_nodes, 1 + (pending * 7 + length) % 11)
        kind = (pending + length) % 3
        tokens = [3 + (pending * 13 + 29 * i + length) % 250 for i in range(n)]
        if kind == 0:
            parents = list(range(-1, n - 1))
        elif kind == 1:
            parents = [-1 if i < 2 else (i - 2) // 2 for i in range(n)]
        else:
            parents = [-1 if i < 3 else i - 3 for i in range(n)]
        return tokens, parents

    def add_taps(self, taps: torch.Tensor) -> None:
        self.taps += taps.shape[0]


@pytest.mark.parametrize("seed", [None, 1234], ids=["greedy", "seed1234"])
def test_tree_decode_equals_draft_decode_and_serial(model, seed):
    from tensorfold.engine.exact_sampling import Sampling

    w = model
    sampling = None if seed is None else Sampling(seed, 1.0, 20, 0.95)
    prompt = _tokens(40, 3)
    outs = []
    runner = TreeGraphs(w)
    for mode in ("serial", "eager", "graphs", "graphs"):
        st = State(w)
        prefill_state(w, prompt, st)
        if mode == "serial":
            r = serial_decode(w, st, 5, 120, sampling, stop_eos=False)
        elif mode == "eager":
            r = draft_decode(w, st, prompt, 5, 120, sampling, _Trees(), max_rows=12, stop_eos=False)
        else:
            r = tree_decode(w, st, prompt, 5, 120, sampling, _Trees(), runner, max_rows=12, stop_eos=False)
        outs.append((mode, r))
    tokens = [r.tokens for _, r in outs]
    assert len(tokens[0]) == 120 and all(t == tokens[0] for t in tokens), [(m, r.tokens[:20]) for m, r in outs]
    assert outs[2][1].rounds == outs[3][1].rounds == outs[1][1].rounds   # the same windows, round for round


def test_draft_graph_proposes_what_the_eager_step_proposes(tmp_path):
    """DFlash2's draft step from a graph proposes the eager step's tree as its context grows."""

    from test_qwen27_dflash2_blocks import _drafter

    d = _drafter(tmp_path)
    gen = torch.Generator(device="cuda").manual_seed(8)
    snaps = []
    for k in range(5):
        taps = (torch.randn(5 + 9 * k, 5 * d.hidden, generator=gen, device="cuda") * 0.5).bfloat16()
        d.add_taps(taps)
        snaps.append(d.snapshot())
        for pending in (3, 77):
            eager = d.finish_tree(d.launch_block(pending, 11), 40 + k, 11, None)
            graph = d.finish_tree(d.graph_block(pending, 11), 40 + k, 11, None)
            assert eager == graph
    assert list(d._graphs) == [(1, 12)]
    for picks in ([0, 2, 4], [4, 1], [3, 0, 1, 2]):          # several streams' contexts, each read in place
        group = [snaps[i] for i in picks]
        pendings = [5 + 11 * i for i in picks]
        eager = d.launch_blocks(group, pendings, 11)
        graph = d.graph_blocks(group, pendings, 11)
        for j in range(len(group)):
            assert d.finish_tree(eager[j], 40, 11, None) == d.finish_tree(graph[j], 40, 11, None)


@pytest.mark.parametrize("together", [False, True])
def test_a_lone_stream_of_the_concurrent_decoder_replays_graphs_exactly(together):
    """``MultiDecoder`` streams decode their serial tokens when a lone stream's rounds replay graphs."""

    from test_qwen27_multi import PROMPTS, SAMPLINGS, _model as multi_model, _Oracle, _serial
    from tensorfold.cuda.streams import Stream

    w = multi_model()
    refs = {tuple(p): _serial(w, p, smp, 24) for p, smp in zip(PROMPTS[:4], SAMPLINGS)}
    dec = _Oracle(w, refs)
    dec.trees = TreeGraphs(w, taps=False)
    outs = []

    def admit(i):
        got: list[int] = []
        s = Stream(PROMPTS[i], 24, SAMPLINGS[i], draft=i != 2, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        outs.append((s, got))

    for i in range(4):
        admit(i)
        if not together:                                 # each alone: graphs, the buffers changing hands
            while dec.live():
                dec.finish(dec.round())
    while dec.live():
        dec.finish(dec.round())
    for s, got in outs:
        assert got == refs[tuple(s.prompt)] and s.out == got, s.prompt
    assert dec.trees.captures > 0 or together            # together, the streams may finish in the same round


def test_several_streams_replay_equals_eager(model):
    """Each stream's rows from ``MultiGraphs.verify_streams`` equal ``multi_tree_forward``'s, padding included."""

    w = model
    states = []
    for k, n in enumerate((9, 505, 40, 600)):
        st = State(w)
        prefill_state(w, _tokens(n, 30 + k), st)
        states.append(st)
    runner = MultiGraphs(w)
    shapes = [s for s in SHAPES if len(s) <= 16]
    for rep in range(3):
        for S in (2, 3, 4):
            wins = []
            for j in range(S):
                parents = shapes[(rep * 5 + 3 * S + j) % len(shapes)]
                wins.append((_tokens(len(parents), 50 * rep + 7 * S + j), parents))
            el, er, et, es = multi_tree_forward(w, [(t, p, st) for (t, p), st in zip(wins, states)], capture_taps=True)
            gl, rows, gr, gt, gs = runner.verify_streams(wins, states[:S])
            assert rows == es
            assert torch.equal(el, gl), f"{S} streams, rep {rep}: logits differ"
            for j in range(S):
                n, a, b = len(wins[j][0]), es[j], gs[j]
                assert torch.equal(et[a:a + n], gt[b:b + n])
                for x, y in zip(er, gr):
                    names = ("q", "k", "v", "g", "beta", "qkv") if isinstance(x, GDNRecord) else ("k", "v")
                    for name in names:
                        assert torch.equal(getattr(x, name)[a:a + n], getattr(y, name)[b:b + n]), (S, j, name)


def test_concurrent_streams_from_graphs_decode_their_serial_tokens():
    """``MultiDecoder`` streams decode their serial tokens when rounds of several streams replay one graph."""

    from test_qwen27_multi import PROMPTS, SAMPLINGS, _model as multi_model, _Oracle, _serial
    from tensorfold.cuda.streams import Stream

    w = multi_model()
    refs = {tuple(p): _serial(w, p, smp, 24) for p, smp in zip(PROMPTS, SAMPLINGS)}
    dec = _Oracle(w, refs)
    dec.trees = MultiGraphs(w, taps=False)
    outs = []
    for i in range(5):
        got: list[int] = []
        s = Stream(PROMPTS[i], 24, SAMPLINGS[i], draft=i != 2, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        outs.append((s, got))
    while dec.live():
        dec.finish(dec.round())
    for s, got in outs:
        assert got == refs[tuple(s.prompt)] and s.out == got, s.prompt
    assert dec.trees.mgraphs                                 # rounds of several streams replayed a graph


# ---- two GPUs: the P2P sum in a graph, and the two-rank tree verify -----------------------------------------------


def _two_gpu_worker(rank: int, port: int):
    import torch.distributed as dist

    from tensorfold.cuda import p2p
    from tensorfold.families.qwen3_5.cuda.distributed import split_weights

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        assert p2p.install(rank)
        peer = p2p.peer()
        gen = torch.Generator(device="cuda").manual_seed(300 + rank)
        local = torch.empty((12, 5120), device="cuda")
        out = []

        def sums():
            out.clear()
            for _ in range(3):
                out.append(peer(local))
            return out

        sums()                                          # warm, both ranks
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        peer.begin_capture()
        with torch.cuda.graph(g):
            captured = sums()
        calls = peer.end_capture()
        assert calls == 3
        for step in range(40):                          # replays among eager sums: every round new on both ranks
            local.copy_(torch.randn((12, 5120), generator=gen, device="cuda") * 3)
            both = torch.empty((24, 5120), device="cuda")
            dist.all_gather_into_tensor(both, local)
            want = (both[:12] + both[12:]).to(torch.bfloat16)
            if step % 3 == 0:
                got = [peer(local)]
            else:
                peer.before_replay(calls)
                g.replay()
                got = captured
            for t in got:
                assert torch.equal(t, want), f"step {step}: the sum differs from the rank-ordered sum"
        torch.cuda.synchronize()
        # the two-rank tree verify: graph replay against eager, both ranks
        w = split_weights(_nvfp4_model(), rank, tiled=True, split_head=True)
        w.tap_layers = (0, 1)
        st = State(w)
        prefill_state(w, _tokens(505, 4), st, tp=True)
        runner = TreeGraphs(w, tp=True)
        view = runner.load(st)
        for rep in range(2):
            for k, parents in enumerate(SHAPES):
                toks = _tokens(len(parents), 11 * k + rep)
                eager = tree_forward(w, torch.tensor(toks, device="cuda", dtype=torch.int32), parents, st, tp=True,
                                     capture_taps=True)
                _same(eager, runner.verify(view, toks, parents), len(parents))
        other = State(w)                                # several streams in one two-rank graph against eager
        prefill_state(w, _tokens(40, 5), other, tp=True)
        multi = MultiGraphs(w, tp=True)
        short = [p for p in SHAPES if len(p) <= 16]
        for k in range(4):
            wins = [(_tokens(len(p), 3 * k + j), p) for j, p in enumerate((short[k % len(short)],
                                                                           short[(k + 3) % len(short)]))]
            el, er, et, es = multi_tree_forward(w, [(t, p, x) for (t, p), x in zip(wins, (st, other))], tp=True,
                                                capture_taps=True)
            gl, rows, gr, gt, gs = multi.verify_streams(wins, [st, other])
            assert rows == es and torch.equal(el, gl), f"two-rank streams, round {k}: logits differ"
            for j, (t, _) in enumerate(wins):
                assert torch.equal(et[es[j]:es[j] + len(t)], gt[gs[j]:gs[j] + len(t)])
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def test_two_gpu_graphs_keep_the_rank_ordered_sums():
    if os.environ.get("TENSORFOLD_TEST_NCCL") != "1":
        pytest.skip("set TENSORFOLD_TEST_NCCL=1 to run the two-GPU process tests")
    if torch.cuda.device_count() < 2 or not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("requires two visible CUDA devices that can map each other")
    from torch.multiprocessing import spawn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    spawn(_two_gpu_worker, args=(port,), nprocs=2)
