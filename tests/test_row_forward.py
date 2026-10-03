"""The lane decoder without tensor units (row_forward): windows, trees and several streams' windows reproduce
one-row steps bit for bit, and stacked projections change no bits and store no weight twice."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.engine.family_common import cache_contents  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import (  # noqa: E402
    exact_attention, lane_glue, lane_tree, row_forward, row_glue, row_matmul)


def _same(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == mx.float32:
        return bool(mx.all(a.view(mx.uint32) == b.view(mx.uint32)).item())
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _tiny_model(seed=21):
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    # Qwen3.8-27B's head shapes and MLP width on a 1024-wide residual (4 layers: 3 recurrent, 1 attention)
    args = TextModelArgs(model_type="qwen3_5_text", hidden_size=1024, intermediate_size=17408, num_hidden_layers=4,
                         num_attention_heads=24, num_key_value_heads=4, head_dim=256, rms_norm_eps=1e-6,
                         vocab_size=512, linear_num_value_heads=48, linear_num_key_heads=16,
                         linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                         full_attention_interval=4, tie_word_embeddings=False, max_position_embeddings=4096)
    mx.random.seed(seed)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def tiny():
    exact_attention.install()
    model = _tiny_model()
    originals = {id(m): (m, mx.array(m["weight"])) for _, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)}
    stacked = row_matmul.install(model)
    return model, originals, stacked


def _run(model, tokens, cache, start, keep=None):
    core, head = model.model, model.lm_head
    parents = [-1] + list(range(len(tokens) - 1))
    logits, record = row_forward.forward(core, head, tokens, parents, cache, start, pipeline_layers=2)
    keep = len(tokens) if keep is None else keep
    row_forward.commit(cache, record, list(range(keep)), len(tokens), start)
    mx.eval(logits, *[a for c in cache for a in cache_contents(c)])
    return logits


def _prefill(model, prompt):
    cache = model.make_cache()
    most = row_matmul.WINDOW_ROWS
    for begin in range(0, len(prompt), most):
        _run(model, prompt[begin:begin + most], cache, begin)
    return cache


def test_stacks_hold_the_weights_once(tiny):
    """The members become views of the stacked weight (same values, nothing stored twice); the stacked projection
    matches each member's to rounding (simd_qmm's split count follows the output count, so stacking is part of the
    arithmetic every path shares)."""

    model, originals, stacked = tiny
    assert stacked == {"in": 3, "qkv": 1, "gu": 4}
    for m, before in originals.values():
        assert _same(m["weight"], before)
    gdn = model.model.layers[0].linear_attn
    stack = row_matmul.stack_of(gdn, "in")
    x = (mx.random.normal((1, 5, 1024), key=mx.random.key(3)) * 0.5).astype(mx.bfloat16)
    y = row_matmul.project_stack(stack, x)
    offset = 0
    for name in row_matmul.GROUPS["in"]:
        member = getattr(gdn, name)
        n = int(member["weight"].shape[0])
        assert bool(mx.allclose(y[..., offset:offset + n], row_matmul.project(member, x), atol=2e-2).item()), name
        offset += n


def test_windows_reproduce_one_row_steps(tiny):
    model, _, _ = tiny
    mx.random.seed(5)
    prompt = [int(t) for t in mx.random.randint(0, 512, (21,)).tolist()]
    tokens = [int(t) for t in mx.random.randint(0, 512, (row_matmul.WINDOW_ROWS,)).tolist()]
    base = _prefill(model, prompt)
    start = len(prompt)
    serial_cache = LaneEngine.copy_single_cache(base)
    serial = [_run(model, [t], serial_cache, start + i)[0, -1] for i, t in enumerate(tokens)]
    for width in range(2, len(tokens) + 1):
        window = _run(model, tokens[:width], LaneEngine.copy_single_cache(base), start)
        for i in range(width):
            assert _same(window[0, i], serial[i]), f"row {i} of a {width}-row window differs from its one-row step"


def test_partly_kept_windows_leave_serial_state(tiny):
    """A window kept to its first rows leaves the cache exactly as that many one-row steps: the next rows agree."""

    model, _, _ = tiny
    mx.random.seed(9)
    prompt = [int(t) for t in mx.random.randint(0, 512, (13,)).tolist()]
    tokens = [int(t) for t in mx.random.randint(0, 512, (8,)).tolist()]
    base = _prefill(model, prompt)
    start = len(prompt)
    serial_cache = LaneEngine.copy_single_cache(base)
    serial = [_run(model, [t], serial_cache, start + i)[0, -1] for i, t in enumerate(tokens)]
    cache = LaneEngine.copy_single_cache(base)
    _run(model, tokens[:6], cache, start, keep=3)          # rows 3-5 rejected
    after = _run(model, tokens[3:8], cache, start + 3)
    for i in range(5):
        assert _same(after[0, i], serial[3 + i]), f"row {3 + i} after a rollback differs from serial"


def test_prefill_chunking_does_not_change_bits(tiny):
    """Chunks of any width give the same caches: 19 rows in one chain (the tree kernel's recurrence) or split
    (the compiled chain kernel's)."""

    model, _, _ = tiny
    mx.random.seed(11)
    prompt = [int(t) for t in mx.random.randint(0, 512, (19,)).tolist()]
    whole = model.make_cache()
    _run(model, prompt, whole, 0)
    split = model.make_cache()
    begin = 0
    for size in (3, 1, 8, 7):
        _run(model, prompt[begin:begin + size], split, begin)
        begin += size
    for a, b in zip(whole, split):
        for x, y in zip(cache_contents(a), cache_contents(b)):
            if x is not None:
                assert _same(x, y)


def _gdn_inputs(gdn, W, seed):
    stack = row_matmul.stack_of(gdn, "in")
    x = (mx.random.normal((1, W, 1024), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    y = row_matmul.project_stack(stack, x)
    conv_state = (mx.random.normal((1, 3, gdn.conv_dim), key=mx.random.key(seed + 1)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), key=mx.random.key(seed + 2))
             * 0.1).astype(mx.float32)
    heads = dict(nk=gdn.num_k_heads, nv=gdn.num_v_heads, dk=gdn.head_k_dim, dv=gdn.head_v_dim)
    return y, conv_state, state, heads


def test_recurrent_glue_keeps_lane_glue_and_tree_bits(tiny):
    """gdn_pre from the stacked rows gives lane_glue's bits and the conv tail; the recurrence gives lane_tree's bits,
    and for a chain the state after its last row equals a replay of all its rows."""

    model, _, _ = tiny
    gdn = model.model.layers[0].linear_attn
    W = 5
    y, conv_state, state, heads = _gdn_inputs(gdn, W, 4)
    C = gdn.conv_dim
    nv, dv = gdn.num_v_heads, gdn.head_v_dim
    parents = list(range(-1, W - 1))
    windows = lane_tree._conv_windows(parents, 3)
    q, k, v, g, beta, conv_out = row_glue.gdn_pre(y, conv_state, gdn.conv1d.weight, windows, gdn.A_log,
                                                     gdn.dt_bias, **heads)
    b = mx.contiguous(y[..., C + nv * dv:C + nv * dv + nv])
    a = mx.contiguous(y[..., C + nv * dv + nv:])
    pre = lane_glue.gdn_pre(mx.contiguous(y[..., :C]), conv_state, gdn.conv1d.weight, windows, a, b, gdn.A_log,
                            gdn.dt_bias, **heads)
    for name, ours, glue in zip("qkvgb", (q, k, v, g, beta), pre):
        assert _same(ours, glue), name
    seq = mx.concatenate([conv_state, y[..., :C]], axis=1)[0]
    for w in range(W):                                  # the conv tail after each row: its last three conv inputs
        assert _same(conv_out[w], seq[w + 1:w + 4])
    rec, state_out = row_glue.gated_delta(q, k, v, g, beta, state, parents)
    assert _same(rec, lane_tree.gated_delta_tree(q, k, v, g, beta, state, parents))
    replayed = lane_tree.replay_path(q, k, v, g, beta, state, mx.arange(W, dtype=mx.int32),
                                     mx.array([W], dtype=mx.int32))
    assert _same(state_out, replayed)


def test_tree_nodes_equal_their_paths(tiny):
    model, _, _ = tiny
    gdn = model.model.layers[1].linear_attn
    parents = [-1, 0, 0, 1, 2, 2, 4, 3]
    W = len(parents)
    y, conv_state, state, heads = _gdn_inputs(gdn, W, 7)
    pre = row_glue.gdn_pre(y, conv_state, gdn.conv1d.weight, lane_tree._conv_windows(parents, 3), gdn.A_log,
                              gdn.dt_bias, **heads)
    rec = row_glue.gated_delta(*pre[:5], state, parents)[0]
    _, paths = lane_tree.tree_paths(parents)
    for node, path in enumerate(paths):
        rows = mx.array(path, dtype=mx.int32)
        chain = list(range(-1, len(path) - 1))
        one_pre = row_glue.gdn_pre(mx.take(y, rows, axis=1), conv_state, gdn.conv1d.weight,
                                      lane_tree._conv_windows(chain, 3), gdn.A_log, gdn.dt_bias, **heads)
        one = row_glue.gated_delta(*one_pre[:5], state, chain)[0]
        assert _same(rec[:, node], one[:, -1]), f"node {node} differs from its path as a chain"


def test_aligned_prefill_resumes_exactly(tiny, monkeypatch):
    """Resumed at a chunk start a prompt gets a fresh prefill's bits; between starts it is not; decodes are not kept."""

    from tensorfold.engine.lane_engine import LaneStream
    from tensorfold.engine.prefill_plan import PrefillPlan
    from tensorfold.families.qwen3_5.family import Qwen35Family

    model, _, _ = tiny
    monkeypatch.setattr(LaneEngine, "prefill_plan", PrefillPlan(8))
    family = Qwen35Family(model, widest=row_matmul.WINDOW_ROWS, rows=True)
    engine = LaneEngine(family, max_rows=family.exact_width, max_draft=family.exact_width - 1,
                        retain_finished_caches=True)
    mx.random.seed(21)
    prompt = [int(t) for t in mx.random.randint(0, 512, (29,)).tolist()]

    def arrays(cache):
        return [a for c in cache for a in cache_contents(c)]

    fresh = engine.prefill_prefix(prompt)
    first = LaneStream("first", prompt[:19], max_new_tokens=1)
    engine.add_stream(first, checkpoints_at=[13, 19])
    assert [len(tokens) for tokens, _ in first.history_checkpoints] == [8, 16]
    tokens, cache = first.history_checkpoints[-1]
    resumed = engine.prefill_prefix(prompt, cache=LaneEngine.copy_single_cache(cache), cached_tokens=len(tokens))
    assert all(_same(a, b) for a, b in zip(arrays(fresh), arrays(resumed)))
    off_grid = engine.prefill_prefix(prompt, cache=LaneEngine.copy_single_cache(cache), cached_tokens=13)
    assert all(_same(a, b) for a, b in zip(arrays(fresh), arrays(off_grid)))
    stream = LaneStream("decoded", prompt, max_new_tokens=3)
    engine.add_stream(stream)
    engine.run()
    assert "decoded" not in engine.finished_caches


def test_tree_window_nodes_equal_serial_paths(tiny, monkeypatch):
    """A draft tree through the whole forward (``row_attention``): every node's logits are the serial steps' along
    its path, and a commit of a path that is not the window's first rows leaves the serial state."""

    monkeypatch.setattr(row_forward, "ROW_ATTENTION", True)
    model, _, _ = tiny
    core, head = model.model, model.lm_head
    mx.random.seed(17)
    prompt = [int(t) for t in mx.random.randint(0, 512, (11,)).tolist()]
    parents = [-1, 0, 0, 1, 2, 2, 4, 3]
    tokens = [int(t) for t in mx.random.randint(0, 512, (len(parents),)).tolist()]
    base = _prefill(model, prompt)
    start = len(prompt)
    cache = LaneEngine.copy_single_cache(base)
    logits, record = row_forward.forward(core, head, tokens, parents, cache, start, pipeline_layers=2)
    mx.eval(logits)
    _, paths = lane_tree.tree_paths(parents)
    for node, path in enumerate(paths):
        serial_cache = LaneEngine.copy_single_cache(base)
        last = None
        for i, r in enumerate(path):
            last = _run(model, [tokens[r]], serial_cache, start + i)
        assert _same(logits[0, node], last[0, -1]), f"node {node} differs from its serial path"
    path = paths[6]                                  # rows 0, 2, 4, 6: not in place
    row_forward.commit(cache, record, path, len(parents), start)
    serial_cache = LaneEngine.copy_single_cache(base)
    for i, r in enumerate(path):
        _run(model, [tokens[r]], serial_cache, start + i)
    for a, b in zip(cache, serial_cache):
        for x, y in zip(cache_contents(a), cache_contents(b)):
            if x is not None:
                assert _same(x, y)


def test_streams_in_one_forward_equal_each_alone(tiny):
    """Several streams' windows in one forward (rows grouped by stream, each over its own caches): every row's
    logits and every stream's caches after keeping part of its window equal the stream's own round, bit for bit."""

    model, _, _ = tiny
    # (40, 2, 1): a prompt chunk riding with two streams' decode windows
    # (2, 1, ..., 3): nine streams, a launch of eight and a launch of one
    mixes = [(1, 1), (1, 4), (3, 1, 2, 2), (8, 8), (3, 1, 8, 5), (16, 2), (1, 2, 3, 1, 1, 2, 1, 1, 3, 1, 2), (40, 2, 1),
             (2, 1, 1, 4, 1, 1, 2, 1, 3)]
    ok, failures = row_forward.check_streams(model.model, model.lm_head, model.make_cache, LaneEngine.copy_single_cache,
                                             mixes=mixes)
    assert ok, failures


def test_stream_check_reads_only_the_table(tiny, monkeypatch):
    """The memory past the embedding table changes between the solo and the shared rounds; MLX gathers are unchecked."""

    model, _, _ = tiny
    emb = model.model.embed_tokens
    vocab = int(emb["weight"].shape[0])
    backing = [{k: mx.concatenate([emb[k], mx.full((2048, *emb[k].shape[1:]), fill, emb[k].dtype)])
                for k in ("weight", "scales", "biases")} for fill in (1, 2)]

    def table(phase):
        for k, whole in backing[phase].items():
            monkeypatch.setattr(emb, k, whole[:vocab])

    copies = []

    def copy(cache):
        copies.append(cache)
        if len(copies) == 3:                       # mix (1, 4): two solo rounds' copies, then the shared round's
            table(1)
        return LaneEngine.copy_single_cache(cache)

    table(0)
    ok, failures = row_forward.check_streams(model.model, model.lm_head, model.make_cache, copy, mixes=[(1, 4)],
                                             prefixes=(9,))
    assert ok, failures


def test_hidden_rows_and_keep_rows(tiny):
    """The family interface: hidden_rows + logits give multi_forward's logits (GPU token arrays taken unread),
    starts default to each stream's cache length, and keep_rows leaves the caches commit leaves."""

    model, _, _ = tiny
    core, head = model.model, model.lm_head
    mx.random.seed(23)
    prompts = [[int(t) for t in mx.random.randint(0, 512, (n,)).tolist()] for n in (7, 12)]
    bases = [_prefill(model, p) for p in prompts]
    windows = [[5, 17, 300], [9, 44]]
    keeps = [2, 1]
    caches = [LaneEngine.copy_single_cache(b) for b in bases]
    ref, records, offsets = row_forward.multi_forward(core, head, windows, [[-1, 0, 1], [-1, 0]], caches,
                                                      [len(p) for p in prompts])
    for cache, record, window, keep, prompt in zip(caches, records, windows, keeps, prompts):
        row_forward.commit(cache, record, list(range(keep)), len(window), len(prompt))
    mine = [LaneEngine.copy_single_cache(b) for b in bases]
    x, recs = row_forward.hidden_rows(core, [mx.array(w, dtype=mx.int32) for w in windows], mine)
    lg = row_matmul.logits(head, x)
    for cache, record, keep in zip(mine, recs, keeps):
        row_forward.keep_rows(cache, record, keep)
    mx.eval(ref, lg)
    assert offsets == [0, 3]
    assert [r.start for r in recs] == [len(p) for p in prompts]
    assert _same(ref, lg)
    for a, b in zip(caches, mine):
        for ia, ib in zip(a, b):
            for u, v in zip(cache_contents(ia), cache_contents(ib)):
                if u is not None:
                    assert _same(u, v)


def test_streams_with_trees_equal_each_alone(tiny, monkeypatch):
    """With ``row_attention`` a stream's window can be a draft tree: a tree and a chain in one forward give each
    stream's rows the bits of its window alone, and a tree path kept off the window's first rows commits exactly."""

    monkeypatch.setattr(row_forward, "ROW_ATTENTION", True)
    model, _, _ = tiny
    core, head = model.model, model.lm_head
    mx.random.seed(29)
    prompts = [[int(t) for t in mx.random.randint(0, 512, (n,)).tolist()] for n in (10, 6)]
    bases = [_prefill(model, p) for p in prompts]
    parents = [[-1, 0, 0, 1, 2], [-1, 0, 1]]
    windows = [[int(t) for t in mx.random.randint(0, 512, (len(p),)).tolist()] for p in parents]
    paths = [[0, 2, 4], [0, 1]]
    alone = []
    for base, window, par, prompt, path in zip(bases, windows, parents, prompts, paths):
        own = LaneEngine.copy_single_cache(base)
        lg, record = row_forward.forward(core, head, window, par, own, len(prompt))
        row_forward.commit(own, record, path, len(window), len(prompt))
        mx.eval(lg)
        alone.append((lg, own))
    caches = [LaneEngine.copy_single_cache(b) for b in bases]
    lg, records, offsets = row_forward.multi_forward(core, head, windows, parents, caches, [len(p) for p in prompts])
    for cache, record, window, prompt, path in zip(caches, records, windows, prompts, paths):
        row_forward.commit(cache, record, path, len(window), len(prompt))
    mx.eval(lg)
    for (ref, own), cache, a, window in zip(alone, caches, offsets, windows):
        assert _same(lg[0, a:a + len(window)], ref[0])
        for ia, ib in zip(cache, own):
            for u, v in zip(cache_contents(ia), cache_contents(ib)):
                if u is not None:
                    assert _same(u, v)


def test_one_kernel_signature_for_every_window(monkeypatch):
    """The row decoder's kernels keep one Metal signature through chains, a row_attention tree and row_streams."""

    from tests.kernel_signatures import changed, recording

    from tensorfold.kernels.qwen.dense.v1 import row_attention, row_streams, simd_qmm

    caches = [row_glue._kernels, row_streams._kernels, lane_tree._kernels, simd_qmm._kernels, simd_qmm._plans,
              row_attention._kernels]
    saved = [dict(c) for c in caches]
    saved_backend = row_matmul.BACKEND
    try:
        with recording() as seen:
            for c in caches:
                c.clear()                                      # made again, inside the recorder
            backend = row_matmul.simd_qmm_backend()
            exact_attention.install()
            model = _tiny_model(seed=31)
            row_matmul.install(model, backend)
            core, head = model.model, model.lm_head
            mx.random.seed(12)
            tokens = [int(t) for t in mx.random.randint(0, 512, (200,)).tolist()]
            cache = model.make_cache()
            _run(model, tokens[:20], cache, 0)                 # a chain past WINDOW_ROWS: the tree kernel's recurrence
            start = 20
            for n, keep in ((7, 3), (2, 2), (1, 1), (row_matmul.WINDOW_ROWS, row_matmul.WINDOW_ROWS), (1, 1), (12, 9)):
                logits = _run(model, tokens[start:start + n], cache, start, keep=keep)
                assert bool(mx.all(mx.isfinite(logits)).item())
                start += keep
            monkeypatch.setattr(row_forward, "ROW_ATTENTION", True)
            parents = [-1, 0, 0, 1, 2, 2, 4, 3]
            logits, record = row_forward.forward(core, head, tokens[start:start + 8], parents, cache, start)
            row_forward.commit(cache, record, [0, 2, 4, 6], len(parents), start)
            mx.eval(logits, *[a for c in cache for a in cache_contents(c)])
            assert bool(mx.all(mx.isfinite(logits)).item())
            monkeypatch.setattr(row_forward, "ROW_ATTENTION", False)
            streams = [_prefill(model, tokens[100 + 20 * s:100 + 20 * s + 9 + 4 * s]) for s in range(3)]
            starts = [9 + 4 * s for s in range(3)]
            at = 160
            for widths in ((5, 1), (1, 3, 9), (2, 2, 2), (1, 1)):
                ids = list(range(len(widths)))
                windows = []
                for w in widths:
                    windows.append(tokens[at:at + w])
                    at += w
                chains = [list(range(-1, w - 1)) for w in widths]
                logits, records, _ = row_forward.multi_forward(core, head, windows, chains, [streams[s] for s in ids],
                                                               [starts[s] for s in ids])
                for s, record, w in zip(ids, records, widths):
                    keep = max(1, w // 2)
                    row_forward.commit(streams[s], record, list(range(keep)), w, starts[s])
                    starts[s] += keep
                mx.eval(logits, *[a for s in ids for c in streams[s] for a in cache_contents(c)])
                assert bool(mx.all(mx.isfinite(logits)).item())
    finally:
        for c, old in zip(caches, saved):
            c.clear()
            c.update(old)
        row_matmul.BACKEND = saved_backend
    names = {name.rsplit("_", 1)[0] for name, _ in seen}
    expect = {"row_forward_tree", "row_forward_chain", "row_forward_gdn_pre", "gated_delta_replay",
              "row_attention_partial", "row_attention_merge", "row_streams_pre2", "row_streams_tree3"}
    assert expect <= names, names
    assert not changed(seen), "kernels called with more than one signature: " + changed(seen)
