"""The lane decoder without tensor units (row_forward): windows reproduce one-row steps bit for bit, stacked
projections change no bits and store no weight twice, and the glue variants read the stacked rows in place."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import exact_attention, lane_glue, lane_tree, row_forward, row_qmv  # noqa: E402


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
    stacked = row_forward.install(model, row_forward.row_qmv_backend())
    return model, originals, stacked


def _run(model, tokens, cache, start, keep=None):
    core, head = model.model, model.lm_head
    parents = [-1] + list(range(len(tokens) - 1))
    logits, record = row_forward.forward(core, head, tokens, parents, cache, start, pipeline_layers=2)
    keep = len(tokens) if keep is None else keep
    row_forward.commit(cache, record, list(range(keep)), len(tokens), start)
    mx.eval(logits, *[a for c in cache for a in c.state if a is not None])
    return logits


def _prefill(model, prompt):
    cache = model.make_cache()
    most = row_forward.BACKEND.max_rows
    for begin in range(0, len(prompt), most):
        _run(model, prompt[begin:begin + most], cache, begin)
    return cache


def test_stacks_are_views_with_the_same_bits(tiny):
    model, originals, stacked = tiny
    assert stacked == {"in": 3, "gu": 4}
    for m, before in originals.values():
        assert _same(m["weight"], before)                 # the members hold views with the same values
    gdn = model.model.layers[0].linear_attn
    stack = row_forward.stack_of(gdn, "in")
    x = (mx.random.normal((1, 5, 1024), key=mx.random.key(3)) * 0.5).astype(mx.bfloat16)
    y = row_forward.project_stack(stack, x)
    offset = 0
    for name in row_forward.GROUPS["in"]:
        member = getattr(gdn, name)
        n = int(member["weight"].shape[0])
        assert _same(y[..., offset:offset + n], row_forward.project(member, x)), name
        offset += n


def test_windows_reproduce_one_row_steps(tiny):
    model, _, _ = tiny
    mx.random.seed(5)
    prompt = [int(t) for t in mx.random.randint(0, 512, (21,)).tolist()]
    tokens = [int(t) for t in mx.random.randint(0, 512, (row_forward.BACKEND.max_rows,)).tolist()]
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
    model, _, _ = tiny
    mx.random.seed(11)
    prompt = [int(t) for t in mx.random.randint(0, 512, (19,)).tolist()]
    whole = _prefill(model, prompt)
    split = model.make_cache()
    begin = 0
    for size in (3, 1, 8, 7):
        _run(model, prompt[begin:begin + size], split, begin)
        begin += size
    for a, b in zip(whole, split):
        for x, y in zip(a.state, b.state):
            if x is not None:
                assert _same(x, y)


def _gdn_inputs(gdn, W, seed):
    stack = row_forward.stack_of(gdn, "in")
    x = (mx.random.normal((1, W, 1024), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    y = row_forward.project_stack(stack, x)
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
    q, k, v, g, beta, conv_out = row_forward.gdn_pre(y, conv_state, gdn.conv1d.weight, windows, gdn.A_log,
                                                     gdn.dt_bias, **heads)
    b = mx.contiguous(y[..., C + nv * dv:C + nv * dv + nv])
    a = mx.contiguous(y[..., C + nv * dv + nv:])
    pre = lane_glue.gdn_pre(mx.contiguous(y[..., :C]), conv_state, gdn.conv1d.weight, windows, a, b, gdn.A_log,
                            gdn.dt_bias, **heads)
    for name, ours, glue in zip("qkvgb", (q, k, v, g, beta), pre):
        assert _same(ours, glue), name
    seq = mx.concatenate([conv_state, y[..., :C]], axis=1)[0]
    assert _same(conv_out[0], seq[-3:])
    rec, state_out = row_forward.gated_delta(q, k, v, g, beta, state, parents)
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
    pre = row_forward.gdn_pre(y, conv_state, gdn.conv1d.weight, lane_tree._conv_windows(parents, 3), gdn.A_log,
                              gdn.dt_bias, **heads)
    rec = row_forward.gated_delta(*pre[:5], state, parents)[0]
    _, paths = lane_tree.tree_paths(parents)
    for node, path in enumerate(paths):
        rows = mx.array(path, dtype=mx.int32)
        chain = list(range(-1, len(path) - 1))
        one_pre = row_forward.gdn_pre(mx.take(y, rows, axis=1), conv_state, gdn.conv1d.weight,
                                      lane_tree._conv_windows(chain, 3), gdn.A_log, gdn.dt_bias, **heads)
        one = row_forward.gated_delta(*one_pre[:5], state, chain)[0]
        assert _same(rec[:, node], one[:, -1]), f"node {node} differs from its path as a chain"


def test_gate_up_act_equals_matmul_then_act(tiny):
    """row_qmv with SiLU(gate) * up as its epilogue gives the bits of the stacked matmul followed by mlp_act."""

    model, _, _ = tiny
    mlp = model.model.layers[2].mlp
    stack = row_forward.stack_of(mlp, "gu")
    for rows in (1, 3, row_qmv.MAX_ROWS):
        x = (mx.random.normal((1, rows, 1024), key=mx.random.key(rows)) * 0.5).astype(mx.bfloat16)
        fused = row_forward.row_qmv_gate_up_act(x, stack.weight, stack.scales, stack.biases, stack.group_size)
        plain = row_forward.mlp_act(row_qmv.qmv(x, stack.weight, stack.scales, stack.biases, stack.group_size))
        assert _same(fused, plain), rows


def test_folded_variants_keep_rows_and_partials(tiny):
    """The matmul with the norm on load / the residual epilogue: a row's bits do not depend on the rows beside it,
    the residual is mlx_lm's bf16 add, and the partial sums are row_parts' of the result."""

    model, _, _ = tiny
    mlp = model.model.layers[0].mlp
    gu = row_forward.stack_of(mlp, "gu")
    down = mlp.down_proj
    rows = row_qmv.MAX_ROWS
    h = (mx.random.normal((1, rows, 1024), key=mx.random.key(31)) * 2).astype(mx.bfloat16)
    parts = row_forward.row_parts(h)
    nw = (mx.random.normal((1024,), key=mx.random.key(32)) * 0.1 + 1).astype(mx.bfloat16)
    norm = (parts, nw, 1e-6)
    act = row_forward.row_qmv_variant(h, gu.weight, gu.scales, gu.biases, 64, norm=norm, epilogue="act")
    for r in range(rows):
        one = row_forward.row_qmv_variant(h[:, r:r + 1], gu.weight, gu.scales, gu.biases, 64,
                                          norm=(parts[:, r:r + 1], nw, 1e-6), epilogue="act")
        assert _same(one[0, 0], act[0, r])
    res = (mx.random.normal((1, rows, 1024), key=mx.random.key(33))).astype(mx.bfloat16)
    h2, parts2 = row_forward.row_qmv_variant(act, down["weight"], down["scales"], down["biases"], 64,
                                             epilogue="residual", res=res)
    y = row_qmv.qmv(act, down["weight"], down["scales"], down["biases"], 64)
    assert _same(h2, (res.astype(mx.float32) + y.astype(mx.float32)).astype(mx.bfloat16))
    assert bool(mx.array_equal(parts2, row_forward.row_parts(h2)).item())


@pytest.mark.parametrize("fold", [False, True])
def test_backends_windows_reproduce_one_row_steps(tiny, fold):
    """Plain row_qmv (separate norm kernels) and row_qmv with the norms folded in (``Backend.variant``)."""

    model, _, _ = tiny
    saved = row_forward.BACKEND
    row_forward.BACKEND = row_forward.Backend("row_qmv", row_qmv.qmv, row_qmv.MAX_ROWS, row_qmv.fits,
                                              variant=row_forward.row_qmv_variant if fold else None)
    try:
        mx.random.seed(15)
        prompt = [int(t) for t in mx.random.randint(0, 512, (9,)).tolist()]
        tokens = [int(t) for t in mx.random.randint(0, 512, (6,)).tolist()]
        base = _prefill(model, prompt)
        serial_cache = LaneEngine.copy_single_cache(base)
        serial = [_run(model, [t], serial_cache, len(prompt) + i)[0, -1] for i, t in enumerate(tokens)]
        window = _run(model, tokens, LaneEngine.copy_single_cache(base), len(prompt))
        for i in range(len(tokens)):
            assert _same(window[0, i], serial[i]), i
    finally:
        row_forward.BACKEND = saved


def test_aligned_prefill_resumes_exactly(tiny, monkeypatch):
    """MLX's prefill on a grid: a prompt resumed from a grid checkpoint gets a fresh prefill's bits; a checkpoint
    asked for off the grid is taken at the grid point below it; decoded states are not kept."""

    from tensorfold.engine.lane_engine import LaneStream

    model, _, _ = tiny
    monkeypatch.setattr(LaneEngine, "lane_prefill", 0)
    monkeypatch.setattr(LaneEngine, "prefill_step", 8)
    monkeypatch.setattr(LaneEngine, "prefill_align", 8)
    engine = LaneEngine(model, max_rows=1, max_draft=0, retain_finished_caches=True)
    mx.random.seed(21)
    prompt = [int(t) for t in mx.random.randint(0, 512, (29,)).tolist()]

    def arrays(cache):
        return [a for c in cache for a in c.state if a is not None]

    fresh = engine.prefill(LaneStream("fresh", prompt, max_new_tokens=1))
    first = LaneStream("first", prompt[:19], max_new_tokens=1)
    engine.prefill(first, checkpoints_at=[13, 19])
    assert [len(tokens) for tokens, _ in first.history_checkpoints] == [8, 16]
    tokens, cache = first.history_checkpoints[-1]
    resumed = engine.prefill(LaneStream("resumed", prompt, max_new_tokens=1), cache=cache, cached_tokens=len(tokens))
    for a, b in zip(arrays(fresh), arrays(resumed)):
        assert _same(a, b)
    off_grid = engine.prefill(LaneStream("off", prompt, max_new_tokens=1),
                              cache=LaneEngine.copy_single_cache(cache), cached_tokens=13)
    for a, b in zip(arrays(fresh), arrays(off_grid)):
        assert _same(a, b)
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
        for x, y in zip(a.state, b.state):
            if x is not None:
                assert _same(x, y)
