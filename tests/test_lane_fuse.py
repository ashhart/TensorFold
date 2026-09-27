"""Stacked projections (lane_fuse): the same bits as separate lane matmuls, no weight stored twice."""

import gc

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.kernels.qwen.dense.v1 import lane_fuse, lane_glue, lane_qmm  # noqa: E402

from tensorfold.kernels.qwen.dense.v1 import (  # noqa: E402
    lane_attention, lane_multi, lane_tree, stream_attention, stream_gdn)
from tests.kernel_signatures import changed, recording  # noqa: E402

K = 5120
ROWS = (1, 7, 16, 17, 32, 64, 128)
# Qwen3.8-27B: the stacked groups' real shapes (members in stacking order)
GROUP_SIZES = {"zba": (6144, 48, 48), "kv": (1024, 1024), "gu": (17408, 17408)}


def _needs_tensor_units():
    try:
        w = mx.zeros((32, 8), dtype=mx.uint32)
        s = mx.ones((32, 1), dtype=mx.bfloat16)
        mx.eval(lane_qmm.lane_matmul(mx.ones((1, 64), dtype=mx.bfloat16), w, lane_qmm.pack_scales(s, s)))
    except Exception as exc:  # noqa: BLE001 - no Metal 4 tensor ops on this machine
        pytest.skip(f"tensor-unit kernels unavailable: {str(exc).splitlines()[0][:80]}")


def _same(a, b, *, finite=True):
    """The same bits, and (``finite``) no inf or NaN: a lost dispatch can leave the same NaN on both sides."""

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if finite and mx.issubdtype(a.dtype, mx.floating) and not bool(mx.all(mx.isfinite(a)).item()):
        return False
    if a.dtype == mx.float32:
        return bool(mx.all(a.view(mx.uint32) == b.view(mx.uint32)).item())
    if a.dtype == mx.uint32 or a.dtype == mx.int32:
        return bool(mx.all(a == b).item())
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _quantized(n, k, seed, bits=4):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=64, bits=bits)       # bf16 scales and biases, as the checkpoint has


def test_split_k_of_the_groups():
    expect = {10240: 4, 6144: 8, 48: 8, 12288: 4, 1024: 8, 17408: 2}
    for n, sk in expect.items():
        assert lane_qmm.split_k(n, K) == sk, n
    # every group's members agree, so a stack computed with that split gives each member its own bits
    for sizes in GROUP_SIZES.values():
        assert len({lane_qmm.split_k(n, K) for n in sizes}) == 1


@pytest.mark.parametrize("bits", [4, 3, 2])
@pytest.mark.parametrize("kind", sorted(GROUP_SIZES))
@pytest.mark.parametrize("tile", [True, False])
def test_stacked_matmul_bits(kind, tile, bits):
    """The stack's columns equal each member's own call, in the layout install() leaves it."""

    _needs_tensor_units()
    sizes = GROUP_SIZES[kind]
    qs = [_quantized(n, K, 100 + i, bits) for i, n in enumerate(sizes)]
    sbts = [lane_qmm.pack_scales(s, b) for _, s, b in qs]
    tiled = [tile and n % lane_qmm.NT == 0 for n in sizes]
    ws = [lane_qmm.tile_weight(q, bits=bits) if t else q for (q, _, _), t in zip(qs, tiled)]
    if all(tiled) or not any(tiled):
        stack = mx.concatenate(ws, axis=0)
    else:                                   # z tiled, [b; a] (96 rows, 3 tiles) tiled into the stack
        tail = mx.concatenate([q for q, _, _ in qs[1:]], axis=0)
        stack = mx.concatenate([ws[0], lane_qmm.tile_weight(tail, bits=bits)])
    sbt = mx.concatenate(sbts, axis=1)
    sk = lane_qmm.split_k(sizes[0], K)
    mx.random.seed(1)
    x = (mx.random.normal((128, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(stack, sbt, x, *ws)
    for m in ROWS:
        fused = lane_qmm.lane_matmul(x[:m], stack, sbt, tiled=any(tiled), sk=sk)
        off = 0
        for w, s, t, n in zip(ws, sbts, tiled, sizes):
            alone = lane_qmm.lane_matmul(x[:m], w, s, tiled=t)
            assert _same(fused[:, off:off + n], alone), f"{kind} {m} rows: columns [{off}, {off + n}) changed"
            off += n


def _xs_of(out):
    hit = lane_qmm._xs_cache.get(id(out))
    assert hit is not None and hit[0] is out
    return hit[1]


@pytest.mark.parametrize("W", [1, 7, 16, 17, 32])
def test_mlp_act_reads_the_stack_in_place(W):
    _needs_tensor_units()
    N = 17408
    mx.random.seed(W)
    gu = (mx.random.normal((1, W, 2 * N)) * 3.0).astype(mx.bfloat16)
    if W >= 4:
        # every bf16 bit pattern as a gate value (inf and NaN included), in the first 4 rows
        flat = gu.reshape(W, 2 * N)
        pattern = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16)
        pattern = mx.concatenate([pattern, mx.zeros((4 * N - 65536,), dtype=mx.bfloat16)]).reshape(4, N)
        flat = mx.concatenate([mx.concatenate([pattern, flat[:4, N:]], axis=1), flat[4:]], axis=0)
        gu = flat.reshape(1, W, 2 * N)
    gate, up = mx.contiguous(gu[..., :N]), mx.contiguous(gu[..., N:])
    ref = lane_glue.mlp_act(gate, up)
    ref_xs = _xs_of(ref)
    ours = lane_fuse.mlp_act(gu)
    ours_xs = _xs_of(ours)
    mx.eval(ref, ref_xs, ours, ours_xs)
    assert _same(ours, ref, finite=False) and _same(ours_xs, ref_xs, finite=False)   # inf and NaN gates, bit for bit


@pytest.mark.parametrize("W", [1, 7, 16, 17, 32])
def test_gdn_glue_reads_the_stack_in_place(W):
    _needs_tensor_units()
    nk, nv, dk, dv, taps = 16, 48, 128, 128, 4
    C = 2 * nk * dk + nv * dv
    zs = nv * dv + 2 * nv
    mx.random.seed(40 + W)
    qkv = (mx.random.normal((1, W, C)) * 2.0).astype(mx.bfloat16)
    conv_state = (mx.random.normal((1, taps - 1, C)) * 2.0).astype(mx.bfloat16)
    conv_weight = (mx.random.normal((C, taps, 1)) * 0.5).astype(mx.bfloat16)
    zba = (mx.random.normal((1, W, zs)) * 4.0).astype(mx.bfloat16)
    a_log = mx.log(mx.random.uniform(low=0.5, high=16.0, shape=(nv,)))                   # fp32, as served
    dt_bias = (mx.random.normal((nv,)) * 2.0).astype(mx.bfloat16)
    parents = [-1] + [max(-1, r - 1 - (r % 3 == 0)) for r in range(1, W)]
    plan = stream_gdn.ConvPlan([parents], taps - 1)
    windows = plan.windows
    z = mx.contiguous(zba[..., :nv * dv])
    b = mx.contiguous(zba[..., nv * dv:nv * dv + nv])
    a = mx.contiguous(zba[..., nv * dv + nv:])
    heads = dict(nk=nk, nv=nv, dk=dk, dv=dv)
    ref = lane_glue.gdn_pre(qkv, conv_state, conv_weight, windows, a, b, a_log, dt_bias, **heads)
    ours = stream_gdn.gdn_pre(qkv, [conv_state], conv_weight, plan, zba, a_log, dt_bias, **heads)
    mx.eval(ref, ours)
    for name, r, o in zip(("q", "k", "v", "g", "beta"), ref, ours):
        assert _same(o, r), f"gdn_pre {name} changed"
    y = (mx.random.normal((1, W, nv, dv)) * 2.0).astype(mx.bfloat16)
    norm_w = (1.0 + 0.1 * mx.random.normal((dv,))).astype(mx.bfloat16)
    ref = lane_glue.gdn_post(y, z, norm_w, 1e-6)
    ref_xs = _xs_of(ref)
    ours = lane_fuse.gdn_post(y, zba, norm_w, 1e-6)
    ours_xs = _xs_of(ours)
    mx.eval(ref, ref_xs, ours, ours_xs)
    assert _same(ours, ref) and _same(ours_xs, ref_xs)


class _Holder(nn.Module):
    def __init__(self, names_sizes):
        super().__init__()
        for name, n in names_sizes:
            setattr(self, name, nn.Linear(K, n, bias=False))


def _real_groups(bits=4):
    root = nn.Module()
    root.linear_attn = _Holder([("in_proj_z", 6144), ("in_proj_b", 48), ("in_proj_a", 48)])
    root.self_attn = _Holder([("k_proj", 1024), ("v_proj", 1024)])
    root.mlp = _Holder([("gate_proj", 17408), ("up_proj", 17408)])
    for i, (_, module) in enumerate(root.named_modules()):
        if isinstance(module, nn.Linear):
            mx.random.seed(300 + i)
            module.weight = (mx.random.normal(module.weight.shape) * 0.02).astype(mx.bfloat16)
    nn.quantize(root, group_size=64, bits=bits)
    mx.eval(root.parameters())
    return root


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_build_keeps_the_weights_and_adds_only_the_scales(bits):
    _needs_tensor_units()
    root = _real_groups(bits)
    members = {kind: [getattr(parent, n) for n in lane_fuse.GROUPS[kind]]
               for kind, parent in (("zba", root.linear_attn), ("kv", root.self_attn), ("gu", root.mlp))}
    originals = {id(m): m["weight"] for ms in members.values() for m in ms}          # MLX's layout
    mx.random.seed(9)
    x = (mx.random.normal((1, 16, K)) * 0.5).astype(mx.bfloat16)
    saved = lane_fuse.enabled
    try:
        lane_qmm.install(root, rows=lane_qmm.MAX_ROWS)
        alone = {id(m): m(x) for ms in members.values() for m in ms}
        mx.eval(alone)
        weight_bytes = sum(m["weight"].nbytes for ms in members.values() for m in ms)
        gc.collect()
        before = mx.get_active_memory()
        counts = lane_fuse.build(root)
        gc.collect()
        grown = mx.get_active_memory() - before
        assert counts == {"zba": 1, "kv": 1, "gu": 1}
        added = lane_fuse.stats(root)["added_bytes"]
        # the stacks' scales and the tiled [b; a] copy, nothing more: the old weight arrays were freed
        expect = sum(sum(lane_qmm.pack_scales(m["scales"], m["biases"]).nbytes for m in ms) for ms in members.values())
        expect += 96 * (K * bits // 32) * 4
        assert added == expect
        # scales are 1/(2 * bits) of the weights' bytes (1/8 at 4 bits): a second copy of the weights would add them all
        assert abs(grown - expect) < 1024**2 and grown < 1.6 / (2 * bits) * weight_bytes, (grown, expect, weight_bytes)
        for ms in members.values():
            for m in ms:
                w = m["weight"]
                seen = lane_qmm.untile_weight(w, bits=bits) if getattr(m, "_lane_tiled", False) else w
                assert _same(seen, originals[id(m)]), "a member's weight changed"
        # each member's own call (on its view of the stack) keeps its bits, and the stack gives them too
        lane_fuse.enabled = True
        fused = {"zba": lane_fuse.gdn_in(root.linear_attn, x), "kv": lane_fuse.attn_kv(root.self_attn, x),
                 "gu": lane_fuse.mlp_gate_up(root.mlp, x)}
        for kind, ms in members.items():
            off = 0
            for m in ms:
                n = int(m["weight"].shape[0])
                assert _same(m(x), alone[id(m)])
                assert _same(fused[kind][..., off:off + n], alone[id(m)]), kind
                off += n
        # switched off, or wider than the lane kernel: the members' own calls
        lane_fuse.enabled = False
        assert lane_fuse.mlp_gate_up(root.mlp, x) is None
        lane_fuse.enabled = True
        assert lane_fuse.mlp_gate_up(root.mlp, mx.zeros((1, lane_qmm.MAX_ROWS + 1, K), dtype=mx.bfloat16)) is None
    finally:
        lane_fuse.enabled = saved
        lane_qmm.uninstall()
    # uninstall gave the members MLX's layout back as new arrays: the stacks are stale, then dropped
    for ms in members.values():
        for m in ms:
            assert _same(m["weight"], originals[id(m)])
    assert lane_fuse._group(root.mlp, "gu", build=False) is None
    lane_fuse.clear(root)


def test_groups_of_mixed_widths_stay_separate():
    """A mixed 3/4-bit checkpoint (gate 3-bit, up 4-bit): no stack, each member keeps its own lane call."""

    _needs_tensor_units()
    root = nn.Module()
    root.mlp = _Holder([("gate_proj", 1024), ("up_proj", 1024)])
    for i, (_, module) in enumerate(root.mlp.named_modules()):
        if isinstance(module, nn.Linear):
            mx.random.seed(400 + i)
            module.weight = (mx.random.normal(module.weight.shape) * 0.02).astype(mx.bfloat16)
    nn.quantize(root.mlp, group_size=64, bits=3, class_predicate=lambda p, m: p == "gate_proj")
    nn.quantize(root.mlp, group_size=64, bits=4, class_predicate=lambda p, m: p == "up_proj")
    mx.eval(root.parameters())
    assert (root.mlp.gate_proj.bits, root.mlp.up_proj.bits) == (3, 4)
    x = (mx.random.normal((1, 16, K)) * 0.5).astype(mx.bfloat16)
    saved = lane_fuse.enabled
    try:
        lane_qmm.install(root, rows=lane_qmm.MAX_ROWS)
        alone = [m(x) for m in (root.mlp.gate_proj, root.mlp.up_proj)]
        mx.eval(alone)
        assert lane_fuse.build(root) == {"zba": 0, "kv": 0, "gu": 0}
        lane_fuse.enabled = True
        assert lane_fuse.mlp_gate_up(root.mlp, x) is None
        for m, y in zip((root.mlp.gate_proj, root.mlp.up_proj), alone):
            assert _same(m(x), y)
    finally:
        lane_fuse.enabled = saved
        lane_qmm.uninstall()
        lane_fuse.clear(root)


def _tiny_model(bits=4):
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    # Qwen3.8's head shapes and MLP width on a 1024-wide residual: the gate/up stack alone would
    # get split 1 (members: 2), the [z; b; a] tail is 3 tiles, attention has 4 kv heads of 256
    args = TextModelArgs(model_type="qwen3_5_text", hidden_size=1024, intermediate_size=17408, num_hidden_layers=4,
                         num_attention_heads=24, num_key_value_heads=4, head_dim=256, rms_norm_eps=1e-6,
                         vocab_size=512, linear_num_value_heads=48, linear_num_key_heads=16,
                         linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                         full_attention_interval=4, tie_word_embeddings=False, max_position_embeddings=4096)
    mx.random.seed(21)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_tree_forward_fused_equals_unfused(bits):
    """Whole lane-decoder rounds: prompt chain, draft tree, commit, chain, one row; every bit the same."""

    _needs_tensor_units()
    _check_fused_rounds(bits, pipeline_layers=2)


def _check_fused_rounds(bits, *, pipeline_layers):
    model = _tiny_model(bits)
    core, head = model.model, model.lm_head
    mx.random.seed(5)
    prompt = [int(t) for t in mx.random.randint(0, 512, (40,)).tolist()]      # gate/up stacked (> 32 rows)
    tree = [int(t) for t in mx.random.randint(0, 512, (16,)).tolist()]        # all stacked
    chain = [int(t) for t in mx.random.randint(0, 512, (20,)).tolist()]       # gate/up separate (17-32 rows)
    parents = [-1, 0, 1, 2, 0, 4, 5, 1, 7, 3, 9, 10, 2, 12, 13, 14]
    path = [0, 1, 2, 3, 9, 10, 11]                    # not in place: the commit moves rows

    def rounds():
        cache = model.make_cache()
        outs = []
        start = 0

        def step(tokens, rows_parents, keep):
            nonlocal start
            logits, record = lane_tree.tree_forward(core, head, tokens, rows_parents, cache, start,
                                                    pipeline_layers=pipeline_layers)
            outs.append(logits)
            for entry in record:
                outs.extend(a for a in entry[1:] if isinstance(a, mx.array))
            lane_tree.commit_tree(cache, record, keep, len(tokens), start)
            start += len(keep)

        step(prompt, [-1] + list(range(len(prompt) - 1)), list(range(len(prompt))))
        step(tree, parents, path)
        step(chain, [-1] + list(range(len(chain) - 1)), list(range(len(chain))))
        step([7], [-1], [0])
        outs.extend(a for c in cache for a in c.state if a is not None)
        mx.eval(outs)
        return outs

    saved = lane_fuse.enabled
    try:
        lane_qmm.install(model, rows=lane_qmm.MAX_ROWS)
        lane_fuse.enabled = False
        plain = rounds()
        assert lane_fuse.stats(model)["groups"] == {"zba": 0, "kv": 0, "gu": 0}
        lane_fuse.enabled = True                      # stacked on first use (auto_build)
        fused = rounds()
        assert lane_fuse.stats(model)["groups"] == {"zba": 3, "kv": 1, "gu": 4}
        lane_fuse.enabled = False                     # the members are views of the stacks now
        after = rounds()
    finally:
        lane_fuse.enabled = saved
        lane_qmm.uninstall()
        lane_fuse.clear(model)
    assert len(plain) == len(fused) == len(after)
    for i, (p, f, a) in enumerate(zip(plain, fused, after)):
        assert _same(f, p), f"output {i} changed with the stacked projections"
        assert _same(a, p), f"output {i} changed after the stacks were built"


@pytest.mark.parametrize("bits", [4, 3, 2])
@pytest.mark.parametrize("fused", [False, True])
def test_chain_rows_equal_one_row_steps(bits, fused):
    """A 16-row chain's logits equal 16 one-row steps bit for bit, through the production install and stacks."""

    _needs_tensor_units()
    model = _tiny_model(bits)
    core, head = model.model, model.lm_head
    mx.random.seed(6)
    prompt = [int(t) for t in mx.random.randint(0, 512, (40,)).tolist()]
    chain = [int(t) for t in mx.random.randint(0, 512, (16,)).tolist()]

    def run(serial):
        cache = model.make_cache()
        start = 0

        def step(tokens):
            nonlocal start
            parents = [-1] + list(range(len(tokens) - 1))
            logits, record = lane_tree.tree_forward(core, head, tokens, parents, cache, start, pipeline_layers=2)
            lane_tree.commit_tree(cache, record, list(range(len(tokens))), len(tokens), start)
            start += len(tokens)
            mx.eval(logits)
            return logits

        step(prompt)
        return mx.concatenate([step([t]) for t in chain], axis=1) if serial else step(chain)

    saved = lane_fuse.enabled
    try:
        lane_qmm.install(model, rows=lane_qmm.MAX_ROWS, wide=True)
        assert lane_qmm.warm(model) > 0
        lane_fuse.enabled = fused
        if fused:
            assert lane_fuse.build(model) == {"zba": 3, "kv": 1, "gu": 4}
            assert lane_fuse.warm(model) > 0
        window, steps = run(serial=False), run(serial=True)
    finally:
        lane_fuse.enabled = saved
        lane_qmm.uninstall()
        lane_fuse.clear(model)
    for i in range(len(chain)):
        assert _same(window[:, i], steps[:, i]), f"chain row {i} differs from its one-row step"


def test_models_of_other_widths_in_one_process():
    """Rounds for a 4-bit, a 3-bit and a 4-bit model in one process, unpipelined: kernels compile mid-evaluation."""

    _needs_tensor_units()
    for bits in (4, 3, 4):
        _check_fused_rounds(bits, pipeline_layers=0)


def test_streams_conv_window_in_one_launch_equals_one_stream_calls():
    """Every row of a multi-stream gdn_pre has the bits of the same row from its own stream's call."""
    _needs_tensor_units()
    nk, nv, dk, dv, taps = 16, 48, 128, 128, 4
    C = 2 * nk * dk + nv * dv
    zs = nv * dv + 2 * nv
    mx.random.seed(77)
    parents = [[-1, 0, 0, 1, 3], [-1], [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    R = sum(len(p) for p in parents)
    qkv = (mx.random.normal((1, R, C)) * 2.0).astype(mx.bfloat16)
    zba = (mx.random.normal((1, R, zs)) * 4.0).astype(mx.bfloat16)
    states = [(mx.random.normal((1, taps - 1, C)) * 2.0).astype(mx.bfloat16) for _ in parents]
    conv_weight = (mx.random.normal((C, taps, 1)) * 0.5).astype(mx.bfloat16)
    a_log = mx.log(mx.random.uniform(low=0.5, high=16.0, shape=(nv,)))
    dt_bias = (mx.random.normal((nv,)) * 2.0).astype(mx.bfloat16)
    heads = dict(nk=nk, nv=nv, dk=dk, dv=dv)
    together = stream_gdn.gdn_pre(qkv, states, conv_weight, stream_gdn.ConvPlan(parents, taps - 1), zba, a_log,
                                  dt_bias, **heads)
    mx.eval(together)
    first = 0
    for s, rp in enumerate(parents):
        W = len(rp)
        alone = stream_gdn.gdn_pre(qkv[:, first:first + W], [states[s]], conv_weight,
                                   stream_gdn.ConvPlan([rp], taps - 1), zba[:, first:first + W], a_log, dt_bias, **heads)
        mx.eval(alone)
        for name, t, a in zip(("q", "k", "v", "g", "beta"), together, alone):
            assert _same(t[:, first:first + W], a), f"stream {s} {name} differs from its own call"
        first += W


def _chain(n):
    return [-1] + list(range(n - 1))


@pytest.mark.parametrize("bits", [4, 3, 2])
@pytest.mark.parametrize("fused", [False, True])
def test_kernel_signatures_do_not_change_between_calls(bits, fused):
    """One Metal signature per lane kernel, from install and warm-ups through one-stream and 1-9-stream rounds."""

    _needs_tensor_units()
    model = _tiny_model(bits)
    core, head = model.model, model.lm_head
    mx.random.seed(8)
    tokens = [int(t) for t in mx.random.randint(0, 512, (600,)).tolist()]
    tree_parents = [-1, 0, 1, 2, 0, 4, 5, 1, 7, 3, 9, 10, 2, 12, 13, 14]
    one = ((40, _chain(40), list(range(40))), (16, tree_parents, [0, 1, 2, 3, 9, 10, 11]),
           (20, _chain(20), list(range(20))), (1, [-1], [0]), (3, _chain(3), [0, 1, 2]), (1, [-1], [0]))
    # shared rounds: (stream, rows' parents, kept path) per stream taking part
    shared = ([(0, tree_parents, [0, 4, 5, 6]), (1, _chain(5), [0, 1]), (2, [-1], [0])],
              [(0, [-1], [0]), (2, _chain(3), [0, 1, 2])],
              [(1, _chain(8), list(range(8)))],
              [(s, [-1] if s % 3 else _chain(2), [0]) for s in range(9)],
              [(0, _chain(2), [0, 1]), (1, [-1], [0]), (2, [-1, 0, 0, 1], [0, 2]), (8, _chain(13), list(range(4)))])
    caches = [lane_qmm._kernels, lane_attention._kernels, lane_glue._kernels, lane_tree._kernels, lane_fuse._variants,
              stream_attention._kernels, stream_gdn._commit_kernels, stream_gdn._kernel_cache]
    saved_kernels = [c.copy() for c in caches]
    saved = lane_fuse.enabled
    at = 0

    def take(n):
        nonlocal at
        at += n
        return tokens[at - n:at]

    try:
        with recording() as seen:
            for c in caches:
                c.clear()                             # made again, inside the recorder
            lane_qmm.install(model, rows=lane_qmm.MAX_ROWS, wide=True)
            lane_qmm.warm(model)
            lane_attention.warm(max_queries=lane_attention.MAX_QUERIES)
            lane_fuse.enabled = fused
            if fused:
                lane_fuse.build(model)
                lane_fuse.warm(model)
            cache = model.make_cache()
            start = 0
            for n, parents, keep in one:
                logits, record = lane_tree.tree_forward(core, head, take(n), parents, cache, start, pipeline_layers=2)
                lane_tree.commit_tree(cache, record, keep, n, start)
                start += len(keep)
                mx.eval(logits, [a for c in cache for a in c.state if a is not None])
                assert bool(mx.all(mx.isfinite(logits)).item())
            streams = [model.make_cache() for _ in range(9)]
            starts = [0] * 9
            prompts = [9 + 7 * s for s in range(9)]
            for group in ((0, 1, 2, 3), (4, 5), (6, 7), (8,)):      # prompts of 9 to 65 rows, up to 128 a forward
                windows = [take(prompts[s]) for s in group]
                chains = [_chain(len(w)) for w in windows]
                logits, records, _ = lane_multi.multi_tree_forward(core, head, windows, chains,
                                                                   [streams[s] for s in group], [0] * len(group))
                lane_multi.commit_streams([streams[s] for s in group], records, [list(range(len(w))) for w in windows],
                                          [len(w) for w in windows], [0] * len(group))
                for s in group:
                    starts[s] = prompts[s]
                mx.eval(logits)
            for plan in shared:
                ids = [s for s, _, _ in plan]
                windows = [take(len(p)) for _, p, _ in plan]
                logits, records, _ = lane_multi.multi_tree_forward(core, head, windows, [p for _, p, _ in plan],
                                                                   [streams[s] for s in ids], [starts[s] for s in ids])
                lane_multi.commit_streams([streams[s] for s in ids], records, [k for _, _, k in plan],
                                          [len(w) for w in windows], [starts[s] for s in ids])
                for s, _, keep in plan:
                    starts[s] += len(keep)
                mx.eval(logits, [a for s in ids for c in streams[s] for a in c.state if a is not None])
                assert bool(mx.all(mx.isfinite(logits)).item())
    finally:
        lane_fuse.enabled = saved
        lane_qmm.uninstall()
        lane_fuse.clear(model)
        for c, old in zip(caches, saved_kernels):
            c.clear()
            if isinstance(c, list):
                c.extend(old)
            else:
                c.update(old)
    names = {name.rsplit("_", 1)[0] for name, _ in seen}
    expect = {"stream_attention_partial", "stream_attention_tail", "stream_attention_merge", "stream_gdn_tree",
              "stream_gdn_replay", "stream_gdn_tails", "stream_gdn_pre" if fused else "lane_glue_gdn_pre",
              "lane_attention_partial_direct", "lane_attention_merge"}
    assert expect <= names, names
    assert not changed(seen), "kernels called with more than one signature: " + changed(seen)
