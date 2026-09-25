"""Stacked projections (lane_fuse): the same bits as separate lane matmuls, no weight stored twice."""

import gc

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.kernels.qwen.dense.v1 import lane_fuse, lane_glue, lane_qmm  # noqa: E402

from tensorfold.kernels.qwen.dense.v1 import lane_tree  # noqa: E402

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


def _same(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == mx.float32:
        return bool(mx.all(a.view(mx.uint32) == b.view(mx.uint32)).item())
    if a.dtype == mx.uint32 or a.dtype == mx.int32:
        return bool(mx.all(a == b).item())
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _quantized(n, k, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=64, bits=4)          # bf16 scales and biases, as the checkpoint has


def test_split_k_of_the_groups():
    expect = {10240: 4, 6144: 8, 48: 8, 12288: 4, 1024: 8, 17408: 2}
    for n, sk in expect.items():
        assert lane_qmm.split_k(n, K) == sk, n
    # every group's members agree, so a stack computed with that split gives each member its own bits
    for sizes in GROUP_SIZES.values():
        assert len({lane_qmm.split_k(n, K) for n in sizes}) == 1


@pytest.mark.parametrize("kind", sorted(GROUP_SIZES))
@pytest.mark.parametrize("tile", [True, False])
def test_stacked_matmul_bits(kind, tile):
    """The stack's columns equal each member's own call, in the layout install() leaves it."""

    _needs_tensor_units()
    sizes = GROUP_SIZES[kind]
    qs = [_quantized(n, K, 100 + i) for i, n in enumerate(sizes)]
    sbts = [lane_qmm.pack_scales(s, b) for _, s, b in qs]
    tiled = [tile and n % lane_qmm.NT == 0 for n in sizes]
    ws = [lane_qmm.tile_weight(q) if t else q for (q, _, _), t in zip(qs, tiled)]
    if all(tiled) or not any(tiled):
        stack = mx.concatenate(ws, axis=0)
    else:                                   # z tiled, [b; a] (96 rows, 3 tiles) tiled into the stack
        stack = mx.concatenate([ws[0], lane_qmm.tile_weight(mx.concatenate([q for q, _, _ in qs[1:]], axis=0))])
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
    assert _same(ours, ref) and _same(ours_xs, ref_xs)


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
    windows = lane_tree._conv_windows(parents, taps - 1)
    z = mx.contiguous(zba[..., :nv * dv])
    b = mx.contiguous(zba[..., nv * dv:nv * dv + nv])
    a = mx.contiguous(zba[..., nv * dv + nv:])
    heads = dict(nk=nk, nv=nv, dk=dk, dv=dv)
    ref = lane_glue.gdn_pre(qkv, conv_state, conv_weight, windows, a, b, a_log, dt_bias, **heads)
    ours = lane_fuse.gdn_pre(qkv, conv_state, conv_weight, windows, zba, a_log, dt_bias, **heads)
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


def _real_groups():
    root = nn.Module()
    root.linear_attn = _Holder([("in_proj_z", 6144), ("in_proj_b", 48), ("in_proj_a", 48)])
    root.self_attn = _Holder([("k_proj", 1024), ("v_proj", 1024)])
    root.mlp = _Holder([("gate_proj", 17408), ("up_proj", 17408)])
    for i, (_, module) in enumerate(root.named_modules()):
        if isinstance(module, nn.Linear):
            mx.random.seed(300 + i)
            module.weight = (mx.random.normal(module.weight.shape) * 0.02).astype(mx.bfloat16)
    nn.quantize(root, group_size=64, bits=4)
    mx.eval(root.parameters())
    return root


def test_build_keeps_the_weights_and_adds_only_the_scales():
    _needs_tensor_units()
    root = _real_groups()
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
        expect += 96 * (K // 8) * 4
        assert added == expect
        assert abs(grown - expect) < 1024**2 and grown < 0.2 * weight_bytes, (grown, expect, weight_bytes)
        for ms in members.values():
            for m in ms:
                w = m["weight"]
                seen = lane_qmm.untile_weight(w) if getattr(m, "_lane_tiled", False) else w
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


def _tiny_model():
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
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


def test_tree_forward_fused_equals_unfused():
    """Whole lane-decoder rounds: prompt chain, draft tree, commit, chain, one row; every bit the same."""

    _needs_tensor_units()
    model = _tiny_model()
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
            logits, record = lane_tree.tree_forward(core, head, tokens, rows_parents, cache, start, pipeline_layers=2)
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
