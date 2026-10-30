//! Flash Next's launches: stream norms, gates and mixes, the n-gram gate, conv and hashes, indexer, sparse attention.

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const I = util.I;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;

pub const Fns = struct {
    hc_norm: Function,
    hc_norm_moe: Function,
    hc_norm_v: Function, // bf16 streams of 2,560, one rank: 128 threads a block, 16-byte accesses, the same bits
    hc_norm_moe_v: Function, // hc_norm_v with the MoE combine as the branch, a block a row (a prompt's rows)
    hc_gates: Function,
    rms_wide: Function,
    hc_mix: Function,
    hc_mix8: Function, // bf16 activations, g bf16 or fp32: 8 outputs a thread, the same bits
    ple_gate: Function,
    ple_conv: Function,
    ngram_ids: Function,
    idx_pool: Function,
    qsa_prep: Function,
    idx_scores: Function,
    idx_scores_quad: Function,
    idx_scores8: Function, // a verify window's 5-15 rows: 8 rows' dots a wave at once, the same scores
    idx_select: Function,
    sparse_parts: [2]Function, // groups of up to 6 and up to 12 query heads
    sparse_merge: Function,

    pub fn load(m: Module) Error!Fns {
        return .{
            .hc_norm = try m.function("tf_fn_hc_norm"),
            .hc_norm_moe = try m.function("tf_fn_hc_norm_moe"),
            .hc_norm_v = try m.function("tf_fn_hc_norm_v"),
            .hc_norm_moe_v = try m.function("tf_fn_hc_norm_moe_v"),
            .hc_gates = try m.function("tf_fn_hc_gates"),
            .rms_wide = try m.function("tf_fn_rms_wide"),
            .hc_mix = try m.function("tf_fn_hc_mix"),
            .hc_mix8 = try m.function("tf_fn_hc_mix8"),
            .ple_gate = try m.function("tf_fn_ple_gate"),
            .ple_conv = try m.function("tf_fn_ple_conv"),
            .ngram_ids = try m.function("tf_fn_ngram_ids"),
            .idx_pool = try m.function("tf_fn_idx_pool"),
            .qsa_prep = try m.function("tf_fn_qsa_prep"),
            .idx_scores = try m.function("tf_fn_idx_scores"),
            .idx_scores_quad = try m.function("tf_fn_idx_scores_quad"),
            .idx_scores8 = try m.function("tf_fn_idx_scores8"),
            .idx_select = try m.function("tf_fn_idx_select"),
            .sparse_parts = .{ try m.function("tf_fn_sparse_parts6"), try m.function("tf_fn_sparse_parts12") },
            .sparse_merge = try m.function("tf_fn_sparse_merge"),
        };
    }
};

/// The widest stream the norm and gate kernels walk: 16 values a thread of 256.
pub const most_dims = 4096;
const threads = 256;

/// `rows` rows of `streams` streams: h += round(branch * inject) when a branch is given, then x = h rinv scale.
pub fn hcNorm(
    l: *const Launcher,
    h: C,
    branch: C,
    inject: C,
    h_out: P,
    x_out: P,
    scale: CF,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    eps: f32,
    s: S,
) Error!void {
    return hcNormShares(l, h, branch, inject, h_out, x_out, scale, kind, rows, streams, dims, eps, 0, 0, s);
}

/// hcNorm with the MoE's combine of fp32 expert rows `y` [rows, slots, dims] by `wts` [rows, slots] as the branch.
pub fn hcNormMoe(
    l: *const Launcher,
    h: C,
    y: CF,
    wts: CF,
    inject: C,
    h_out: P,
    x_out: P,
    scale: CF,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    eps: f32,
    slots: c_int,
    s: S,
) Error!void {
    if (dims > most_dims or slots < 1) return util.invalid("hcNormMoe");
    var a: Args = .{};
    inline for (.{ ad(h), ad(y), ad(wts), ad(inject), ad(h_out), ad(x_out), ad(scale), kind, streams, dims, eps, slots }) |v| try a.add(v);
    try l.go(l.fnx.hc_norm_moe, dim(streams, rows, 1), dim(1024, 1, 1), 0, s, &a);
}

/// hcNormMoe for a prompt's bf16 streams of 2,560 on one rank, a block a row, same bits; the caller checks the shape.
pub fn hcNormMoeV(
    l: *const Launcher,
    h: C,
    y: CF,
    wts: CF,
    inject: C,
    h_out: P,
    x_out: P,
    scale: CF,
    rows: c_int,
    streams: c_int,
    eps: f32,
    slots: c_int,
    s: S,
) Error!void {
    if (streams < 1 or streams > 4 or slots < 1) return util.invalid("hcNormMoeV");
    var a: Args = .{};
    inline for (.{ ad(h), ad(y), ad(wts), ad(inject), ad(h_out), ad(x_out), ad(scale), streams, eps, slots }) |v| try a.add(v);
    try l.go(l.fnx.hc_norm_moe_v, dim(1, rows, 1), dim(128 * streams, 1, 1), 0, s, &a);
}

/// hcNorm with the branch as `world` fp32 shares `stride` values apart (a tensor-parallel gather), summed in order.
pub fn hcNormShares(
    l: *const Launcher,
    h: C,
    branch: C,
    inject: C,
    h_out: P,
    x_out: P,
    scale: CF,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    eps: f32,
    world: c_int,
    stride: i64,
    s: S,
) Error!void {
    if (dims > most_dims) return util.invalid("hcNorm");
    if (world == 0 and kind == 2 and dims == 2560) {
        var v: Args = .{};
        inline for (.{ ad(h), ad(branch), ad(inject), ad(h_out), ad(x_out), ad(scale), streams, eps }) |x| try v.add(x);
        return l.go(l.fnx.hc_norm_v, dim(streams, rows, 1), dim(128, 1, 1), 0, s, &v);
    }
    var a: Args = .{};
    inline for (.{
        ad(h),
        ad(branch),
        ad(inject),
        ad(h_out),
        ad(x_out),
        ad(scale),
        kind,
        streams,
        dims,
        eps,
        world,
        stride,
    }) |v| try a.add(v);
    try l.go(l.fnx.hc_norm, dim(streams, rows, 1), dim(1024, 1, 1), 0, s, &a);
}

/// act = silu(z / streams) and inject = 2 sigmoid(z / streams) from the down projection's fp32 sums (no inject: null).
pub fn hcGates(
    l: *const Launcher,
    z: CF,
    act: P,
    inject: P,
    kind: c_int,
    rows: c_int,
    low: c_int,
    streams: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{ ad(z), ad(act), ad(inject), kind, rows, low, streams }) |v| try a.add(v);
    try l.flat(l.fnx.hc_gates, @as(i64, rows) * (low + if (inject != null) streams else 0), s, &a);
}

/// Each of `rows` rows normed over its whole `width` (any width) times the fp32 `scale`.
pub fn rmsWide(
    l: *const Launcher,
    x: C,
    scale: CF,
    y: P,
    kind: c_int,
    rows: c_int,
    width: c_int,
    eps: f32,
    s: S,
) Error!void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ad(x), ad(scale), ad(y), kind, width, eps }) |v| try a.add(v);
    try l.go(l.fnx.rms_wide, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// The block input: the mean over the streams of sigmoid(g) times the normed stream.
pub fn hcMix(
    l: *const Launcher,
    g: C,
    gkind: c_int,
    x: C,
    mixed: P,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    s: S,
) Error!void {
    if (kind == 2 and (gkind == 0 or gkind == 2) and @rem(dims, 8) == 0) {
        var v: Args = .{};
        inline for (.{ ad(g), gkind, ad(x), ad(mixed), rows, streams, dims }) |w| try v.add(w);
        return l.flat(l.fnx.hc_mix8, @divExact(@as(i64, rows) * dims, 8), s, &v);
    }
    return hcMixScalar(l, g, gkind, x, mixed, kind, rows, streams, dims, s);
}

/// hcMix on the per-element kernel (any kinds): the reference the bf16 path's bytes are tested against.
pub fn hcMixScalar(l: *const Launcher, g: C, gkind: c_int, x: C, mixed: P, kind: c_int, rows: c_int, streams: c_int, dims: c_int, s: S) Error!void {
    var a: Args = .{};
    inline for (.{ ad(g), gkind, ad(x), ad(mixed), kind, rows, streams, dims }) |v| try a.add(v);
    try l.flat(l.fnx.hc_mix, @as(i64, rows) * dims, s, &a);
}

/// The n-gram gate of every stream; `normed` is the conv's input rows.
pub fn pleGate(
    l: *const Launcher,
    kv: C,
    h: C,
    key_scale: CF,
    query_scale: CF,
    conv_scale: CF,
    gated: P,
    normed: P,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    eps: f32,
    s: S,
) Error!void {
    if (dims > most_dims) return util.invalid("pleGate");
    var a: Args = .{};
    inline for (.{
        ad(kv),
        ad(h),
        ad(key_scale),
        ad(query_scale),
        ad(conv_scale),
        ad(gated),
        ad(normed),
        kind,
        streams,
        dims,
        eps,
    }) |v| try a.add(v);
    try l.go(l.fnx.ple_gate, dim(streams, rows, 1), dim(threads, 1, 1), 0, s, &a);
}

/// out = h + (gated + silu(conv)): the dilated causal conv over `cin`, its tail rows first.
pub fn pleConv(
    l: *const Launcher,
    cin: C,
    weight: CF,
    gated: C,
    h: C,
    out: P,
    kind: c_int,
    rows: c_int,
    width: c_int,
    taps: c_int,
    dilation: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{
        ad(cin),
        ad(weight),
        ad(gated),
        ad(h),
        ad(out),
        kind,
        rows,
        width,
        taps,
        dilation,
    }) |v| try a.add(v);
    try l.flat(l.fnx.ple_conv, @as(i64, rows) * width, s, &a);
}

/// Each row's n-gram table rows; `meta` holds the history, eos, multipliers, sizes and offsets (int64).
pub fn ngramIds(
    l: *const Launcher,
    tokens: CI,
    meta: ?*const anyopaque,
    ids: P,
    rows: c_int,
    heads: c_int,
    per: c_int,
    n: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{ ad(tokens), ad(meta), ad(ids), rows, heads, per, n }) |v| try a.add(v);
    try l.flat(l.fnx.ngram_ids, @as(i64, rows) * heads, s, &a);
}

/// Blocks first .. first + blocks of the raw indexer keys, pooled, normed and rotated into `pooled`.
pub fn idxPool(
    l: *const Launcher,
    raw: C,
    pooled: P,
    scale: CF,
    kind: c_int,
    first: c_int,
    blocks: c_int,
    dims: c_int,
    ratio: c_int,
    rotary: c_int,
    theta: f32,
    eps: f32,
    s: S,
) Error!void {
    if (blocks == 0) return;
    if (dims > threads) return util.invalid("idxPool");
    var a: Args = .{};
    inline for (.{
        ad(raw),
        ad(pooled),
        ad(scale),
        kind,
        first,
        @as(u64, 0),
        dims,
        ratio,
        rotary,
        theta,
        eps,
    }) |v| try a.add(v);
    try l.go(l.fnx.idx_pool, dim(blocks, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// The blocks `rows` rows at device positions `pos` complete, pooled into their slots of `pooled` (all blocks').
pub fn idxPoolAt(
    l: *const Launcher,
    raw: C,
    pooled: P,
    scale: CF,
    kind: c_int,
    pos: CI,
    rows: c_int,
    dims: c_int,
    ratio: c_int,
    rotary: c_int,
    theta: f32,
    eps: f32,
    s: S,
) Error!void {
    if (rows == 0) return;
    if (dims > threads) return util.invalid("idxPoolAt");
    var a: Args = .{};
    inline for (.{
        ad(raw),
        ad(pooled),
        ad(scale),
        kind,
        @as(c_int, 0),
        ad(pos),
        dims,
        ratio,
        rotary,
        theta,
        eps,
    }) |v| try a.add(v);
    try l.go(l.fnx.idx_pool, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// flashnext/indexer.hpp's QsaRope and QsaPrep: a QSA layer's decode prep at device positions in one launch.
pub const QsaRope = extern struct { src: u64 = 0, s_row: c_longlong = 0, s_head: c_int = 0, heads: c_int = 0, width: c_int = 0, weight: u64 = 0, wide: u64 = 0, cache: u64 = 0 };
pub const QsaPrep = extern struct {
    q: QsaRope = .{},
    iq: QsaRope = .{},
    k: QsaRope,
    v: u64,
    v_cache: u64,
    v_width: c_int,
    raw: u64,
    raw_stride: c_longlong,
    raw_offset: c_int,
    raw_cache: u64,
    raw_width: c_int,
    pooled: u64,
    pool_scale: u64,
    ratio: c_int,
    pos: u64,
    seg_pos: u64,
    rows: c_int,
    n: c_int,
    total: c_int,
    kind: c_int,
    eps: f32,
    rotary: c_int,
    theta: f32,
};

comptime {
    std.debug.assert(@sizeOf(QsaRope) == 56 and @sizeOf(QsaPrep) == 304);
}

/// tf_fn_qsa_prep: a QSA layer's q/k rope, v and raw index key cache writes and completed-block pools in one launch.
pub fn qsaPrep(l: *const Launcher, a: QsaPrep, s: S) Error!void {
    if (a.n < 1 or a.k.width > 512 or a.q.width > 512 or a.iq.width > 512 or a.raw_width > threads or a.kind == 0) return util.invalid("qsaPrep");
    const qt: c_int = if (a.q.src != 0) a.rows * (a.q.heads + a.iq.heads) else 0;
    const tasks: c_int = qt + a.n * a.k.heads;
    var args: Args = .{};
    try args.add(a);
    try l.go(l.fnx.qsa_prep, dim(cdiv(tasks, threads / 32) + 1, 1, 1), dim(threads, 1, 1), 0, s, &args);
}

/// Rows from which the block scores take tf_fn_idx_scores_quad (fewer: a verify window's, a wave a block).
pub const quad_rows = 16;
/// Rows from which (to quad_rows) the block scores take tf_fn_idx_scores8; fewer rows keep the plain kernel.
pub const scores8_rows = 5;

/// Every row's score of each of `blocks` pooled blocks it has complete, for rows past `top` complete blocks.
pub fn idxScores(
    l: *const Launcher,
    q: CF,
    pooled: C,
    complete: CI,
    scores: F,
    kind: c_int,
    rows: c_int,
    heads: c_int,
    dims: c_int,
    blocks: c_int,
    stride: i64,
    top: c_int,
    s: S,
) Error!void {
    if (blocks == 0) return;
    if (dims > threads) return util.invalid("idxScores");
    // a prompt slice of Flash Next's 4 x 128 index heads: 4 lanes a score, the same sums (indexer.hpp)
    if (rows >= quad_rows and heads == 4 and dims == 128) {
        var b: Args = .{};
        inline for (.{ ad(q), ad(pooled), ad(complete), ad(scores), kind, rows, blocks, stride, top }) |v| try b.add(v);
        return l.go(l.fnx.idx_scores_quad, dim(cdiv(blocks, 64), cdiv(rows, 32), 1), dim(128, 1, 1), 0, s, &b);
    }
    // a verify window's rows of the same heads: a wave's 8 rows' dots at once, the same sums (indexer.hpp)
    if (rows >= scores8_rows and heads == 4 and dims == 128) {
        var c: Args = .{};
        inline for (.{ ad(q), ad(pooled), ad(complete), ad(scores), kind, rows, blocks, stride, top }) |v| try c.add(v);
        return l.go(l.fnx.idx_scores8, dim(cdiv(blocks, threads / 32), cdiv(rows, 8), 1), dim(threads, 1, 1), 0, s, &c);
    }
    var a: Args = .{};
    inline for (.{
        ad(q),
        ad(pooled),
        ad(complete),
        ad(scores),
        kind,
        rows,
        heads,
        dims,
        blocks,
        stride,
        top,
    }) |v| try a.add(v);
    try l.go(l.fnx.idx_scores, dim(cdiv(blocks, 8), cdiv(rows, 8), 1), dim(threads, 1, 1), 0, s, &a);
}

/// Each row's keys: its `top` best blocks and its tail past `top` complete blocks, else its first ends[r] keys.
pub fn idxSelect(
    l: *const Launcher,
    scores: CF,
    stride: i64,
    complete: CI,
    ends: CI,
    keys: I,
    key_stride: i64,
    count: I,
    sparse: I,
    rows: c_int,
    top: c_int,
    ratio: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{
        ad(scores),
        stride,
        ad(complete),
        ad(ends),
        ad(keys),
        key_stride,
        ad(count),
        ad(sparse),
        top,
        ratio,
    }) |v| try a.add(v);
    try l.go(l.fnx.idx_select, dim(rows, 1, 1), dim(1024, 1, 1), 0, s, &a);
}

const sparse_attn = @import("flashnext_sparse.zig");
pub const sparseAttention = sparse_attn.sparseAttention;
