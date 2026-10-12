//! Flash Next's launches: stream norms, gates and mixes, the n-gram gate, conv and hashes, indexer, sparse attention.

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
    hc_gates: Function,
    rms_wide: Function,
    hc_mix: Function,
    ple_gate: Function,
    ple_conv: Function,
    ngram_ids: Function,
    idx_pool: Function,
    idx_scores: Function,
    idx_select: Function,
    sparse_parts: [2]Function, // groups of up to 6 and up to 12 query heads
    sparse_merge: Function,

    pub fn load(m: Module) Error!Fns {
        return .{
            .hc_norm = try m.function("tf_fn_hc_norm"),
            .hc_gates = try m.function("tf_fn_hc_gates"),
            .rms_wide = try m.function("tf_fn_rms_wide"),
            .hc_mix = try m.function("tf_fn_hc_mix"),
            .ple_gate = try m.function("tf_fn_ple_gate"),
            .ple_conv = try m.function("tf_fn_ple_conv"),
            .ngram_ids = try m.function("tf_fn_ngram_ids"),
            .idx_pool = try m.function("tf_fn_idx_pool"),
            .idx_scores = try m.function("tf_fn_idx_scores"),
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
pub fn hcNorm(l: *const Launcher, h: C, branch: C, inject: C, h_out: P, x_out: P, scale: CF, kind: c_int, rows: c_int, streams: c_int, dims: c_int, eps: f32, s: S) Error!void {
    return hcNormShares(l, h, branch, inject, h_out, x_out, scale, kind, rows, streams, dims, eps, 0, 0, s);
}

/// hcNorm with the branch as `world` fp32 shares `stride` values apart (a tensor-parallel gather), summed in order.
pub fn hcNormShares(l: *const Launcher, h: C, branch: C, inject: C, h_out: P, x_out: P, scale: CF, kind: c_int, rows: c_int, streams: c_int, dims: c_int, eps: f32, world: c_int, stride: i64, s: S) Error!void {
    if (dims > most_dims) return util.invalid("hcNorm");
    var a: Args = .{};
    inline for (.{ ad(h), ad(branch), ad(inject), ad(h_out), ad(x_out), ad(scale), kind, streams, dims, eps, world, stride }) |v| try a.add(v);
    try l.go(l.fnx.hc_norm, dim(streams, rows, 1), dim(1024, 1, 1), 0, s, &a);
}

/// act = silu(z / streams) and inject = 2 sigmoid(z / streams) from the down projection's fp32 sums (no inject: null).
pub fn hcGates(l: *const Launcher, z: CF, act: P, inject: P, kind: c_int, rows: c_int, low: c_int, streams: c_int, s: S) Error!void {
    var a: Args = .{};
    inline for (.{ ad(z), ad(act), ad(inject), kind, rows, low, streams }) |v| try a.add(v);
    try l.flat(l.fnx.hc_gates, @as(i64, rows) * (low + if (inject != null) streams else 0), s, &a);
}

/// Each of `rows` rows normed over its whole `width` (any width) times the fp32 `scale`.
pub fn rmsWide(l: *const Launcher, x: C, scale: CF, y: P, kind: c_int, rows: c_int, width: c_int, eps: f32, s: S) Error!void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ad(x), ad(scale), ad(y), kind, width, eps }) |v| try a.add(v);
    try l.go(l.fnx.rms_wide, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// The block input: the mean over the streams of sigmoid(g) times the normed stream.
pub fn hcMix(l: *const Launcher, g: CF, x: C, mixed: P, kind: c_int, rows: c_int, streams: c_int, dims: c_int, s: S) Error!void {
    var a: Args = .{};
    inline for (.{ ad(g), ad(x), ad(mixed), kind, rows, streams, dims }) |v| try a.add(v);
    try l.flat(l.fnx.hc_mix, @as(i64, rows) * dims, s, &a);
}

/// The n-gram gate of every stream; `normed` is the conv's input rows.
pub fn pleGate(l: *const Launcher, kv: C, h: C, key_scale: CF, query_scale: CF, conv_scale: CF, gated: P, normed: P, kind: c_int, rows: c_int, streams: c_int, dims: c_int, eps: f32, s: S) Error!void {
    if (dims > most_dims) return util.invalid("pleGate");
    var a: Args = .{};
    inline for (.{ ad(kv), ad(h), ad(key_scale), ad(query_scale), ad(conv_scale), ad(gated), ad(normed), kind, streams, dims, eps }) |v| try a.add(v);
    try l.go(l.fnx.ple_gate, dim(streams, rows, 1), dim(threads, 1, 1), 0, s, &a);
}

/// out = h + (gated + silu(conv)): the dilated causal conv over `cin`, its tail rows first.
pub fn pleConv(l: *const Launcher, cin: C, weight: CF, gated: C, h: C, out: P, kind: c_int, rows: c_int, width: c_int, taps: c_int, dilation: c_int, s: S) Error!void {
    var a: Args = .{};
    inline for (.{ ad(cin), ad(weight), ad(gated), ad(h), ad(out), kind, rows, width, taps, dilation }) |v| try a.add(v);
    try l.flat(l.fnx.ple_conv, @as(i64, rows) * width, s, &a);
}

/// Each row's n-gram table rows; `meta` holds the history, eos, multipliers, sizes and offsets (int64).
pub fn ngramIds(l: *const Launcher, tokens: CI, meta: ?*const anyopaque, ids: P, rows: c_int, heads: c_int, per: c_int, n: c_int, s: S) Error!void {
    var a: Args = .{};
    inline for (.{ ad(tokens), ad(meta), ad(ids), rows, heads, per, n }) |v| try a.add(v);
    try l.flat(l.fnx.ngram_ids, @as(i64, rows) * heads, s, &a);
}

/// Blocks first .. first + blocks of the raw indexer keys, pooled, normed and rotated into `pooled`.
pub fn idxPool(l: *const Launcher, raw: C, pooled: P, scale: CF, kind: c_int, first: c_int, blocks: c_int, dims: c_int, ratio: c_int, rotary: c_int, theta: f32, eps: f32, s: S) Error!void {
    if (blocks == 0) return;
    if (dims > threads) return util.invalid("idxPool");
    var a: Args = .{};
    inline for (.{ ad(raw), ad(pooled), ad(scale), kind, first, @as(u64, 0), dims, ratio, rotary, theta, eps }) |v| try a.add(v);
    try l.go(l.fnx.idx_pool, dim(blocks, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// The blocks `rows` rows at device positions `pos` complete, pooled into their slots of `pooled` (all blocks').
pub fn idxPoolAt(l: *const Launcher, raw: C, pooled: P, scale: CF, kind: c_int, pos: CI, rows: c_int, dims: c_int, ratio: c_int, rotary: c_int, theta: f32, eps: f32, s: S) Error!void {
    if (rows == 0) return;
    if (dims > threads) return util.invalid("idxPoolAt");
    var a: Args = .{};
    inline for (.{ ad(raw), ad(pooled), ad(scale), kind, @as(c_int, 0), ad(pos), dims, ratio, rotary, theta, eps }) |v| try a.add(v);
    try l.go(l.fnx.idx_pool, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// Every row's score of each of `blocks` pooled blocks it has complete, for rows past `top` complete blocks.
pub fn idxScores(l: *const Launcher, q: CF, pooled: C, complete: CI, scores: F, kind: c_int, rows: c_int, heads: c_int, dims: c_int, blocks: c_int, stride: i64, top: c_int, s: S) Error!void {
    if (blocks == 0) return;
    if (dims > threads) return util.invalid("idxScores");
    var a: Args = .{};
    inline for (.{ ad(q), ad(pooled), ad(complete), ad(scores), kind, rows, heads, dims, blocks, stride, top }) |v| try a.add(v);
    try l.go(l.fnx.idx_scores, dim(cdiv(blocks, 8), cdiv(rows, 8), 1), dim(threads, 1, 1), 0, s, &a);
}

/// Each row's keys: its `top` best blocks and its tail past `top` complete blocks, else its first ends[r] keys.
pub fn idxSelect(l: *const Launcher, scores: CF, stride: i64, complete: CI, ends: CI, keys: I, key_stride: i64, count: I, sparse: I, rows: c_int, top: c_int, ratio: c_int, s: S) Error!void {
    var a: Args = .{};
    inline for (.{ ad(scores), stride, ad(complete), ad(ends), ad(keys), key_stride, ad(count), ad(sparse), top, ratio }) |v| try a.add(v);
    try l.go(l.fnx.idx_select, dim(rows, 1, 1), dim(1024, 1, 1), 0, s, &a);
}

/// Attention of each row's heads over its listed keys (or its first count[r] keys) in `parts` parts, then merged.
pub fn sparseAttention(l: *const Launcher, q: CF, k: C, v: C, cache_kind: c_int, ids: CI, id_stride: i64, count: CI, sparse: CI, parts_o: F, parts_ms: F, out: F, rows: c_int, heads: c_int, kv_heads: c_int, dims: c_int, cap: i64, scale: f32, parts: c_int, s: S) Error!void {
    if (dims > 256 or @rem(heads, kv_heads) != 0 or @divExact(heads, kv_heads) > 12) return util.invalid("sparseAttention");
    var a: Args = .{};
    inline for (.{ ad(q), ad(k), ad(v), cache_kind, ad(ids), id_stride, ad(count), ad(sparse), ad(parts_o), ad(parts_ms), heads, kv_heads, dims, cap, scale, parts }) |x| try a.add(x);
    const wide: usize = if (@divExact(heads, kv_heads) <= 6) 0 else 1;
    try l.go(l.fnx.sparse_parts[wide], dim(kv_heads, rows, parts), dim(256, 1, 1), 0, s, &a);
    var b: Args = .{};
    inline for (.{ ad(parts_o), ad(parts_ms), ad(out), heads, dims, parts }) |x| try b.add(x);
    try l.go(l.fnx.sparse_merge, dim(heads, rows, 1), dim(256, 1, 1), 0, s, &b);
}
