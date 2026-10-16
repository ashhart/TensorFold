//! Flash Next's glue launches: our own kernels (cuda_glue.zig), or the captured Triton cubins where a set is loaded.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

pub const embed = @import("cuda_embed.zig");
pub const projection = @import("cuda_projection.zig");
pub const modelopt_experts = @import("cuda_modelopt_experts.zig");
pub const hc = @import("cuda_hc.zig");
pub const qmm = @import("cuda_qmm.zig");
pub const gdn = @import("cuda_gdn.zig");
pub const mlp = @import("cuda_mlp.zig");
pub const moe = @import("cuda_moe.zig");
pub const attn = @import("cuda_attn.zig");
pub const ple = @import("cuda_ple.zig");
pub const ple_check = @import("cuda_ple_check.zig");
pub const mix = @import("cuda_mix.zig");
pub const prompt = @import("cuda_prompt.zig");
/// The row-by-row reference path, for flashnext-prompt
pub const row_path = @import("cuda_rows.zig");
/// The MTP head's drafts: its forward, the chain, the verify window's rows and their rollback
pub const mtp = @import("cuda_mtp.zig");
pub const native = @import("cuda_lanes.zig").Namespace(Tri);
/// Our own glue kernels, which every launch takes when no captured set is loaded
pub const Glue = glue_.Glue;
const glue_ = @import("cuda_glue.zig");

const p = aot.ptr;

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) u32 {
    return u((a + b - 1) / b);
}

/// One program per token row and per group of 32 hidden features.
pub fn embedGrid(rows: usize, dims: usize) [3]u32 {
    return .{ u(rows), u(dims / 32), 1 };
}

/// RMSNorm's BLOCK is the next power of two at or above the group width.
pub fn rmsBlock(group: usize) usize {
    return std.math.ceilPowerOfTwo(usize, group) catch unreachable;
}

/// Each launch's grid and constexprs match the Python wrapper's when a captured set is loaded; without one, ours run.
pub const Tri = struct {
    set: ?*const aot.Set = null,
    own: ?*const Glue = null,
    s: cuda.Stream,

    fn run(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        try t.set.?.run(t.s, name, grid, args, consts);
    }

    /// Ours for `name` without a captured set, or with one when TF_FLASHNEXT_OWNED names it; else null: captured.
    fn glue(t: Tri, comptime name: []const u8) ?*const Glue {
        if (t.set == null) return t.own.?;
        const g = t.own orelse return null;
        return if (g.mixed & glue_.entryBit(name) != 0) g else null;
    }

    /// glue.embed: ids (R,) int32 into (R, copies * dims) bf16, MLX 4-bit group 32.
    pub fn embed(t: Tri, ids: u64, weight: u64, scales: u64, biases: u64, out: u64, dims: usize, copies: usize, rows: usize) !void {
        if (t.glue("_embed")) |g| return g.embed(t.s, ids, weight, scales, biases, out, dims, copies, rows);
        try t.run("_embed", embedGrid(rows, dims), &.{
            p("IDS", "*i32", ids),
            p("W", "*i32", weight),
            p("S", "*bf16", scales),
            p("B", "*bf16", biases),
            p("OUT", "*bf16", out),
        }, &.{ ci("D", dims), ci("S_COPIES", copies) });
    }

    pub fn embedDense(t: Tri, ids: u64, w: u64, out: u64, dims: usize, copies: usize, rows: usize) !void {
        const g = t.own orelse return error.NativeGlueRequired;
        try g.embedDense(t.s, ids, w, out, dims, copies, rows);
    }

    /// glue.rmsnorm per row and `group` features: bf16 out [rows, dims] and 32-group sums; x rows `x_stride` apart.
    pub fn rmsnorm(t: Tri, x: u64, w: u64, out: u64, xs: u64, eps: f32, x_stride: usize, rows: usize, dims: usize, group: usize) !void {
        if (t.glue("_rmsnorm")) |g| return g.rmsnorm(t.s, x, w, out, xs, eps, x_stride, rows, dims, group);
        try t.run("_rmsnorm", .{ u(rows), u(dims / group), 1 }, &.{
            p("X", "*bf16", x),
            p("W", "*fp32", w),
            p("OUT", "*bf16", out),
            p("XS", "*fp32", xs),
            aot.float("eps", eps),
            int("x_stride", x_stride),
        }, &.{ ci("D", dims), ci("G", group), ci("BLOCK", rmsBlock(group)) });
    }

    /// glue.add_streams: out[r, s] = bf16(e[r] + hs[r, s]), the MTP head's input streams. Program (row, 256 columns).
    pub fn addStreams(t: Tri, e: u64, hs: u64, out: u64, rows: usize, dims: usize, streams: usize) !void {
        if (t.glue("_add_streams")) |g| return g.addStreams(t.s, e, hs, out, rows, dims, streams);
        try t.run("_add_streams", .{ u(rows), u(dims / 256), 1 }, &.{
            p("E", "*bf16", e),
            p("HS", "*bf16", hs),
            p("OUT", "*bf16", out),
        }, &.{ ci("D", dims), ci("S", streams), ci("BLOCK", 256) });
    }

    /// glue.attn_gate: one program per row and per query head.
    pub fn attnGate(t: Tri, o: u64, proj: u64, out: u64, xs: u64, rows: usize, pair_width: usize, q_heads: usize, head_dim: usize) !void {
        if (t.glue("_attn_gate")) |g| return g.attnGate(t.s, o, proj, out, xs, rows, pair_width, q_heads, head_dim);
        try t.run("_attn_gate", .{ u(rows), u(q_heads), 1 }, &.{
            p("O", "*bf16", o),
            p("P", "*bf16", proj),
            p("OUT", "*bf16", out),
            p("XS", "*fp32", xs),
        }, &.{ ci("PW", pair_width), ci("NQ", q_heads), ci("HD", head_dim) });
    }

    /// glue.attn_prep for text positions, bf16 KV (BITS 0, the scale pointers unread); `length` unused at MODE 0.
    pub fn attnPrep(t: Tri, proj: u64, pos: u64, q_scale: u64, k_scale: u64, i_scale: u64, inv: u64, q: u64, kc: u64, vc: u64, iq: u64, ikc: u64, rope: u64, delta: u64, eps: f32, rows: usize, length: usize) !void {
        if (t.glue("_attn_prep")) |g| return g.attnPrep(t.s, proj, pos, q_scale, k_scale, i_scale, inv, q, kc, vc, iq, ikc, eps, rows);
        try t.run("_attn_prep", .{ u(rows), u(24 + 2 + 4 + 1), 1 }, &.{
            p("P", "*bf16", proj),
            p("POS0", "*i32", pos),
            p("QW", "*fp32", q_scale),
            p("KW", "*fp32", k_scale),
            p("IW", "*fp32", i_scale),
            p("INV", "*fp32", inv),
            p("Q", "*bf16", q),
            p("KC", "*bf16", kc),
            p("VC", "*bf16", vc),
            p("KS", "*fp16", rope),
            p("VS", "*fp16", rope),
            p("IQ", "*bf16", iq),
            p("IKC", "*bf16", ikc),
            p("ROPE", "*i32", rope),
            p("DELTA", "*i32", delta),
            int("length", length),
            aot.float("eps", eps),
        }, &.{ ci("PW", 13952), ci("NQ", 24), ci("NKV", 2), ci("HD", 256), ci("NI", 4), ci("IHD", 128), ci("HALF", 32), ci("BITS", 0), ci("MODE", 0), ci("S1", 11), ci("S2", 10) });
    }

    /// attention._chunks over 512-key chunks, bf16 KV; `qsa` the sparse form (NCH 5, QSA 1), else dense (NCH 3, QSA 0).
    pub fn attnChunks(t: Tri, q: u64, kc: u64, vc: u64, pos: u64, po: u64, pm: u64, pl: u64, ids: u64, nk: u64, sparse: u64, rows: usize, kv: usize, chunks: usize, qsa: bool) !void {
        if (t.glue("_chunks")) |g| return g.attnChunks(t.s, q, kc, vc, pos, po, pm, pl, ids, nk, sparse, rows, kv, chunks, qsa);
        try t.run("_chunks", .{ u(rows), u(kv), u(chunks) }, &.{
            p("Q", "*bf16", q),
            p("KC", "*bf16", kc),
            p("VC", "*bf16", vc),
            p("KSC", "*fp16", sparse),
            p("VSC", "*fp16", sparse),
            p("POS0", "*i32", pos),
            p("PO", "*fp32", po),
            p("PM", "*fp32", pm),
            p("PL", "*fp32", pl),
            p("IDS", "*i32", ids),
            p("NKR", "*i32", nk),
            p("SPR", "*i32", sparse),
        }, &.{ ci("H", 24), ci("HK", 2), ci("D", 256), ci("G", 12), ci("CH", 512), ci("NCH", if (qsa) 5 else 3), aot.cf("SCALE", 0.0625), ci("IDW", 2052), ci("QSA", @intFromBool(qsa)), ci("BITS", 0) });
    }

    /// attention._merge: one program per row and per KV head, in attnChunks' form.
    pub fn attnMerge(t: Tri, po: u64, pm: u64, pl: u64, pos: u64, out: u64, nk: u64, sparse: u64, rows: usize, kv: usize, qsa: bool) !void {
        if (t.glue("_merge")) |g| return g.attnMerge(t.s, po, pm, pl, pos, out, nk, sparse, rows, kv, qsa);
        try t.run("_merge", .{ u(rows), u(kv), 1 }, &.{
            p("PO", "*fp32", po),
            p("PM", "*fp32", pm),
            p("PL", "*fp32", pl),
            p("POS0", "*i32", pos),
            p("OUT", "*bf16", out),
            p("NKR", "*i32", nk),
            p("SPR", "*i32", sparse),
        }, &.{ ci("H", 24), ci("HK", 2), ci("D", 256), ci("G", 12), ci("CH", 512), ci("NCH", if (qsa) 5 else 3), ci("QSA", @intFromBool(qsa)), ci("BITS", 0) });
    }

    /// attention.qsa_pool for text rows (MODE 0): every complete block of 4 indexer keys pooled, normed and roped.
    pub fn qsaPool(t: Tri, ikc: u64, pooled: u64, pos: u64, k_scale: u64, inv: u64, eps: f32, rows: usize) !void {
        if (t.glue("_pool")) |g| return g.qsaPool(t.s, ikc, pooled, pos, k_scale, inv, eps, rows);
        try t.run("_pool", .{ u(rows / 4 + 2), 1, 1 }, &.{
            p("IKC", "*bf16", ikc),
            p("POOLED", "*bf16", pooled),
            p("POS0", "*i32", pos),
            p("W", "*fp32", k_scale),
            p("INV", "*fp32", inv),
            aot.float("eps", eps),
            int("R", rows),
            int("length", 0),
        }, &.{ ci("DI", 128), ci("HALF", 32), ci("RATIO", 4), ci("MODE", 0), ci("S1", 11), ci("S2", 10) });
    }

    /// attention._scores: rows' complete blocks scored; `nb` is the row stride (a multiple of 16), `blocks` the grid.
    pub fn qsaScores(t: Tri, iq: u64, pooled: u64, pos: u64, sc: u64, nb: usize, rows: usize, blocks: usize) !void {
        if (t.glue("_scores")) |g| return g.qsaScores(t.s, iq, pooled, pos, sc, nb, rows, blocks);
        try t.run("_scores", .{ u(rows), cdiv(blocks, 64), 1 }, &.{
            p("IQ", "*bf16", iq),
            p("POOLED", "*bf16", pooled),
            p("POS0", "*i32", pos),
            p("SC", "*fp32", sc),
            int("NB", nb),
        }, &.{ ci("HI", 4), ci("DI", 128), ci("RATIO", 4), ci("TOP", 512), ci("BB", 64) });
    }

    /// Dense connections always use native glue; a captured affine projection cannot replace these operators.
    pub fn hcNormalize(t: Tri, h: u64, pss: u64, scale: u64, out: u64, eps: f32, rows: usize, dims: usize, streams: usize) !void {
        const g = t.own orelse return error.NativeGlueRequired;
        return g.hcNormalize(t.s, h, pss, scale, out, eps, rows, dims, streams);
    }

    pub fn hcDenseMix(t: Tri, up: u64, norm: u64, mixed: u64, sums: u64, rows: usize, dims: usize, streams: usize) !void {
        const g = t.own orelse return error.NativeGlueRequired;
        return g.hcDenseMix(t.s, up, norm, mixed, sums, rows, dims, streams);
    }

    /// glue.hc_reduce_act: split-K sum fused with hc_act. M folds to a constexpr when it is 1.
    pub fn hcReduceAct(t: Tri, part: u64, act: u64, xs: u64, inj: u64, sk: usize, rows: usize, streams: usize, low: usize, ndn: usize, has_inj: usize) !void {
        if (t.glue("_hc_reduce_act")) |g| return g.hcReduceAct(t.s, part, act, xs, inj, sk, rows, streams, low, ndn, has_inj);
        try t.run("_hc_reduce_act", .{ u(rows), 1, 1 }, &.{
            p("PART", "*fp32", part),
            p("ACT", "*bf16", act),
            p("XS", "*fp32", xs),
            p("INJ", "*bf16", inj),
            int("M", rows),
        }, &.{ ci("SK", sk), ci("S", streams), ci("LOW", low), ci("LOWP", rmsBlock(low)), ci("HAS_INJ", has_inj), ci("NDN", ndn) });
    }

    /// qmm.hc_upmix: the up projection and the stream mix. N is the wide row, K is the low rank. M folds when it is 1.
    pub fn hcUpmix(t: Tri, act: u64, xs: u64, weight: u64, scales: u64, biases: u64, normed: u64, mixed: u64, xs_mixed: u64, rows: usize, n: usize, k: usize, dims: usize, streams: usize) !void {
        if (t.glue("_qmm_upmix")) |g| return g.hcUpmix(t.s, act, xs, weight, scales, biases, normed, mixed, xs_mixed, rows, n, k, dims, streams);
        try t.run("_qmm_upmix", .{ cdiv(rows, 16), u(dims / 32), 1 }, &.{
            p("X", "*bf16", act),
            p("XS", "*fp32", xs),
            p("W", "*i32", weight),
            p("S", "*bf16", scales),
            p("B", "*bf16", biases),
            p("NORMED", "*bf16", normed),
            p("MIXED", "*bf16", mixed),
            p("XSM", "*fp32", xs_mixed),
            int("M", rows),
        }, &.{ ci("N", n), ci("K", k), ci("D", dims), ci("SS", streams), ci("BM", 16), ci("DB", 32), ci("GPI", 2), ci("SBN", 64) });
    }

    /// qmm.hc_down for one decode row. N=324 is the layer down stacked with inject. M folds when it is 1.
    pub fn hcDown(t: Tri, h: u64, pss: u64, scale: u64, normed: u64, weight: u64, scales: u64, biases: u64, out: u64, part: u64, eps: f32, rows: usize, n: usize, k: usize, dims: usize, nc: usize, streams: usize, sk: usize) !void {
        if (t.glue("_qmm_hcdown")) |g| return g.hcDown(t.s, h, pss, scale, normed, weight, scales, biases, out, part, eps, rows, n, k, dims, nc, streams, sk);
        try t.run("_qmm_hcdown", .{ cdiv(rows, 16), cdiv(n, 64), u(sk) }, &.{
            p("H", "*bf16", h),
            p("PSS", "*fp32", pss),
            p("SCALE", "*fp32", scale),
            p("NORMED", "*bf16", normed),
            p("W", "*i32", weight),
            p("S", "*bf16", scales),
            p("B", "*bf16", biases),
            p("OUT", "*bf16", out),
            p("PART", "*fp32", part),
            int("M", rows),
            aot.float("eps", eps),
        }, &.{ ci("N", n), ci("K", k), ci("D", dims), ci("NC", nc), ci("SS", streams), ci("SK", sk), ci("BM", 16), ci("BLOCK_N", 64), ci("GPI", 2), ci("SBN", 64) });
    }

    /// glue.hc_writeback. Pointer types follow the tensor dtypes Triton specialized (mode 0..4).
    pub fn hcWriteback(t: Tri, h: u64, hout: u64, pss: u64, branch: u64, branch_ty: []const u8, inj: u64, y: u64, y_ty: []const u8, wts: u64, wts_ty: []const u8, rs: usize, rows: usize, dims: usize, streams: usize, mode: usize, topk: usize, slots: usize, world: usize) !void {
        if (t.glue("_hc_writeback")) |g| return g.hcWriteback(t.s, h, hout, pss, branch, inj, y, wts, rs, rows, dims, streams, mode, topk, slots, world);
        try t.run("_hc_writeback", .{ u(rows), u(dims / 256), 1 }, &.{
            p("H", "*bf16", h),
            p("HOUT", "*bf16", hout),
            p("PSS", "*fp32", pss),
            p("BR", branch_ty, branch),
            p("INJ", "*bf16", inj),
            p("Y", y_ty, y),
            p("WTS", wts_ty, wts),
            int("RS", rs),
        }, &.{ ci("D", dims), ci("S", streams), ci("MODE", mode), ci("TOPK", topk), ci("SLOTS", slots), ci("BLOCK", 256), ci("WORLD", world) });
    }

    /// glue.ple_embed: gathered n-gram rows, MLX 4-bit group 32. `scale` is a constexpr.
    pub fn pleEmbed(t: Tri, weight: u64, scales: u64, biases: u64, out: u64, xs: u64, rows: usize, heads: usize, dh: usize, scale: f32) !void {
        if (t.set == null) return error.CapturedKernelsRequired; // the row path's n-gram check only
        try t.run("_ple_embed", .{ u(rows), u(heads), 1 }, &.{
            p("W", "*i32", weight),
            p("S", "*bf16", scales),
            p("B", "*bf16", biases),
            p("OUT", "*bf16", out),
            p("XS", "*fp32", xs),
        }, &.{ ci("HEADS", heads), ci("DH", dh), aot.cf("SCALE", scale) });
    }

    /// glue.ple_gate: one program per row.
    pub fn pleGate(t: Tri, keys: u64, vals: u64, h: u64, norm_key: u64, norm_query: u64, gated: u64, pss: u64, eps: f32, rows: usize, dims: usize, streams: usize) !void {
        if (t.glue("_ple_gate")) |g| return g.pleGate(t.s, keys, vals, h, norm_key, norm_query, gated, pss, eps, rows, dims, streams);
        try t.run("_ple_gate", .{ u(rows), 1, 1 }, &.{
            p("KEYS", "*bf16", keys),
            p("VALS", "*bf16", vals),
            p("H", "*bf16", h),
            p("NK", "*fp32", norm_key),
            p("NQ", "*fp32", norm_query),
            p("GATED", "*bf16", gated),
            p("PSS", "*fp32", pss),
            aot.float("eps", eps),
        }, &.{ ci("D", dims), ci("S", streams), ci("BLOCK", 512) });
    }

    /// glue.ple_conv. R folds to a constexpr when it is 1.
    pub fn pleConv(t: Tri, gated: u64, pss: u64, norm_conv: u64, tail: u64, conv_w: u64, h: u64, hout: u64, nrow: u64, eps: f32, rows: usize, dims: usize, streams: usize, taps: usize, dilation: usize) !void {
        if (t.glue("_ple_conv")) |g| return g.pleConv(t.s, gated, pss, norm_conv, tail, conv_w, h, hout, nrow, eps, rows, dims, streams, taps, dilation);
        try t.run("_ple_conv", .{ u(rows), u(streams * dims / 512), 1 }, &.{
            p("GATED", "*bf16", gated),
            p("PSS", "*fp32", pss),
            p("NC", "*fp32", norm_conv),
            p("TAIL", "*bf16", tail),
            p("CW", "*bf16", conv_w),
            p("H", "*bf16", h),
            p("HOUT", "*bf16", hout),
            p("NROW", "*bf16", nrow),
            aot.float("eps", eps),
            int("R", rows),
        }, &.{ ci("D", dims), ci("S", streams), ci("TAPS", taps), ci("DIL", dilation), ci("BLOCK", 512) });
    }

    /// moe.router for one decode row. M folds to a constexpr. 513 rows: the experts, then the shared gate.
    pub fn router(t: Tri, x: u64, w: u64, out: u64, x_stride: usize) !void {
        try t.routerMany(x, w, out, x_stride, 1);
    }

    /// moe._topk_rows: one program, the row's 10 experts and the shared expert.
    pub fn topkRows(t: Tri, logits: u64, pick: u64, wts: u64) !void {
        try t.topkMany(logits, pick, wts, 1);
    }

    /// moe._topk_rows over `rows` rows of [rows, 513] logits: picks and weights [rows, 11].
    pub fn topkMany(t: Tri, logits: u64, pick: u64, wts: u64, rows: usize) !void {
        if (t.glue("_topk_rows")) |g| return g.topkMany(t.s, logits, pick, wts, rows);
        try t.run("_topk_rows", .{ u(rows), 1, 1 }, &.{
            p("L", "*fp32", logits),
            p("PICK", "*i32", pick),
            p("WTS", "*fp32", wts),
        }, &.{ ci("NE", 512), ci("NL", 513), ci("TOPK", 10), ci("SLOTS", 11), ci("BLOCK", 1024), ci("SLOTP", 16) });
    }

    /// moe.router over `rows` rows at the one-row tiles (BM 16, BLOCK_E 32, BK 256), each row keeping its own bits.
    pub fn routerMany(t: Tri, x: u64, w: u64, out: u64, x_stride: usize, rows: usize) !void {
        if (t.glue("_router")) |g| return g.routerMany(t.s, x, w, out, x_stride, rows);
        try t.run("_router", .{ cdiv(rows, 16), cdiv(513, 32), 1 }, &.{
            p("X", "*bf16", x),
            p("W", "*bf16", w),
            p("OUT", "*fp32", out),
            int("M", rows),
            int("x_stride", x_stride),
        }, &.{ ci("D", 2560), ci("NE", 513), ci("BM", 16), ci("BLOCK_E", 32), ci("BK", 256) });
    }
};

test {
    _ = embed;
    _ = hc;
    _ = qmm;
    _ = gdn;
    _ = mlp;
    _ = moe;
    _ = attn;
    _ = ple;
    _ = mix;
    _ = prompt;
    _ = prompt.batch;
    _ = mtp;
    _ = native;
    _ = @import("cuda_snap.zig");
    _ = @import("cuda_config.zig");
    _ = @import("cuda_projection_load.zig");
    _ = @import("cuda_modelopt_experts.zig");
    _ = row_path;
    _ = @import("cuda_mtp_check.zig");
}

test "embed grid is one program per row and per group of 32" {
    try std.testing.expectEqual([3]u32{ 4, 80, 1 }, embedGrid(4, 2560));
}

test "rmsnorm block is the next power of two at or above the group" {
    try std.testing.expectEqual(@as(usize, 4096), rmsBlock(2560));
    try std.testing.expectEqual(@as(usize, 16384), rmsBlock(10240));
}
