//! The attention head split across the two expert-parallel ranks (docs/plans/exl3-kv-memory.md lever (b), §3): rank r

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const load = @import("../pack/cuda_exl3_load.zig");
const tp = @import("../tp/cuda_exl3_tp.zig");
const projection = @import("../../cuda_projection.zig");
const Glue = @import("../../cuda_glue.zig").Glue;

const head_dim = wt.head_dim;
/// Query heads and KV heads a rank keeps at world 2.
pub const rank_q: usize = wt.q_heads / 2;
pub const rank_kv: usize = wt.kv_heads / 2;
const index_cols: usize = wt.index_heads * wt.index_dim + wt.index_dim;
/// A rank's projection row: its queries and gates, its K, its V, the whole indexer.
pub const rank_pw: usize = rank_q * 2 * head_dim + 2 * rank_kv * head_dim + index_cols;
/// A rank's o_proj inputs.
pub const rank_out_k: usize = rank_q * head_dim;

comptime {
    std.debug.assert(wt.q_heads * 2 * head_dim + 2 * wt.kv_heads * head_dim + index_cols == wt.proj_n);
    std.debug.assert(rank_pw == 7296 and rank_out_k == 3072);
}

/// Whether attention layers split at `world`: at 2 only with TF_FLASHNEXT_ATTN_SPLIT=1; never at 1.
pub fn splitAt(world: usize) bool {
    if (world != 2) return false;
    const v = std.c.getenv("TF_FLASHNEXT_ATTN_SPLIT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Whether a loaded layer's attention is a rank's half (its projection row is the rank's).
pub fn isSplit(lw: anytype) bool {
    return !lw.linear and lw.proj.n == rank_pw;
}

/// KV heads an engine's sequences hold: 1 when its attention layers are split.
pub fn kvHeads(layers: anytype) usize {
    for (layers) |*l| if (!l.linear) return if (isSplit(l)) rank_kv else wt.kv_heads;
    return wt.kv_heads;
}

// -- loader ---------------------------------------------------------------------------------------------------------

/// Input rows [lo, hi) of an EXL3 matrix in the stored layout [K/16][N/16][tile words]: contiguous whole tile rows.
pub fn shardRows(m: load.Mat, lo: usize, hi: usize) ![]const u8 {
    if (lo % 128 != 0 or hi % 128 != 0 or hi <= lo or hi > m.k) return error.BadHeadSharding;
    const row = (m.n / 16) * @as(usize, m.halves) * 16;
    if (m.trellis.len != (m.k / 16) * row) return error.UnexpectedTensor;
    return m.trellis[(lo / 16) * row .. (hi / 16) * row];
}

fn upPart(p: *load.Proj, d: *cuda.Driver, trellis: []const u8, suh: []const u8, svh: []const u8, k: usize, n: usize, halves: u32) !cuda.exl3.Weight {
    var ptrs: [3]u64 = undefined;
    for (&ptrs, [3][]const u8{ trellis, suh, svh }) |*b, bytes| {
        try p.bufs.ensureUnusedCapacity(p.gpa, 1);
        const db = try cuda.DeviceBuffer.fromHost(d, bytes);
        p.bufs.appendAssumeCapacity(db);
        b.* = db.ptr;
    }
    const st = cuda.exl3.strides(k, n, halves, false);
    return .{ .trellis = ptrs[0], .suh = ptrs[1], .svh = ptrs[2], .k = @intCast(k), .n = @intCast(n), .halves = halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
}

fn wrapProj(d: *cuda.Driver, p: *load.Proj) projection.Projection {
    var out = projection.Projection.empty(d);
    out.n = p.n;
    out.k = p.k;
    out.exl3 = p;
    return out;
}

/// The rank's q|k|v|indexer row of attention `ap` (".self_attn" prefix): four EXL3 column shards side by side.
pub fn rankProjection(gpa: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, ap: []const u8, rank: usize) !projection.Projection {
    if (rank > 1) return error.BadRank;
    const p = try gpa.create(load.Proj);
    p.* = .{ .gpa = gpa, .ops = try load.opsOf(store, d), .k = wt.dims, .n = rank_pw, .parts = &.{} };
    errdefer p.destroy();
    const Cut = struct { name: []const u8, lo: usize, hi: usize };
    const qw = rank_q * 2 * head_dim;
    const kw = rank_kv * head_dim;
    const cuts = [_]Cut{
        .{ .name = ".q_proj", .lo = rank * qw, .hi = (rank + 1) * qw },
        .{ .name = ".k_proj", .lo = rank * kw, .hi = (rank + 1) * kw },
        .{ .name = ".v_proj", .lo = rank * kw, .hi = (rank + 1) * kw },
        .{ .name = ".indexer.index_qk_proj", .lo = 0, .hi = index_cols },
    };
    var parts: [cuts.len]load.Part = undefined;
    var col: usize = 0;
    var buf: [256]u8 = undefined;
    for (cuts, &parts) |c, *out| {
        const m = try load.mat(store, try std.fmt.bufPrint(&buf, "{s}{s}", .{ ap, c.name }));
        if (m.k != wt.dims) return error.UnexpectedTensor;
        const trellis = try tp.shardTrellis(gpa, m, c.lo, c.hi);
        defer gpa.free(trellis);
        const n = c.hi - c.lo;
        const w = try upPart(p, d, trellis, m.suh, m.svh[c.lo * 2 .. c.hi * 2], m.k, n, m.halves);
        p.ops.noteWeight(w, rank_pw);
        out.* = .{ .col = col, .n = n, .kind = .{ .exl3 = w } };
        col += n;
    }
    if (col != rank_pw) return error.UnexpectedTensor;
    p.parts = try gpa.dupe(load.Part, &parts);
    return wrapProj(d, p);
}

/// The rank's o_proj rows: inputs [3072r, 3072r + 3072) of `prefix` (".self_attn.o_proj"), all 2560 outputs.
pub fn rankOut(gpa: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, prefix: []const u8, rank: usize) !projection.Projection {
    if (rank > 1) return error.BadRank;
    const m = try load.mat(store, prefix);
    if (m.k != wt.out_k or m.n != wt.dims) return error.UnexpectedTensor;
    const lo = rank * rank_out_k;
    const hi = lo + rank_out_k;
    const p = try gpa.create(load.Proj);
    p.* = .{ .gpa = gpa, .ops = try load.opsOf(store, d), .k = rank_out_k, .n = wt.dims, .parts = &.{} };
    errdefer p.destroy();
    const w = try upPart(p, d, try shardRows(m, lo, hi), m.suh[lo * 2 .. hi * 2], m.svh, rank_out_k, m.n, m.halves);
    p.ops.noteWeight(w, wt.dims);
    p.parts = try gpa.alloc(load.Part, 1);
    p.parts[0] = .{ .col = 0, .n = wt.dims, .kind = .{ .exl3 = w } };
    return wrapProj(d, p);
}

/// cuda_prompt_load.zig's hook: the rank's attention projections when `world` splits them (an EXL3 pack), else null.
pub fn loadRank(gpa: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, ap: []const u8, world: usize, rank: usize) !?[2]projection.Projection {
    if (!splitAt(world) or !load.has(store, (std.fmt.bufPrint(&name_buf, "{s}.q_proj", .{ap}) catch return null))) return null;
    var proj = try rankProjection(gpa, d, store, ap, rank);
    errdefer proj.free();
    const out = try rankOut(gpa, d, store, try std.fmt.bufPrint(&name_buf, "{s}.o_proj", .{ap}), rank);
    return .{ proj, out };
}

var name_buf: [256]u8 = undefined;

// -- pass -----------------------------------------------------------------------------------------------------------

/// fn_attn_chunks' dynamic smem (cuda_glue.zig chunkSmem) and its read-ahead row bound.
fn chunkSmem(buffers: u32) u32 {
    return (16 + buffers * 2 * 32) * (256 + 8) * 2 + 16 * (32 + 8) * 2;
}
const ahead_rows = 16;

fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: u32, shared: u32, a: *cuda.Args) !void {
    if (grid[0] == 0 or grid[1] == 0 or grid[2] == 0) return;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared }, s, a);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// cuda_batch.zig Pass.attnBlock for a split layer: the rank's projection row, its heads' attention on its KV head.
pub fn attnBlock(p: anytype, lw: anytype, ai: usize) !void {
    const s = p.s;
    const n = p.rows;
    const pa = s.proj;
    const g: *const Glue = if (comptime @hasField(@TypeOf(p.tri), "own")) (p.tri.own orelse return error.NativeGlueRequired) else return error.NativeGlueRequired;
    if (p.tri.set != null) return error.NativeGlueRequired;
    const c = p.m.collective orelse return error.TensorParallelUnavailable;
    _ = try p.lane(s.mixed, s.xsm, lw.proj, pa, s.dummy, n, 1);
    if (p.lanes.len > 0) {
        const own_seq = p.seq;
        const own_pos0 = p.pos0;
        defer {
            p.seq = own_seq;
            p.pos0 = own_pos0;
        }
        for (p.lanes, 0..) |l, r| {
            p.seq = l.seq;
            p.pos0 = l.pos;
            try rows(p, g, lw, ai, pa + r * rank_pw * 2, 1, s.pos + r * 16, s.q + r * rank_q * head_dim * 2, s.iq + r * wt.index_heads * wt.index_dim * 2, s.ao + r * rank_out_k * 2);
        }
    } else try rows(p, g, lw, ai, pa, n, s.pos, s.q, s.iq, s.ao);
    try p.tri.attnGate(s.ao, pa, s.gated, s.gxs_attn, n, rank_pw, rank_q, head_dim);
    _ = try p.lane(s.gated, s.gxs_attn, lw.out, s.dummy, s.part, n, wt.out_sk);
    try c.sum(p.stream, s.part, n * wt.dims);
}

/// `n` rows of p.seq from p.pos0 on the rank's heads: upstream's attnRows with NQ 12, NKV 1, PW 7296.
fn rows(p: anytype, g: *const Glue, lw: anytype, ai: usize, pa: u64, n: usize, pos_at: u64, q_at: u64, iq_at: u64, ao_at: u64) !void {
    const s = p.s;
    if (p.seq.media != null or p.seq.kv8 != null) return error.SplitAttentionUnsupported;
    const kc = p.seq.kc[ai].ptr;
    const vc = p.seq.vc[ai].ptr;
    const ikc = p.seq.ikc[ai].ptr;
    const pooled = p.seq.pooled[ai].ptr;
    if (p.seq.kc[ai].len < p.seq.cap * rank_kv * head_dim * 2) return error.UnexpectedTensor;
    const st = p.stream;
    {
        var a: cuda.Args = .{};
        for ([_]u64{ pa, pos_at, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, q_at, kc, vc, iq_at, ikc }) |v| a.add(v);
        a.add(@as(f32, wt.eps));
        for ([_]usize{ rank_pw, rank_q, rank_kv, wt.index_heads, wt.index_dim, wt.half }) |v| a.add(ci(v));
        try go(g.f.prep, st, .{ n, rank_q + rank_kv + wt.index_heads + 1, 1 }, 256, 0, &a);
    }
    try p.tri.qsaPool(ikc, pooled, pos_at, lw.ik_scale.ptr, s.inv, wt.eps, n, .{});
    const ratio = wt.index_ratio;
    const top = wt.index_top;
    const att_rows: usize = 256;
    const nb = (cdiv(p.seq.cap, ratio) + 15) / 16 * 16;
    var r0: usize = 0;
    var b: usize = 0;
    while (r0 < n) : ({
        r0 += att_rows;
        b += 1;
    }) {
        const m = @min(att_rows, n - r0);
        const ends = p.pos0 + r0 + m;
        if (ends > p.seq.cap) return error.UnexpectedTensor;
        const pos = pos_at + b * 16;
        const iq = iq_at + r0 * wt.index_heads * wt.index_dim * 2;
        if (ends / ratio > top) try p.tri.qsaScores(iq, pooled, pos, p.seq.scores, nb, m, @min(nb, cdiv(ends, ratio)), .{});
        var a: cuda.Args = .{};
        a.add(p.seq.scores);
        a.add(pos);
        a.add(s.qids);
        a.add(s.qnk);
        a.add(s.qspr);
        for ([_]usize{ nb, wt.index_idw, top, ratio }) |v| a.add(ci(v));
        try cuda.launch.launch(p.k.select, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1024 } }, st, &a);
        const chunks = cdiv(@min(ends, wt.dense_keys), 512);
        {
            var ca: cuda.Args = .{};
            for ([_]u64{ q_at + r0 * rank_q * head_dim * 2, kc, vc, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr }) |v| ca.add(v);
            for ([_]usize{ rank_q, rank_kv, rank_q / rank_kv, 512, wt.nch }) |v| ca.add(ci(v));
            ca.add(@as(f32, 0.0625));
            for ([_]usize{ wt.index_idw, 1 }) |v| ca.add(ci(v));
            const ahead = m <= ahead_rows;
            try go(if (ahead) g.f.chunks_ahead else g.f.chunks, st, .{ m, rank_kv, chunks }, 128, chunkSmem(if (ahead) 2 else 1), &ca);
        }
        {
            var ma: cuda.Args = .{};
            for ([_]u64{ s.po, s.pm, s.pl, pos, ao_at + r0 * rank_out_k * 2, s.qnk, s.qspr }) |v| ma.add(v);
            for ([_]usize{ rank_q, rank_kv, rank_q / rank_kv, 512, wt.nch, 1 }) |v| ma.add(ci(v));
            try go(g.f.merge, st, .{ m, rank_q, 1 }, 256, 0, &ma);
        }
    }
}

test "exl3: a rank's attention row is half the heads and the whole indexer, in whole 128-column blocks" {
    try std.testing.expectEqual(@as(usize, 7296), rank_pw);
    try std.testing.expectEqual(@as(usize, 0), rank_pw % 128);
    try std.testing.expectEqual(@as(usize, 0), rank_out_k % 128);
    try std.testing.expectEqual(@as(usize, 12), rank_q / rank_kv);
}

test "exl3: o_proj's input-row shard is the contiguous tile rows" {
    var tr: [4 * 2 * 4 * 16]u8 = undefined; // K 512 (32 tile rows), N 32 (2 tiles) at 2 halves... (shape-only check)
    _ = &tr;
    const words = try std.testing.allocator.alloc(u8, (256 / 16) * (128 / 16) * 2 * 16);
    defer std.testing.allocator.free(words);
    for (words, 0..) |*w, i| w.* = @truncate(i);
    const m: load.Mat = .{ .trellis = words, .suh = &.{}, .svh = &.{}, .k = 256, .n = 128, .halves = 2 };
    const row = (128 / 16) * 2 * 16;
    const hi = try shardRows(m, 128, 256);
    try std.testing.expectEqual(@as(usize, 8 * row), hi.len);
    try std.testing.expectEqual(words[8 * row], hi[0]);
    try std.testing.expectError(error.BadHeadSharding, shardRows(m, 64, 128));
}
