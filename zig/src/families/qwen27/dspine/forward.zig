//! DSpine's two passes on Metal: committed taps into the K/V caches, then one block, its bridge and the greedy table.
const std = @import("std");
const core = @import("core");
const mtl = @import("metal");
const ck = @import("../checkpoint.zig");
const config = @import("config.zig");
const runtime = @import("runtime.zig");
const Model = runtime.Model;
const Ref = runtime.Ref;
const topk_rows = core.bf16_topk;
const row_words = @sizeOf(topk_rows.Row) / 4;
/// TF_DSPINE_PROFILE: each layer, the head and the tail in their own command buffers, their GPU times printed.
var profile = false;
pub fn setProfile(on: bool) void {
    profile = on;
}

fn scratch(m: *Model, rows: usize, width: usize, dtype: ck.DType) !Ref {
    return m.frame.ref(try m.frame.alloc(@intCast(rows), @intCast(width), dtype));
}

fn at(r: Ref, bytes: usize) Ref {
    return .{ .buf = r.buf, .off = r.off + bytes };
}

fn norm(m: *Model, e: mtl.ComputeEncoder, x: Ref, w: ck.Tensor, rows: usize, heads: usize, dim: usize) !Ref {
    const y = try scratch(m, rows, heads * dim, .bf16);
    try m.primitives.norm(e, x, try m.weight(w.bytes), y, .{ .rows = @intCast(rows), .heads = @intCast(heads), .dim = @intCast(dim), .x_stride = @intCast(heads * dim), .y_stride = @intCast(heads * dim), .eps = m.graph.shape.eps });
    e.barrier();
    return y;
}

fn rope(m: *Model, e: mtl.ComputeEncoder, x: Ref, positions: Ref, rows: usize, heads: usize) !Ref {
    const s = m.graph.shape;
    const y = try scratch(m, rows, heads * s.head_dim, .bf16);
    try m.primitives.rope(e, x, positions, y, .{ .rows = @intCast(rows), .heads = @intCast(heads), .dim = @intCast(s.head_dim), .x_stride = @intCast(heads * s.head_dim), .y_stride = @intCast(heads * s.head_dim), .theta = @floatCast(s.theta) });
    e.barrier();
    return y;
}

fn project(m: *Model, e: mtl.ComputeEncoder, x: Ref, w: ck.Tensor, rows: usize) !Ref {
    const y = try scratch(m, rows, w.shape[0], .bf16);
    try m.linear(e, x, w, y, @intCast(rows));
    return y;
}

/// A bridge linear y [rows, n] = x [rows, k] w^T, w the matrix at host `bytes` (GPU `w`): affine-4 when prepared.
fn bridgeLinear(m: *Model, e: mtl.ComputeEncoder, x: Ref, bytes: []const u8, w: Ref, rows: usize, n: usize, k: usize) !Ref {
    const y = try scratch(m, rows, n, .bf16);
    try m.matmul(e, x, bytes, w, y, @intCast(rows), @intCast(n), @intCast(k));
    return y;
}

fn positionsOf(m: *Model, first: u64, rows: usize) !Ref {
    const p = try scratch(m, rows, 1, .u64);
    for (p.buf.slice(u64, p.off / 8 + rows)[p.off / 8 ..][0..rows], 0..) |*v, i| v.* = first + i;
    return p;
}

fn finish(m: *Model) !void {
    m.frame.finish() catch |err| {
        m.broken = true;
        return err;
    };
}

/// Committed taps [rows, taps * hidden], the rows after m.committed_end, into every layer's K/V cache.
pub fn append(m: *Model, taps: Ref, rows: usize) !void {
    const s = m.graph.shape;
    if (m.broken or m.frame.active or rows == 0) return error.BadCommittedTaps;
    _ = try m.frame.begin();
    defer m.frame.release();
    const e = m.frame.encoder;
    const positions = try positionsOf(m, m.committed_end, rows);
    const fused = try project(m, e, taps, m.graph.fusion, rows);
    const hidden = try norm(m, e, fused, m.graph.hidden_norm, rows, 1, s.hidden);
    const kv = config.kvWidth(s);
    for (m.graph.layers, m.caches) |layer, cache| {
        const raw_keys = try project(m, e, hidden, layer.attention.k, rows);
        const keys = try rope(m, e, try norm(m, e, raw_keys, layer.attention.k_norm, rows, s.kv_heads, s.head_dim), positions, rows, s.kv_heads);
        const values = try project(m, e, hidden, layer.attention.v, rows);
        var first: usize = 0;
        while (first < rows) : (first += 16) {
            const count = @min(rows - first, 16);
            var slots: [16]u32 = undefined;
            for (slots[0..count], 0..) |*slot, i| slot.* = @intCast((m.committed_end + first + i) % s.window);
            const k_at = at(keys, first * kv * 2);
            const v_at = at(values, first * kv * 2);
            try m.attention.appendKv(e, slots[0..count], .{ .buf = k_at.buf, .off = k_at.off }, .{ .buf = v_at.buf, .off = v_at.off }, .{ .buf = cache.keys }, .{ .buf = cache.values });
            e.barrier();
        }
    }
    try finish(m);
    m.committed_end += rows;
    m.begin = @max(m.begin, m.committed_end -| (s.window - 1));
}

/// Query row i sees the context rows its window holds and the block's rows up to its own.
pub fn fillMask(out: []u8, context: usize, queries: usize, window: usize) !void {
    const keys = context + queries;
    if (out.len != queries * keys or context >= window) return error.BadDraftContext;
    @memset(out, 0);
    for (0..queries) |q| {
        const hidden = @min(context, (context + q + 1) -| window);
        @memset(out[q * keys + hidden ..][0 .. context + q + 1 - hidden], 1);
    }
}

/// One proposal: the greedy chain, each head row's 16 candidates ascending, and their score table.
pub const Proposal = struct {
    drafts: [15]u32,
    /// each row's top logit alone: the drafts with the second pass off (the gate's planted control)
    tops: [15]u32,
    candidates: [15][16]u32,
    table: [15][16][16]f32,
};

/// The block after `anchor` over the committed context: drafts and candidates as dspine_mlx's RefinementTail.greedy.
pub fn propose(m: *Model, anchor: u32) !Proposal {
    const s = m.graph.shape;
    const t = s.block;
    const n = t - 1;
    const h = s.hidden;
    const k = s.topk;
    if (m.broken or m.frame.active or anchor >= s.vocab or m.committed_end == 0) return error.BadDraftContext;
    const context: usize = @intCast(m.committed_end - m.begin);
    // unwrapped: the block's own rows go behind the context in the caches' spare rows
    if (m.begin != 0 or m.committed_end > s.window - 1) return error.UnsupportedDSpineContext;
    _ = try m.frame.begin();
    defer m.frame.release();
    var e = m.frame.encoder;
    const ids = try scratch(m, t, 1, .u32);
    const id_slice = ids.buf.slice(u32, ids.off / 4 + t)[ids.off / 4 ..][0..t];
    @memset(id_slice, s.mask);
    id_slice[0] = anchor;
    const positions = try positionsOf(m, m.committed_end, t);
    const x = try scratch(m, t, h, .bf16);
    try m.target.glue.embedding(e, m.target.config, m.target.weights.embed, .{ .buffer = ids.buf, .offset = ids.off }, .{ .buffer = x.buf, .offset = x.off }, @intCast(t));
    e.barrier();
    var residual = x;
    var xin = try norm(m, e, x, m.graph.layers[0].input_norm, t, 1, h);
    const keys = context + t;
    const mask = try scratch(m, t, keys, .u8);
    try fillMask(mask.buf.contents()[mask.off..][0 .. t * keys], context, t, s.window);
    const kv = config.kvWidth(s);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(s.head_dim)));
    const packed_w: Ref = .{ .buf = m.packed_w };
    const packed_bytes = m.packed_w.contents()[0 .. 3 * s.rank * s.code * 2];
    const codebook = try m.weight(m.graph.bridge.codebook.bytes);
    const bias = try m.weight(m.graph.bridge.gate_bias.bytes);
    var raw: Ref = undefined;
    var rms: Ref = undefined;
    var packed_rows: Ref = undefined;
    for (m.graph.layers, m.caches, 0..) |layer, cache, l| {
        const q = try rope(m, e, try norm(m, e, try project(m, e, xin, layer.attention.q, t), layer.attention.q_norm, t, s.heads, s.head_dim), positions, t, s.heads);
        const own_k = try rope(m, e, try norm(m, e, try project(m, e, xin, layer.attention.k, t), layer.attention.k_norm, t, s.kv_heads, s.head_dim), positions, t, s.kv_heads);
        const own_v = try project(m, e, xin, layer.attention.v, t);
        try m.copy(e, own_k, .{ .buf = cache.keys, .off = context * kv * 2 }, t, kv);
        try m.copy(e, own_v, .{ .buf = cache.values, .off = context * kv * 2 }, t, kv);
        const attended = try scratch(m, t, config.qWidth(s), .bf16);
        const A = core.shared_attention.Ref;
        const views = [_]A{ .{ .buf = q.buf, .off = q.off }, .{ .buf = cache.keys }, .{ .buf = cache.values }, .{ .buf = mask.buf, .off = mask.off }, .{ .buf = attended.buf, .off = attended.off } };
        if (keys > 1024) try m.attention.encodeSplit(e, t, keys, keys, scale, views[0], views[1], views[2], views[3], views[4]) else try m.attention.encode(e, t, keys, keys, scale, views[0], views[1], views[2], views[3], views[4]);
        e.barrier();
        const hsum = try scratch(m, t, h, .bf16);
        m.kernels.add(e, residual, try project(m, e, attended, layer.attention.o, t), hsum, @intCast(t), @intCast(h));
        e.barrier();
        const post = try norm(m, e, hsum, layer.post_norm, t, 1, h);
        const gate = try project(m, e, post, layer.mlp.gate, t);
        const up = try project(m, e, post, layer.mlp.up, t);
        const act = try scratch(m, t, s.intermediate, .bf16);
        try m.primitives.swiglu(e, gate, up, act, .{ .rows = @intCast(t), .width = @intCast(s.intermediate), .gate_stride = @intCast(s.intermediate), .up_stride = @intCast(s.intermediate), .y_stride = @intCast(s.intermediate) });
        e.barrier();
        raw = try scratch(m, t, h, .bf16);
        m.kernels.add(e, hsum, try project(m, e, act, layer.mlp.down, t), raw, @intCast(t), @intCast(h));
        e.barrier();
        // the bridge's write after layer l, closed by the next layer's input norm (the final norm after the last)
        const xn = try scratch(m, t, h, .bf16);
        rms = try scratch(m, t, 1, .f32);
        m.kernels.read(e, raw, xn, rms, @intCast(t), @intCast(h), s.bridge_eps);
        e.barrier();
        const projected = try bridgeLinear(m, e, xn, m.graph.readOf(l), try m.weight(m.graph.readOf(l)), t, s.code, h);
        const codes = try scratch(m, t, s.code, .bf16);
        m.kernels.code(e, projected, codebook, ids, codes, @intCast(t), @intCast(s.code), @intCast(t), s.bridge_eps);
        e.barrier();
        packed_rows = try bridgeLinear(m, e, codes, packed_bytes, packed_w, t, 3 * s.rank, s.code);
        const msg = try scratch(m, t, s.rank, .bf16);
        m.kernels.message(e, packed_rows, bias, msg, @intCast(t), @intCast(s.rank), @intCast(t));
        e.barrier();
        const delta = try bridgeLinear(m, e, msg, m.graph.writeOf(l), try m.weight(m.graph.writeOf(l)), t, h, s.rank);
        const last = l + 1 == s.layers;
        const next = if (last) m.graph.norm else m.graph.layers[l + 1].input_norm;
        const res = try scratch(m, t, h, .bf16);
        xin = try scratch(m, t, h, .bf16);
        m.kernels.write(e, raw, delta, rms, try m.weight(next.bytes), res, xin, @intCast(t), @intCast(h), @intCast(t), s.eps);
        e.barrier();
        residual = res;
        if (profile) {
            try m.frame.part();
            e = m.frame.encoder;
        }
    }
    // the head over rows 1.. (the anchor drafts nothing) for the candidates
    const v = s.vocab;
    const logits = try scratch(m, n, v, .bf16);
    try m.head(e, at(xin, h * 2), logits, @intCast(n));
    if (profile) {
        try m.frame.part();
        e = m.frame.encoder;
    }
    const top = try scratch(m, n, @sizeOf(topk_rows.Row), .u8);
    try m.topk.encode(e, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = top.buf, .off = top.off }, .{ .rows = @intCast(n), .vocab = @intCast(v), .k = @intCast(k), .stride = @intCast(v) });
    e.barrier();
    const senders = try scratch(m, n * k, s.code, .bf16);
    const sids = try scratch(m, n * k, 1, .u32);
    m.kernels.senders(e, top, codebook, senders, sids, .{ .rows = @intCast(n), .k = @intCast(k), .code = @intCast(s.code), .row_words = row_words, .anchor = anchor, .scale = @sqrt(@as(f32, @floatFromInt(s.code))) });
    e.barrier();
    const tsv = try bridgeLinear(m, e, senders, packed_bytes[s.rank * s.code * 2 ..], at(packed_w, s.rank * s.code * 2), n * k, 2 * s.rank, s.code);
    const base = try scratch(m, n * k, 1, .f32);
    m.kernels.base(e, raw, try m.weight(m.graph.norm.bytes), m.head_rows orelse return error.DSpineNeedsTargetDir, top, base, .{ .rows = @intCast(n), .k = @intCast(k), .hidden = @intCast(h), .row_words = row_words, .raw_first = 1 });
    const table = try scratch(m, n * k * k, 1, .f32);
    m.kernels.table(e, packed_rows, tsv, bias, .{ .buf = m.tg }, top, base, rms, table, .{ .rows = @intCast(n), .k = @intCast(k), .rank = @intCast(s.rank), .row_words = row_words, .rms_first = 1 });
    e.barrier();
    try finish(m);
    if (profile) {
        for (m.frame.parts[0..m.frame.part_count], 0..) |cb, i| std.debug.print("dspine part {d}: {d:.2} ms\n", .{ i, cb.gpuSeconds() * 1e3 });
        std.debug.print("dspine tail: {d:.2} ms\n", .{m.frame.cb.gpuSeconds() * 1e3});
    }
    return read(top, table, n, k);
}

/// Candidates ascending per row (ties to the lower id, as dspine_mlx's argmax), the table permuted, the chain walked.
fn read(top: Ref, table: Ref, n: usize, k: usize) !Proposal {
    var p: Proposal = undefined;
    const rows = std.mem.bytesAsSlice(topk_rows.Row, top.buf.contents()[top.off..][0 .. n * @sizeOf(topk_rows.Row)]);
    const scores = std.mem.bytesAsSlice(f32, table.buf.contents()[table.off..][0 .. n * k * k * 4]);
    var order: [15][16]u8 = undefined;
    for (0..n) |i| {
        if (rows[i].count != k or rows[i].nonfinite != 0) return error.DraftLatticeFinite;
        for (0..k) |j| order[i][j] = @intCast(j);
        const ids = rows[i].entries;
        p.tops[i] = ids[0].token;
        std.mem.sort(u8, order[i][0..k], ids, struct {
            fn lt(c: [16]topk_rows.Entry, a: u8, b: u8) bool {
                return c[a].token < c[b].token;
            }
        }.lt);
        for (0..k) |j| p.candidates[i][j] = ids[order[i][j]].token;
    }
    for (0..n) |i| for (0..k) |sj| for (0..k) |j| {
        const sender = if (i == 0) 0 else order[i - 1][sj];
        p.table[i][sj][j] = scores[(i * k + sender) * k + order[i][j]];
    };
    const flat_candidates: *const [15 * 16]u32 = @ptrCast(&p.candidates);
    const flat_table: *const [15 * 16 * 16]f32 = @ptrCast(&p.table);
    core.dspine.chain(flat_candidates[0 .. n * k], flat_table[0 .. n * k * k], k, p.drafts[0..n]);
    return p;
}

test "DSpine mask: block rows see the context and the block up to their own row; the window hides old context" {
    var mask: [4 * 7]u8 = undefined;
    try fillMask(&mask, 3, 4, 8192);
    for (0..4) |q| for (0..7) |key| try std.testing.expectEqual(@as(u8, @intFromBool(key < 3 + q + 1)), mask[q * 7 + key]);
    try fillMask(&mask, 3, 4, 5);
    // row q sits at absolute 3 + q: it sees rows past 3 + q - 5
    for (0..4) |q| for (0..7) |key| try std.testing.expectEqual(@as(u8, @intFromBool(key + 5 > 3 + q and key < 3 + q + 1)), mask[q * 7 + key]);
}
