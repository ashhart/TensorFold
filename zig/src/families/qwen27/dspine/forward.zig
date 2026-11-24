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

/// Committed taps [rows, taps * hidden], the rows after the state's context_end, into every layer's K/V cache.
pub fn append(m: *Model, st: runtime.State, taps: Ref, rows: usize) !void {
    const s = m.graph.shape;
    if (m.broken or m.frame.active or rows == 0) return error.BadCommittedTaps;
    _ = try m.frame.begin();
    defer m.frame.release();
    const e = m.frame.encoder;
    const positions = try positionsOf(m, st.context_end.*, rows);
    const fused = try project(m, e, taps, m.graph.fusion, rows);
    const hidden = try norm(m, e, fused, m.graph.hidden_norm, rows, 1, s.hidden);
    const kv = config.kvWidth(s);
    for (m.graph.layers, st.caches) |layer, cache| {
        const raw_keys = try project(m, e, hidden, layer.attention.k, rows);
        const keys = try rope(m, e, try norm(m, e, raw_keys, layer.attention.k_norm, rows, s.kv_heads, s.head_dim), positions, rows, s.kv_heads);
        const values = try project(m, e, hidden, layer.attention.v, rows);
        var first: usize = 0;
        while (first < rows) : (first += 16) {
            const count = @min(rows - first, 16);
            var slots: [16]u32 = undefined;
            for (slots[0..count], 0..) |*slot, i| slot.* = @intCast((st.context_end.* + first + i) % s.window);
            const k_at = at(keys, first * kv * 2);
            const v_at = at(values, first * kv * 2);
            try m.attention.appendKv(e, slots[0..count], .{ .buf = k_at.buf, .off = k_at.off }, .{ .buf = v_at.buf, .off = v_at.off }, .{ .buf = cache.keys }, .{ .buf = cache.values });
            e.barrier();
        }
    }
    try finish(m);
    st.context_end.* += rows;
    st.begin.* = @max(st.begin.*, st.context_end.* -| (s.window - 1));
}

/// Query row i (absolute end + i) sees the cached rows at [begin, end) its window reaches, and the block to its own.
pub fn fillMask(out: []u8, end: u64, begin: u64, queries: usize, window: usize) !void {
    const context: usize = @intCast(@min(end, window));
    const keys = context + queries;
    if (out.len != queries * keys or begin > end) return error.BadDraftContext;
    @memset(out, 0);
    for (0..queries) |q| {
        const row = out[q * keys ..][0..keys];
        const oldest = @max(begin, (end + q + 1) -| window);
        for (row[0..context], 0..) |*v, j| {
            // slot j holds the latest absolute row below end that is j modulo the window
            const held: u64 = if (end <= window) j else j + window * ((end - 1 - j) / window);
            v.* = @intFromBool(held >= oldest);
        }
        @memset(row[context..][0 .. q + 1], 1);
    }
}

/// One proposal: the greedy chain, each head row's 16 candidates ascending, and their score table.
pub const Proposal = struct {
    drafts: [15]u32,
    /// each row's top logit alone: the drafts with the second pass off (the gate's planted control)
    tops: [15]u32,
    candidates: [15][16]u32,
    table: [15][16][16]f32,
    /// each head row's final-norm scale 1 / rms(written row): the table's scores times it are logits
    scales: [15]f32,
};

/// Each stream's block after its anchor over its committed context, in one pass: as dspine_mlx's RefinementTail.greedy.
pub fn propose(m: *Model, states: []const runtime.State, anchors: []const u32, out: []Proposal) !void {
    const s = m.graph.shape;
    const t = s.block;
    const n = t - 1;
    const h = s.hidden;
    const k = s.topk;
    const b = states.len;
    const rows = b * t;
    if (m.broken or m.frame.active or b == 0 or anchors.len != b or out.len != b) return error.BadDraftContext;
    for (states, anchors) |st, anchor| if (anchor >= s.vocab or st.committed_end == 0) return error.BadDraftContext;
    for (states) |st| try m.flush(st);
    _ = try m.frame.begin();
    defer m.frame.release();
    var e = m.frame.encoder;
    const ids = try scratch(m, rows, 1, .u32);
    const id_slice = ids.buf.slice(u32, ids.off / 4 + rows)[ids.off / 4 ..][0..rows];
    @memset(id_slice, s.mask);
    const positions = try scratch(m, rows, 1, .u64);
    const position_slice = positions.buf.slice(u64, positions.off / 8 + rows)[positions.off / 8 ..][0..rows];
    // each stream: its context's slots, then the block's own rows behind them (past the window: the caches' spare rows)
    var contexts: [8]usize = undefined;
    var masks: [8]Ref = undefined;
    for (states, anchors, 0..) |st, anchor, j| {
        id_slice[j * t] = anchor;
        for (position_slice[j * t ..][0..t], 0..) |*v, i| v.* = st.context_end.* + i;
        contexts[j] = @intCast(@min(st.context_end.*, s.window));
        const keys = contexts[j] + t;
        masks[j] = try scratch(m, t, keys, .u8);
        try fillMask(masks[j].buf.contents()[masks[j].off..][0 .. t * keys], st.context_end.*, st.begin.*, t, s.window);
    }
    const x = try scratch(m, rows, h, .bf16);
    try m.target.glue.embedding(e, m.target.config, m.target.weights.embed, .{ .buffer = ids.buf, .offset = ids.off }, .{ .buffer = x.buf, .offset = x.off }, @intCast(rows));
    e.barrier();
    var residual = x;
    var xin = try norm(m, e, x, m.graph.layers[0].input_norm, rows, 1, h);
    const kv = config.kvWidth(s);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(s.head_dim)));
    const packed_w: Ref = .{ .buf = m.packed_w };
    const packed_bytes = m.packed_w.contents()[0 .. 3 * s.rank * s.code * 2];
    const codebook = try m.weight(m.graph.bridge.codebook.bytes);
    const bias = try m.weight(m.graph.bridge.gate_bias.bytes);
    var raw: Ref = undefined;
    var rms: Ref = undefined;
    var packed_rows: Ref = undefined;
    for (m.graph.layers, 0..) |layer, l| {
        const q = try rope(m, e, try norm(m, e, try project(m, e, xin, layer.attention.q, rows), layer.attention.q_norm, rows, s.heads, s.head_dim), positions, rows, s.heads);
        const own_k = try rope(m, e, try norm(m, e, try project(m, e, xin, layer.attention.k, rows), layer.attention.k_norm, rows, s.kv_heads, s.head_dim), positions, rows, s.kv_heads);
        const own_v = try project(m, e, xin, layer.attention.v, rows);
        const attended = try scratch(m, rows, config.qWidth(s), .bf16);
        for (states, 0..) |st, j| {
            const cache = st.caches[l];
            const context = contexts[j];
            try m.copy(e, at(own_k, j * t * kv * 2), .{ .buf = cache.keys, .off = context * kv * 2 }, t, kv);
            try m.copy(e, at(own_v, j * t * kv * 2), .{ .buf = cache.values, .off = context * kv * 2 }, t, kv);
            const keys = context + t;
            const A = core.shared_attention.Ref;
            const qj = at(q, j * t * config.qWidth(s) * 2);
            const oj = at(attended, j * t * config.qWidth(s) * 2);
            const views = [_]A{ .{ .buf = qj.buf, .off = qj.off }, .{ .buf = cache.keys }, .{ .buf = cache.values }, .{ .buf = masks[j].buf, .off = masks[j].off }, .{ .buf = oj.buf, .off = oj.off } };
            if (keys > 1024) try m.attention.encodeSplit(e, t, keys, keys, scale, views[0], views[1], views[2], views[3], views[4]) else try m.attention.encode(e, t, keys, keys, scale, views[0], views[1], views[2], views[3], views[4]);
        }
        e.barrier();
        const hsum = try scratch(m, rows, h, .bf16);
        m.kernels.add(e, residual, try project(m, e, attended, layer.attention.o, rows), hsum, @intCast(rows), @intCast(h));
        e.barrier();
        const post = try norm(m, e, hsum, layer.post_norm, rows, 1, h);
        const gate = try project(m, e, post, layer.mlp.gate, rows);
        const up = try project(m, e, post, layer.mlp.up, rows);
        const act = try scratch(m, rows, s.intermediate, .bf16);
        try m.primitives.swiglu(e, gate, up, act, .{ .rows = @intCast(rows), .width = @intCast(s.intermediate), .gate_stride = @intCast(s.intermediate), .up_stride = @intCast(s.intermediate), .y_stride = @intCast(s.intermediate) });
        e.barrier();
        raw = try scratch(m, rows, h, .bf16);
        m.kernels.add(e, hsum, try project(m, e, act, layer.mlp.down, rows), raw, @intCast(rows), @intCast(h));
        e.barrier();
        // the bridge's write after layer l, closed by the next layer's input norm (the final norm after the last)
        const xn = try scratch(m, rows, h, .bf16);
        rms = try scratch(m, rows, 1, .f32);
        m.kernels.read(e, raw, xn, rms, @intCast(rows), @intCast(h), s.bridge_eps);
        e.barrier();
        const projected = try bridgeLinear(m, e, xn, m.graph.readOf(l), try m.weight(m.graph.readOf(l)), rows, s.code, h);
        const codes = try scratch(m, rows, s.code, .bf16);
        m.kernels.code(e, projected, codebook, ids, codes, @intCast(rows), @intCast(s.code), @intCast(t), s.bridge_eps);
        e.barrier();
        packed_rows = try bridgeLinear(m, e, codes, packed_bytes, packed_w, rows, 3 * s.rank, s.code);
        const msg = try scratch(m, rows, s.rank, .bf16);
        m.kernels.message(e, packed_rows, bias, msg, @intCast(rows), @intCast(s.rank), @intCast(t));
        e.barrier();
        const delta = try bridgeLinear(m, e, msg, m.graph.writeOf(l), try m.weight(m.graph.writeOf(l)), rows, h, s.rank);
        const last = l + 1 == s.layers;
        const next = if (last) m.graph.norm else m.graph.layers[l + 1].input_norm;
        const res = try scratch(m, rows, h, .bf16);
        xin = try scratch(m, rows, h, .bf16);
        m.kernels.write(e, raw, delta, rms, try m.weight(next.bytes), res, xin, @intCast(rows), @intCast(h), @intCast(t), s.eps);
        e.barrier();
        residual = res;
        if (profile) {
            try m.frame.part();
            e = m.frame.encoder;
        }
    }
    // the head over each block's rows 1.. (the anchor drafts nothing), gathered, for the candidates
    const v = s.vocab;
    const heads = try scratch(m, b * n, h, .bf16);
    for (0..b) |j| try m.copy(e, at(xin, (j * t + 1) * h * 2), at(heads, j * n * h * 2), n, h);
    const logits = try scratch(m, b * n, v, .bf16);
    try m.head(e, heads, logits, @intCast(b * n));
    if (profile) {
        try m.frame.part();
        e = m.frame.encoder;
    }
    const row_bytes = @sizeOf(topk_rows.Row);
    const top = try scratch(m, b * n, row_bytes, .u8);
    for (0..b) |j| { // the top-k takes at most 16 rows: one stream's 15 at a time
        const lj = at(logits, j * n * v * 2);
        const tj = at(top, j * n * row_bytes);
        try m.topk.encode(e, .{ .buf = lj.buf, .off = lj.off }, .{ .buf = tj.buf, .off = tj.off }, .{ .rows = @intCast(n), .vocab = @intCast(v), .k = @intCast(k), .stride = @intCast(v) });
    }
    e.barrier();
    const senders = try scratch(m, b * n * k, s.code, .bf16);
    const sids = try scratch(m, b * n * k, 1, .u32);
    const scaled = @sqrt(@as(f32, @floatFromInt(s.code)));
    for (anchors, 0..) |anchor, j| m.kernels.senders(e, at(top, j * n * row_bytes), codebook, at(senders, j * n * k * s.code * 2), at(sids, j * n * k * 4), .{ .rows = @intCast(n), .k = @intCast(k), .code = @intCast(s.code), .row_words = row_words, .anchor = anchor, .scale = scaled });
    e.barrier();
    const tsv = try bridgeLinear(m, e, senders, packed_bytes[s.rank * s.code * 2 ..], at(packed_w, s.rank * s.code * 2), b * n * k, 2 * s.rank, s.code);
    const base = try scratch(m, b * n * k, 1, .f32);
    const head_rows = m.head_rows orelse return error.DSpineNeedsTargetDir;
    const norm_w = try m.weight(m.graph.norm.bytes);
    for (0..b) |j| m.kernels.base(e, at(raw, j * t * h * 2), norm_w, head_rows, at(top, j * n * row_bytes), at(base, j * n * k * 4), .{ .rows = @intCast(n), .k = @intCast(k), .hidden = @intCast(h), .row_words = row_words, .raw_first = 1 });
    const table = try scratch(m, b * n * k * k, 1, .f32);
    for (0..b) |j| m.kernels.table(e, at(packed_rows, j * t * 3 * s.rank * 2), at(tsv, j * n * k * 2 * s.rank * 2), bias, .{ .buf = m.tg }, at(top, j * n * row_bytes), at(base, j * n * k * 4), at(rms, j * t * 4), at(table, j * n * k * k * 4), .{ .rows = @intCast(n), .k = @intCast(k), .rank = @intCast(s.rank), .row_words = row_words, .rms_first = 1 });
    e.barrier();
    try finish(m);
    if (profile) {
        for (m.frame.parts[0..m.frame.part_count], 0..) |cb, i| std.debug.print("dspine part {d}: {d:.2} ms\n", .{ i, cb.gpuSeconds() * 1e3 });
        std.debug.print("dspine tail: {d:.2} ms\n", .{m.frame.cb.gpuSeconds() * 1e3});
    }
    for (out, 0..) |*p, j| p.* = try read(at(top, j * n * row_bytes), at(table, j * n * k * k * 4), at(residual, j * t * h * 2), n, k, h, s.eps);
}

/// Candidates ascending per row (ties to the lower id, as dspine_mlx's argmax), the table permuted, the chain walked.
fn read(top: Ref, table: Ref, written: Ref, n: usize, k: usize, h: usize, eps: f32) !Proposal {
    var p: Proposal = undefined;
    const rows = written.buf.slice(u16, written.off / 2 + (n + 1) * h)[written.off / 2 ..];
    for (0..n) |i| {
        var ss: f32 = 0;
        for (rows[(i + 1) * h ..][0..h]) |v| ss += @import("host.zig").float(v) * @import("host.zig").float(v);
        p.scales[i] = 1 / @sqrt(ss / @as(f32, @floatFromInt(h)) + eps);
    }
    const picks = std.mem.bytesAsSlice(topk_rows.Row, top.buf.contents()[top.off..][0 .. n * @sizeOf(topk_rows.Row)]);
    const scores = std.mem.bytesAsSlice(f32, table.buf.contents()[table.off..][0 .. n * k * k * 4]);
    var order: [15][16]u8 = undefined;
    for (0..n) |i| {
        if (picks[i].count != k or picks[i].nonfinite != 0) return error.DraftLatticeFinite;
        for (0..k) |j| order[i][j] = @intCast(j);
        const ids = picks[i].entries;
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
    try fillMask(&mask, 3, 0, 4, 8192);
    for (0..4) |q| for (0..7) |key| try std.testing.expectEqual(@as(u8, @intFromBool(key < 3 + q + 1)), mask[q * 7 + key]);
    try fillMask(&mask, 3, 0, 4, 5);
    // row q sits at absolute 3 + q: it sees rows past 3 + q - 5
    for (0..4) |q| for (0..7) |key| try std.testing.expectEqual(@as(u8, @intFromBool(key + 5 > 3 + q and key < 3 + q + 1)), mask[q * 7 + key]);
    // wrapped: end 7 over window 5, slots 0..4 hold rows 5, 6, 2, 3, 4; row q sees rows past 7 + q - 5
    var wrapped: [2 * 7]u8 = undefined;
    try fillMask(&wrapped, 7, 0, 2, 5);
    const rows = [_]u64{ 5, 6, 2, 3, 4 };
    for (0..2) |q| {
        for (rows, 0..) |r, j| try std.testing.expectEqual(@as(u8, @intFromBool(r + 5 > 7 + q)), wrapped[q * 7 + j]);
        for (0..2) |b| try std.testing.expectEqual(@as(u8, @intFromBool(b <= q)), wrapped[q * 7 + 5 + b]);
    }
    // a restored context's rows before `begin` were never cached
    try fillMask(&wrapped, 7, 4, 2, 5);
    for (rows, 0..) |r, j| try std.testing.expectEqual(@as(u8, @intFromBool(r >= 4)), wrapped[j]);
}
