//! The prompt path on the host, sums in S (f64 or f32), rounded to bf16 where the Metal kernels round: the oracle.
const std = @import("std");
const rules = @import("picture_prompt.zig");
const bf16_round = @import("bf16_round");

const D = rules.head_dim;
pub const Mode = enum { qwen, h3 };
/// Activations and weights as fp32 values that bf16 holds exactly.
pub const Layer = rules.Layer([]const f32);
pub const Block = rules.Block([]const f32);
pub const Text = struct { embed: []const f32, layers: []const Layer };
/// prompt.Vision's rows as fp32 values.
pub const Vision = struct { pads: []const u32, rows: []const f32, deepstack: [3][]const f32 };
pub const Refiner = struct { cond_w: []const f32, cond_b: []const f32, final_norm: []const f32, blocks: []const Block };

fn bf(v: anytype) f32 {
    return @floatCast(bf16_round.bf16(v));
}

/// y [M, N] = bf16(x [M, K] w [N, K]^T (+ b)).
pub fn gemm(comptime S: type, x: []const f32, w: []const f32, b: ?[]const f32, y: []f32, M: usize, N: usize, K: usize) void {
    for (0..M) |m| for (0..N) |n| {
        var s: S = 0;
        for (x[m * K ..][0..K], w[n * K ..][0..K]) |p, q| s += @as(S, p) * q;
        y[m * N + n] = bf(if (b) |v| s + v[n] else s);
    };
}

fn normed(comptime S: type, x: f32, r: S, w: f32, mode: Mode) f32 {
    return switch (mode) {
        .qwen => bf(@as(S, w) * bf(x * r)),
        .h3 => bf(x * r * w),
    };
}

fn inverse(comptime S: type, x: []const f32, eps: f32) S {
    var s: S = 0;
    for (x) |v| s += @as(S, v) * v;
    return 1.0 / @sqrt(s / @as(S, @floatFromInt(x.len)) + eps);
}

/// y [rows, width] = RMSNorm(x) by w.
pub fn norm(comptime S: type, x: []const f32, w: []const f32, y: []f32, width: usize, eps: f32, mode: Mode) void {
    var at: usize = 0;
    while (at < x.len) : (at += width) {
        const r = inverse(S, x[at..][0..width], eps);
        for (x[at..][0..width], w, y[at..][0..width]) |v, g, *o| o.* = normed(S, v, r, g, mode);
    }
}

/// Heads of 128 `stride` apart in rows `row` apart, normed in place and turned by cos, sin [rows, 64] if given.
pub fn heads(comptime S: type, x: []f32, count: usize, row: usize, stride: usize, w: []const f32, rot: ?[2][]const f32, rows: usize, eps: f32, mode: Mode) void {
    for (0..rows) |r| for (0..count) |h| {
        const v = x[r * row + h * stride ..][0..D];
        const inv = inverse(S, v, eps);
        for (v, w) |*o, g| o.* = normed(S, o.*, inv, g, mode);
        const t = rot orelse continue;
        var turned: [D]f32 = undefined;
        for (0..D) |c| {
            const f = r * 64 + c % 64;
            const half: S = if (c < 64) -v[c + 64] else v[c - 64];
            turned[c] = bf(@as(S, bf(@as(S, v[c]) * t[0][f])) + bf(half * t[1][f]));
        }
        @memcpy(v, &turned);
    };
}

/// Where a projection's heads sit, as prompt_kernels.Heads.
pub const Heads = struct { count: usize, row: usize, head: usize };

/// o [rows, q heads x 128] = softmax(q k^T / sqrt(128)) v over earlier rows when `causal`, else every row.
pub fn attention(comptime S: type, a: std.mem.Allocator, q: []const f32, qh: Heads, k: []const f32, v: []const f32, kh: Heads, o: []f32, rows: usize, causal: bool) !void {
    const p = try a.alloc(S, rows);
    defer a.free(p);
    const group = qh.count / kh.count;
    for (0..rows) |r| for (0..qh.count) |h| {
        const qr = q[r * qh.row + h * qh.head ..][0..D];
        const kv = h / group;
        const end = if (causal) r + 1 else rows;
        var top: S = -std.math.inf(S);
        for (0..end) |j| {
            var s: S = 0;
            for (qr, k[j * kh.row + kv * kh.head ..][0..D]) |x, y| s += @as(S, x) * y;
            p[j] = s * (1.0 / @sqrt(@as(S, 128.0)));
            top = @max(top, p[j]);
        }
        var total: S = 0;
        for (p[0..end]) |*e| {
            e.* = @exp(e.* - top);
            total += e.*;
        }
        for (0..D) |c| {
            var s: S = 0;
            for (0..end) |j| s += p[j] * v[j * kh.row + kv * kh.head + c];
            o[r * qh.count * D + h * D + c] = bf(s / total);
        }
    };
}

/// o [rows, width] = silu(g) v, rounded as `mode`.
pub fn glu(comptime S: type, g: []const f32, v: []const f32, o: []f32, rows: usize, width: usize, g_row: usize, v_row: usize, mode: Mode) void {
    for (0..rows) |r| for (0..width) |c| {
        const x: S = g[r * g_row + c];
        const e = 1.0 + @exp(-x);
        const s: S = switch (mode) {
            .qwen => bf(x / e),
            .h3 => bf(x * bf(1.0 / e)),
        };
        o[r * width + c] = bf(s * v[r * v_row + c]);
    };
}

pub fn add(x: []f32, y: []const f32) void {
    for (x, y) |*p, q| p.* = bf(@as(f64, p.*) + q);
}

/// The text model's rows leaving its last layer [T, hidden] for `ids`, cos and sin [T, 64].
pub fn encode(comptime S: type, a: std.mem.Allocator, g: rules.Text, w: Text, ids: []const u32, cos: []const f32, sin: []const f32, vision: ?Vision, out: []f32) !void {
    const T = ids.len;
    const H = g.hidden;
    const qn = g.heads * D;
    const kvn = g.kv_heads * D;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const x = out;
    for (ids, 0..) |id, r| @memcpy(x[r * H ..][0..H], w.embed[id * H ..][0..H]);
    if (vision) |v| for (v.pads, 0..) |at, i| @memcpy(x[at * H ..][0..H], v.rows[i * H ..][0..H]);
    const xn = try ar.alloc(f32, T * H);
    const y = try ar.alloc(f32, T * H);
    const q = try ar.alloc(f32, T * qn);
    const k = try ar.alloc(f32, T * kvn);
    const v = try ar.alloc(f32, T * kvn);
    const att = try ar.alloc(f32, T * qn);
    const gate = try ar.alloc(f32, T * g.ffn);
    const up = try ar.alloc(f32, T * g.ffn);
    const act = try ar.alloc(f32, T * g.ffn);
    for (w.layers[0..g.layers], 0..) |l, li| {
        norm(S, x, l.norm1, xn, H, g.eps, .qwen);
        gemm(S, xn, l.q, null, q, T, qn, H);
        gemm(S, xn, l.k, null, k, T, kvn, H);
        gemm(S, xn, l.v, null, v, T, kvn, H);
        heads(S, q, g.heads, qn, D, l.q_norm, .{ cos, sin }, T, g.eps, .qwen);
        heads(S, k, g.kv_heads, kvn, D, l.k_norm, .{ cos, sin }, T, g.eps, .qwen);
        try attention(S, ar, q, .{ .count = g.heads, .row = qn, .head = D }, k, v, .{ .count = g.kv_heads, .row = kvn, .head = D }, att, T, true);
        gemm(S, att, l.o, null, y, T, H, qn);
        add(x, y);
        norm(S, x, l.norm2, xn, H, g.eps, .qwen);
        gemm(S, xn, l.gate, null, gate, T, g.ffn, H);
        gemm(S, xn, l.up, null, up, T, g.ffn, H);
        glu(S, gate, up, act, T, g.ffn, g.ffn, g.ffn, .qwen);
        gemm(S, act, l.down, null, y, T, H, g.ffn);
        add(x, y);
        const seen = vision orelse continue;
        if (li < 3) for (seen.pads, 0..) |at, i| add(x[at * H ..][0..H], seen.deepstack[li][i * H ..][0..H]);
    }
}

/// condition_proj, the refiner's blocks over every row and its final norm: rows [T, text] to out [T, hidden].
pub fn refine(comptime S: type, a: std.mem.Allocator, g: rules.Refiner, w: Refiner, rows: []const f32, out: []f32) !void {
    const T = rows.len / g.text;
    const H = g.hidden;
    const inner = g.heads * D;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const x = try ar.alloc(f32, T * H);
    const xn = try ar.alloc(f32, T * H);
    const y = try ar.alloc(f32, T * H);
    const qkv = try ar.alloc(f32, T * 3 * inner);
    const att = try ar.alloc(f32, T * inner);
    const wide = try ar.alloc(f32, T * 2 * g.mlp);
    const act = try ar.alloc(f32, T * g.mlp);
    gemm(S, rows, w.cond_w, w.cond_b, x, T, H, g.text);
    const fused: Heads = .{ .count = g.heads, .row = 3 * inner, .head = 3 * D };
    for (w.blocks[0..g.blocks]) |b| {
        norm(S, x, b.norm1, xn, H, g.eps, .h3);
        gemm(S, xn, b.qkv, null, qkv, T, 3 * inner, H);
        heads(S, qkv, g.heads, 3 * inner, 3 * D, b.q_norm, null, T, g.qk_eps, .h3);
        heads(S, qkv[D..], g.heads, 3 * inner, 3 * D, b.k_norm, null, T, g.qk_eps, .h3);
        try attention(S, ar, qkv, fused, qkv[D..], qkv[2 * D ..], fused, att, T, false);
        gemm(S, att, b.out, null, y, T, H, inner);
        add(x, y);
        norm(S, x, b.norm2, xn, H, g.eps, .h3);
        gemm(S, xn, b.fc1, null, wide, T, 2 * g.mlp, H);
        glu(S, wide, wide[g.mlp..], act, T, g.mlp, 2 * g.mlp, 2 * g.mlp, .h3);
        gemm(S, act, b.fc2, null, y, T, H, g.mlp);
        add(x, y);
    }
    norm(S, x, w.final_norm, out, H, g.final_eps, .h3);
}
