//! A Qwen-VL tower on the host, sums in S (f64 or f32), rounded to bf16 where the Metal tower rounds: its oracle.
const std = @import("std");
const bf16_round = @import("bf16_round");
const erf = @import("../act_ref.zig").erf;
const Config = @import("config.zig").Config;
const grid = @import("grid.zig");

/// Weights as fp32 values bf16 holds exactly, in the Metal tower's layout (the patch kernel [O, C, T, Y, X]).
pub const Block = struct { norm1_w: []const f32, norm1_b: []const f32, qkv_w: []const f32, qkv_b: []const f32, proj_w: []const f32, proj_b: []const f32, norm2_w: []const f32, norm2_b: []const f32, fc1_w: []const f32, fc1_b: []const f32, fc2_w: []const f32, fc2_b: []const f32 };
pub const Merger = struct { norm_w: []const f32, norm_b: []const f32, fc1_w: []const f32, fc1_b: []const f32, fc2_w: []const f32, fc2_b: []const f32 };
pub const Weights = struct { patch_w: []const f32, patch_b: []const f32, pos: []const f32, blocks: []const Block, merger: Merger, deep: []const Merger };

fn bf(v: anytype) f32 {
    return @floatCast(bf16_round.bf16(v));
}

/// y [M, N] = bf16(x [M, K] w [N, K]^T), as the tower's bf16 products round once.
fn gemm(comptime S: type, x: []const f32, w: []const f32, y: []f32, M: usize, N: usize, K: usize) void {
    for (0..M) |m| for (0..N) |n| {
        var s: S = 0;
        for (x[m * K ..][0..K], w[n * K ..][0..K]) |p, q| s += @as(S, p) * q;
        y[m * N + n] = bf(s);
    };
}

/// y = bf16 of LayerNorm(x) by w and b, rows `width` wide (eps 1e-6).
fn norm(comptime S: type, x: []const S, w: []const f32, b: []const f32, y: []f32, width: usize) void {
    var at: usize = 0;
    while (at < x.len) : (at += width) {
        const r = x[at..][0..width];
        var sum: S = 0;
        for (r) |v| sum += v;
        const mean = sum / @as(S, @floatFromInt(width));
        var sq: S = 0;
        for (r) |v| sq += (v - mean) * (v - mean);
        const inv = 1.0 / @sqrt(sq / @as(S, @floatFromInt(width)) + 1e-6);
        for (r, w, b, y[at..][0..width]) |v, g, c, *o| o.* = bf((v - mean) * inv * g + c);
    }
}

/// y = bf16(gelu(y + b)): exact (erf) or the tanh form, rows `width` wide.
fn gelu(comptime S: type, y: []f32, b: []const f32, width: usize, exact: bool) void {
    for (y, 0..) |*o, i| {
        const v: S = @as(S, o.*) + b[i % width];
        const inner: S = if (exact) @floatCast(erf(@as(f64, v) / std.math.sqrt2)) else std.math.tanh(0.79788456080286536 * (v + 0.044715 * v * v * v));
        o.* = bf(0.5 * v * (1 + inner));
    }
}

/// x += y + b on the residual stream.
fn residual(comptime S: type, x: []S, y: []const f32, b: []const f32) void {
    for (x, y, 0..) |*p, q, i| p.* += @as(S, q) + b[i % b.len];
}

/// The residual's merged rows through `m` into `out` [T, out_hidden]: normed per patch, or after the shuffle.
fn merger(comptime S: type, a: std.mem.Allocator, c: Config, m: Merger, x: []const S, out: []f32, post_shuffle: bool) !void {
    const width: usize = c.merged();
    const T = x.len / width;
    const xn = try a.alloc(f32, x.len);
    defer a.free(xn);
    norm(S, x, m.norm_w, m.norm_b, xn, if (post_shuffle) width else c.hidden);
    const mid = try a.alloc(f32, T * width);
    defer a.free(mid);
    gemm(S, xn, m.fc1_w, mid, T, width, width);
    gelu(S, mid, m.fc1_b, width, true);
    gemm(S, mid, m.fc2_w, out, T, c.out_hidden, width);
    for (out, 0..) |*o, i| o.* = bf(@as(S, o.*) + m.fc2_b[i % c.out_hidden]);
}

/// Every head's attention over all P patches: att [P, hidden] = bf16(softmax(q k^T / sqrt(dim)) v).
fn attention(comptime S: type, a: std.mem.Allocator, c: Config, q: []const f32, k: []const f32, v: []const f32, att: []f32) !void {
    const hd: usize = c.headDim();
    const P = q.len / c.hidden;
    const p = try a.alloc(S, P);
    defer a.free(p);
    const scale = 1.0 / @sqrt(@as(S, @floatFromInt(hd)));
    for (0..c.heads) |h| for (0..P) |r| {
        const qr = q[(h * P + r) * hd ..][0..hd];
        var top = -std.math.inf(S);
        for (p, 0..) |*s, j| {
            var dot: S = 0;
            for (qr, k[(h * P + j) * hd ..][0..hd]) |x, y| dot += @as(S, x) * y;
            s.* = dot * scale;
            top = @max(top, s.*);
        }
        var total: S = 0;
        for (p) |*s| {
            s.* = @exp(s.* - top);
            total += s.*;
        }
        for (0..hd) |d| {
            var o: S = 0;
            for (p, 0..) |s, j| o += s * v[(h * P + j) * hd + d];
            att[r * c.hidden + h * hd + d] = bf(o / total);
        }
    };
}

/// An image's patches (merge order) to merged rows `out` and each DeepStack tap's in `deep` (none, or one per tap).
pub fn image(comptime S: type, a: std.mem.Allocator, c: Config, w: Weights, pixels: []const f32, gh: u32, gw: u32, out: []f32, deep: []const []f32) !void {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const P: usize = @as(usize, gh) * gw;
    const h: usize = c.hidden;
    const hd: usize = c.headDim();
    const pw: usize = c.patchWidth();
    if (P == 0 or pixels.len != P * pw or out.len * c.merge * c.merge != P * c.out_hidden) return error.ImageGrid;
    if (deep.len != 0 and (deep.len != c.taps().len or w.deep.len != deep.len)) return error.DeepStackOutputs;
    const patches = try ar.alloc(f32, pixels.len);
    for (patches, pixels) |*o, v| o.* = bf(v);
    const xn = try ar.alloc(f32, P * h);
    gemm(S, patches, w.patch_w, xn, P, h, pw);
    // the residual: the projection, its bias, and the patch's four learned-position taps
    const x = try ar.alloc(S, P * h);
    for (0..P) |i| {
        const at = grid.place(i, gw, c.merge);
        const tp = grid.taps(at[0], at[1], gh, gw, c.side());
        for (0..h) |d| {
            var pos: S = 0;
            for (tp.index, tp.weight) |n, wt| pos += @as(S, w.pos[n * h + d]) * wt;
            x[i * h + d] = @as(S, xn[i * h + d]) + w.patch_b[d] + pos;
        }
    }
    const inv = try ar.alloc(f32, hd / 4);
    grid.invFreq(inv, c.headDim(), grid.axial_theta);
    const cos = try ar.alloc(f32, P * hd);
    const sin = try ar.alloc(f32, P * hd);
    grid.rotary(cos, sin, gh, gw, c.merge, inv);
    const qkv = try ar.alloc(f32, P * 3 * h);
    const q = try ar.alloc(f32, P * h);
    const k = try ar.alloc(f32, P * h);
    const v = try ar.alloc(f32, P * h);
    const att = try ar.alloc(f32, P * h);
    const mid = try ar.alloc(f32, P * c.intermediate);
    for (w.blocks, 0..) |b, bi| {
        norm(S, x, b.norm1_w, b.norm1_b, xn, h);
        gemm(S, xn, b.qkv_w, qkv, P, 3 * h, h);
        // q, k and v head-major with their bias; q and k turned by rotate_half, each rounded to bf16
        const mid_d = hd / 2;
        for (0..P) |r| for (0..3) |which| for (0..c.heads) |hh| for (0..hd) |d| {
            const col = (which * c.heads + hh) * hd;
            const val: S = @as(S, qkv[r * 3 * h + col + d]) + b.qkv_b[col + d];
            const o = (hh * P + r) * hd + d;
            if (which == 2) {
                v[o] = bf(val);
                continue;
            }
            const e = if (d < mid_d) d + mid_d else d - mid_d;
            const partner: S = @as(S, qkv[r * 3 * h + col + e]) + b.qkv_b[col + e];
            const cs: S = cos[r * hd + d];
            const sn: S = sin[r * hd + d];
            (if (which == 0) q else k)[o] = bf(if (d < mid_d) val * cs - partner * sn else val * cs + partner * sn);
        };
        try attention(S, ar, c, q, k, v, att);
        gemm(S, att, b.proj_w, xn, P, h, h);
        residual(S, x, xn, b.proj_b);
        norm(S, x, b.norm2_w, b.norm2_b, xn, h);
        gemm(S, xn, b.fc1_w, mid, P, c.intermediate, h);
        gelu(S, mid, b.fc1_b, c.intermediate, false);
        gemm(S, mid, b.fc2_w, xn, P, h, c.intermediate);
        residual(S, x, xn, b.fc2_b);
        if (deep.len != 0) for (c.taps(), w.deep, deep) |tap, m, o| if (tap == bi) try merger(S, ar, c, m, x, o, true);
    }
    try merger(S, ar, c, w.merger, x, out, false);
}

test "the reference's LayerNorm, GELUs and merged-row norm on hand values" {
    var y: [4]f32 = undefined;
    norm(f64, &[_]f64{ 1, 2, 3, 4 }, &.{ 1, 1, 1, 1 }, &.{ 0, 0, 0, 0 }, &y, 4);
    const s = @sqrt(1.25 + 1e-6);
    try std.testing.expectEqual(bf(-1.5 / s), y[0]);
    try std.testing.expectEqual(bf(0.5 / s), y[2]);
    var g = [_]f32{ 1, -1 };
    gelu(f64, &g, &.{ 0, 0 }, 2, true);
    try std.testing.expectEqual(bf(0.5 * (1 + erf(1.0 / std.math.sqrt2))), g[0]); // 0.8413...
    var t = [_]f32{1};
    gelu(f64, &t, &.{0.5}, 1, false);
    try std.testing.expectEqual(bf(0.75 * (1 + std.math.tanh(@as(f64, 0.79788456080286536 * (1.5 + 0.044715 * 3.375))))), t[0]);
    // the post-shuffle norm reads four patches' rows as one: a constant row normalises to its bias
    var row: [8]f32 = undefined;
    const flat: [8]f64 = @splat(3);
    const ones: [8]f32 = @splat(1);
    const bias: [8]f32 = @splat(0.25);
    norm(f64, &flat, &ones, &bias, &row, 8);
    for (row) |r| try std.testing.expectEqual(@as(f32, 0.25), r);
}
