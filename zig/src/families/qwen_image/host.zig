//! The host's part of a Qwen-Image-2.1 pass: rotary tables, the time embedding and the modulation all blocks share.
const std = @import("std");
const cfg = @import("config.zig");

/// fp32 to bf16, round to nearest even (torch's and MLX's cast); NaN stays NaN.
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (b & 0x7fffffff > 0x7f800000) return @truncate((b >> 16) | 0x40);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn round(x: f32) f32 {
    return widen(bf16(x));
}

/// Rotary cos, sin (rows, 64): prompt rows on the diagonal of all three axes, then the image centred at frame `text`.
pub fn rotary(cos: []f32, sin: []f32, text: usize, rows: usize, columns: usize) void {
    const half = cfg.head_dim / 2;
    std.debug.assert(cos.len == (text + rows * columns) * half and sin.len == cos.len);
    for (0..text + rows * columns) |r| {
        var pos: [3]f64 = .{ @floatFromInt(r), @floatFromInt(r), @floatFromInt(r) };
        if (r >= text) {
            const i = r - text;
            const top: i64 = @intCast(rows - rows / 2);
            const left: i64 = @intCast(columns - columns / 2);
            pos = .{ @floatFromInt(text), @floatFromInt(@as(i64, @intCast(i / columns)) - top), @floatFromInt(@as(i64, @intCast(i % columns)) - left) };
        }
        var at: usize = 0;
        for (cfg.axes, pos) |dim, p| {
            for (0..dim / 2) |j| {
                const freq: f32 = @floatCast(1.0 / std.math.pow(f64, cfg.theta, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dim))));
                const angle: f32 = @as(f32, @floatCast(p)) * freq;
                cos[r * half + at] = @cos(angle);
                sin[r * half + at] = @sin(angle);
                at += 1;
            }
        }
    }
}

/// y[o] = bf16(sum over k of x[k] w[o, k]) for a bf16 projection stored (outputs, inputs), fp32 sums.
pub fn project(w: []const u16, x: []const f32, y: []f32) void {
    std.debug.assert(w.len == x.len * y.len);
    const V = @Vector(8, f32);
    for (y, 0..) |*o, row| {
        const ws = w[row * x.len ..][0..x.len];
        var acc: V = @splat(0);
        var k: usize = 0;
        while (k + 8 <= x.len) : (k += 8) {
            const bits: @Vector(8, u32) = @as(@Vector(8, u16), ws[k..][0..8].*);
            const wv: V = @bitCast(bits << @splat(16));
            acc += wv * @as(V, x[k..][0..8].*);
        }
        var sum = @reduce(.Add, acc);
        while (k < x.len) : (k += 1) sum += widen(ws[k]) * x[k];
        o.* = round(sum);
    }
}

/// silu in place, as x * sigmoid(x) with each stage rounded to bf16.
pub fn silu(v: []f32) void {
    for (v) |*x| x.* = round(x.* * round(1.0 / (1.0 + @exp(-x.*))));
}

/// The bf16 projections the modulation reads, each stored (outputs, inputs).
pub const Weights = struct { time_1: []const u16, time_2: []const u16, modulation: []const u16, norm_out: []const u16 };

/// What a pass at one noise level gives every block: one plus each norm's scale, each gate's tanh, the output's scale.
pub const Modulation = struct {
    scale_1: [cfg.hidden]u16,
    gate_1: [cfg.hidden]u16,
    scale_2: [cfg.hidden]u16,
    gate_2: [cfg.hidden]u16,
    scale_out: [cfg.hidden]u16,

    /// At noise level `sigma` (prompt rows read 0): the sinusoid in bf16, the time MLP, silu, then both projections.
    pub fn at(m: *Modulation, gpa: std.mem.Allocator, w: Weights, sigma: f32) !void {
        const emb = try gpa.alloc(f32, cfg.time_dim);
        defer gpa.free(emb);
        const mid = try gpa.alloc(f32, cfg.hidden);
        defer gpa.free(mid);
        const temb = try gpa.alloc(f32, cfg.hidden);
        defer gpa.free(temb);
        const table = try gpa.alloc(f32, 4 * cfg.hidden);
        defer gpa.free(table);
        const half = cfg.time_dim / 2;
        for (0..half) |i| {
            const freq: f32 = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, half));
            const angle = sigma * 1000.0 * freq;
            emb[i] = round(@cos(angle));
            emb[half + i] = round(@sin(angle));
        }
        project(w.time_1, emb, mid);
        silu(mid);
        project(w.time_2, mid, temb);
        silu(temb);
        project(w.modulation, temb, table);
        const parts = [_]*[cfg.hidden]u16{ &m.scale_1, &m.gate_1, &m.scale_2, &m.gate_2 };
        for (parts, 0..) |out, p| for (out, table[p * cfg.hidden ..][0..cfg.hidden]) |*o, v| {
            o.* = bf16(if (p % 2 == 0) 1.0 + v else std.math.tanh(v));
        };
        project(w.norm_out, temb, mid);
        for (&m.scale_out, mid) |*o, v| o.* = bf16(1.0 + v);
    }
};

test "bf16 rounds to nearest even and keeps NaN" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expect(std.math.isNan(widen(bf16(std.math.nan(f32)))));
}

test "rotary positions: the prompt on the diagonal, the image centred, the frame held at the prompt's length" {
    const a = std.testing.allocator;
    const text = 3;
    const rows = 4;
    const columns = 6;
    const n = (text + rows * columns) * 64;
    const cos = try a.alloc(f32, n);
    defer a.free(cos);
    const sin = try a.alloc(f32, n);
    defer a.free(sin);
    rotary(cos, sin, text, rows, columns);
    // channel 0 of each axis turns by its position (frequency 1): frame at 0, height at 8, width at 36
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 2)), sin[2 * 64 + 0], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 2)), sin[2 * 64 + 8], 1e-6);
    const first = text * 64;
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 3)), sin[first + 0], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, -2)), sin[first + 8], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, -3)), sin[first + 36], 1e-6);
    const last = (text + rows * columns - 1) * 64;
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1)), sin[last + 8], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 2)), sin[last + 36], 1e-6);
}

test "project sums in fp32 and rounds once" {
    const row = [_]u16{ bf16(1), bf16(2), bf16(3), bf16(4), bf16(5), bf16(6), bf16(7), bf16(8), bf16(9), bf16(0.5) };
    const w = row ++ row;
    const x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    var y: [2]f32 = undefined;
    project(&w, &x, &y);
    try std.testing.expectEqual(@as(f32, 45.5), y[0]);
    try std.testing.expectEqual(@as(f32, 45.5), y[1]);
}
