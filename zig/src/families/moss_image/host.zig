//! The host's part of a MOSS-Image pass: the rotary tables, and the time embedding's modulation all blocks share.
const std = @import("std");
const cfg = @import("config.zig");

/// fp32 to bf16, round to nearest even (torch's cast); NaN stays NaN.
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

/// Rotary cos, sin (image + text rows, 64) in bf16: image rows at frame `text`, centred; then prompt rows by frame.
pub fn rotary(cos: []f32, sin: []f32, bands: *const [cfg.bands]f32, text: usize, rows: usize, columns: usize) void {
    const half = cfg.head_dim / 2;
    const image = rows * columns;
    std.debug.assert(cos.len == (image + text) * half and sin.len == cos.len);
    const top = @as(f32, @floatFromInt(rows - 1)) / 2;
    const left = @as(f32, @floatFromInt(columns - 1)) / 2;
    for (0..image + text) |r| {
        const at: [4]f32 = if (r < image)
            .{ 0, @floatFromInt(text), @as(f32, @floatFromInt(r / columns)) - top, @as(f32, @floatFromInt(r % columns)) - left }
        else
            .{ 0, @floatFromInt(r - image), 0, 0 };
        for (0..half) |j| {
            const angle = at[j / cfg.bands] * bands[j % cfg.bands];
            cos[r * half + j] = round(@cos(angle));
            sin[r * half + j] = round(@sin(angle));
        }
    }
}

/// y[o] = bf16(sum over k of x[k] w[o, k] + b[o]), w stored (outputs, inputs): fp32 sums, one rounding after the bias.
pub fn project(w: []const u16, b: []const u16, x: []const f32, y: []f32) void {
    std.debug.assert(w.len == x.len * y.len and b.len == y.len);
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
        o.* = round(sum + widen(b[row]));
    }
}

/// The bf16 projections the modulation reads, each stored (outputs, inputs) with its bias.
pub const Weights = struct {
    time_1: []const u16,
    time_1_b: []const u16,
    time_2: []const u16,
    time_2_b: []const u16,
    blocks: []const u16,
    blocks_b: []const u16,
    head: []const u16,
    head_b: []const u16,
};

/// The shared vectors a pass at one timestep gives the kernels, each `hidden` bf16 values.
pub const Vector = enum(usize) { scale_a, shift_a, gate_a, scale_f, shift_f, gate_f, scale_out, shift_out, ones };
pub const vectors = @typeInfo(Vector).@"enum".field_names.len;

/// The blocks' and head's modulation at `timestep` (sigma x 1,000); scales as bf16 `1 + scale`, as torch makes them.
pub fn modulation(out: *[vectors][cfg.hidden]u16, gpa: std.mem.Allocator, w: Weights, timestep: f32) !void {
    const h = cfg.hidden;
    const emb = try gpa.alloc(f32, cfg.time_dim);
    defer gpa.free(emb);
    const mid = try gpa.alloc(f32, h);
    defer gpa.free(mid);
    const time = try gpa.alloc(f32, h);
    defer gpa.free(time);
    const table = try gpa.alloc(f32, 6 * h);
    defer gpa.free(table);
    const head = try gpa.alloc(f32, 2 * h);
    defer gpa.free(head);
    const half = cfg.time_dim / 2;
    for (0..half) |i| {
        const freq = std.math.pow(f64, 10000, -@as(f64, @floatFromInt(i)) / half);
        const phase = @as(f64, timestep) * freq;
        emb[i] = round(@floatCast(@cos(phase)));
        emb[half + i] = round(@floatCast(@sin(phase)));
    }
    project(w.time_1, w.time_1_b, emb, mid);
    for (mid) |*x| x.* = round(x.* / (1.0 + @exp(-x.*)));
    project(w.time_2, w.time_2_b, mid, time);
    project(w.blocks, w.blocks_b, time, table);
    project(w.head, w.head_b, time, head);
    // the blocks' table is shift, scale, gate for attention then the MLP; the head's is shift, scale
    const from = [_]struct { Vector, []const f32, bool }{
        .{ .shift_a, table[0..h], false },          .{ .scale_a, table[h .. 2 * h], true },
        .{ .gate_a, table[2 * h .. 3 * h], false }, .{ .shift_f, table[3 * h .. 4 * h], false },
        .{ .scale_f, table[4 * h .. 5 * h], true }, .{ .gate_f, table[5 * h .. 6 * h], false },
        .{ .shift_out, head[0..h], false },         .{ .scale_out, head[h .. 2 * h], true },
    };
    for (from) |f| for (&out[@backingInt(f[0])], f[1]) |*o, v| {
        o.* = bf16(if (f[2]) 1.0 + v else v);
    };
    @memset(&out[@backingInt(Vector.ones)], bf16(1.0));
}

test "bf16 rounds to nearest even and keeps NaN" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expect(std.math.isNan(widen(bf16(std.math.nan(f32)))));
}

test "rotary positions: the image centred at the prompt's length, then the prompt along the frame axis" {
    const a = std.testing.allocator;
    const text = 3;
    const rows = 4;
    const columns = 6;
    const n = (text + rows * columns) * 64;
    const cos = try a.alloc(f32, n);
    defer a.free(cos);
    const sin = try a.alloc(f32, n);
    defer a.free(sin);
    var bands: [cfg.bands]f32 = undefined;
    for (&bands, 0..) |*b, i| b.* = 1.0 / std.math.pow(f32, 10000, @as(f32, @floatFromInt(i)) / cfg.bands);
    rotary(cos, sin, &bands, text, rows, columns);
    // band 0 of each axis turns by its coordinate: axis 0 never, frame at 16, height at 32, width at 48
    try std.testing.expectEqual(@as(f32, 0), sin[0]);
    try std.testing.expectEqual(@as(f32, 1), cos[0]);
    try std.testing.expectEqual(round(@sin(@as(f32, 3))), sin[16]);
    try std.testing.expectEqual(round(@sin(@as(f32, -1.5))), sin[32]);
    try std.testing.expectEqual(round(@sin(@as(f32, -2.5))), sin[48]);
    const last_image = (rows * columns - 1) * 64;
    try std.testing.expectEqual(round(@sin(@as(f32, 1.5))), sin[last_image + 32]);
    try std.testing.expectEqual(round(@sin(@as(f32, 2.5))), sin[last_image + 48]);
    const second_text = (rows * columns + 1) * 64;
    try std.testing.expectEqual(round(@sin(@as(f32, 1))), sin[second_text + 16]);
    try std.testing.expectEqual(@as(f32, 0), sin[second_text + 32]);
}

test "project sums in fp32 and adds the bias before its one rounding" {
    const row = [_]u16{ bf16(1), bf16(2), bf16(3), bf16(4), bf16(5), bf16(6), bf16(7), bf16(8), bf16(9), bf16(0.5) };
    const w = row ++ row;
    const x = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const b = [_]u16{ bf16(0.25), bf16(0.0) };
    var y: [2]f32 = undefined;
    project(&w, &b, &x, &y);
    try std.testing.expectEqual(@as(f32, 45.75), y[0]);
    try std.testing.expectEqual(@as(f32, 45.5), y[1]);
    // 45.6 + 0.125 rounds to 45.75; rounding 45.6 first (45.5) would leave a tie at 45.625 that goes to 45.5
    const one = [_]u16{bf16(1)};
    const eighth = [_]u16{bf16(0.125)};
    var z: [1]f32 = undefined;
    project(&one, &eighth, &[_]f32{45.6}, &z);
    try std.testing.expectEqual(@as(f32, 45.75), z[0]);
}
