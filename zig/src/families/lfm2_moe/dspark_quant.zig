//! BF16 drafter rows quantized at load to MLX's affine codes in groups of 64, the layout core/tile_qmm reads.
const std = @import("std");

/// MLX's affine codes of 64 bf16 values at `bits`: bias the larger edge, scale |edge| / round(|edge| / step).
pub fn group(x: []const u16, bits: u8, codes: []u8, scale: *u16, bias: *u16) void {
    var lo: f32 = std.math.inf(f32);
    var hi: f32 = -std.math.inf(f32);
    for (x) |v| {
        lo = @min(lo, toF32(v));
        hi = @max(hi, toF32(v));
    }
    const bins: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    const step = @max(hi - lo, 1e-7) / bins;
    const from_min = @abs(lo) > @abs(hi);
    const edge = if (from_min) lo else hi;
    const q0 = @round(@abs(edge) / step);
    const s_f: f32 = if (q0 == 0) step else (if (from_min) @abs(edge) else -@abs(edge)) / q0;
    scale.* = toBf16(s_f);
    bias.* = toBf16(if (q0 == 0) lo else edge);
    const s = toF32(scale.*);
    const b = toF32(bias.*);
    @memset(codes, 0);
    for (x, 0..) |v, i| {
        const q: u8 = @intFromFloat(std.math.clamp(@round((toF32(v) - b) / s), 0, bins));
        if (bits == 8) codes[i] = q else codes[i / 2] |= q << @intCast((i % 2) * 4);
    }
}

pub fn toF32(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Round to nearest even, as MLX casts.
pub fn toBf16(f: f32) u16 {
    const b: u32 = @bitCast(f);
    if ((b & 0x7f80_0000) == 0x7f80_0000) return @truncate(b >> 16);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// A matrix's codes [n][k * bits / 8], scales and biases [n][k / 64], each written by one thread's rows.
pub const Job = struct {
    src: []const u16,
    k: usize,
    bits: u8,
    codes: []u8,
    scales: []u16,
    biases: []u16,
    fn run(j: Job, from: usize, to: usize) void {
        const groups = j.k / 64;
        const per = 64 * @as(usize, j.bits) / 8;
        for (from..to) |r| for (0..groups) |g| {
            const at = r * groups + g;
            group(j.src[r * j.k + g * 64 ..][0..64], j.bits, j.codes[at * per ..][0..per], &j.scales[at], &j.biases[at]);
        };
    }

    /// The job over its rows on every CPU (at most 16 threads).
    pub fn all(j: Job, gpa: std.mem.Allocator, rows: usize) !void {
        const threads = @min(std.Thread.getCpuCount() catch 1, 16);
        const per = (rows + threads - 1) / threads;
        const handles = try gpa.alloc(std.Thread, threads);
        defer gpa.free(handles);
        var spawned: usize = 0;
        defer for (handles[0..spawned]) |h| h.join();
        var at: usize = 0;
        while (at < rows) : (at += per) {
            handles[spawned] = try std.Thread.spawn(.{}, run, .{ j, at, @min(rows, at + per) });
            spawned += 1;
        }
    }
};

/// Head `h`'s interleaved RoPE pairs (2i, 2i + 1) as half-split pairs (i, i + dim / 2): row h * dim + j's source.
pub fn halfSplit(row: usize, dim: usize) usize {
    const h = row / dim;
    const j = row % dim;
    const half = dim / 2;
    return h * dim + if (j < half) 2 * j else 2 * (j - half) + 1;
}

test "lfm2 dspark: 4- and 8-bit codes keep the larger edge exact, as MLX quantizes" {
    var x: [64]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = toBf16(@as(f32, @floatFromInt(i)) * 0.125 - 2.0);
    for ([_]u8{ 4, 8 }) |bits| {
        var codes: [64]u8 = undefined;
        var s: u16 = undefined;
        var b: u16 = undefined;
        group(&x, bits, codes[0 .. 64 * @as(usize, bits) / 8], &s, &b);
        try std.testing.expect(toF32(s) < 0 and toF32(b) == 5.875); // the max is the edge: codes descend from it
        const last: u8 = if (bits == 8) codes[63] else codes[31] >> 4;
        try std.testing.expectEqual(@as(u8, 0), last);
        for (x, 0..) |v, i| {
            const q: u8 = if (bits == 8) codes[i] else (codes[i / 2] >> @intCast((i % 2) * 4)) & 0xf;
            const back = toF32(s) * @as(f32, @floatFromInt(q)) + toF32(b);
            try std.testing.expect(@abs(back - toF32(v)) <= @abs(toF32(s)) * 0.51);
        }
    }
}

test "lfm2 dspark: half-split rows rotated by the half-split rule equal interleaved pairs rotated, row for row" {
    const dim = 64;
    var x: [2 * dim]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 3;
    var moved: [2 * dim]f32 = undefined;
    for (&moved, 0..) |*v, r| v.* = x[halfSplit(r, dim)];
    const pos: f32 = 1234;
    for (0..2) |h| for (0..dim / 2) |i| {
        const angle = pos * std.math.pow(f32, 1e7, -@as(f32, @floatFromInt(i)) / (dim / 2));
        const cs = @cos(angle);
        const sn = @sin(angle);
        const a = x[h * dim + 2 * i];
        const b = x[h * dim + 2 * i + 1];
        const l = moved[h * dim + i];
        const r = moved[h * dim + i + dim / 2];
        try std.testing.expectEqual(a * cs - b * sn, l * cs - r * sn);
        try std.testing.expectEqual(a * sn + b * cs, l * sn + r * cs);
    };
}
