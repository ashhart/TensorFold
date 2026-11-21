//! bf16 helpers, and a 640 tile's resized tables in the release's own fp32 steps (measured bit for bit, black box).
const std = @import("std");

pub fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Keys' cubic kernel (a = -0.5) at x >= 0, in fp32 with fused steps, as the release's resize rounds it.
fn cubic(x: f32) f32 {
    const a: f32 = -0.5;
    if (x < 1) return @mulAdd(f32, @mulAdd(f32, a + 2, x, -(a + 3)) * x, x, 1);
    if (x < 2) return @mulAdd(f32, @mulAdd(f32, @mulAdd(f32, a, x, -(5 * a)), x, 8 * a), x, -(4 * a));
    return 0;
}

/// One output's taps: inputs `lo` .. `lo + w.len` with their weights.
const Taps = struct { lo: usize, w: []f32 };

/// Antialiased bicubic shrink taps, all in fp32: output i centred at (i + 0.5) scale, weights divided by their sum.
fn bicubicTaps(a: std.mem.Allocator, n_in: usize, n_out: usize) ![]Taps {
    std.debug.assert(n_out <= n_in);
    const scale = @as(f32, @floatFromInt(n_in)) / @as(f32, @floatFromInt(n_out));
    const support = 2 * scale;
    const inv = 1 / scale;
    const out = try a.alloc(Taps, n_out);
    for (out, 0..) |*t, i| {
        const centre = scale * (@as(f32, @floatFromInt(i)) + 0.5);
        const lo: usize = @intCast(@max(@as(i64, @intFromFloat(centre - support + 0.5)), 0));
        const hi: usize = @intCast(@min(@as(i64, @intFromFloat(centre + support + 0.5)), @as(i64, @intCast(n_in))));
        t.* = .{ .lo = lo, .w = try a.alloc(f32, hi - lo) };
        var total: f32 = 0;
        for (t.w, lo..) |*w, j| {
            w.* = cubic(@abs(@mulAdd(f32, @as(f32, @floatFromInt(j)) - centre, inv, 0.5 * inv)));
            total += w.*;
        }
        for (t.w) |*w| w.* /= total;
    }
    return out;
}

/// One output's fp32 sum over its taps, left to right, each tap after the first fused: `at(j)` reads input j.
fn tapSum(t: Taps, at: anytype) f32 {
    var s = at.get(t.lo) * t.w[0];
    for (t.w[1..], t.lo + 1..) |w, j| s = @mulAdd(f32, at.get(j), w, s);
    return s;
}

/// Rows [n_in^2, ch] bf16 shrunk to fp32 [n_out^2, ch] by the bicubic taps, along each row first, then down columns.
pub fn bicubicGrid32(gpa: std.mem.Allocator, src: []const u16, n_in: usize, n_out: usize, ch: usize, out: []f32) !void {
    std.debug.assert(src.len == n_in * n_in * ch and out.len == n_out * n_out * ch);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const taps = try bicubicTaps(a, n_in, n_out);
    const mid = try a.alloc(f32, n_in * n_out * ch);
    const Src = struct {
        rows: []const u16,
        step: usize,
        fn get(r: @This(), j: usize) f32 {
            return widen(r.rows[j * r.step]);
        }
    };
    const Mid = struct {
        rows: []const f32,
        step: usize,
        fn get(r: @This(), j: usize) f32 {
            return r.rows[j * r.step];
        }
    };
    for (0..n_in) |y| for (taps, 0..) |t, x| for (0..ch) |c| {
        mid[(y * n_out + x) * ch + c] = tapSum(t, Src{ .rows = src[y * n_in * ch + c ..], .step = ch });
    };
    for (taps, 0..) |t, y| for (0..n_out) |x| for (0..ch) |c| {
        out[(y * n_out + x) * ch + c] = tapSum(t, Mid{ .rows = mid[x * ch + c ..], .step = n_out * ch });
    };
}

/// bicubicGrid32, each value then rounded to bf16 (the tables the encoder adds).
pub fn bicubicGrid(gpa: std.mem.Allocator, src: []const u16, n_in: usize, n_out: usize, ch: usize, out: []u16) !void {
    const wide = try gpa.alloc(f32, out.len);
    defer gpa.free(wide);
    try bicubicGrid32(gpa, src, n_in, n_out, ch, wide);
    for (out, wide) |*o, v| o.* = bf16(v);
}

/// Rows [n_in, ch] bf16 resized linearly to fp32 [n_out, ch]: output i at max(fma(scale, i + 0.5, -0.5), 0), fused.
pub fn linearRows32(src: []const u16, n_in: usize, n_out: usize, ch: usize, out: []f32) void {
    std.debug.assert(src.len == n_in * ch and out.len == n_out * ch);
    const scale = @as(f32, @floatFromInt(n_in)) / @as(f32, @floatFromInt(n_out));
    for (0..n_out) |i| {
        const x = @max(@mulAdd(f32, scale, @as(f32, @floatFromInt(i)) + 0.5, -0.5), 0);
        const lo: usize = @intFromFloat(x);
        const hi = if (lo < n_in - 1) lo + 1 else lo;
        const f1 = x - @as(f32, @floatFromInt(lo));
        const f0 = 1 - f1;
        for (out[i * ch ..][0..ch], 0..) |*o, c| {
            o.* = @mulAdd(f32, widen(src[lo * ch + c]), f0, widen(src[hi * ch + c]) * f1);
        }
    }
}

/// linearRows32, each value then rounded to bf16.
pub fn linearRows(gpa: std.mem.Allocator, src: []const u16, n_in: usize, n_out: usize, ch: usize, out: []u16) !void {
    const wide = try gpa.alloc(f32, out.len);
    defer gpa.free(wide);
    linearRows32(src, n_in, n_out, ch, wide);
    for (out, wide) |*o, v| o.* = bf16(v);
}

/// The test inputs: bf16 bits of mixed signs and magnitudes (2^-7 to 2^8) from a fixed formula.
fn probe(i: usize) u16 {
    return @intCast((i * 7919) % 2 << 15 | (120 + (i * 31) % 16) << 7 | (i * 97) % 128);
}

test "uocr: the tile resizes give the release's fp32 bits (outputs of its resize on these inputs)" {
    const a = std.testing.allocator;
    // a 9 x 9 grid of 2 channels shrunk to 5 x 5; rounding the float64 result once to fp32 misses 47 of these 50
    const cubic_bits = [_]u32{
        0x42da1050, 0xc330c4d0, 0x42077b28, 0xc136b7e0, 0xbd89eb9f, 0x40c69083, 0xc0a210e3, 0xc1399d6a, 0x42723fcf, 0xc35a83b9,
        0x421b757f, 0xc11424d3, 0x3f69afdc, 0x40c5d996, 0xc0e78ef1, 0xc1970b63, 0x423f2169, 0xc32323c4, 0x42d444b4, 0xc2797333,
        0x3ed21167, 0x410cca44, 0xc0d807ef, 0xc1b4f941, 0x42366ec4, 0xc33b1445, 0x428fb517, 0xc292637b, 0x4094e1d6, 0x40615566,
        0xc0c52b32, 0xc1f6e533, 0x424cf01c, 0xc34c55bb, 0x42a7d07c, 0xc279521c, 0x40e8f5f0, 0x409e3179, 0xbf5466a4, 0x40a52bf4,
        0x425e016a, 0xc380e610, 0x42b32dff, 0xc29be992, 0x409ee987, 0x4098fa32, 0xbf727214, 0x40a0c01f, 0x3caa4021, 0xc291bca3,
    };
    var grid: [9 * 9 * 2]u16 = undefined;
    for (&grid, 0..) |*v, i| v.* = probe(i);
    var got: [5 * 5 * 2]f32 = undefined;
    try bicubicGrid32(a, &grid, 9, 5, 2, &got);
    for (cubic_bits, got) |want, v| try std.testing.expectEqual(want, @as(u32, @bitCast(v)));
    // 13 rows of 3 resized to 7; float64 rounded once misses 18 of these 21
    const linear_bits = [_]u32{
        0x3ffa4001, 0xbf4c6db8, 0x3eac524a, 0xc2eecbdd, 0x424b60eb, 0xc1a7ea8d, 0x424e724a, 0xc1b22493, 0x41182001, 0x3f3a0000,
        0xbe9b0000, 0x3e7c0000, 0xbc0edba0, 0xc3c0da8f, 0x43264869, 0xc1672484, 0x40b66dac, 0xc0d5248a, 0xbf33b6e4, 0x3e8b492d,
        0xbdea4932,
    };
    var line: [13 * 3]u16 = undefined;
    for (&line, 0..) |*v, i| v.* = probe(i + 1000);
    var rows: [7 * 3]f32 = undefined;
    linearRows32(&line, 13, 7, 3, &rows);
    for (linear_bits, rows) |want, v| try std.testing.expectEqual(want, @as(u32, @bitCast(v)));
    // the bf16 entry points round those same values
    var half: [5 * 5 * 2]u16 = undefined;
    try bicubicGrid(a, &grid, 9, 5, 2, &half);
    for (half, got) |h, v| try std.testing.expectEqual(bf16(v), h);
}
