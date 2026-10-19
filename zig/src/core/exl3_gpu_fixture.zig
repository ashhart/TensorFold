//! Explicit GPU qualification tests; run through the native test module on a Metal SIMDgroup-matrix device.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("exl3_projection.zig");
const decoder = @import("exl3_projection_decode.zig");
pub const k = 256;
pub const n = 128;
pub const rows = 9;

pub const Owned = struct {
    matrix: ex.Matrix,
    buffers: [4]mtl.Buffer,

    pub fn init(device: mtl.Device, halves: u16, book: ex.format.Codebook, seed: u64) !Owned {
        const words = @as(usize, halves) * 8 * (k / 16) * (n / 16);
        const t = try device.buffer(words * 2 + 16, mtl.ResourceOptions.shared);
        errdefer t.deinit();
        const su = try device.buffer(k * 2 + 16, mtl.ResourceOptions.shared);
        errdefer su.deinit();
        const sv = try device.buffer(n * 2 + 16, mtl.ResourceOptions.shared);
        errdefer sv.deinit();
        const bias = try device.buffer(n * 2 + 16, mtl.ResourceOptions.shared);
        errdefer bias.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        for (t.slice(i16, words + 8)[8..]) |*x| x.* = prng.random().int(i16);
        for (su.slice(f16, k + 8)[8..], 0..) |*x, i| x.* = if (i % 3 == 0) -0.75 else 1.125;
        for (sv.slice(f16, n + 8)[8..], 0..) |*x, i| x.* = if (i % 5 == 0) -0.25 else 0.875;
        for (bias.slice(f16, n + 8)[8..], 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(i % 7)) / 16);
        return .{ .matrix = .{ .trellis = .{ .buffer = t, .offset = 16 }, .suh = .{ .buffer = su, .offset = 16 }, .svh = .{ .buffer = sv, .offset = 16 }, .bias = .{ .buffer = bias, .offset = 16 }, .k = k, .n = n, .width = .{ .halves = halves }, .codebook = book }, .buffers = .{ t, su, sv, bias } };
    }
    pub fn deinit(w: Owned) void {
        for (w.buffers) |b| b.deinit();
    }
};

pub fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}
pub fn input(device: mtl.Device, count: usize) !mtl.Buffer {
    const x = try device.buffer(count * (k + 16) * 2 + 32, mtl.ResourceOptions.shared);
    for (x.slice(u16, count * (k + 16) + 16), 0..) |*v, i|
        v.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 32);
    return x;
}
pub fn complete(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Exl3GpuFailure;
}

pub fn reference(w: Owned, x: []const u16, out: *[n]f32) void {
    var rotated: [k]f16 = undefined;
    const su = w.buffers[1].slice(f16, k + 8)[8..];
    for (0..k / 128) |b| {
        var block: [128]f32 = undefined;
        for (&block, 0..) |*v, i| v.* = @as(f32, @bitCast(@as(u32, x[b * 128 + i]) << 16)) * @as(f32, su[b * 128 + i]);
        decoder.hadamard(&block);
        for (block, 0..) |v, i| rotated[b * 128 + i] = @floatCast(v);
    }
    const word_count = w.matrix.width.tileWords() * (k / 16) * (n / 16);
    const words = w.buffers[0].slice(i16, word_count + 8)[8..];
    var unpacked: [k * n]f16 = undefined;
    ex.format.unpack(w.matrix.width, w.matrix.codebook, words, k / 16, n / 16, &unpacked);
    for (out, 0..) |*v, col| {
        var sum: f64 = 0;
        for (0..k) |j| sum += @as(f64, rotated[j]) * @as(f64, unpacked[j * n + col]);
        v.* = @floatCast(sum);
    }
    decoder.hadamard(out);
    const sv = w.buffers[2].slice(f16, n + 8)[8..];
    const bias = w.buffers[3].slice(f16, n + 8)[8..];
    for (out, 0..) |*v, i| v.* = v.* * @as(f32, sv[i]) + @as(f32, bias[i]);
}
