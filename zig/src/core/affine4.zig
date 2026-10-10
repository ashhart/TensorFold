//! CPU affine q4/group64 preparation preserves BF16 metadata and low-bit-first MLX words.
const std = @import("std");
pub const Prepared = struct {
    allocator: std.mem.Allocator,
    n: usize,
    k: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,
    pub fn deinit(p: *Prepared) void {
        p.allocator.free(p.words);
        p.allocator.free(p.scales);
        p.allocator.free(p.biases);
        p.* = undefined;
    }
};
pub fn asFloat(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
pub fn asBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
pub fn prepare(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize) !Prepared {
    if (n == 0 or k == 0 or k % 64 != 0 or weights.len != try std.math.mul(usize, n, k)) return error.BadAffineShape;
    const words = try a.alloc(u32, weights.len / 8);
    errdefer a.free(words);
    const scales = try a.alloc(u16, weights.len / 64);
    errdefer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    errdefer a.free(biases);
    // groups are independent: workers take contiguous ranges, so every byte matches one serial pass
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, scales.len / 4096));
    const per = (scales.len + workers - 1) / workers;
    var results: [max_workers]Error!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |w| threads[w] = std.Thread.spawn(.{}, run, .{ &results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per) }) catch null;
    run(&results[0], weights, words, scales, biases, 0, @min(scales.len, per));
    for (1..workers) |w| if (threads[w]) |t| t.join() else run(&results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per));
    for (results[0..workers]) |r| try r;
    return .{ .allocator = a, .n = n, .k = k, .words = words, .scales = scales, .biases = biases };
}
const max_workers = 16;
const Error = error{ NonFiniteAffineInput, NonFiniteAffineRange };
fn run(out: *Error!void, weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize) void {
    out.* = groups(weights, words, scales, biases, first, last);
}
/// Groups [first, last): each group's 64 inputs to one scale, one bias and eight low-bit-first words.
fn groups(weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize) Error!void {
    for (first..last) |group| {
        const values = weights[group * 64 ..][0..64];
        for (values) |bits| if (!std.math.isFinite(asFloat(bits))) return error.NonFiniteAffineInput;
        var lower: f32 = std.math.inf(f32);
        var upper: f32 = 0;
        for (values) |bits| {
            const v = asFloat(bits);
            lower = @min(lower, v);
            upper = @max(upper, v);
        }
        var bias: f32 = if (@abs(lower) > @abs(upper)) lower else upper;
        var scale: f32 = @max((upper - lower) / @as(f32, 15), @as(f32, 1e-7));
        if (bias >= 0) scale = -scale;
        const zero = @round(bias / scale);
        if (zero != 0) scale = bias / zero else bias = 0;
        if (!std.math.isFinite(scale) or scale == 0) return error.NonFiniteAffineRange;
        scales[group] = asBf16(scale);
        biases[group] = asBf16(bias);
        @memset(words[group * 8 ..][0..8], 0);
        for (values, 0..) |bits, index| {
            const code: u32 = @intFromFloat(@min(@as(f32, 15), @max(@as(f32, 0), @round((asFloat(bits) - bias) / scale))));
            words[group * 8 + index / 8] |= code << @as(u5, @intCast(4 * (index % 8)));
        }
    }
}
test "q4 shape and finite input checks occur before preparation" {
    var w: [64]u16 = @splat(0);
    try std.testing.expectError(error.BadAffineShape, prepare(std.testing.allocator, &w, 1, 63));
    w[7] = 0x7fc0;
    try std.testing.expectError(error.NonFiniteAffineInput, prepare(std.testing.allocator, &w, 1, 64));
}
test "a threaded preparation gives the serial pass's bytes" {
    const a = std.testing.allocator;
    const n = 64;
    const k = 4096;
    const w = try a.alloc(u16, n * k);
    defer a.free(w);
    var prng = std.Random.DefaultPrng.init(7);
    for (w) |*x| x.* = asBf16(prng.random().floatNorm(f32));
    var p = try prepare(a, w, n, k);
    defer p.deinit();
    const words = try a.alloc(u32, w.len / 8);
    defer a.free(words);
    const scales = try a.alloc(u16, w.len / 64);
    defer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    defer a.free(biases);
    try groups(w, words, scales, biases, 0, scales.len);
    try std.testing.expectEqualSlices(u32, words, p.words);
    try std.testing.expectEqualSlices(u16, scales, p.scales);
    try std.testing.expectEqualSlices(u16, biases, p.biases);
}
test "constant positive, negative and tiny groups retain signed scales and exact zero codes" {
    const a = std.testing.allocator;
    var w: [3 * 64]u16 = undefined;
    @memset(w[0..64], 0x3f80);
    @memset(w[64..128], 0xbf80);
    @memset(w[128..], asBf16(1e-20));
    var p = try prepare(a, &w, 3, 64);
    defer p.deinit();
    try std.testing.expectEqual(@as(u16, 0xb3d7), p.scales[0]);
    try std.testing.expectEqual(@as(u16, 0x3d89), p.scales[1]);
    try std.testing.expectEqual(@as(u16, 0), p.biases[2]);
    for (p.words) |word| try std.testing.expectEqual(@as(u32, 0), word);
}

pub fn signature() [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("affine4.zig"), &result, .{});
    return result;
}
