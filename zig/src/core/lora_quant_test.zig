//! The MLX quantizer and its comparison helpers on synthetic tensors: no model, no GPU.
const std = @import("std");
const host4 = @import("affine4_host.zig");
const lq = @import("lora_quant.zig");

const gpa = std.testing.allocator;

fn alloc(n: usize, group: usize) !lq.Quant {
    return .{ .words = try gpa.alloc(u32, n / 8), .scales = try gpa.alloc(u16, n / group), .biases = try gpa.alloc(u16, n / group) };
}

fn free(q: lq.Quant) void {
    gpa.free(q.words);
    gpa.free(q.scales);
    gpa.free(q.biases);
}

fn same(a: lq.Quant, b: lq.Quant) bool {
    return std.mem.eql(u32, a.words, b.words) and std.mem.eql(u16, a.scales, b.scales) and std.mem.eql(u16, a.biases, b.biases);
}

/// A bf16 dense weight (as the public checkpoints ship) of pseudo-random values, as f32.
fn dense(n: usize, k: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const w = try gpa.alloc(f32, n * k);
    for (w) |*v| v.* = host4.f32of(host4.bf16of((prng.random().float(f32) - 0.5) * 0.2));
    return w;
}

fn codes(q: lq.Quant, i: usize) u32 {
    return (q.words[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
}

test "the quantizer's hand-checked groups: edge at the maximum, at the minimum, and a tie rounded away from zero" {
    var v: [32]f32 = @splat(0);
    const q = try alloc(32, 32);
    defer free(q);
    // 0..15 twice over: the larger-magnitude edge is the maximum, so the scale is -1 and the bias 15
    for (v[0..16], 0..) |*x, i| x.* = @floatFromInt(i);
    for (v[16..], 0..) |*x, i| x.* = @floatFromInt(i);
    try std.testing.expect(try lq.quantize(32, &v, q) == null);
    try std.testing.expectEqual(host4.bf16of(-1), q.scales[0]);
    try std.testing.expectEqual(host4.bf16of(15), q.biases[0]);
    for (0..32) |i| try std.testing.expectEqual(@as(u32, 15 - @as(u32, @intFromFloat(v[i]))), codes(q, i));
    // -15..0: the minimum is the larger edge, scale +1, bias -15, codes count up
    for (v[0..16], 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) - 15;
    for (v[16..], 0..) |*x, i| x.* = @as(f32, @floatFromInt(i)) - 15;
    try std.testing.expect(try lq.quantize(32, &v, q) == null);
    try std.testing.expectEqual(host4.bf16of(1), q.scales[0]);
    try std.testing.expectEqual(host4.bf16of(-15), q.biases[0]);
    for (0..32) |i| try std.testing.expectEqual(@as(u32, @intCast(i % 16)), codes(q, i));
    // 2.5 sits halfway between two codes of the 0..15 group (scale -1, bias 15): 12.5 rounds away to 13, not to even 12
    @memset(&v, 0);
    v[0] = 15;
    v[1] = 2.5;
    try std.testing.expect(try lq.quantize(32, &v, q) == null);
    try std.testing.expectEqual(@as(u32, 13), codes(q, 1));
    // a flat group is its own bias, exactly, and every code is zero
    @memset(&v, 0.5);
    try std.testing.expect(try lq.quantize(32, &v, q) == null);
    for (0..32) |i| try std.testing.expectEqual(@as(u32, 0), codes(q, i));
    try std.testing.expectEqual(host4.bf16of(0.5), q.biases[0]);
    // a group of zeros keeps zero as an exact grid point
    @memset(&v, 0);
    try std.testing.expect(try lq.quantize(32, &v, q) == null);
    try std.testing.expectEqual(@as(u16, 0), q.biases[0]);
}

test "agreement counts what differs: one scale, one code" {
    const w = try dense(2, 64, 4);
    defer gpa.free(w);
    const q = try alloc(w.len, 64);
    defer free(q);
    try std.testing.expect(try lq.quantize(64, w, q) == null);
    const other = try alloc(w.len, 64);
    defer free(other);
    @memcpy(other.words, q.words);
    @memcpy(other.scales, q.scales);
    @memcpy(other.biases, q.biases);
    other.scales[0] ^= 1;
    other.words[8] ^= 1;
    const ag = lq.agreement(64, lq.viewOf(q), lq.viewOf(other));
    try std.testing.expectEqual(@as(f64, 0.5), ag.scales);
    try std.testing.expectEqual(@as(f64, 1), ag.biases);
    try std.testing.expectEqual(@as(f64, 0), ag.groups);
    try std.testing.expect(ag.codes < 1 and !ag.identical());
}

test "adapter factors read as f32 from f32, bf16 and f16 bytes" {
    var out: [2]f32 = undefined;
    try lq.toF32(.bf16, &.{ 0x80, 0x3f, 0x00, 0xc0 }, &out); // 1.0, -2.0
    try std.testing.expectEqualSlices(f32, &.{ 1, -2 }, &out);
    try lq.toF32(.f16, &.{ 0x00, 0x3c, 0x00, 0xc0 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1, -2 }, &out);
    try lq.toF32(.f32, &.{ 0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1, -2 }, &out);
    try std.testing.expectError(error.UnsupportedDType, lq.toF32(.u8, &.{ 0, 0 }, &out));
}

test "the quantizer refuses a non-finite value, a range that overflows and a scale that collapses, naming the group" {
    var v: [128]f32 = @splat(0.25);
    const q = try alloc(128, 64);
    defer free(q);
    try std.testing.expect(try lq.quantize(64, &v, q) == null);
    v[70] = std.math.nan(f32);
    try std.testing.expectEqual(@as(?usize, 1), try lq.quantize(64, &v, q));
    v[70] = std.math.inf(f32);
    try std.testing.expectEqual(@as(?usize, 1), try lq.quantize(64, &v, q));
    v[70] = 0.25;
    v[3] = -3e38; // hi - lo overflows
    v[4] = 3e38;
    try std.testing.expectEqual(@as(?usize, 0), try lq.quantize(64, &v, q));
    @memset(&v, 1e38); // edge / q0 overflows: a zero scale
    try std.testing.expectEqual(@as(?usize, 0), try lq.quantize(64, &v, q));
}

test "a scale or bias that rounds to a bf16 infinity is refused, naming the group" {
    var v: [128]f32 = @splat(3.3e38);
    @memset(v[0..64], 0.25);
    v[64] = 3.4e38; // finite in f32, above the largest finite bf16: the stored bias would be infinite
    const q = try alloc(128, 64);
    defer free(q);
    try std.testing.expectEqual(@as(?usize, 1), try lq.quantize(64, &v, q));
    v[64] = host4.f32of(0x7f7f); // the largest finite bf16 is fine
    try std.testing.expect(std.math.isFinite(host4.f32of(0x7f7f)));
    try std.testing.expectEqual(@as(?usize, null), try lq.quantize(64, &v, q));
}

test "a group the engine does not serve is refused before anything is written, a table sized for it never overrun" {
    // 256 is the case that wrote count / 128 scales into a table of count / 256
    for ([_]usize{ 0, 16, 48, 96, 256 }) |group| {
        const n = 512;
        const w = try dense(1, n, 7);
        defer gpa.free(w);
        const q = try lq.allocQuant(gpa, 1, n, if (group == 0) 128 else group);
        defer lq.freeQuant(gpa, q);
        @memset(q.words, 0xA5A5A5A5);
        @memset(q.scales, 0x1234);
        @memset(q.biases, 0x5678);
        try std.testing.expectError(error.BadGroup, lq.quantize(group, w, q));
        for (q.words) |x| try std.testing.expectEqual(@as(u32, 0xA5A5A5A5), x);
        for (q.scales) |x| try std.testing.expectEqual(@as(u16, 0x1234), x);
        for (q.biases) |x| try std.testing.expectEqual(@as(u16, 0x5678), x);
    }
    for ([_]usize{ 32, 64, 128 }) |group| {
        const w = try dense(1, 512, 9);
        defer gpa.free(w);
        const q = try lq.allocQuant(gpa, 1, 512, group);
        defer lq.freeQuant(gpa, q);
        try std.testing.expectEqual(@as(?usize, null), try lq.quantize(group, w, q));
    }
}
