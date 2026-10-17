//! Vision launch formulas and an independent scalar M-RoPE reference for the device contract.
const std = @import("std");
fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}
fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
pub const Which = enum { mask, mrope };
pub const packed_width = 13952;
pub const output_width = 24 * 256 + 2 * 256 + 4 * 128;

pub fn symbol(w: Which) []const u8 {
    return if (w == .mask) "fz_mask_rows" else "fn_attn_prep_mrope";
}
pub fn source(w: Which) []const u8 {
    return if (w == .mask) @import("grammar.zig").source else @import("kernel_sources").flashnext_mrope;
}
pub fn spec(comptime C: type, w: Which) C.Spec {
    const D = struct {
        fn depth(_: C.Shape) usize {
            return 14;
        }
    };
    return switch (w) {
        .mask => .{ .name = "flashnext vision fz_mask_rows", .accum = .none, .depth = D.depth, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 1, .n = 33, .k = 1 }, .{ .rows = 16, .n = 248320, .k = 1 } }, .tiles = .{ .{}, .{ .size = 32 }, .{} }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "a mask copies the original bits or writes negative infinity; it has no sums" } },
        .mrope => .{ .name = "flashnext vision fn_attn_prep_mrope", .accum = .f32, .depth = D.depth, .ops = 16, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 1, .n = output_width, .k = 256 }, .{ .rows = 16, .n = output_width, .k = 256 }, .{ .rows = 128, .n = output_width, .k = 256 } }, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "normalization rounds to bf16 before rotary transcendental operations; reduced sums need device probes" } },
    };
}
pub fn axis(i: usize) usize {
    return if (i % 3 == 1 and i < 33) 1 else if (i % 3 == 2 and i < 30) 2 else 0;
}

pub const Accum = enum { f64, f32, f16, bf16 };
pub const Order = enum { sequential, pairwise, strided };
fn round(v: f64, acc: Accum) f64 {
    return switch (acc) {
        .f64 => v,
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => fromBf16(toBf16(@floatCast(v))),
    };
}
fn sum(terms: []const f64, acc: Accum, order: Order) f64 {
    var tmp: [256]f64 = undefined;
    if (order == .pairwise) {
        @memcpy(tmp[0..terms.len], terms);
        var n = terms.len;
        while (n > 1) {
            for (0..n / 2) |i| tmp[i] = round(tmp[i * 2] + tmp[i * 2 + 1], acc);
            n /= 2;
        }
        return tmp[0];
    }
    if (order == .strided) {
        @memset(tmp[0..32], 0);
        for (terms, 0..) |v, i| tmp[i % 32] = round(tmp[i % 32] + v, acc);
        return sum(tmp[0..32], acc, .pairwise);
    }
    var result: f64 = 0;
    for (terms) |v| result = round(result + v, acc);
    return result;
}

pub fn reference(p: []const u16, pos: [3]i32, weights: [3][]const f32, eps: f32, log2base: f32, out: []u16) void {
    referenceWith(p, pos, weights, eps, log2base, .f64, .sequential, out, null);
}

pub fn referenceWith(p: []const u16, pos: [3]i32, weights: [3][]const f32, eps: f32, log2base: f32, acc: Accum, order: Order, out: []u16, magnitude: ?[]f64) void {
    for (0..30) |head| {
        const kind: usize = if (head < 24) 0 else if (head < 26) 1 else 2;
        const width: usize = if (kind == 2) 128 else 256;
        const src: usize = if (kind == 0) head * 512 else if (kind == 1) 24 * 512 + (head - 24) * 256 else 24 * 512 + 4 * 256 + (head - 26) * 128;
        const dst: usize = if (kind == 0) head * 256 else if (kind == 1) 24 * 256 + (head - 24) * 256 else 26 * 256 + (head - 26) * 128;
        var terms: [256]f64 = undefined;
        for (p[src..][0..width], terms[0..width]) |h, *term| {
            const x: f64 = fromBf16(h);
            term.* = x * x;
        }
        const ss = sum(terms[0..width], acc, order);
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
        var normed: [256]f64 = undefined;
        for (0..width) |d| normed[d] = fromBf16(toBf16(@floatCast(fromBf16(p[src + d]) * inv * weights[kind][d])));
        for (0..width) |d| {
            var x = normed[d];
            var mag = @abs(x);
            if (d < 64) {
                const i = d % 32;
                const freq = @exp2(-(@as(f64, @floatFromInt(i)) / 32) * log2base);
                const angle = @as(f64, @floatFromInt(pos[axis(i)])) * freq;
                const co = @cos(angle);
                const si = @sin(angle);
                mag = if (d < 32) @abs(normed[d] * co) + @abs(normed[d + 32] * si) else @abs(normed[d - 32] * si) + @abs(normed[d] * co);
                x = if (d < 32) normed[d] * co - normed[d + 32] * si else normed[d - 32] * si + normed[d] * co;
            }
            out[dst + d] = toBf16(@floatCast(x));
            if (magnitude) |m| m[dst + d] = mag;
        }
    }
}

test "flashnext vision: M-RoPE scalar reference keeps text axes and distinguishes image axes" {
    const p: [packed_width]u16 = @splat(0x3f80);
    const w: [256]f32 = @splat(1);
    var text: [output_width]u16 = undefined;
    var image: [output_width]u16 = undefined;
    reference(&p, .{ 0, 0, 0 }, .{ &w, &w, w[0..128] }, 0, 0, &text);
    for (text) |h| try std.testing.expectEqual(@as(u16, 0x3f80), h);
    reference(&p, .{ 3, 5, 7 }, .{ &w, &w, w[0..128] }, 0, 0, &image);
    try std.testing.expectEqualSlices(u16, &.{ 0xbf91, 0x3f9f, 0x3dc6 }, image[0..3]);
    try std.testing.expectEqualSlices(u16, &.{ 0xbf59, 0xbf2d, 0x3fb5 }, image[32..35]);
    try std.testing.expectEqual(@as(usize, 0), axis(30));
    try std.testing.expectEqual(@as(usize, 1), axis(31));
    for (0..30) |head| {
        const start: usize = if (head < 26) head * 256 else 26 * 256 + (head - 26) * 128;
        try std.testing.expectEqualSlices(u16, image[0..64], image[start..][0..64]);
    }
}
