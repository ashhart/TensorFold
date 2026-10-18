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
pub const Which = enum { mask, mrope, target, mtp, pooled, absolute };
pub const table_words = @import("rotary_config.zig").words;
pub const packed_width = 13952;
pub const output_width = 24 * 256 + 2 * 256 + 4 * 128;

const prep_symbol = "custom_kernel_q4_attn_prep_63a42a6cd8e0434a_bfloat16_t_int32_t_float_float_float_floatc_floatc_bfloat16_t_bfloat16_t_bfloat16_t";
const pool_symbol = "custom_kernel_q4_idx_pool_6d826f5ebecec197_bfloat16_t_int32_tc_float_floatc_floatc_bfloat16_t";
pub fn isPool(w: Which) bool {
    return w == .pooled or w == .absolute;
}
pub fn symbol(w: Which) []const u8 {
    return switch (w) {
        .mask => "fz_mask_rows",
        .mrope => "fn_attn_prep_mrope",
        .target, .mtp => prep_symbol,
        .pooled => pool_symbol,
        .absolute => "fz_idx_pool_abs",
    };
}
fn recorded(file: []const u8) []const u8 {
    for (@import("kernel_sources").flashnext_gen.sources) |entry| if (std.mem.eql(u8, entry.name, file)) return entry.text;
    unreachable;
}
pub fn source(w: Which) []const u8 {
    return switch (w) {
        .mask => @import("grammar.zig").source,
        .mrope => @import("kernel_sources").flashnext_mrope,
        .target, .mtp => recorded("q4_attn_prep_63a42a6cd8e0434a.metal"),
        .pooled => recorded("q4_idx_pool_6d826f5ebecec197.metal"),
        .absolute => @import("kernel_sources").flashnext_select,
    };
}
pub fn name(w: Which, scaled: bool) []const u8 {
    return switch (w) {
        .mask => "flashnext vision fz_mask_rows",
        .mrope => if (scaled) "flashnext vision fn_attn_prep_mrope YaRN x4" else "flashnext vision fn_attn_prep_mrope",
        .target => if (scaled) "flashnext rotary target YaRN x4" else "flashnext rotary target",
        .mtp => if (scaled) "flashnext rotary mtp YaRN x4" else "flashnext rotary mtp",
        .pooled => if (scaled) "flashnext rotary pooled YaRN x4" else "flashnext rotary pooled",
        .absolute => if (scaled) "flashnext rotary absolute YaRN x4" else "flashnext rotary absolute",
    };
}
pub fn table(log2base: f32, factor: f64) [table_words]f32 {
    return @import("rotary_config.zig").table(log2base, factor, @import("window_config.zig").trained);
}
pub fn spec(comptime C: type, w: Which) C.Spec {
    return specScaled(C, w, false);
}
pub fn specScaled(comptime C: type, w: Which, scaled: bool) C.Spec {
    const D = struct {
        fn depth(_: C.Shape) usize {
            return 14;
        }
    };
    if (w == .mask) return .{ .name = name(w, false), .accum = .none, .depth = D.depth, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 1, .n = 33, .k = 1 }, .{ .rows = 16, .n = 248320, .k = 1 } }, .tiles = .{ .{}, .{ .size = 32 }, .{} }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "a mask copies the original bits or writes negative infinity; it has no sums" } };
    return .{ .name = name(w, scaled), .accum = .f32, .depth = D.depth, .ops = if (isPool(w)) 22 else 16, .mids = if (isPool(w)) 2 else 1, .out = .bf16, .served = if (isPool(w)) &.{ .{ .rows = 1, .n = 128, .k = 128 }, .{ .rows = 128, .n = 128, .k = 128 } } else if (w == .mtp) &.{ .{ .rows = 1, .n = output_width, .k = 256 }, .{ .rows = 16, .n = output_width, .k = 256 } } else &.{ .{ .rows = 1, .n = output_width, .k = 256 }, .{ .rows = 16, .n = output_width, .k = 256 }, .{ .rows = 128, .n = output_width, .k = 256 } }, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "pooling or normalization rounds to bf16 before rotary phases; the probe reports reduced-sum margins" } };
}
pub fn axis(i: usize) usize {
    return if (i % 3 == 1 and i < 33) 1 else if (i % 3 == 2 and i < 30) 2 else 0;
}

pub const Accum = enum { f64, f32, f16, bf16 };
pub const Order = enum { sequential, pairwise, strided };
pub fn round(v: f64, acc: Accum) f64 {
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
    const rotary = table(log2base, 0);
    referenceTableWith(p, pos, weights, eps, &rotary, acc, order, out, magnitude);
}

pub fn normalizedRotation(input: []const u16, pos: [3]i32, weights: []const f32, eps: f32, rotary: []const f32, acc: Accum, order: Order, out: []u16, magnitude: ?[]f64) void {
    const width = input.len;
    std.debug.assert((width == 128 or width == 256) and rotary.len == table_words);
    var terms: [256]f64 = undefined;
    for (input, terms[0..width]) |h, *term| {
        const x: f64 = fromBf16(h);
        term.* = x * x;
    }
    const inv = 1 / @sqrt(sum(terms[0..width], acc, order) / @as(f64, @floatFromInt(width)) + eps);
    var normed: [256]f64 = undefined;
    for (0..width) |d| normed[d] = fromBf16(toBf16(@floatCast(fromBf16(input[d]) * inv * weights[d])));
    for (0..width) |d| {
        var x = normed[d];
        var mag = @abs(x);
        if (d < 64) {
            const i = d % 32;
            const freq: f64 = if (rotary[1] != 0) rotary[3 + i] else @exp2(-(@as(f64, @floatFromInt(i)) / 32) * rotary[0]);
            const angle = @as(f64, @floatFromInt(pos[axis(i)])) * freq;
            const scale: f64 = if (rotary[1] != 0) rotary[2] else 1;
            const co = @cos(angle) * scale;
            const si = @sin(angle) * scale;
            mag = (if (d < 32) @abs(normed[d] * co) + @abs(normed[d + 32] * si) else @abs(normed[d - 32] * si) + @abs(normed[d] * co)) * (1 + @abs(angle));
            x = if (d < 32) normed[d] * co - normed[d + 32] * si else normed[d - 32] * si + normed[d] * co;
        }
        out[d] = toBf16(@floatCast(x));
        if (magnitude) |m| m[d] = mag;
    }
}

pub fn referenceTableWith(p: []const u16, pos: [3]i32, weights: [3][]const f32, eps: f32, rotary: []const f32, acc: Accum, order: Order, out: []u16, magnitude: ?[]f64) void {
    for (0..30) |head| {
        const kind: usize = if (head < 24) 0 else if (head < 26) 1 else 2;
        const width: usize = if (kind == 2) 128 else 256;
        const src: usize = if (kind == 0) head * 512 else if (kind == 1) 24 * 512 + (head - 24) * 256 else 24 * 512 + 4 * 256 + (head - 26) * 128;
        const dst: usize = if (kind == 0) head * 256 else if (kind == 1) 24 * 256 + (head - 24) * 256 else 26 * 256 + (head - 26) * 128;
        normalizedRotation(p[src..][0..width], pos, weights[kind], eps, rotary, acc, order, out[dst..][0..width], if (magnitude) |m| m[dst..][0..width] else null);
    }
}

test "Flash Next contracts: M-RoPE scalar reference keeps text axes and distinguishes image axes" {
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
