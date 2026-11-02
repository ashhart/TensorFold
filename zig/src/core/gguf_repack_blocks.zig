//! Serialized field extraction only. All input/output sizes and finite half fields are checked by the public API.
const std = @import("std");
const Kind = @import("gguf_tensor.zig").Kind;
fn half(b: []const u8, at: usize) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[at..][0..2], .little))));
}
pub fn validate(kind: Kind, src: []const u8) !void {
    const size = kind.block().bytes;
    var at: usize = 0;
    while (at < src.len) : (at += size) {
        const b = src[at..][0..size];
        const d_at: usize = switch (kind) {
            .q2_k => 80,
            .q3_k => 108,
            .q6_k => 208,
            else => 0,
        };
        if (!std.math.isFinite(half(b, d_at))) return error.NonFiniteQuantScale;
        switch (kind) {
            .q4_1, .q5_1, .q4_k, .q5_k => if (!std.math.isFinite(half(b, 2))) return error.NonFiniteQuantScale,
            .q2_k => if (!std.math.isFinite(half(b, 82))) return error.NonFiniteQuantScale,
            else => {},
        }
    }
}
const Output = struct {
    bits: u8,
    group: usize,
    base: usize = 0,
    words: []u32,
    scales: []f32,
    biases: []f32,
    fn coefficient(o: Output, g: usize, s: f32, b: f32) void {
        const at = o.base / o.group + g;
        o.scales[at] = s;
        o.biases[at] = b;
    }
    fn code(o: Output, i: usize, v: u8) void {
        const element = o.base + i;
        const bit = (element % 32) * o.bits;
        const at = element / 32 * o.bits + bit / 32;
        const shift: u5 = @intCast(bit % 32);
        o.words[at] |= @as(u32, v) << shift;
        if (@as(usize, shift) + o.bits > 32) o.words[at + 1] |= @as(u32, v) >> @intCast(32 - @as(u6, shift));
    }
};
fn small(kind: Kind, b: []const u8, out: Output) void {
    const d = half(b, 0);
    const center: f32 = switch (kind) {
        .q4_0 => 8,
        .q5_0 => 16,
        .q8_0 => 128,
        else => 0,
    };
    const bias = if (kind == .q4_1 or kind == .q5_1) half(b, 2) else -center * d;
    out.coefficient(0, d, bias);
    if (kind == .q8_0) {
        for (b[2..], 0..) |v, i| out.code(i, v ^ 0x80);
        return;
    }
    const q_at: usize = switch (kind) {
        .q4_0 => 2,
        .q4_1 => 4,
        .q5_0 => 6,
        .q5_1 => 8,
        else => unreachable,
    };
    const h_at: usize = if (kind == .q5_1) 4 else 2;
    const h: u32 = if (kind == .q5_0 or kind == .q5_1) std.mem.readInt(u32, b[h_at..][0..4], .little) else 0;
    for (0..16) |i| {
        const lo: u8 = @intCast(((h >> @intCast(i)) & 1) * 16);
        const hi: u8 = @intCast(((h >> @intCast(i + 16)) & 1) * 16);
        out.code(i, (b[q_at + i] & 15) | lo);
        out.code(i + 16, (b[q_at + i] >> 4) | hi);
    }
}
fn sixBit(q: []const u8, g: usize) struct { s: u8, m: u8 } {
    if (g < 4) return .{ .s = q[g] & 63, .m = q[g + 4] & 63 };
    return .{ .s = (q[g + 4] & 15) | ((q[g - 4] >> 6) << 4), .m = (q[g + 4] >> 4) | ((q[g] >> 6) << 4) };
}
fn fourFive(kind: Kind, b: []const u8, out: Output) void {
    const d = half(b, 0);
    const m = half(b, 2);
    const q_at: usize = if (kind == .q5_k) 48 else 16;
    for (0..8) |g| {
        const sm = sixBit(b[4..16], g);
        out.coefficient(g, d * @as(f32, @floatFromInt(sm.s)), -m * @as(f32, @floatFromInt(sm.m)));
        for (0..32) |j| {
            const shift: u3 = @intCast(4 * (g % 2));
            var q = (b[q_at + 32 * (g / 2) + j] >> shift) & 15;
            if (kind == .q5_k) q |= ((b[16 + j] >> @intCast(g)) & 1) << 4;
            out.code(g * 32 + j, q);
        }
    }
}
fn six(b: []const u8, out: Output) void {
    const d = half(b, 208);
    for (0..16) |g| {
        const signed: i8 = @bitCast(b[192 + g]);
        const scale = d * @as(f32, @floatFromInt(signed));
        out.coefficient(g, scale, -32 * scale);
    }
    for (0..2) |n| for (0..32) |j| for (0..4) |quarter| {
        const low = b[n * 64 + j + (quarter % 2) * 32];
        const l = (low >> @intCast((quarter / 2) * 4)) & 15;
        const h = (b[128 + n * 32 + j] >> @intCast(quarter * 2)) & 3;
        out.code(n * 128 + quarter * 32 + j, l | (h << 4));
    };
}
fn three(b: []const u8, out: Output) void {
    const d = half(b, 108);
    // The scale bytes split into four groups of four low nibbles, plus four 2-bit planes.
    for (0..16) |g| {
        const byte = b[96 + g % 8];
        const low = (byte >> @intCast((g / 8) * 4)) & 15;
        const high = (b[104 + g % 4] >> @intCast(2 * (g / 4))) & 3;
        const signed = @as(i16, low | (high << 4)) - 32;
        const scale = d * @as(f32, @floatFromInt(signed));
        out.coefficient(g, scale, -4 * scale);
    }
    for (0..2) |n| for (0..4) |run| for (0..32) |j| {
        const low = (b[32 + n * 32 + j] >> @intCast(2 * run)) & 3;
        const high = (b[j] >> @intCast(n * 4 + run)) & 1;
        out.code(n * 128 + run * 32 + j, low | (high << 2));
    };
}
fn two(b: []const u8, out: Output) void {
    const d = half(b, 80);
    const m = half(b, 82);
    for (b[0..16], 0..) |v, g| out.coefficient(g, d * @as(f32, @floatFromInt(v & 15)), -m * @as(f32, @floatFromInt(v >> 4)));
    for (0..2) |n| for (0..4) |run| for (0..32) |j| {
        out.code(n * 128 + run * 32 + j, (b[16 + n * 32 + j] >> @intCast(2 * run)) & 3);
    };
}
pub fn write(kind: Kind, src: []const u8, bits: u8, group: usize, words: []u32, scales: []f32, biases: []f32) void {
    @memset(words, 0);
    const block = kind.block();
    var out: Output = .{ .bits = bits, .group = group, .words = words, .scales = scales, .biases = biases };
    var at: usize = 0;
    while (at < src.len) : (at += block.bytes) {
        const b = src[at..][0..block.bytes];
        switch (kind) {
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0 => small(kind, b, out),
            .q4_k, .q5_k => fourFive(kind, b, out),
            .q6_k => six(b, out),
            .q3_k => three(b, out),
            .q2_k => two(b, out),
            else => unreachable,
        }
        out.base += block.values;
    }
}
