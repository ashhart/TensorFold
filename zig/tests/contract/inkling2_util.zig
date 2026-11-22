//! Shared pieces of the Inkling 2 operator contract probes: views, values and planted controls (see inkling2_math).
const std = @import("std");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

pub const c = contract.spec;
pub const g = contract.gates;
pub const ops = tf.inkling2_metal.ops;
pub const View = ops.View;

pub const nan_bits: u16 = 0x7fc0;
pub const pinf_bits: u16 = 0x7f80;
pub const ninf_bits: u16 = 0xff80;

pub fn view(r: rig_.Ref) View {
    return .{ .buffer = r.buf, .offset = r.off };
}

/// float64 of a bf16 word.
pub fn val(v: u16) f64 {
    return c.fromBf16(v);
}

/// A float64 rounded to bf16 (round to nearest even) and back: the reference's node rounding.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

pub fn word(v: f64) u16 {
    return c.toBf16(@floatCast(v));
}

/// A random multiple of 1/denom with magnitude at most span/denom, exact in bf16 when span <= 256.
pub fn dyadic(rng: std.Random, span: i32, denom: f32) u16 {
    const k: f32 = @floatFromInt(rng.intRangeAtMost(i32, -span, span));
    return c.toBf16(k / denom);
}

pub fn uniform(rng: std.Random, scale: f32) u16 {
    return c.toBf16((rng.float(f32) * 2 - 1) * scale);
}

pub fn fillUniform(rng: std.Random, out: []u16, scale: f32) void {
    for (out) |*v| v.* = uniform(rng, scale);
}

pub fn fillDyadic(rng: std.Random, out: []u16, span: i32, denom: f32) void {
    for (out) |*v| v.* = dyadic(rng, span, denom);
}

pub fn decode(a: std.mem.Allocator, words: []const u16) ![]f64 {
    const out = try a.alloc(f64, words.len);
    for (out, words) |*o, w| o.* = val(w);
    return out;
}

/// The production text with its one edit; an edit that is absent, or ambiguous, would prove nothing.
pub fn mutate(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]u8 {
    if (std.mem.count(u8, text, from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// Output element i of a bf16 result.
pub fn outBf(bytes: []const u8, i: usize) f64 {
    return c.read(.bf16, bytes, i);
}

/// Probes call this first in `prepare`: frees the last shape's buffers and host data; returns the arena's allocator.
pub fn reset(arena: *std.heap.ArenaAllocator, rig: *rig_.Rig) std.mem.Allocator {
    rig.reset();
    _ = arena.reset(.retain_capacity);
    return arena.allocator();
}

test "dyadic values are exact in bf16" {
    var prng = std.Random.DefaultPrng.init(1);
    for (0..1000) |_| {
        const w = dyadic(prng.random(), 64, 16);
        try std.testing.expectEqual(w, word(val(w)));
    }
}

test "a control edit must match once" {
    const a = std.testing.allocator;
    const out = try mutate(a, "x + y", "+", "-");
    defer a.free(out);
    try std.testing.expectEqualStrings("x - y", out);
    try std.testing.expectError(error.ControlEditMissing, mutate(a, "x + y + z", "+", "-"));
    try std.testing.expectError(error.ControlEditMissing, mutate(a, "x", "+", "-"));
}
