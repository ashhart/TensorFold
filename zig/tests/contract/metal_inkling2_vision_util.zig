//! Values and buffers shared by the Inkling 2 Mini image-tower contract probes.
const std = @import("std");
const contract = @import("contract");
const rig_ = @import("rig.zig");

pub const c = contract.spec;
pub const g = contract.gates;

pub const nan_bits: u16 = 0x7fc0;
pub const pinf_bits: u16 = 0x7f80;
pub const ninf_bits: u16 = 0xff80;

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

pub fn fillDyadic(rng: std.Random, out: []u16, span: i32, denom: f32) void {
    for (out) |*v| v.* = dyadic(rng, span, denom);
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
