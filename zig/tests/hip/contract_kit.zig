//! What the HIP contract tests share: the activation kind, values exact in it, and device pointers from addresses.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");

const c = core.contract.spec;

/// The kernels' activation kinds: 0 fp32, 1 fp16, 2 bf16.
pub const Kind = enum(c_int) { f32 = 0, f16 = 1, bf16 = 2 };

/// The activation kind this build's chip serves: bf16 with a bf16 dot2, fp16 on RDNA2.
pub fn act() Kind {
    const caps = hip.Caps.of(hip.kernels.arch) orelse return .bf16;
    return if (caps.act == .bf16) .bf16 else .f16;
}

pub fn elem(k: Kind) c.Elem {
    return switch (k) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
    };
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .f32 => "fp32",
        .f16 => "fp16",
        .bf16 => "bf16",
    };
}

pub fn size(k: Kind) usize {
    return if (k == .f32) 4 else 2;
}

/// `v` rounded to the kind (to nearest even, NaN and infinities kept), as the kernels store it.
pub fn round(k: Kind, v: f64) f64 {
    return switch (k) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
    };
}

/// Element `i` of `bytes` written as the kind.
pub fn put(k: Kind, bytes: []u8, i: usize, v: f64) void {
    switch (k) {
        .f32 => std.mem.writeInt(u32, bytes[4 * i ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little),
        .f16 => std.mem.writeInt(u16, bytes[2 * i ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
        .bf16 => std.mem.writeInt(u16, bytes[2 * i ..][0..2], c.toBf16(@floatCast(v)), .little),
    }
}

/// Element `i` of `bytes` read as the kind.
pub fn get(k: Kind, bytes: []const u8, i: usize) f64 {
    return c.read(elem(k), bytes, i);
}

/// `n` values of the kind, uniform in [lo, hi) and exact in the kind, as float64 and as the kind's bytes.
pub fn values(a: std.mem.Allocator, rng: std.Random, k: Kind, n: usize, lo: f64, hi: f64) !struct { []f64, []u8 } {
    const v = try a.alloc(f64, n);
    errdefer a.free(v);
    const b = try a.alloc(u8, n * size(k));
    for (v, 0..) |*x, i| {
        x.* = round(k, lo + rng.float(f64) * (hi - lo));
        if (k == .f16 and @abs(x.*) < 0x1p-14) x.* = 0; // fp16 normals only, as bf16 values are
        put(k, b, i, x.*);
    }
    return .{ v, b };
}

/// `n` fp32 values uniform in [lo, hi), as float64.
pub fn floats(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f64, hi: f64) ![]f64 {
    const v = try a.alloc(f64, n);
    for (v) |*x| x.* = @as(f32, @floatCast(lo + rng.float(f64) * (hi - lo)));
    return v;
}

/// float64 values as fp32 bytes.
pub fn f32Bytes(a: std.mem.Allocator, v: []const f64) ![]u8 {
    const b = try a.alloc(u8, v.len * 4);
    for (v, 0..) |x, i| put(.f32, b, i, x);
    return b;
}

/// A device address as the pointer types the launchers take.
pub fn ptr(comptime T: type, at: u64) T {
    return @ptrFromInt(at);
}

/// x / (1 + e^-x), torch's silu on its formula; NaN for -Inf as x * sigmoid(x) gives.
pub fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// |d silu / dx|: how far an error in silu's input moves its output.
pub fn siluSlope(x: f64) f64 {
    const s = 1 / (1 + @exp(-x));
    return @abs(s * (1 + x * (1 - s)));
}
