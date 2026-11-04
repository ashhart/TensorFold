//! What the XPU contract tests share: the glue's value kinds, values exact in them, and the glue's own formulas.
const std = @import("std");
const core = @import("core");
const c = core.spec;

/// The kinds the glue stores: fp16 (half), bf16 bits, and fp32.
pub const Kind = enum { f16, bf16, f32 };

pub fn elem(k: Kind) c.Elem {
    return switch (k) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
    };
}

pub fn size(k: Kind) usize {
    return if (k == .f32) 4 else 2;
}

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .f16 => "fp16",
        .bf16 => "bf16",
        .f32 => "fp32",
    };
}

/// `v` rounded to the kind (to nearest even, NaN and infinities kept), as the glue stores it.
pub fn round(k: Kind, v: f64) f64 {
    return switch (k) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
    };
}

/// Element `i` of `bytes` written as the kind.
pub fn put(k: Kind, b: []u8, i: usize, v: f64) void {
    switch (k) {
        .f32 => std.mem.writeInt(u32, b[4 * i ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little),
        .f16 => std.mem.writeInt(u16, b[2 * i ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
        .bf16 => std.mem.writeInt(u16, b[2 * i ..][0..2], c.toBf16(@floatCast(v)), .little),
    }
}

/// Element `i` of `bytes` read as the kind.
pub fn get(k: Kind, b: []const u8, i: usize) f64 {
    return c.read(elem(k), b, i);
}

/// `n` values of the kind, uniform in [lo, hi) and exact in the kind, as float64 and as the kind's bytes.
pub fn values(a: std.mem.Allocator, rng: std.Random, k: Kind, n: usize, lo: f64, hi: f64) !struct { []f64, []u8 } {
    const v = try a.alloc(f64, n);
    errdefer a.free(v);
    const b = try a.alloc(u8, n * size(k));
    for (v, 0..) |*x, i| {
        x.* = round(k, lo + rng.float(f64) * (hi - lo));
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

/// float64 values as the kind's bytes.
pub fn bytes(a: std.mem.Allocator, v: []const f64, k: Kind) ![]u8 {
    const b = try a.alloc(u8, v.len * size(k));
    for (v, 0..) |x, i| put(k, b, i, x);
    return b;
}

/// A device address as the pointer types the launcher takes.
pub fn ptr(comptime T: type, at: u64) T {
    return @ptrFromInt(at);
}
