//! Shared pieces of MiMo-V2's contract tests: host values in the kernels' types, uploads, derived allowances.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;

/// fp32's unit roundoff, and the relative error of rsqrtf, expf and tanhf (2 ulps).
pub const u = 0x1p-24;
pub const fast = 0x1p-22;

pub fn gamma(d: f64) f64 {
    return c.gamma(d, u);
}

pub fn h16(v: f64) f64 {
    return @as(f16, @floatCast(v));
}

pub fn f32r(v: f64) f64 {
    return @as(f32, @floatCast(v));
}

pub fn bf16r(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// The allowance of an output stored as `e` whose value before the store is within `err` of `want`.
pub fn allow(e: c.Elem, want: f64, err: f64) f64 {
    const unit: f64, const floor: f64 = switch (e) {
        .f32 => .{ 0x1p-24, 0x1p-149 },
        .f16 => .{ 0x1p-11, 0x1p-24 },
        .bf16 => .{ 0x1p-8, 0x1p-133 },
        .raw => .{ 0, 0 },
    };
    return err + unit * (@abs(want) + err) + floor;
}

/// Uploads values as the kernel reads them (each must be exact in that type).
pub fn put(rig: *rig_.Rig, a: std.mem.Allocator, comptime T: type, vals: []const f64) !u64 {
    const buf = try a.alloc(T, vals.len);
    defer a.free(buf);
    for (buf, vals) |*o, v| o.* = switch (T) {
        u16 => c.toBf16(@floatCast(v)),
        i32, i64 => @intFromFloat(v),
        else => @floatCast(v),
    };
    return rig.input(std.mem.sliceAsBytes(buf));
}

pub fn ints(rig: *rig_.Rig, comptime T: type, vals: []const T) !u64 {
    return rig.input(std.mem.sliceAsBytes(vals));
}

/// Copies `len` bytes between device buffers on the rig's stream (a kernel's in-place state, set before a run).
pub fn copy(rig: *rig_.Rig, dst: u64, src: u64, len: usize) !void {
    const b: cuda.DeviceBuffer = .{ .d = rig.gpu.d, .ptr = dst, .len = len };
    try b.copyFrom(0, src, len, rig.s.handle);
}

/// Integers in [lo, hi] as f64 (gate 4's terms).
pub fn int(rng: std.Random, lo: i32, hi: i32) f64 {
    return @floatFromInt(rng.intRangeAtMost(i32, lo, hi));
}

/// Uniform in [lo, hi) rounded to T's values.
pub fn uni(rng: std.Random, comptime T: type, lo: f64, hi: f64) f64 {
    const v = lo + (hi - lo) * rng.float(f64);
    return switch (T) {
        f16 => h16(v),
        u16 => bf16r(v),
        else => f32r(v),
    };
}

/// The outputs equal another launch's bytes (a replacement's gates 1 and 3); at gate 1 some must be non-finite.
pub fn sameBytes(msg: []u8, got: []const u8, want: []const u8, what: []const u8, e: c.Elem, at: core.contract.gates.Gate) !?[]const u8 {
    if (std.mem.indexOfDiff(u8, got, want)) |i| return try std.fmt.bufPrint(msg, "byte {d} of {d} differs from {s}'s", .{ i, got.len, what });
    if (at != .nonfinite) return null;
    for (0..got.len / c.size(e)) |i| if (c.classOf(c.read(e, got, i)) != .finite) return null;
    return try std.fmt.bufPrint(msg, "no output is non-finite: the planted values prove nothing", .{});
}

/// fmaxf's maximum: a NaN operand gives the other (the kernels' warp maxima drop NaN scores).
pub fn fmax(a: f64, b: f64) f64 {
    if (std.math.isNan(b)) return a;
    if (std.math.isNan(a)) return b;
    return @max(a, b);
}
