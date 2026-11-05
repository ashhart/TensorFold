//! The per-element MOSS teacher envelope, independent of the device and of byte-repeat checks.
const std = @import("std");

pub const Status = enum {
    pass,
    fail,
    inconclusive,

    pub fn label(self: Status) []const u8 {
        return switch (self) {
            .pass => "PASS",
            .fail => "FAIL",
            .inconclusive => "INCONCLUSIVE",
        };
    }
    pub fn exitCode(self: Status) u8 {
        return switch (self) {
            .pass => 0,
            .fail => 1,
            .inconclusive => 2,
        };
    }
};

pub fn combine(a: Status, b: Status) Status {
    if (a == .fail or b == .fail) return .fail;
    if (a == .inconclusive or b == .inconclusive) return .inconclusive;
    return .pass;
}

pub const Result = struct {
    status: Status,
    repeat: ?bool,

    pub fn overall(self: Result) Status {
        return combine(self.status, if (self.repeat) |r| (if (r) .pass else .fail) else .inconclusive);
    }
};

pub const Verdict = struct { status: Status = .pass, worst: usize = 0, error_abs: f64 = 0, bound: f64 = 0, ratio: f64 = 0, finite: bool = true };

/// Half a bf16 ULP at the fp32 value: 2^(floor(log2(|x|))-8), floored at half a bf16 subnormal step.
pub fn rounding(x: f32) f64 {
    const exponent: i32 = @intCast((@as(u32, @bitCast(x)) >> 23) & 0xff);
    return @exp2(@as(f64, @floatFromInt(if (exponent == 0) -134 else exponent - 135)));
}

/// Twice the larger of the recorded bf16 error and its local half-ULP; the family's reference factor is two.
pub fn compare(actual: []const f32, fp32: []const f32, bf16: []const f32) Verdict {
    std.debug.assert(actual.len == fp32.len and actual.len == bf16.len and actual.len > 0);
    var result: Verdict = .{};
    for (actual, fp32, bf16, 0..) |a, t, b, i| {
        if (!std.math.isFinite(a) or !std.math.isFinite(t) or !std.math.isFinite(b)) {
            return .{ .status = .fail, .worst = i, .finite = false };
        }
        const limit = 2 * @max(@abs(@as(f64, b) - t), rounding(t));
        const err = @abs(@as(f64, a) - t);
        const ratio = err / limit;
        if (i == 0 or ratio > result.ratio) result = .{ .worst = i, .error_abs = err, .bound = limit, .ratio = ratio };
    }
    if (result.ratio > 1) result.status = .fail;
    return result;
}

test "bf16 half-ULP follows magnitude, sign, binade and subnormal spacing" {
    const t = std.testing;
    try t.expectEqual(@as(f64, 1.0 / 256.0), rounding(1));
    try t.expectEqual(rounding(1), rounding(-1));
    try t.expectEqual(@as(f64, 1.0 / 128.0), rounding(2));
    try t.expectEqual(@as(f64, 1.0 / 512.0), rounding(@bitCast(@as(u32, 0x3f7fffff))));
    try t.expectEqual(@exp2(@as(f64, -134)), rounding(0));
    try t.expectEqual(rounding(0), rounding(@bitCast(@as(u32, 1))));
    try t.expectEqual(rounding(0), rounding(@bitCast(@as(u32, 0x00800000))));
}

test "teacher envelope includes its boundary and the measured bf16 error" {
    const t = std.testing;
    try t.expectEqual(Status.pass, compare(&.{1.0078125}, &.{1}, &.{1}).status);
    try t.expectEqual(Status.fail, compare(&.{1.015625}, &.{1}, &.{1}).status);
    const v = compare(&.{1.125}, &.{1}, &.{1.0625});
    try t.expectEqual(Status.pass, v.status);
    try t.expectEqual(@as(f64, 0.125), v.bound);
}

test "worst element uses its own bound rather than another element's magnitude" {
    const v = compare(&.{ 1028, 1.015625, 2 }, &.{ 1024, 1, 2 }, &.{ 1024, 1, 2 });
    try std.testing.expectEqual(Status.fail, v.status);
    try std.testing.expectEqual(@as(usize, 1), v.worst);
    try std.testing.expectEqual(@as(f64, 2), v.ratio);
}

test "nonfinite actual, fp32 or bf16 cannot pass" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    for ([_][3]f32{ .{ nan, 1, 1 }, .{ 1, inf, 1 }, .{ 1, 1, -inf } }) |v| {
        const result = compare(&.{v[0]}, &.{v[1]}, &.{v[2]});
        try std.testing.expectEqual(Status.fail, result.status);
        try std.testing.expect(!result.finite);
    }
}

test "missing coverage never passes and numerical or repeat failure takes precedence" {
    const t = std.testing;
    try t.expectEqual(Status.pass, (Result{ .status = .pass, .repeat = true }).overall());
    try t.expectEqual(Status.inconclusive, (Result{ .status = .inconclusive, .repeat = true }).overall());
    try t.expectEqual(Status.inconclusive, (Result{ .status = .pass, .repeat = null }).overall());
    try t.expectEqual(Status.fail, (Result{ .status = .fail, .repeat = true }).overall());
    try t.expectEqual(Status.fail, (Result{ .status = .inconclusive, .repeat = false }).overall());
}

test "zero and bf16 subnormal values use the absolute rounding floor" {
    const one = @exp2(@as(f32, -133));
    try std.testing.expectEqual(Status.pass, compare(&.{one}, &.{0}, &.{0}).status);
    try std.testing.expectEqual(Status.fail, compare(&.{2 * one}, &.{0}, &.{0}).status);
}
