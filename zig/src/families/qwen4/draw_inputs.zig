//! Sampling admission rejects invalid positions, rules and empty grammar rows before any metadata upload.
const std = @import("std");
pub fn check(vocab: usize, positions: []const u64, sampling: anytype, bits: []const u32) !void {
    for (positions) |p| if (p == 0 or p > std.math.maxInt(i32)) return error.Qwen4DrawPosition;
    if (sampling) |s| {
        if (!std.math.isFinite(s.temperature) or s.temperature < 0 or !std.math.isFinite(s.top_p) or s.top_p < 0 or s.top_p > 1 or !std.math.isFinite(s.min_p) or s.min_p < 0 or s.min_p > 1) return error.Qwen4DrawRule;
    }
    if (bits.len == 0) return;
    if (vocab == 0) return error.Qwen4DrawMask;
    const words = (vocab + 31) / 32;
    if (bits.len != positions.len * words) return error.Qwen4DrawMask;
    for (0..positions.len) |r| {
        var any = false;
        for (bits[r * words ..][0..words], 0..) |b, i| {
            const valid: u32 = if (i + 1 == words and vocab % 32 != 0) (@as(u32, 1) << @as(u5, @intCast(vocab % 32))) - 1 else std.math.maxInt(u32);
            any = any or b & valid != 0;
        }
        if (!any) return error.Qwen4EmptyDrawMask;
    }
}
test "Qwen4 draw input: padding bits cannot turn an empty vocabulary mask into an allowed row" {
    const T = std.testing;
    const Rule = struct { temperature: f64 = 1, top_p: f64 = 1, min_p: f64 = 0 };
    try check(33, &.{1}, @as(?Rule, .{}), &.{ 0, 1 });
    try T.expectError(error.Qwen4EmptyDrawMask, check(33, &.{1}, @as(?Rule, null), &.{ 0, 2 }));
    try T.expectError(error.Qwen4DrawPosition, check(33, &.{0}, @as(?Rule, null), &.{}));
    try T.expectError(error.Qwen4DrawRule, check(33, &.{1}, @as(?Rule, .{ .temperature = std.math.nan(f64) }), &.{}));
}
