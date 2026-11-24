//! tf-idol-generate's numeric flags, checked as parsed: a value that is not finite or out of range is refused.
const std = @import("std");
const dc = @import("decoder_config.zig");
const pre = @import("preprocess.zig");

/// The most sampler steps a run takes (the SDK's default is 50).
pub const max_steps = 1000;

pub const Error = error{Usage};

pub fn steps(v: []const u8) Error!usize {
    const n = std.fmt.parseInt(usize, v, 10) catch return error.Usage;
    if (n == 0 or n > max_steps) return error.Usage;
    return n;
}

/// Classifier-free guidance: finite (the SDK's default is 3).
pub fn guidance(v: []const u8) Error!f32 {
    const g = std.fmt.parseFloat(f32, v) catch return error.Usage;
    if (!std.math.isFinite(g)) return error.Usage;
    return g;
}

/// Gaussians within the decoders' range (dc.anchors).
pub fn gaussians(v: []const u8) Error!usize {
    const n = std.fmt.parseInt(usize, v, 10) catch return error.Usage;
    _ = dc.anchors(n) catch return error.Usage;
    return n;
}

/// The crop margin: finite and above 0 (finish bounds the crop it makes).
pub fn cropMargin(v: []const u8) Error!f64 {
    const m = std.fmt.parseFloat(f64, v) catch return error.Usage;
    return pre.checkMargin(m) catch error.Usage;
}

pub fn seed(v: []const u8) Error!u64 {
    return std.fmt.parseInt(u64, v, 10) catch error.Usage;
}

test "numeric flags refuse values that are not finite, overflow or leave their range" {
    for ([_][]const u8{ "1e300", "inf", "-inf", "nan", "0", "-1", "x" }) |v| {
        if (!std.mem.eql(u8, v, "1e300")) try std.testing.expectError(error.Usage, cropMargin(v));
    }
    try std.testing.expectEqual(@as(f64, 1e300), try cropMargin("1e300")); // finite: finish refuses its crop
    try std.testing.expectEqual(@as(f64, 1.7), try cropMargin("1.7"));
    for ([_][]const u8{ "inf", "nan", "-inf", "1e40" }) |v| try std.testing.expectError(error.Usage, guidance(v));
    try std.testing.expectEqual(@as(f32, 3), try guidance("3"));
    for ([_][]const u8{ "0", "1001", "18446744073709551615", "18446744073709551616", "-3" }) |v| try std.testing.expectError(error.Usage, steps(v));
    try std.testing.expectEqual(@as(usize, 50), try steps("50"));
    for ([_][]const u8{ "1", "18446744073709551615", "99999999999999999999" }) |v| try std.testing.expectError(error.Usage, gaussians(v));
    try std.testing.expectEqual(@as(usize, 262144), try gaussians("262144"));
    try std.testing.expectError(error.Usage, seed("18446744073709551616"));
    try std.testing.expectEqual(@as(u64, 42), try seed("42"));
}
