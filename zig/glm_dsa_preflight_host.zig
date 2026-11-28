//! These host tests assert expected loader errors and their allocation counts; ignore only their test-log receipts.
const std = @import("std");

fn expectedErrors(comptime _: std.log.Level, comptime _: @EnumLiteral(), comptime _: []const u8, _: anytype) void {}
pub const std_options: std.Options = .{ .logFn = expectedErrors };

test {
    _ = @import("src/families/glm/cuda_dsa_preflight_test.zig");
}
