//! The environment setters the platform shim exposes, exercised on the host: set, read back, overwrite, clear.
const std = @import("std");
const env = @import("sys").env;

fn value(name: [:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    return std.mem.span(raw);
}

test "the platform environment shim sets a variable, keeps it on a no-overwrite write and clears it" {
    const name = "TF_PLATFORM_ENV_SHIM_TEST";
    defer _ = env.unsetenv(name);
    try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, "1", 1));
    try std.testing.expectEqualStrings("1", value(name).?);
    // a write that may not overwrite leaves the value alone
    try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, "2", 0));
    try std.testing.expectEqualStrings("1", value(name).?);
    try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, "2", 1));
    try std.testing.expectEqualStrings("2", value(name).?);
    try std.testing.expectEqual(@as(c_int, 0), env.unsetenv(name));
    // Windows clears a variable to the empty string, elsewhere it is gone
    const left = value(name);
    try std.testing.expect(left == null or left.?.len == 0);
}
