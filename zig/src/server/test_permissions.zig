const std = @import("std");
const builtin = @import("builtin");

pub fn skipIfRoot() error{SkipZigTest}!void {
    // Root bypasses permission bits.
    if (comptime builtin.os.tag != .windows) {
        if (std.c.geteuid() == 0) return error.SkipZigTest;
    }
}
