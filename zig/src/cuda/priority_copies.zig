//! Load-time opt-in for unqualified root/context copies; suffix copies keep the existing lane facts.
const std = @import("std");

pub fn enabled(drafts: bool, value: ?[]const u8) !bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return drafts;
    return error.BadCopyPolicy;
}

pub fn read(name: [*:0]const u8, drafts: bool) !bool {
    const value = if (std.c.getenv(name)) |v| std.mem.span(v) else null;
    return enabled(drafts, value) catch |err| {
        std.log.err("{s}: expected 0 or 1 for verified root/context copies", .{name});
        return err;
    };
}

test "CUDA lane copies: priority policy defaults off and respects the serving drafts switch" {
    for ([_]bool{ false, true }) |drafts| {
        try std.testing.expect(!try enabled(drafts, null));
        try std.testing.expect(!try enabled(drafts, "0"));
        try std.testing.expectEqual(drafts, try enabled(drafts, "1"));
        for ([_][]const u8{ "", "2", "true", "-1", "1 " }) |bad|
            try std.testing.expectError(error.BadCopyPolicy, enabled(drafts, bad));
    }
}
