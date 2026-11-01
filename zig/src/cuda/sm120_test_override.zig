//! Experimental SM120 tile overrides; automatic dispatch remains limited to the measured device class.
const std = @import("std");

pub const Error = error{InvalidTestOverride};

pub fn enabled(sm: u32, sms: usize, setting: ?[]const u8) bool {
    if (sm != 120) return false;
    if (setting) |value| {
        if (std.mem.eql(u8, value, "0")) return false;
        if (std.mem.eql(u8, value, "1")) return true;
    }
    return sms == 82;
}

/// An override is "0" or "1"; anything else is refused rather than silently falling back to automatic selection.
pub fn check(setting: ?[]const u8) Error!void {
    const value = setting orelse return;
    if (!std.mem.eql(u8, value, "0") and !std.mem.eql(u8, value, "1")) return error.InvalidTestOverride;
}

/// The tile choice for the environment variable `name` on this device.
pub fn select(comptime name: [:0]const u8, sm: u32, sms: usize) Error!bool {
    const setting: ?[]const u8 = if (std.c.getenv(name.ptr)) |v| std.mem.span(v) else null;
    check(setting) catch |err| {
        std.log.err("{s} must be 0 or 1, not '{s}'", .{ name, setting.? });
        return err;
    };
    const on = enabled(sm, sms, setting);
    if (setting) |v| std.log.info("experimental {s}={s}, enabled={}", .{ name, v, on });
    return on;
}

test "SM120 test overrides preserve automatic selection and cannot enable another architecture" {
    try std.testing.expect(enabled(120, 82, null));
    try std.testing.expect(!enabled(120, 170, null));
    try std.testing.expect(!enabled(120, 82, "0"));
    try std.testing.expect(enabled(120, 170, "1"));
    try std.testing.expect(enabled(120, 188, "1"));
    try std.testing.expect(!enabled(89, 128, "1"));
    try std.testing.expect(!enabled(121, 48, "1"));
}

test "an override other than 0 or 1 is refused" {
    try check(null);
    try check("0");
    try check("1");
    for ([_][]const u8{ "", "2", "true", "01", "invalid", "1 " }) |bad|
        try std.testing.expectError(error.InvalidTestOverride, check(bad));
}
