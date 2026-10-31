//! Experimental SM120 tile overrides; automatic dispatch remains limited to the measured device class.
const std = @import("std");

pub fn enabled(sm: u32, sms: usize, setting: ?[]const u8) bool {
    if (sm != 120) return false;
    if (setting) |value| {
        if (std.mem.eql(u8, value, "0")) return false;
        if (std.mem.eql(u8, value, "1")) return true;
    }
    return sms == 82;
}

test "SM120 test overrides preserve automatic selection and cannot enable another architecture" {
    try std.testing.expect(enabled(120, 82, null));
    try std.testing.expect(!enabled(120, 170, null));
    try std.testing.expect(!enabled(120, 82, "0"));
    try std.testing.expect(enabled(120, 170, "1"));
    try std.testing.expect(enabled(120, 188, "1"));
    try std.testing.expect(!enabled(89, 128, "1"));
    try std.testing.expect(!enabled(121, 48, "1"));
    try std.testing.expect(!enabled(120, 170, "invalid"));
}
