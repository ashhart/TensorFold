//! Explicit residency policy shared by load validation and host controls.
const std = @import("std");

pub fn flag(text: ?[]const u8) error{BadResidency}!bool {
    const value = text orelse return true;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadResidency;
}

pub const invalid_flag = "GLM_RESIDENCY must be 0 or 1; unset keeps eligible Metal buffers resident by default";

test "residency GLM: the unset default and exact zero and one overrides" {
    try std.testing.expect(try flag(null));
    try std.testing.expect(!try flag("0"));
    try std.testing.expect(try flag("1"));
}

test "residency GLM: every other flag value refuses" {
    for ([_][]const u8{ "", "00", "01", "yes", "true", "2", " 0", "1 " }) |text|
        try std.testing.expectError(error.BadResidency, flag(text));
}
