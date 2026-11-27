//! A strict 0-or-1 GLM switch shared by every rank's settings reader.
const std = @import("std");

pub fn read(comptime name: [:0]const u8, fallback: bool, value: ?[]const u8, problem: *[]const u8) !bool {
    const v = value orelse return fallback;
    if (std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    problem.* = name ++ ": 0 or 1";
    return error.BadSetting;
}
