//! TENSORFOLD_KOLIBRI_GRAPHS through the registry: Kolibri-1's CUDA step and round graphs, on unless exactly 0.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;
const name = "TENSORFOLD_KOLIBRI_GRAPHS";

const One = struct {
    raw: ?[]const u8,
    pub fn get(v: One, key: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, key, name)) v.raw else null;
    }
};

test "environment: TENSORFOLD_KOLIBRI_GRAPHS is 0 or 1 on CUDA, blank and unset keep graphs on, other text refused" {
    // unset and blank keep the default (graphs on), as before the row
    for ([_]?[]const u8{ null, "", " \t" }) |raw| try t.expectEqual(true, try e.value(bool, name, raw, true));
    try t.expectEqual(false, try e.value(bool, name, "0", true)); // the one text that turned graphs off, still does
    try t.expectEqual(true, try e.value(bool, name, "1", true));
    // other text read as on before the row; now it is refused at a CUDA start, by name
    for ([_][]const u8{ "false", "off", "no", "2", " 0", "0 ", "yes", "true" }) |raw|
        try t.expectError(error.InvalidEnvironment, e.parse(name, raw));
    const why = (try e.refusal(t.allocator, One{ .raw = "off" }, .{ .cuda = true })).?;
    defer t.allocator.free(why);
    try t.expect(std.mem.indexOf(u8, why, "TENSORFOLD_KOLIBRI_GRAPHS=\"off\"") != null);
    try t.expectError(error.InvalidEnvironment, e.checkScope(.cuda, One{ .raw = "off" }));
    // a Kolibri setting: a Metal start does not check it
    try t.expect(try e.refusal(t.allocator, One{ .raw = "off" }, .{ .metal = true }) == null);
}
