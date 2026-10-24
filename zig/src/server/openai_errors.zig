//! OpenAI request refusal logging and untyped error payloads.
const std = @import("std");
const json = @import("json");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// Why a request got an error reply, logged before it: its access line shows only the status.
pub fn logRefused(id: []const u8, message: []const u8) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, message)});
}

pub fn errorOther(a: Allocator, message: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "message", .{ .string = message });
    return .{ .object = o };
}
