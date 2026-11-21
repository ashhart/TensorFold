//! Positions where the Metal prompt path can keep an exact state for a later continuation.
const std = @import("std");
const Config = @import("dsv41").config.Config;

pub const Marks = struct {
    ends: [3]u32 = @splat(0),
    thinking: u32 = 0,
};

pub fn of(c: *const Config, ids: []const u32, at: u32, prefix: u32) Marks {
    const user = std.mem.indexOfScalar(u32, ids, c.user) orelse 0;
    const shared: u32 = if (user >= 256 and user > at) @intCast(user) else 0;
    const n = ids.len;
    const thinking: u32 = if (n >= 2 and ids[n - 1] == c.think_open and
        ids[n - 2] == c.assistant and n - 1 > @max(at, shared)) @intCast(n - 1) else 0;
    const extra = if (prefix > at and prefix < n and prefix != shared and prefix != thinking) prefix else 0;
    return .{ .ends = .{ @min(shared, extra), @max(shared, extra), thinking }, .thinking = thinking };
}
