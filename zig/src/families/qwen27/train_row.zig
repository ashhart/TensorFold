//! A learner's single-row Round view has the same metadata as Round.init, with no owning heap allocations.
const std = @import("std");
const Round = @import("round_plan.zig").Round;
const Window = @import("state.zig").Window;

pub const Row = struct {
    ids: [1]u32 = undefined,
    feed: [1]i32 = undefined,
    draw: [1]u64 = undefined,
    slots: [1]u32 = undefined,
    firsts: [1]u32 = .{0},
    parents: [1]i32 = .{-1},
    depths: [1]u32 = .{0},
    conv: [16]u32 = undefined,
    windows: [1]Window = undefined,

    /// Borrowed until this Row dies; do not call Round.deinit on the view.
    pub fn view(r: *Row, a: std.mem.Allocator, id: u32, slot: u32, at: u32, capacity: u32, taps: u32, slots: u32, vocab: u32) !Round {
        if (slots == 0 or slots > 128 or vocab == 0 or slot >= slots or @as(u64, at) + 1 > capacity) return error.BadRoundPlan;
        if (id >= vocab) return error.BadTokenId;
        if (taps < 1 or taps > 16) return error.BadWindow;
        r.ids = .{id};
        r.feed = .{std.math.cast(i32, at) orelse return error.PositionOverflow};
        r.draw = .{@as(u64, at) + 1};
        r.slots = .{slot};
        for (r.conv[0..taps], 0..) |*v, i| v.* = @intCast(i);
        r.windows = .{.{ .gpa = a, .start = at, .parents = &r.parents, .depths = &r.depths, .conv = r.conv[0..taps], .taps = taps }};
        return .{ .allocator = a, .ids = &r.ids, .feed_positions = &r.feed, .draw_positions = &r.draw, .windows = &r.windows, .slots = &r.slots, .firsts = &r.firsts };
    }
};
