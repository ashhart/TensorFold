//! Feed positions and keyed draw positions stay separate when sibling lanes share a depth.
const std = @import("std");
const state = @import("state.zig");
const gdn = @import("gdn_plan.zig");
const abi = @import("gdn_contract.zig");

pub const Input = struct { slot: u32, start: u32, capacity: u32, ids: []const u32, parents: ?[]const i32 = null };

pub const Round = struct {
    allocator: std.mem.Allocator,
    ids: []u32,
    feed_positions: []i32,
    draw_positions: []u64,
    windows: []state.Window,
    slots: []u32,
    firsts: []u32,

    pub fn init(a: std.mem.Allocator, inputs: []const Input, taps: u32, slots: u32, vocab: u32) !Round {
        if (inputs.len == 0 or inputs.len > 64 or slots == 0 or slots > 64 or vocab == 0) return error.BadRoundPlan;
        var rows: usize = 0;
        var used: u64 = 0;
        for (inputs) |input| {
            if (input.slot >= slots or input.ids.len == 0 or input.ids.len > 128 or @as(u64, input.start) + input.ids.len > input.capacity) return error.BadRoundPlan;
            const bit = @as(u64, 1) << @intCast(input.slot);
            if (used & bit != 0) return error.DuplicateRoundSlot;
            used |= bit;
            rows = try std.math.add(usize, rows, input.ids.len);
            for (input.ids) |id| if (id >= vocab) return error.BadTokenId;
        }
        if (rows > 128) return error.BadRoundPlan;
        const ids = try a.alloc(u32, rows);
        errdefer a.free(ids);
        const feed = try a.alloc(i32, rows);
        errdefer a.free(feed);
        const draws = try a.alloc(u64, rows);
        errdefer a.free(draws);
        const windows = try a.alloc(state.Window, inputs.len);
        var made: usize = 0;
        errdefer {
            for (windows[0..made]) |*w| w.deinit();
            a.free(windows);
        }
        const slot_ids = try a.alloc(u32, inputs.len);
        errdefer a.free(slot_ids);
        const firsts = try a.alloc(u32, inputs.len);
        errdefer a.free(firsts);
        var first: usize = 0;
        for (inputs, 0..) |input, i| {
            var chain: [128]i32 = undefined;
            chain[0] = -1;
            for (1..input.ids.len) |r| chain[r] = @intCast(r - 1);
            const parents = input.parents orelse chain[0..input.ids.len];
            if (parents.len != input.ids.len) return error.BadRoundPlan;
            windows[i] = try state.Window.init(a, input.start, parents, taps, input.capacity);
            made += 1;
            slot_ids[i] = input.slot;
            firsts[i] = @intCast(first);
            @memcpy(ids[first..][0..input.ids.len], input.ids);
            for (0..input.ids.len) |r| {
                const p = windows[i].position(r);
                feed[first + r] = std.math.cast(i32, p) orelse return error.PositionOverflow;
                draws[first + r] = @as(u64, p) + 1;
            }
            first += input.ids.len;
        }
        return .{ .allocator = a, .ids = ids, .feed_positions = feed, .draw_positions = draws, .windows = windows, .slots = slot_ids, .firsts = firsts };
    }

    pub fn deinit(r: *Round) void {
        for (r.windows) |*w| w.deinit();
        r.allocator.free(r.ids);
        r.allocator.free(r.feed_positions);
        r.allocator.free(r.draw_positions);
        r.allocator.free(r.windows);
        r.allocator.free(r.slots);
        r.allocator.free(r.firsts);
        r.* = undefined;
    }

    pub fn recurrent(r: Round, p: abi.Params, paths: []const []const u32) !gdn.Plan {
        if (paths.len != r.windows.len or p.rows != r.ids.len) return error.BadRoundPlan;
        const items = try r.allocator.alloc(gdn.Input, paths.len);
        defer r.allocator.free(items);
        for (items, r.windows, r.slots, paths) |*item, window, slot, path| item.* = .{ .window = window, .state_slot = slot, .next_slot = slot, .path = path };
        return gdn.Plan.init(r.allocator, p, items);
    }
};

test "sibling feeds share position and every draw uses its successor position" {
    var r = try Round.init(std.testing.allocator, &.{ .{ .slot = 1, .start = 17, .capacity = 64, .ids = &.{ 3, 4, 5, 6, 7 }, .parents = &.{ -1, 0, 0, 1, 2 } }, .{ .slot = 3, .start = 30, .capacity = 64, .ids = &.{ 8, 9 } } }, 4, 4, 16);
    defer r.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 17, 18, 18, 19, 19, 30, 31 }, r.feed_positions);
    try std.testing.expectEqualSlices(u64, &.{ 18, 19, 19, 20, 20, 31, 32 }, r.draw_positions);
    try std.testing.expectEqualSlices(u32, &.{ 0, 5 }, r.firsts);
    const p = abi.Params{ .rows = 7, .slots = 4, .nk = 16, .nv = 48, .dk = 128, .dv = 128, .taps = 4, .weight_flags = 0, .eps = 1e-6 };
    var plan = try r.recurrent(p, &.{ &.{ 0, 2, 4 }, &.{ 0, 1 } });
    defer plan.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4, 5, 6 }, plan.kept_rows);
}

test "physical tree cache width and unique stream ownership are checked before dispatch" {
    const a = std.testing.allocator;
    const input = Input{ .slot = 0, .start = 62, .capacity = 64, .ids = &.{ 1, 2, 3 }, .parents = &.{ -1, 0, 0 } };
    try std.testing.expectError(error.BadRoundPlan, Round.init(a, &.{input}, 4, 4, 16));
    const valid = Input{ .slot = 0, .start = 0, .capacity = 64, .ids = &.{1} };
    try std.testing.expectError(error.DuplicateRoundSlot, Round.init(a, &.{ valid, valid }, 4, 4, 16));
}
