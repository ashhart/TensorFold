//! Pack lane requests into the buffers used by the native target and DFly2 graph.
const std = @import("std");
const slots = @import("attention_slots.zig");
const groups = @import("index_groups.zig");

pub const Query = struct {
    token: u32,
    coordinates: [3]i32,
    media: bool = false,
};

pub const Lane = struct {
    /// Reused or freshly prefetched prefix. Speculative rows are supplied separately.
    prefix: []const u32,
    queries: []const Query,
    cache: slots.Cache,
};

pub const Rows = struct {
    ids: []u32,
    coordinates: [][3]i32,
    windows: [][3]u32,
    media: []u32,
    writes: []u32,

    /// Metadata is identical for a lane in a solo batch or one that shares the forward.
    pub fn stage(self: Rows, lanes: []const Lane, pad: u32) !usize {
        var row: usize = 0;
        for (lanes) |lane| {
            for (lane.queries, 0..) |q, i| {
                if (row >= self.ids.len or row >= self.coordinates.len or row >= self.windows.len or
                    row >= self.media.len or row >= self.writes.len) return error.ShortHy4ForwardRows;
                const position = try std.math.add(usize, lane.prefix.len, i);
                const pos = std.math.cast(u32, position) orelse return error.Hy4PositionTooLarge;
                self.ids[row] = q.token;
                self.coordinates[row] = q.coordinates;
                self.windows[row][0] = q.token;
                for (1..3) |lag| {
                    self.windows[row][lag] = if (lag <= i)
                        lane.queries[i - lag].token
                    else if (lag <= position)
                        lane.prefix[position - lag]
                    else
                        pad;
                }
                self.media[row] = @intFromBool(q.media);
                self.writes[row] = try lane.cache.slot(pos);
                row += 1;
            }
        }
        return row;
    }
};

pub const Attention = union(enum) {
    sliding: u32,
    sparse: struct {
        group_budget: u32,
        /// Selected logical token indices for rows beyond the all-keys frontier.
        retained: []const []const i32 = &.{},
    },
};

pub const Keys = struct {
    slots: []u32,
    counts: []u32,
    stride: u32,

    pub fn stage(self: Keys, lanes: []const Lane, mode: Attention) !usize {
        if (self.stride == 0) return error.ShortHy4KeyList;
        var row: usize = 0;
        for (lanes) |lane| {
            for (0..lane.queries.len) |i| {
                if (row >= self.counts.len or row >= self.slots.len / self.stride) return error.ShortHy4KeyList;
                const buffer = self.slots[row * self.stride ..][0..self.stride];
                const pos = try std.math.add(usize, lane.prefix.len, i);
                const end = std.math.cast(u32, try std.math.add(usize, pos, 1)) orelse return error.Hy4PositionTooLarge;
                self.counts[row] = switch (mode) {
                    .sliding => |window| try lane.cache.sliding(end, window, buffer),
                    .sparse => |s| blk: {
                        const f = try groups.frontier(pos, s.group_budget);
                        if (f.groups_to_recall == f.indexed_groups) break :blk try lane.cache.sliding(end, end, buffer);
                        if (row >= s.retained.len) return error.Hy4IndexRowsRequired;
                        const n = try lane.cache.retained(end, s.retained[row], buffer);
                        const expected = f.groups_to_recall * groups.slots + f.visible_end - f.current_group_start;
                        if (n != expected) return error.BadHy4RetainedCount;
                        break :blk n;
                    },
                };
                row += 1;
            }
        }
        return row;
    }
};

test "sharing a target pass preserves tokenizer history and physical cache isolation" {
    const t = std.testing;
    const one: Lane = .{
        .prefix = &.{ 7, 8, 9 },
        .queries = &.{
            .{ .token = 20, .coordinates = .{ 3, 7, 10 }, .media = true },
            .{ .token = 21, .coordinates = .{ 4, 4, 4 } },
        },
        .cache = .{ .page_bases = &.{ 16, 32 }, .page_tokens = 4, .capacity = 64 },
    };
    const two: Lane = .{
        .prefix = &.{11},
        .queries = &.{.{ .token = 22, .coordinates = .{ 1, 1, 1 } }},
        .cache = .{ .page_bases = &.{48}, .page_tokens = 4, .capacity = 64 },
    };
    var ids: [3]u32 = undefined;
    var coords: [3][3]i32 = undefined;
    var windows: [3][3]u32 = undefined;
    var media: [3]u32 = undefined;
    var writes: [3]u32 = undefined;
    const rows: Rows = .{ .ids = &ids, .coordinates = &coords, .windows = &windows, .media = &media, .writes = &writes };
    try t.expectEqual(@as(usize, 3), try rows.stage(&.{ two, one }, 2));
    try t.expectEqualSlices(u32, &.{ 22, 20, 21 }, &ids);
    try t.expectEqualSlices(u32, &.{ 49, 19, 32 }, &writes);
    try t.expectEqualSlices(u32, &.{ 21, 20, 9 }, &windows[2]);
    try t.expectEqualSlices(u32, &.{ 0, 1, 0 }, &media);
    try t.expectEqualSlices(i32, &.{ 3, 7, 10 }, &coords[1]);

    var keys: [3 * 4]u32 = undefined;
    var counts: [3]u32 = undefined;
    const k: Keys = .{ .slots = &keys, .counts = &counts, .stride = 4 };
    try t.expectEqual(@as(usize, 3), try k.stage(&.{ two, one }, .{ .sliding = 4 }));
    try t.expectEqualSlices(u32, &.{ 17, 18, 19, 32 }, keys[8..]);
    try t.expectEqualSlices(u32, &.{ 2, 4, 4 }, &counts);

    try t.expectEqual(@as(usize, 2), try rows.stage(&.{one}, 2));
    try t.expectEqualSlices(u32, &.{ 21, 20, 9 }, &windows[1]);
    try t.expectEqualSlices(u32, &.{ 19, 32 }, writes[0..2]);
    try t.expectError(error.Hy4IndexRowsRequired, k.stage(&.{one}, .{ .sparse = .{ .group_budget = 1 } }));
}
