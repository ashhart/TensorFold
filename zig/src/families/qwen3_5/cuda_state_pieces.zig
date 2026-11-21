//! A pool lane's state as pieces on its lease's pages (core kv_pieces); the copy lies in log rows 1.. (Overlay).
const std = @import("std");
const kv = @import("core").kv_pool;
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const pieces = kv.pieces;
const most_window = engine.most_window;
const alignment = 256;

pub const Map = struct {
    /// The exclusive pieces on the lane's state pages: nl recurrences, nl tails, row 0, the row groups.
    own: pieces.Map,
    /// The copy's pieces (nl recurrences, nl tails) inside the groups: a page of `own` is one group here.
    copy: ?pieces.Map,
    /// Log rows a full group holds, and the groups (the last may hold fewer).
    group_rows: usize,
    groups: usize,

    pub fn deinit(m: *Map, gpa: std.mem.Allocator) void {
        m.own.deinit(gpa);
        if (m.copy) |*c| c.deinit(gpa);
        m.* = undefined;
    }

    /// The layout's and the load report's count: the exclusive pieces, the copy's among them when it has its own.
    pub fn summary(m: Map) kv.groups.Pieces {
        return m.own.summary();
    }

    /// The state table (cuda_states.zig) of a lane whose state pages start at `ids` in a pool at `base`.
    pub fn fill(m: Map, t: []u64, g: geometry.Geometry, base: u64, ids: []const u32) void {
        const nl = g.deltas();
        for (0..2 * nl) |i| t[i] = m.own.addr(i, base, ids);
        const row = nl * g.logRowBytes();
        t[4 * nl] = m.own.addr(2 * nl, base, ids);
        for (1..most_window) |r| {
            const k = (r - 1) / m.group_rows;
            t[4 * nl + r] = m.own.addr(2 * nl + 1 + k, base, ids) + (r - 1 - k * m.group_rows) * row;
        }
        for (0..2 * nl) |i| t[2 * nl + i] = if (m.copy) |c|
            m.own.addr(2 * nl + 1 + c.at[i].page, base, ids) + c.at[i].offset
        else
            m.own.addr(2 * nl + 1 + m.groups + i, base, ids);
    }
};

/// The map at `page` bytes a page: the copy inside the full groups if it fits there, else pieces of its own.
pub fn plan(gpa: std.mem.Allocator, g: geometry.Geometry, page: u64) !Map {
    const nl = g.deltas();
    const row: u64 = nl * g.logRowBytes();
    if (row > page) return error.QwenLogRowOverPage;
    const group_rows: usize = if (row == 0) most_window - 1 else @min(most_window - 1, page / row);
    const groups = (most_window - 1 + group_rows - 1) / group_rows;
    var state: std.ArrayList(u64) = .empty;
    defer state.deinit(gpa);
    try state.appendNTimes(gpa, g.recBytes(), nl);
    try state.appendNTimes(gpa, g.convBytes(), nl);
    // the copy inside the full groups, or (a piece wider than a group, too few groups) pieces of its own
    var copy: ?pieces.Map = pieces.pack(gpa, state.items, group_rows * row, alignment) catch |err| switch (err) {
        error.PieceOverPage => null,
        else => |x| return x,
    };
    errdefer if (copy) |*c| c.deinit(gpa);
    if (copy != null and copy.?.pages > (most_window - 1) / group_rows) {
        copy.?.deinit(gpa);
        copy = null;
    }
    var sizes: std.ArrayList(u64) = .empty;
    defer sizes.deinit(gpa);
    try sizes.appendSlice(gpa, state.items);
    try sizes.append(gpa, row);
    for (0..groups) |k| try sizes.append(gpa, @min(group_rows, most_window - 1 - k * group_rows) * row);
    if (copy == null) try sizes.appendSlice(gpa, state.items);
    return .{ .own = try pieces.pack(gpa, sizes.items, page, alignment), .copy = copy, .group_rows = group_rows, .groups = groups };
}

test {
    _ = @import("cuda_state_pieces_test.zig");
}
