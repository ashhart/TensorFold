//! A stream window's tree, conv lookup and accepted-path commit, shared by native recurrent and KV state owners.
const std = @import("std");
const Config = @import("config.zig").Config;
const Float = @import("glue.zig").Float;

pub const Shape = struct {
    conv_elements: usize,
    recurrent_elements: usize,
    kv_elements_per_token: usize,
    activation: Float,
    recurrent: Float = .f32,

    pub fn init(c: Config, activation: Float) !Shape {
        return .{
            .conv_elements = try std.math.mul(usize, c.conv_kernel - 1, c.gdnQkvDim()),
            .recurrent_elements = try std.math.mul(usize, c.gdnValueDim(), c.dk),
            .kv_elements_per_token = try std.math.mul(usize, c.kvDim(), 2),
            .activation = activation,
        };
    }
};

pub const Window = struct {
    gpa: std.mem.Allocator,
    start: u32,
    parents: []i32,
    depths: []u32,
    conv: []u32,
    taps: u32,

    pub fn init(gpa: std.mem.Allocator, start: u32, parents: []const i32, taps: u32, capacity: u32) !Window {
        if (parents.len == 0 or parents.len > 128 or taps < 1 or taps > 16 or start > capacity) return error.BadWindow;
        if (parents[0] != -1) return error.BadParents;
        const own = try gpa.dupe(i32, parents);
        errdefer gpa.free(own);
        const depths = try gpa.alloc(u32, parents.len);
        errdefer gpa.free(depths);
        const conv = try gpa.alloc(u32, parents.len * taps);
        errdefer gpa.free(conv);
        for (parents, 0..) |parent, row| {
            if (row > 0 and (parent < 0 or parent >= row)) return error.BadParents;
            depths[row] = if (parent < 0) 0 else depths[@intCast(parent)] + 1;
            if (@as(u64, start) + depths[row] >= capacity) return error.ContextFull;
            var path: [128]u32 = undefined;
            var at: i32 = @intCast(row);
            var n: usize = 0;
            while (at >= 0) {
                path[n] = @intCast(at);
                n += 1;
                at = parents[@intCast(at)];
            }
            for (0..taps) |j| {
                const behind = taps - 1 - j;
                conv[row * taps + j] = if (behind < n) taps - 1 + path[behind] else @intCast(taps - 1 - (behind - n + 1));
            }
        }
        return .{ .gpa = gpa, .start = start, .parents = own, .depths = depths, .conv = conv, .taps = taps };
    }

    pub fn deinit(w: *Window) void {
        w.gpa.free(w.parents);
        w.gpa.free(w.depths);
        w.gpa.free(w.conv);
        w.* = undefined;
    }

    pub fn position(w: Window, row: usize) u32 {
        return w.start + w.depths[row];
    }

    pub fn keep(w: Window, path: []const u32) !Commit {
        if (path.len > w.parents.len) return error.BadPath;
        var previous: i32 = -1;
        for (path) |row| {
            if (row >= w.parents.len or w.parents[row] != previous) return error.BadPath;
            previous = @intCast(row);
        }
        return .{ .length = w.start + @as(u32, @intCast(path.len)), .replay_rows = path, .conv_tail_row = if (path.len == 0) null else path[path.len - 1] };
    }
};

pub const Commit = struct {
    length: u32,
    replay_rows: []const u32,
    conv_tail_row: ?u32,
};

test "tree conv windows and positions follow ancestors rather than row offsets" {
    var w = try Window.init(std.testing.allocator, 17, &.{ -1, 0, 0, 1, 2 }, 4, 64);
    defer w.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 2, 2 }, w.depths);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, w.conv[0..4]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5 }, w.conv[8..12]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 5, 7 }, w.conv[16..20]);
    try std.testing.expectEqual(@as(u32, 19), w.position(4));
}

test "non-prefix keep names exact recurrence replay and final conv row" {
    var w = try Window.init(std.testing.allocator, 17, &.{ -1, 0, 0, 1, 2 }, 4, 64);
    defer w.deinit();
    const commit = try w.keep(&.{ 0, 2, 4 });
    try std.testing.expectEqual(@as(u32, 20), commit.length);
    try std.testing.expectEqual(@as(?u32, 4), commit.conv_tail_row);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4 }, commit.replay_rows);
    const empty = try w.keep(&.{});
    try std.testing.expectEqual(@as(u32, 17), empty.length);
    try std.testing.expectEqual(@as(?u32, null), empty.conv_tail_row);
    try std.testing.expectError(error.BadPath, w.keep(&.{ 0, 1, 4 }));
    try std.testing.expectError(error.BadPath, w.keep(&.{2}));
}

test "invalid parents and context overflow refuse before state mutation" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.BadParents, Window.init(a, 0, &.{ -1, 2 }, 4, 64));
    try std.testing.expectError(error.BadParents, Window.init(a, 0, &.{ -1, -1 }, 4, 64));
    try std.testing.expectError(error.ContextFull, Window.init(a, 63, &.{ -1, 0 }, 4, 64));
}
