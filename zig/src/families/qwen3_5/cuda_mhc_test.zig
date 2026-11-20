//! Host launch recording for the two-stream mHC adapter.
const std = @import("std");
const cuda = @import("cuda");
const mhc = @import("cuda_mhc.zig");
const t = std.testing;

const Fake = struct {
    g: struct { hidden: usize } = .{ .hidden = 4096 },
    h: u64 = 100,
    r: u64 = 200,
    mhc: mhc.Runtime = .{},
    calls: usize = 0,
    ids: [6]usize = undefined,
    values: [6][10]u64 = @splat(@splat(0)),
    counts: [6]usize = undefined,
    grids: [6]usize = undefined,

    pub fn run(e: *Fake, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        const n = e.calls;
        e.calls += 1;
        e.ids[n] = @intFromPtr(f.handle.?);
        e.grids[n] = grid[0];
        try t.expectEqual(@as(usize, 256), block[0]);
        const fields = @typeInfo(@TypeOf(args)).@"struct".field_names;
        e.counts[n] = fields.len;
        inline for (fields, 0..) |field, i| e.values[n][i] = if (@TypeOf(@field(args, field)) == f32) @as(u32, @bitCast(@field(args, field))) else @intCast(@field(args, field));
    }
};

fn function(id: usize) cuda.Function {
    return .{ .d = undefined, .handle = @ptrFromInt(id) };
}

test "SenseNova mHC: checked launches pass pointers, row count and parameters" {
    var e: Fake = .{};
    e.mhc.enter = function(1);
    e.mhc.route = function(2);
    e.mhc.merge = function(3);
    e.mhc.leave = function(4);
    e.mhc.streams = 300;
    e.mhc.post = 400;
    e.mhc.mix = 500;
    try mhc.enter(&e, true, 7, 600);
    try mhc.enter(&e, false, 7, 600);
    try mhc.route(&e, 700, 800, 7, 600);
    e.mhc.iterations = 3;
    e.mhc.pre_scale = 1.5;
    try mhc.route(&e, 700, 800, 7, 600);
    try mhc.merge(&e, 7, 600);
    try mhc.leave(&e, 7, 600);
    try t.expectEqualSlices(usize, &.{ 1, 1, 2, 2, 3, 4 }, &e.ids);
    try t.expectEqualSlices(u64, &.{ 100, 200, 300, 4096, 600 }, e.values[0][0..5]);
    try t.expectEqualSlices(u64, &.{ 100, 0, 300, 4096, 600 }, e.values[1][0..5]);
    try t.expectEqualSlices(u64, &.{ 300, 700, 800, 100, 400, 500, 4096, 20, @as(u32, @bitCast(@as(f32, 1))), 600 }, &e.values[2]);
    try t.expectEqualSlices(u64, &.{ 300, 700, 800, 100, 400, 500, 4096, 3, @as(u32, @bitCast(@as(f32, 1.5))), 600 }, &e.values[3]);
    try t.expectEqualSlices(u64, &.{ 300, 200, 400, 500, 4096, 600 }, e.values[4][0..6]);
    try t.expectEqualSlices(u64, &.{ 300, 100, 4096, 600 }, e.values[5][0..4]);
    for (e.grids) |grid| try t.expectEqual(@as(usize, 7), grid);
}
