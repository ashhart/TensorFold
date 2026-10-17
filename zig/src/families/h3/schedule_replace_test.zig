//! The production replacement coordinator under later-block, projection, completion and Steps allocation faults.
const std = @import("std");
const replace = @import("schedule_replace.zig");
const Ref = struct { buffer: u32 };
const Model = struct {
    blocks: [2]struct { tables: Ref } = .{ .{ .tables = .{ .buffer = 1 } }, .{ .tables = .{ .buffer = 2 } } },
    final_table: Ref = .{ .buffer = 3 },
    adaln: []const i32 = &.{7},
    times: []const i32 = &.{8},
    lines: usize = 9,
};
const Op = struct {
    allocator: std.mem.Allocator = std.testing.allocator,
    fail: u8 = 0,
    releases: usize = 0,
    adaln: []const i32 = &.{10},
    times: []const i32 = &.{11},
    lines: usize = 12,
    pub fn makeSteps(o: *@This()) !u32 {
        if (o.fail == 1) return error.InjectedSteps;
        return 42;
    }
    pub fn freeSteps(_: *@This(), _: u32) void {}
    pub fn block(o: *@This(), i: usize) !Ref {
        if (i == 1 and o.fail == 2) return error.InjectedProjection;
        return .{ .buffer = @intCast(4 + i) };
    }
    pub fn final(o: *@This()) !Ref {
        if (o.fail == 3) return error.InjectedFinal;
        return .{ .buffer = 6 };
    }
    pub fn finish(o: *@This()) !void {
        if (o.fail == 4) return error.InjectedCompletion;
    }
    pub fn release(o: *@This(), _: Ref) void {
        o.releases += 1;
    }
};
test "H3 schedule replacement faults leave all live tables and metadata unchanged for retry" {
    for ([_]u8{ 1, 2, 3, 4 }) |failure| {
        var m: Model = .{};
        var op: Op = .{ .fail = failure };
        const result = replace.run(&m, &op);
        if (result) |_| return error.TestUnexpectedResult else |_| {}
        try std.testing.expectEqual(@as(u32, 1), m.blocks[0].tables.buffer);
        try std.testing.expectEqual(@as(u32, 2), m.blocks[1].tables.buffer);
        try std.testing.expectEqual(@as(u32, 3), m.final_table.buffer);
        try std.testing.expectEqualSlices(i32, &.{7}, m.adaln);
        try std.testing.expectEqualSlices(i32, &.{8}, m.times);
        try std.testing.expectEqual(@as(usize, 9), m.lines);
        try std.testing.expectEqual(@as(usize, failure - 1), op.releases);
        op.releases = 0;
        op.fail = 0;
        try std.testing.expectEqual(@as(u32, 42), try replace.run(&m, &op));
        try std.testing.expectEqual(@as(u32, 4), m.blocks[0].tables.buffer);
        try std.testing.expectEqual(@as(u32, 5), m.blocks[1].tables.buffer);
        try std.testing.expectEqual(@as(u32, 6), m.final_table.buffer);
        try std.testing.expectEqual(@as(usize, 3), op.releases);
        try std.testing.expectEqualSlices(i32, &.{10}, m.adaln);
        try std.testing.expectEqualSlices(i32, &.{11}, m.times);
        try std.testing.expectEqual(@as(usize, 12), m.lines);
    }
}
