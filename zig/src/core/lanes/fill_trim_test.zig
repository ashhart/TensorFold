//! The family trim path must keep the same held prefixes and graft metadata as the common allocator.
const std = @import("std");
const win = @import("windows.zig");
const sm = @import("stream.zig");
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;
const fake = @import("fake.zig");
const trim = @import("trim.zig");
const a = std.testing.allocator;
const Policy = struct {
    count: u32,
    nodes: ?[]const u32 = null,
    fn choose(ptr: *anyopaque, _: []const trim.Window, _: bool, _: std.mem.Allocator, out: []trim.Choice) !void {
        const p: *Policy = @ptrCast(@alignCast(ptr));
        for (out) |*o| o.* = .{ .count = p.count, .nodes = p.nodes };
    }
    fn commit(_: *anyopaque, _: trim.Commit) !void {}
    fn rule(p: *Policy) trim.Trim {
        return .{ .ptr = p, .vtable = &.{ .choose = choose, .commit = commit } };
    }
};
test "graft trim: family prefix policies keep held and host trees at every count including zero" {
    var cfg = try Config.init(a, .{ .exact_width = 16, .mtp = true, .drafts = 6, .hidden_rows = true }, 16, 15);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(a, .{ .id = "trim", .prompt = &.{ 3, 1, 4 }, .max_new = 64 });
    defer s.deinit(a);
    s.pending = 4;
    const parents = [_]i32{ -1, 0, 1, 2, 0, 4 };
    const tokens = [_]u32{ 4, 1, 5, 7, 8, 9 };
    for ([_]u32{ 3, 0 }) |held| for ([_]usize{ 1, 2, 3, 4, 5, 6, 0 }) |count| {
        var policy: Policy = .{ .count = @intCast(count) };
        engine.trim = policy.rule();
        var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .held = held, .tokens = tokens[held..], .parents = &parents, .branch_from = 4, .tried = true }};
        try win.allocate(&engine, &plans);
        const p = plans[0];
        try std.testing.expectEqual(count, p.count());
        try std.testing.expectEqual(@min(held, count), p.held);
        try std.testing.expectEqualSlices(u32, tokens[held .. held + count - @min(held, count)], p.tokens);
        if (count > 0) try std.testing.expectEqualSlices(i32, parents[0..count], p.parents.?);
        try std.testing.expectEqual(@as(u32, if (count >= 4) 4 else 0), p.branch_from);
        try std.testing.expect(p.tried);
        const built = try win.build(&engine, p, false);
        try std.testing.expectEqual(count + 1, built.rows());
    };
}
test "graft trim: node selection reindexes the first graft and Recall flags" {
    var cfg = try Config.init(a, .{ .exact_width = 16, .mtp = true, .drafts = 6, .hidden_rows = true }, 16, 15);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(a, .{ .id = "selected", .prompt = &.{ 3, 1, 4 }, .max_new = 64 });
    defer s.deinit(a);
    var policy: Policy = .{ .count = 3, .nodes = &.{ 0, 4, 5 } };
    engine.trim = policy.rule();
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .tokens = &.{ 4, 1, 5, 7, 8, 9 }, .parents = &.{ -1, 0, 1, 2, 0, 4 }, .branch_from = 4, .tried = true, .graft_recalled = &.{ false, true, false } }};
    try win.allocate(&engine, &plans);
    const p = plans[0];
    try std.testing.expectEqualSlices(u32, &.{ 4, 8, 9 }, p.tokens);
    try std.testing.expectEqual(@as(u32, 2), p.branch_from);
    try std.testing.expectEqualSlices(bool, &.{ true, false }, p.graft_recalled);
    try std.testing.expect(p.recallRow(2) and !p.recallRow(3));
}
