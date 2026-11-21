const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");
const fake = @import("fake.zig");
const t = std.testing;

test "window positions reach the u64 boundary for pending-only, held and host chains" {
    const a = t.allocator;
    var cfg = try @import("config.zig").Config.init(a, .{ .exact_width = 32, .drafts = 31 }, 32, 31);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try Stream.init(a, .{ .id = "positions", .prompt = &.{1}, .max_new = 64 });
    defer s.deinit(a);
    s.pending = 7;
    const tokens: [31]u32 = @splat(9);
    for ([_]usize{ 1, 32 }) |rows| {
        for ([_]u64{ 0, std.math.maxInt(u64) - rows }) |position| {
            for ([_]bool{ false, true }) |held| {
                try t.expect(e.arena.reset(.retain_capacity));
                const p: win.Plan = .{ .stream = &s, .position = position, .kind = .head, .held = if (held) @intCast(rows - 1) else 0, .tokens = if (held) &.{} else tokens[0 .. rows - 1] };
                const w = try win.build(&e, p, held);
                try t.expectEqual(rows, w.rows());
                try t.expect(w.parents == null);
                for (w.positions, 0..) |actual, row| try t.expectEqual(position + 1 + row, actual);
            }
        }
    }
}

test "window tree depth survives sibling branches and invalid parents before a chain" {
    const a = t.allocator;
    var cfg = try @import("config.zig").Config.init(a, .{ .exact_width = 8, .drafts = 7 }, 8, 7);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try Stream.init(a, .{ .id = "tree-positions", .prompt = &.{1}, .max_new = 64 });
    defer s.deinit(a);
    s.pending = 7;
    const end = std.math.maxInt(u64);
    var p: win.Plan = .{ .stream = &s, .position = end - 4, .kind = .head, .tokens = &.{ 8, 9, 10, 11, 12 }, .parents = &.{ -1, -1, 0, 1, 2 } };
    const tree = try win.build(&e, p, false);
    try t.expectEqualSlices(i32, &.{ -1, 0, 0, 1, 2, 3 }, tree.parents.?);
    try t.expectEqualSlices(u64, &.{ end - 3, end - 2, end - 2, end - 1, end - 1, end }, tree.positions);
    for ([_][]const i32{ &.{0}, &.{1}, &.{ -1, 1 } }) |parents| {
        p.parents = parents;
        p.tokens = &.{};
        p.held = @intCast(parents.len);
        try t.expectError(error.ParentAfterChild, win.build(&e, p, false));
    }
    p.parents = null;
    p.held = 3;
    const chain = try win.build(&e, p, true);
    try t.expect(chain.parents == null);
    try t.expectEqualSlices(u64, &.{ end - 3, end - 2, end - 1, end }, chain.positions);
    try t.expectEqualSlices(u64, &.{ end - 3, end - 2, end - 2, end - 1, end - 1, end }, tree.positions);
}
