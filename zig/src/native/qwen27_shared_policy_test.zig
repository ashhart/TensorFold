//! Actual native facts drive a counted shared head through the production lane allocator without a device.
const std = @import("std");
const tf = @import("tensorfold");
const lanes = @import("lanes");
const native = @import("qwen27_lanes.zig");
const a = std.testing.allocator;
const Counted = struct {
    fake: lanes.fake.Fake = .{ .gpa = a },
    nodes: usize = 0,
    requests: usize = 0,

    fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) !void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *Counted = @fieldParentPtr("fake", f);
        for (requests) |r| c.nodes += r.depth;
        c.requests += requests.len;
        try f.backend().vtable.draft(ptr, requests);
    }
};

test "qwen27 grants: native shared proposals do not spend nodes beyond the verified grants" {
    var runner: tf.qwen27.decode_round.Runner = undefined;
    runner.slots = 4;
    var b: native.Metal = undefined;
    b.runner = &runner;
    b.draft = @import("draft_sources_check.zig").present(@TypeOf(b.draft.?));
    b.window = 16;
    b.nodes = 15;
    b.copies = false;
    b.tensor_units = true;
    b.draft_slots = 8;
    b.timed = 64;
    for (&b.costs, 1..) |*cost, rows| cost.* = .{
        .width = @intCast(rows),
        .ms = if (rows <= 16) 1 else 1000,
    };
    var cfg = try lanes.Config.init(a, b.facts(), 16, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var counted: Counted = .{};
    defer counted.fake.deinit();
    var ss: [4]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (ss[0..made]) |*s| s.deinit(a);
    var table = counted.fake.backend().vtable.*;
    table.draft = Counted.draft;
    var backend = counted.fake.backend();
    backend.vtable = &table;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    for (&ss) |*s| {
        s.* = try lanes.Stream.init(a, .{ .id = "granted", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
        made += 1;
        try e.addStream(s);
    }
    counted.nodes = 0;
    counted.requests = 0;
    try e.step();
    var granted: usize = 0;
    for (&ss) |*s| granted += s.granted orelse 0;
    try std.testing.expectEqual(@as(usize, 4), counted.requests);
    try std.testing.expect(granted > 0 and counted.nodes > 0);
    if (counted.nodes > granted) std.debug.print("shared nodes: proposed={d}, granted={d}\n", .{ counted.nodes, granted });
    try std.testing.expect(counted.nodes <= granted);
    while (e.activeCount() > 0) try e.step();
    for (&ss) |*s| {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(a);
        try history.appendSlice(a, &.{ 7, 13, 29 });
        for (s.emitted()) |token| {
            const expected = lanes.fake.next(history.items, null, history.items.len);
            try std.testing.expectEqual(expected, token);
            try history.append(a, expected);
        }
    }
}

test "qwen27 grants: lone landing history cannot reprice shared proposals or their tails" {
    var s = try lanes.Stream.init(a, .{ .id = "lone-history", .prompt = &.{1}, .max_new = 64 });
    defer s.deinit(a);
    var other = try lanes.Stream.init(a, .{ .id = "shared-peer", .prompt = &.{2}, .max_new = 64 });
    defer other.deinit(a);
    var runner: tf.qwen27.decode_round.Runner = undefined;
    runner.slots = 2;
    var b: native.Metal = undefined;
    b.runner = &runner;
    b.owners = @splat(null);
    b.owners[0] = &s;
    b.trees = @splat(null);
    var tokens = [_]u32{ 10, 11, 12 };
    var parents = [_]i32{ -1, 0, 0 };
    var scores = [_]f64{ -0.25, -0.5, -0.75 };
    b.trees[0] = .{ .gpa = a, .tokens = &tokens, .parents = &parents, .scores = &scores };
    b.landing = .{};
    for (0..200) |_| {
        b.landing.observe(false, -0.25, false);
        b.landing.observe(false, -0.5, false);
    }
    const root = @exp(@as(f64, -0.25));
    const expected = [_]f64{ root, root * root, root * @exp(@as(f64, -0.5)) };
    var out: [3]f64 = undefined;
    for ([_]bool{ false, true }) |keyed| {
        s.sampling = if (keyed) .{ .seed = 7 } else null;
        s.shared_round = false;
        b.owners[1] = null;
        try std.testing.expect(try @import("qwen27_drafts.zig").probabilitiesFn(&b, &s, &out));
        try std.testing.expect(out[0] < root / 50);
        s.shared_round = true;
        b.owners[1] = &other;
        try std.testing.expect(try @import("qwen27_drafts.zig").probabilitiesFn(&b, &s, &out));
        if (!std.mem.eql(f64, &expected, &out)) std.debug.print("shared landing inherited a lone proposal's estimates\n", .{});
        try std.testing.expectEqualSlices(f64, &expected, &out);
        b.owners[1] = null; // its last peer ended; this request's proposals still came from shared rounds
        try std.testing.expect(try @import("qwen27_drafts.zig").probabilitiesFn(&b, &s, &out));
        try std.testing.expectEqualSlices(f64, &expected, &out);
        s.shared_round = false;
        try std.testing.expect(try @import("qwen27_drafts.zig").probabilitiesFn(&b, &s, &out));
        try std.testing.expect(out[0] < root / 50);
    }
}
