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
