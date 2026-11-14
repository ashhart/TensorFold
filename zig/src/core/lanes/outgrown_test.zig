//! Backend.grow: a stream whose next rows find no memory ends with `length`; the round's other streams run on.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const be = @import("backend.zig");
const fake = @import("fake.zig");
const model = @import("engine_test.zig").modelWith;
const t = std.testing;
const a = t.allocator;

var refuse: ?[]const u8 = null; // the stream id whose rows find no memory once it has emitted `after` tokens
var after: usize = 0;
var asked: usize = 0;

fn grow(_: *anyopaque, w: be.Window) bool {
    asked += 1;
    const id = refuse orelse return true;
    return !(std.mem.eql(u8, w.stream.id, id) and w.stream.emitted().len >= after);
}

const Run = struct {
    tokens: [2][]u32,
    reasons: [2]@import("stream.zig").Reason,
    problems: [2][]const u8,
    min_rows: [2]u32,
    steps: u64,
    shared: u64,
    fn deinit(r: Run) void {
        for (r.tokens) |x| a.free(x);
    }
};

/// Streams "a" (24 tokens) and "b" (40), from `first`, `count` of them, to their ends; `with_grow` sets the hook.
fn run(with_grow: bool, first: usize, count: usize) !Run {
    var cfg = try model(false);
    defer cfg.deinit(a);
    cfg.pipelined = false;
    cfg.timed = false;
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var vt = target.backend().vtable.*;
    if (with_grow) vt.grow = grow;
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, .{ .ptr = &target, .vtable = &vt, .chunk_invariant = true }, clock.clock());
    defer e.deinit();
    var streams: [2]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    const ids = [_][]const u8{ "a", "b" };
    const lens = [_]u32{ 24, 40 };
    for (streams[0..count], ids[first..][0..count], lens[first..][0..count]) |*s, id, n| {
        s.* = try Stream.init(a, .{ .id = id, .prompt = &.{ 3, 1, 4 }, .max_new = n });
        made += 1;
        try e.addStream(s);
    }
    var guard: usize = 0;
    while (guard < 200) : (guard += 1) {
        var live = false;
        for (streams[0..count]) |s| live = live or !s.finished;
        if (!live) break;
        try e.step();
    }
    var r: Run = .{ .tokens = .{ &.{}, &.{} }, .reasons = .{ .none, .none }, .problems = .{ "", "" }, .min_rows = .{ 0, 0 }, .steps = e.steps, .shared = e.shared_rounds };
    for (streams[0..count], 0..) |s, i| {
        r.tokens[i] = try a.dupe(u32, s.emitted());
        r.reasons[i] = s.reason;
        r.problems[i] = s.problem;
        r.min_rows[i] = s.min_rows;
    }
    return r;
}

test "a stream whose rows find no memory in a shared round ends with length; the other gives the tokens it gives alone" {
    refuse = null;
    const alone = try run(false, 1, 1);
    defer alone.deinit();
    refuse = "a";
    after = 5;
    defer refuse = null;
    const cut = try run(true, 0, 2);
    defer cut.deinit();
    try t.expect(cut.shared > 0); // "a" and "b" shared rounds before "a" ran out
    try t.expectEqual(@as(@TypeOf(cut.reasons[0]), .length), cut.reasons[0]);
    try t.expectEqualStrings("the server had no KV memory left for this reply's next rows", cut.problems[0]);
    try t.expectEqual(@as(usize, after), cut.tokens[0].len); // what it had when its rows found none
    try t.expectEqual(alone.reasons[0], cut.reasons[1]);
    try t.expectEqualSlices(u32, alone.tokens[0], cut.tokens[1]);
}

test "a lone stream whose rows find no memory ends with length before its round" {
    refuse = "a";
    after = 3;
    defer refuse = null;
    const cut = try run(true, 0, 1);
    defer cut.deinit();
    try t.expectEqual(@as(@TypeOf(cut.reasons[0]), .length), cut.reasons[0]);
    try t.expectEqual(@as(usize, 3), cut.tokens[0].len);
    try t.expectEqual(@as(u32, 1), cut.min_rows[0]); // the cut round ran no rows: min_rows keeps its rounds' 1, not 0
}

test "a backend without grow, or one that always holds the rows, runs exactly as before" {
    refuse = null;
    asked = 0;
    const plain = try run(false, 0, 2);
    defer plain.deinit();
    try t.expectEqual(@as(usize, 0), asked);
    const held = try run(true, 0, 2);
    defer held.deinit();
    try t.expect(asked > 0);
    for (0..2) |i| {
        try t.expectEqualSlices(u32, plain.tokens[i], held.tokens[i]);
        try t.expectEqual(plain.reasons[i], held.reasons[i]);
    }
    try t.expectEqual(plain.steps, held.steps);
    try t.expectEqual(plain.shared, held.shared);
}
