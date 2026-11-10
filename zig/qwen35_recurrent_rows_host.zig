//! The served lane planner reaches the backend within the 128-row frame and 64-row per-stream contracts.
const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
const Mode = enum { plain, forced, copied, drafted, grammar };
const Probe = struct {
    var inner: @FieldType(lanes.backend.Backend.VTable, "verify") = undefined;
    var calls: usize = 0;
    var inner_keep: @FieldType(lanes.backend.Backend.VTable, "keep") = undefined;
    var keeps: usize = 0;
    var most: usize = 0;
    var widest: usize = 0;
    fn verify(p: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) !void {
        var rows: usize = 0;
        for (windows) |window| {
            try std.testing.expect(window.rows() > 0 and window.rows() <= 64);
            widest = @max(widest, window.rows());
            rows += window.rows();
        }
        try std.testing.expect(rows <= 128);
        most = @max(most, rows);
        calls += 1;
        try inner(p, windows, out);
    }
    fn keep(p: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) !void {
        keeps += 1;
        try inner_keep(p, windows, paths);
    }
};
const Copy = struct {
    bytes: [63]u32 = @splat(1),
    fn propose(p: *anyopaque, _: []const u32, maximum: i64) ![]const u32 {
        const c: *Copy = @ptrCast(@alignCast(p));
        return c.bytes[0..@min(c.bytes.len, @as(usize, @intCast(@max(0, maximum))))];
    }
    fn matched(_: *anyopaque) i64 {
        return 100;
    }
    fn handle(c: *Copy) lanes.proposer.Proposer {
        return .{ .ptr = c, .vtable = &.{ .propose = propose, .last_match = matched } };
    }
};
const Matcher = struct {
    tag: u8 = 0,
    fn accept(_: *anyopaque, _: u32) !bool {
        return true;
    }
    fn rollback(_: *anyopaque, _: usize) !void {}
    fn terminated(_: *anyopaque) bool {
        return false;
    }
    fn fill(_: *anyopaque, words: []u32) !void {
        @memset(words, std.math.maxInt(u32));
    }
    fn drafts(_: *anyopaque, out: []u32) !usize {
        @memset(out, 1);
        return out.len;
    }
    fn free(_: *anyopaque) void {}
    fn rules(m: *Matcher) lanes.grammar.Rules {
        return .{ .ptr = m, .parallel_walk = true, .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free } };
    }
};

const Result = struct { keeps: usize, verifies: usize, hash: u64 };
fn run(n: usize, mode: Mode, full_keep: bool) !Result {
    var costs: [64]lanes.config.Cost = undefined;
    for (&costs, 1..) |*cost, width| cost.* = .{ .width = @intCast(width), .ms = 1 };
    var config = try lanes.Config.init(a, .{ .exact_width = 64, .mtp = true, .speculate = true, .drafts = 63, .hidden_rows = true, .streams_exact = true, .max_streams = 16, .batch_rows = 128, .draft_streams = true, .window_costs = &costs }, 64, 63);
    defer config.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .head_accepts = 63 };
    defer target.deinit();
    var backend = target.backend();
    var vtable = backend.vtable.*;
    Probe.inner = vtable.verify;
    Probe.inner_keep = vtable.keep;
    Probe.calls = 0;
    Probe.keeps = 0;
    Probe.most = 0;
    Probe.widest = 0;
    vtable.verify = Probe.verify;
    vtable.keep = Probe.keep;
    backend.vtable = &vtable;
    backend.needs_full_keep = full_keep;
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &config, backend, clock.clock());
    defer engine.deinit();
    var streams: [16]lanes.Stream = undefined;
    var copies: [16]Copy = @splat(.{});
    var matchers: [16]Matcher = @splat(.{});
    var constraints: [16]lanes.grammar.Constraint = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| {
        if (!s.finished) backend.release(s);
        s.deinit(a);
    };
    for (streams[0..n], 0..) |*s, i| {
        constraints[i] = .init(matchers[i].rules(), 4, null);
        s.* = try lanes.Stream.init(a, .{ .id = "bounded", .prompt = &.{ 3, 1, 4 }, .max_new = 80, .drafts = mode != .plain, .proposer = if (mode == .copied) copies[i].handle() else null, .grammar = if (mode == .grammar) &constraints[i] else null });
        made += 1;
        try engine.addStream(s);
        if (mode == .forced) try s.force.appendNTimes(a, 1, 79);
    }
    var steps: usize = 0;
    while (engine.activeCount() > 0) : (steps += 1) {
        try std.testing.expect(steps < 512);
        try engine.step();
    }
    try std.testing.expect(Probe.calls > 0);
    if (mode != .plain) try std.testing.expect(Probe.widest > 1);
    if (mode == .forced and n > 1) try std.testing.expectEqual(@as(usize, 128), Probe.most);
    var hash = std.hash.Wyhash.init(0);
    for (streams[0..made]) |*s| hash.update(std.mem.sliceAsBytes(s.emitted()));
    return .{ .keeps = Probe.keeps, .verifies = Probe.calls, .hash = hash.final() };
}

test "Shared verification: recurrent budget: actual core plans bound plain forced copied held and grammar windows" {
    for ([_]usize{ 1, 4, 8, 16 }) |n| for (std.enums.values(Mode)) |mode| {
        _ = try run(n, mode, true);
    };
}

test "Shared verification: recurrent budget: explicit full-keep preserves tokens and commits every lone verify" {
    const old = try run(1, .plain, false);
    const full = try run(1, .plain, true);
    try std.testing.expectEqual(old.hash, full.hash);
    try std.testing.expect(old.keeps < old.verifies);
    try std.testing.expectEqual(full.verifies, full.keeps);
}

test "Shared verification: recurrent budget: draft and recording wrappers retain full-keep policy without enabling it by default" {
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    try std.testing.expect(!target.backend().needs_full_keep);
    for ([_]bool{ false, true }) |needed| {
        var backend = target.backend();
        backend.needs_full_keep = needed;
        var wrapped: lanes.drafted.Drafted = .{ .gpa = a, .target = backend, .drafter = undefined };
        wrapped.drafter.proposal_head = null;
        try std.testing.expectEqual(needed, wrapped.backend().needs_full_keep);
        var recorded: lanes.recorder.Recorded = .{ .inner = backend, .recorder = undefined };
        try std.testing.expectEqual(needed, recorded.backend().needs_full_keep);
    }
}
