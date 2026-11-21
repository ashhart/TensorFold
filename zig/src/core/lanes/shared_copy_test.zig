//! Shared-round copy widths: first_copy as before, or with Model.shared_copy_ramp the lone ramp to a stream's share.
const std = @import("std");
const cfg = @import("config.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const fake = @import("fake.zig");
const Suffix = @import("proposer.zig").SuffixLookup;
const a = std.testing.allocator;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// The fake target's reply (its pattern): forty distinct ids, so a copy can run past three; the prompt is it twice.
const reply: [40]u32 = blk: {
    var r: [40]u32 = undefined;
    for (&r, 0..) |*t, i| t.* = @intCast(20 + i);
    break :blk r;
};
const prompt = reply ++ reply;

/// From step `at` on, the target replies with `pattern` instead, so the copies it holds miss.
const Turn = struct { at: usize, pattern: []const u32 };

fn model(ramp: bool, shared: bool) cfg.Model {
    return .{ .exact_width = 32, .first_copy_rows = 4, .batch_rows = 32, .hidden_rows = shared, .copy_match = 4, .shared_copy_ramp = ramp };
}

/// Run a stream for each of `copying` (true: a suffix proposer) and note each one's window rows a step in `rows`.
fn run(m: cfg.Model, copying: []const bool, rows: [][8]usize, turn: ?Turn) !void {
    var conf = try cfg.Config.init(a, m, 32, 31);
    defer conf.deinit(a);
    var target: fake.Fake = .{ .gpa = a, .probe_prompt = prompt.len, .cycle_after = 0, .pattern = &reply };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var core = Engine.init(a, &conf, target.backend(), clock.clock());
    defer core.deinit();
    const n = copying.len;
    var suffixes: [8]Suffix = undefined;
    var streams: [8]Stream = undefined;
    for (suffixes[0..n], streams[0..n], copying) |*p, *s, copy| {
        p.* = try Suffix.init(a, .{ .min_match = 4 });
        s.* = try Stream.init(a, .{ .id = "copy", .prompt = &prompt, .max_new = 400, .proposer = if (copy) p.proposer() else null });
    }
    defer for (suffixes[0..n], streams[0..n]) |*p, *s| {
        s.deinit(a);
        p.deinit();
    };
    for (streams[0..n]) |*s| try core.addStream(s);
    var before: [8]usize = @splat(std.math.maxInt(usize));
    for (rows, 0..) |*step, i| {
        if (turn) |t| if (i == t.at) {
            target.pattern = t.pattern;
            for (before[0..n], streams[0..n]) |*b, *s| b.* = s.emitted().len;
        };
        step.* = @splat(0);
        try core.step();
        for (step[0..n], streams[0..n]) |*r, *s| r.* = target.lanes.get(s).?.rows.items.len;
    }
    // every token kept is the target's own reply, so a missed copy's rows were rolled back
    for (streams[0..n], before[0..n]) |*s, b| for (s.emitted(), 0..) |t, k| {
        const want = if (k < b) reply[k % reply.len] else turn.?.pattern[k % turn.?.pattern.len];
        try expectEqual(want, t);
    };
}

test "shared copies: without the ramp a shared round's copy keeps first_copy's four rows" {
    var rows: [6][8]usize = undefined;
    try run(model(false, true), &.{ true, true }, &rows, null);
    for (rows) |step| for (step[0..2]) |r| try expect(r <= 4);
    try expectEqual(@as(usize, 4), rows[0][0]);
}

test "shared copies: the ramp grows two streams' copies to their share and starts over after a miss" {
    const other = [_]u32{ 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75 };
    var rows: [40][8]usize = undefined;
    try run(model(true, true), &.{ true, true }, &rows, .{ .at = 5, .pattern = &other });
    for (0..2) |i| {
        // 32 rows shared by two streams: 16 a window, its copy 15; step 5's copy misses the new reply
        for ([_]usize{ 4, 8, 16, 16, 16, 16 }, 0..) |want, j| try expectEqual(want, rows[j][i]);
        var j: usize = 6;
        while (j < rows.len and rows[j][i] == 1) j += 1;
        try expect(j + 3 <= rows.len);
        for ([_]usize{ 4, 8, 16 }, j..) |want, k| try expectEqual(want, rows[k][i]);
    }
}

test "shared copies: eight streams keep four-row copies, and so do streams that share no round" {
    var rows: [6][8]usize = undefined;
    const eight: [8]bool = @splat(true);
    try run(model(true, true), &eight, &rows, null);
    for (rows) |step| for (step) |r| try expect(r <= 4);
    try expectEqual(@as(usize, 4), rows[5][7]);
    var conf = try cfg.Config.init(a, model(true, false), 32, 31);
    defer conf.deinit(a);
    try expect(!conf.family_streams and !conf.shared_copy_ramp);
    try run(model(true, false), &.{ true, true }, &rows, null);
    for (rows) |step| for (step[0..2]) |r| try expect(r <= 4);
    try expectEqual(@as(usize, 4), rows[5][1]);
}

test "shared copies: a copy keeps to its stream's share of the round, leaving the other streams their rows" {
    const copying = [_]bool{ true, false, true, false };
    var rows: [6][8]usize = undefined;
    try run(model(true, true), &copying, &rows, null);
    for (rows) |step| {
        var copied: usize = 0;
        var total: usize = 0;
        for (step[0..4], copying) |r, copy| {
            if (copy) copied += r - 1 else try expectEqual(@as(usize, 1), r);
            total += r;
        }
        try expect(copied <= 2 * (32 / 4 - 1)); // each copy at most its share of the 32 rows less its pending row
        try expect(total <= 32);
    }
    try expectEqual(@as(usize, 8), rows[5][0]);
}
