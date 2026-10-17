//! A calibration times the window a drafted round verifies: the pending token, then drafts that all land.
const std = @import("std");
const window = @This();

/// The head drafts as a round does, then rows 1.. take real text: a looping head chain shares experts, times too cheap.
pub fn prepare(w: anytype, rows: usize) !void {
    try w.restoreHead();
    if (rows < 2) return;
    try w.draft(rows - 1);
    try w.follow(rows - 1);
}

pub fn time(w: anytype, rows: usize) !f64 {
    try w.restore();
    try prepare(w, rows);
    try w.sync();
    try w.start();
    try w.verify(rows);
    return w.elapsed();
}

test "cost timing starts at the pending token, without a serial advance, and its drafts are text that follows" {
    const Fake = struct {
        pos: usize = 16,
        prepared: bool = false,
        inputs: [16]u32 = undefined,
        fn restore(w: *@This()) !void {
            w.pos = 16;
            w.prepared = false;
        }
        fn advance(w: *@This()) !void {
            w.pos += 1;
        }
        fn restoreHead(w: *@This()) !void {
            w.prepared = true;
        }
        fn draft(w: *@This(), n: usize) !void {
            for (1..n + 1) |i| w.inputs[i] = @intCast(400 + i);
        }
        fn follow(w: *@This(), n: usize) !void {
            for (1..n + 1) |i| w.inputs[i] = @intCast(500 + i);
        }
        fn sync(_: *@This()) !void {}
        fn start(_: *@This()) !void {}
        fn verify(w: *@This(), _: usize) !void {
            w.inputs[0] = if (w.prepared) 101 else 303;
        }
        fn elapsed(_: *@This()) !f64 {
            return 1.0;
        }
    };
    for ([_]usize{ 1, 4 }) |rows| {
        var fake: Fake = .{};
        _ = try time(&fake, rows);
        try std.testing.expectEqual(@as(usize, 16), fake.pos); // the decoder has not consumed its pending token yet
        try std.testing.expectEqual(@as(u32, 101), fake.inputs[0]);
        for (1..rows) |i| try std.testing.expectEqual(@as(u32, @intCast(500 + i)), fake.inputs[i]);
    }
}

test "a prose round's seventh level is priced on decoding's windows, not on the head's looping chain" {
    const core = @import("core");
    const rule = @import("lanes").cost_rule;
    // GB10 ms by width: windows over the head's looping chain, and the median windows decoding verified
    const chain = [16]f64{ 9.5155, 11.7257, 13.5219, 15.3061, 17.0903, 18.2354, 19.3805, 20.5256, 21.6707, 22.8158, 23.9609, 25.106, 26.2511, 27.0797, 27.7831, 28.13 };
    const decoding = [16]f64{ 9.6113, 12.495, 14.389, 16.212, 17.975, 19.813, 21.597, 23.289, 24.878, 26.896, 27.774, 28.584, 29.828, 31.072, 32.317, 33.561 };
    const Gpu = struct {
        rows: usize = 0,
        text: bool = false,
        fn restore(g: *@This()) !void {
            g.text = false;
        }
        fn restoreHead(_: *@This()) !void {}
        fn draft(_: *@This(), _: usize) !void {}
        fn follow(g: *@This(), _: usize) !void {
            g.text = true;
        }
        fn sync(_: *@This()) !void {}
        fn start(_: *@This()) !void {}
        fn verify(g: *@This(), rows: usize) !void {
            g.rows = rows;
        }
        fn elapsed(g: *@This()) !f64 {
            return (if (g.text) decoding else chain)[g.rows - 1];
        }
        pub fn time(g: @This(), i: usize) !f64 {
            var x = g;
            return window.time(&x, i + 1);
        }
    };
    const Levels = struct {
        pub fn afterWidth(_: *@This(), _: usize) !void {}
    };
    var ms: [17]f64 = @splat(0);
    var levels: Levels = .{};
    try rule.measureWidths(Gpu{}, &levels, ms[1..], false);
    try rule.smooth(Gpu{}, ms[1..]);
    var r = @import("cuda_drafts.zig").depthRule(core.draft_depth.Costs.measured(&ms, 0.7108, 16));
    for ([_][3]usize{ .{ 2, 3, 3 }, .{ 4, 11, 10 }, .{ 2, 2, 2 }, .{ 4, 4, 3 } }) |x| r.done(x[0], x[1], x[2]); // rounds 0-3
    var run: f64 = 1.0;
    for ([_]f64{ 0.99943, 0.99219, 0.76497, 0.78981, 0.99808, 0.9435 }) |c| run *= c; // round 4's six drafts
    try std.testing.expect(!r.more(6, run)); // priced on the looping chain's windows, level 7 looks cheap enough to draft
}
