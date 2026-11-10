//! Calibration spacing follows executed trials even where the exact forward width has no price.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const prices = @import("shared_price.zig");
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;
const fake = @import("fake.zig");
const win = @import("windows.zig");

/// 192 rounds of two streams, the two-row forward priced from round `priced_from` on (null: never).
fn calibrations(priced_from: ?usize, want: []const usize, priced_at: ?usize) !void {
    var cfg = try Config.init(a, .{ .exact_width = 4, .mtp = true, .drafts = 1, .plain_guard = true, .hidden_rows = true, .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 4, .ms = 10 } } }, 4, 3);
    defer cfg.deinit(a);
    const two = cfg.shared_costs.slots[2];
    cfg.shared_costs.slots[2] = null;
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var streams: [2]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var plans: [2]win.Plan = undefined;
    for (&streams, &plans) |*s, *p| {
        s.* = try Stream.init(a, .{ .id = "calibration", .prompt = &.{1}, .max_new = 1000 });
        made += 1;
        s.rounds = 1;
        p.* = .{ .stream = s, .position = 1, .kind = .none };
    }
    var trials: [4]usize = undefined;
    var n: usize = 0;
    for (1..193) |round| {
        if (priced_from == round) cfg.shared_costs.slots[2] = two;
        const due = try prices.calibrate(&e, &plans);
        if (due) {
            trials[n] = round;
            n += 1;
            try t.expect(!try prices.calibrate(&e, &plans));
        }
        // between trials, drafted work at plain width: it never moves the plain point
        try e.rule.observeShared(2, 2, 12, false, 0, due, due);
        try t.expectEqual(if (priced_at) |at| round >= at else false, e.rule.sharedPrice(2, false) != null);
    }
    try t.expectEqualSlices(usize, want, trials[0..n]);
}

test "shared prices: a missing forward width cannot repeat a calibration or stop its clock" {
    try calibrations(null, &.{ 64, 128, 192 }, null);
}

test "shared prices: a count's first plain trial warms it up, measured or not, and the next one prices it" {
    try calibrations(1, &.{ 64, 128 }, 128);
    try calibrations(100, &.{ 64, 128 }, 128);
}
