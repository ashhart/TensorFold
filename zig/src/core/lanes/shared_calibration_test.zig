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

test "shared prices: a missing forward width cannot repeat a calibration or stop its clock" {
    var cfg = try Config.init(a, .{ .exact_width = 4, .mtp = true, .drafts = 1, .plain_guard = true, .hidden_rows = true, .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 4, .ms = 10 } } }, 4, 3);
    defer cfg.deinit(a);
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
    var trials: usize = 0;
    for (1..193) |round| {
        const due = try prices.calibrate(&e, &plans);
        try t.expectEqual(round % 64 == 0, due);
        if (due) {
            trials += 1;
            try t.expect(!try prices.calibrate(&e, &plans));
        }
        try e.rule.observeShared(2, 2, 12, false, 0, true, due);
        try t.expect(e.rule.sharedPrice(2, false) == null);
    }
    try t.expectEqual(@as(usize, 3), trials);
}
