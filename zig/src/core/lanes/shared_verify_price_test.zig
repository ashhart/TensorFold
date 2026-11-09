//! Width selection spends rows on drafts already held; producing the next drafts is priced separately.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const Config = @import("config.zig").Config;
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const fake = @import("fake.zig");
const win = @import("windows.zig");

fn held(step_ms: f64, observed_head: f64, residual: f64, chance: f64) ![2]usize {
    var cfg = try Config.init(a, .{ .exact_width = 4, .mtp = true, .drafts = 1, .draft_prior = &.{chance}, .mtp_step_ms = step_ms, .plain_guard = true, .hidden_rows = true, .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 4, .ms = 10 } } }, 4, 3);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    try e.rule.observeShared(2, 2, 12, false, 0, true, false);
    try e.rule.observeShared(2, 4, 12 + residual + observed_head, false, observed_head, false, false);
    var streams: [2]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var plans: [2]win.Plan = undefined;
    for (&streams, &plans) |*s, *p| {
        s.* = try Stream.init(a, .{ .id = "held", .prompt = &.{1}, .max_new = 64 });
        made += 1;
        p.* = .{ .stream = s, .position = 1, .kind = .head, .tokens = &.{7} };
    }
    try win.allocate(&e, &plans);
    return .{ plans[0].count(), plans[1].count() };
}

test "shared prices: verify never pays another step for an existing draft" {
    try t.expectEqual([2]usize{ 1, 1 }, try held(3, 0, 1, 0.5));
    try t.expectEqual([2]usize{ 1, 1 }, try held(300, 0, 1, 0.5));
}

test "shared prices: each verify option retains the same measured head fixed cost" {
    try t.expectEqual([2]usize{ 0, 0 }, try held(0, 0, 8, 0.4));
    try t.expectEqual([2]usize{ 1, 1 }, try held(0, 200, 8, 0.4));
}
