//! A drafter's timed hold pass under a legacy cap (limit_shared_drafts): the rule keeps its holds while they pay.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const Config = @import("config.zig").Config;
const HoldCost = @import("hold_price.zig").HoldCost;
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const fake = @import("fake.zig");
const win = @import("windows.zig");

/// Four drafting streams, three drafts each at 0.8 a depth; `holds` the drafter's pass. Each stream's reserved depth.
fn round(holds: []const HoldCost) ![4]i64 {
    // forward 12 ms a row plus about 1 ms a row more; plain 4 rows ~15 ms, 16 drafted rows ~27.5 ms, fixed 8 ms
    const model: @import("config.zig").Model = .{ .exact_width = 32, .mtp = true, .drafts = 3, .draft_prior = &.{0.8}, .plain_guard = true, .hidden_rows = true, .allocate_shared_drafts = true, .limit_shared_drafts = true, .hold_costs = holds, .window_costs = &.{ .{ .width = 1, .ms = 12 }, .{ .width = 32, .ms = 44 } } };
    var cfg = try Config.init(a, model, 32, 3);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var streams: [4]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var plans: [4]win.Plan = undefined;
    for (&streams, &plans) |*s, *p| {
        s.* = try Stream.init(a, .{ .id = "pass", .prompt = &.{1}, .max_new = 64 });
        made += 1;
        p.* = .{ .stream = s, .position = 1, .kind = .none };
    }
    var out: [4]i64 = undefined;
    try @import("shared_drafts.zig").budgets(&e, &plans, &.{ 0, 1, 2, 3 }, &out);
    return out;
}

// Break-even: 4 tokens / 23.1 ms plain vs 11.8 / (35.5 ms + pass); the pass may cost up to ~32.7 ms at 4 streams.
const cheap = [_]HoldCost{ .{ .streams = 1, .depth = 1, .ms = 0.5 }, .{ .streams = 1, .depth = 3, .ms = 1 }, .{ .streams = 4, .depth = 1, .ms = 1 }, .{ .streams = 4, .depth = 3, .ms = 2 } };
const dear = [_]HoldCost{ .{ .streams = 1, .depth = 1, .ms = 20 }, .{ .streams = 1, .depth = 3, .ms = 30 }, .{ .streams = 4, .depth = 1, .ms = 40 }, .{ .streams = 4, .depth = 3, .ms = 60 } };

test "hold pass: a pass cheaper than the drafts it buys keeps every stream's drafter depth" {
    try t.expectEqualSlices(i64, &.{ 3, 3, 3, 3 }, &try round(&cheap));
}

test "hold pass: a pass dearer than the drafts it buys falls back to copies (no drafter hold)" {
    try t.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, &try round(&dear));
}

test "hold pass: no timed pass leaves the legacy cap as it was (the dear case's control)" {
    try t.expectEqualSlices(i64, &.{ 3, 3, 3, 3 }, &try round(&.{}));
}
