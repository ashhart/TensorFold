//! Eight-stream calibration regression with a deterministic host cost record.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const Config = @import("config.zig").Config;
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const fake = @import("fake.zig");
const win = @import("windows.zig");

const costs = [_]@import("allocate.zig").Cost{
    .{ .width = 1, .ms = 38 },
    .{ .width = 8, .ms = 40 },
    .{ .width = 16, .ms = 42 },
    .{ .width = 32, .ms = 48 },
    .{ .width = 64, .ms = 65 },
    .{ .width = 68, .ms = 67 },
    .{ .width = 128, .ms = 120 },
};
const chances = [_]f64{ 0.9, 0.75, 0.6, 0.48, 0.39, 0.32, 0.26, 0.24, 0.17, 0.14, 0.115, 0.095, 0.08, 0.06, 0.04 };
const heads = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 };
const tokens: [15]u32 = @splat(7);
const want = [_]i64{ 8, 8, 8, 8, 7, 7, 7, 7 }; // 60 drafts over eight streams: 7.5 each

fn probabilities(_: *anyopaque, _: *Stream, out: []f64) !bool {
    @memcpy(out, chances[0..out.len]);
    return true;
}

fn check(e: *Engine, plans: *[8]win.Plan, streams: *[8]Stream) !void {
    var budgets: [8]i64 = undefined;
    try @import("shared_drafts.zig").budgets(e, plans, &heads, &budgets);
    try t.expectEqualSlices(i64, &want, &budgets);
    for (plans, streams) |*p, *s| p.* = .{ .stream = s, .position = 1, .kind = .head, .tokens = &tokens };
    try win.allocate(e, plans);
    for (plans, want) |p, n| try t.expectEqual(@as(usize, @intCast(n)), p.count());
}

test "shared combined prices: eight streams keep 7.5 drafts across calibration of a fixed block cost" {
    for ([_]bool{ false, true }) |sampled| {
        var cfg = try Config.init(a, .{ .fixed_draft_block = true, .exact_width = 64, .mtp = true, .drafts = 15, .draft_probabilities = true, .allocate_shared_drafts = true, .draft_streams = true, .mtp_step_ms = 1, .copy_drafts = false, .plain_guard = true, .hidden_rows = true, .batch_rows = 128, .window_costs = &costs }, 64, 63);
        defer cfg.deinit(a);
        var target: fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var backend = target.backend();
        var vtable = backend.vtable.*;
        vtable.probabilities = probabilities;
        backend.vtable = &vtable;
        var e = Engine.init(a, &cfg, backend, clock.clock());
        defer e.deinit();
        var streams: [8]Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(a);
        var plans: [8]win.Plan = undefined;
        for (&streams, &plans) |*s, *p| {
            s.* = try Stream.init(a, .{ .id = "eight", .prompt = &.{ 1, 2, 3 }, .max_new = 1000, .sampling = if (sampled) .{ .seed = 19, .temperature = 1 } else null });
            made += 1;
            s.rounds = 1;
            try s.draft_chances.appendSlice(a, &chances);
            p.* = .{ .stream = s, .position = 1, .kind = .none };
        }
        for (1..129) |round| {
            const trial = try @import("shared_price.zig").calibrate(&e, &plans);
            try t.expectEqual(round == 64 or round == 128, trial);
            // Recorded host costs: 67 forward + 40 fixed draft work; plain is 40 forward + 2 overhead.
            try e.rule.observeShared(8, if (trial) 8 else 68, if (trial) 42 else 107, sampled, if (trial) 0 else 8, trial, trial);
            if (round == 63 or round == 64) {
                try t.expect(e.rule.sharedPrice(8, sampled) == null);
                try check(&e, &plans, &streams);
            }
        }
        try check(&e, &plans, &streams);
        const p = e.rule.sharedPrice(8, sampled).?;
        try t.expectApproxEqAbs(@as(f64, 2), p.plain, 1e-9);
        try t.expectApproxEqAbs(@as(f64, 0), p.row, 1e-9);
        // More plain observations cannot make the positive drafted options forget the block's fixed work.
        for (0..32) |_| try e.rule.observeShared(8, 8, 42, sampled, 0, true, false);
        try check(&e, &plans, &streams);
    }
}
