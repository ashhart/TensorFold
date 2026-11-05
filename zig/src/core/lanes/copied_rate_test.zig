//! Copy windows and grafted rows follow source-specific and trusted-match landing histories.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const win = @import("windows.zig");
const fake = @import("fake.zig");

const gpa = std.testing.allocator;
const copied = [_]u32{ 5, 6, 7, 8, 9, 10, 11 };

/// The copy rows allocation grants each of four streams' seven-row copy windows in one shared round.
fn granted(e: *Engine, streams: []Stream, out: []usize) !void {
    _ = e.arena.reset(.retain_capacity);
    var plans: [4]win.Plan = undefined;
    for (plans[0..streams.len], streams) |*p, *s| p.* = .{ .stream = s, .position = s.context.items.len, .kind = .copy, .tokens = &copied };
    try win.allocate(e, plans[0..streams.len]);
    for (plans[0..streams.len], out) |p, *o| o.* = p.count();
}

test "own_copy_rates: copies that keep breaking are cut, by source; without it every copy is priced at copy_rate" {
    // plain rows 10 ms, 0.6 ms a row past four: seven copy rows a stream pay at the prior, not landing 1 a copy
    var costs: [32]Cost = undefined;
    for (&costs, 1..) |*c, r| c.* = .{ .width = @intCast(r), .ms = 10 + 0.6 * @as(f64, @floatFromInt(@max(r, 4) - 4)) };
    for ([_]bool{ false, true }) |own| {
        var cfg = try Config.init(gpa, .{ .exact_width = 16, .hidden_rows = true, .window_costs = &costs, .batch_rows = 32, .own_copy_rates = own }, 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        var streams: [4]Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(gpa);
        for (&streams) |*s| {
            s.* = try Stream.init(gpa, .{ .id = "copied", .prompt = &.{ 1, 2 }, .max_new = 64 });
            made += 1;
        }
        var rows: [4]usize = undefined;
        try granted(&e, &streams, &rows);
        try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7, 7 }, &rows);
        for (0..8) |_| win.observeCopied(&e, &streams[0], false, 1, 7); // stream 0's context copies land one row of seven
        try granted(&e, &streams, &rows);
        if (own) {
            try std.testing.expect(rows[0] <= 1);
            try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7 }, rows[1..]);
            try std.testing.expect(win.copiedRate(&e, &streams[0], false) < 0.6);
            try std.testing.expectApproxEqAbs(cfg.copy_rate, win.copiedRate(&e, &streams[0], true), 1e-12); // Recall's keep theirs
        } else {
            try std.testing.expectEqualSlices(usize, &.{ 7, 7, 7, 7 }, &rows);
            try std.testing.expectEqual(@as(f64, -1), streams[0].copied_landed[0]); // no history kept
        }
    }
}

test "strong context copies recover independently of earlier weak misses" {
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .own_copy_rates = true, .matched_copy_rates = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try Stream.init(gpa, .{ .id = "strong", .prompt = &.{ 1, 2 }, .max_new = 64 });
    defer s.deinit(gpa);
    for (0..8) |_| win.observePricedCopied(&e, &s, false, false, 0, 4);
    try std.testing.expect(win.pricedCopiedRate(&e, &s, false, false) < 0.5);
    try std.testing.expectApproxEqAbs(cfg.copy_rate, win.pricedCopiedRate(&e, &s, false, true), 1e-12);
    try std.testing.expectApproxEqAbs(cfg.copy_rate, win.pricedCopiedRate(&e, &s, true, true), 1e-12);
    for (0..8) |_| win.observePricedCopied(&e, &s, false, true, 0, 4);
    try std.testing.expect(win.pricedCopiedRate(&e, &s, false, true) < cfg.copy_rate);
    cfg.matched_copy_rates = false;
    try std.testing.expectApproxEqAbs(win.copiedRate(&e, &s, false), win.pricedCopiedRate(&e, &s, false, true), 1e-12);
}

test "a strong match can be allocated after weaker matches failed" {
    var costs: [8]Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 10 + 6 * @as(f64, @floatFromInt(width - 1)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 8, .batch_rows = 8, .window_costs = &costs, .own_copy_rates = true, .matched_copy_rates = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try Stream.init(gpa, .{ .id = "match", .prompt = &.{ 1, 2 }, .max_new = 64 });
    defer s.deinit(gpa);
    for (0..12) |_| win.observePricedCopied(&e, &s, false, false, 0, 4);
    var lookup = try @import("proposer.zig").SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 3 });
    defer lookup.deinit();
    var history: [51]u32 = undefined;
    for (0..25) |i| {
        history[i] = @intCast(i + 1);
        history[26 + i] = @intCast(i + 1);
    }
    history[25] = 123;
    try std.testing.expect((try lookup.propose(&history, 7)).len > 0);
    s.proposer = lookup.proposer();
    try std.testing.expect(win.trustedCopy(&s));
    var plan: [1]win.Plan = .{.{ .stream = &s, .position = 0, .kind = .copy, .tokens = &copied }};
    try win.allocate(&e, &plan);
    try std.testing.expect(plan[0].count() > 0);
    cfg.matched_copy_rates = false;
    _ = e.arena.reset(.retain_capacity);
    plan = .{.{ .stream = &s, .position = 0, .kind = .copy, .tokens = &copied }};
    try win.allocate(&e, &plan);
    try std.testing.expectEqual(@as(usize, 0), plan[0].count());
}

pub fn checkRows(comptime planChances: anytype, comptime short_rate: f64) !void {
    const a = std.testing.allocator;
    var cfg = try Config.init(a, .{ .exact_width = 64, .drafts = 15, .batch_rows = 128 }, 64, 15);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try Stream.init(a, .{ .id = "copy", .prompt = &.{ 1, 2 }, .max_new = 64 });
    defer s.deinit(a);
    // two drafts, then a copy of two rows from the second draft (row parents -1, 0, 1, 2)
    const parents = [_]i32{ -1, 0, 1, 2 };
    const ids = [_]u32{ 5, 6, 7, 8 };
    const p: win.Plan = .{ .stream = &s, .position = 2, .kind = .head, .tokens = &ids, .parents = &parents, .branch_from = 3, .context_copy = true };
    const drafts = try win.nodeChances(&e, &s, 2);
    const out = try planChances(&e, p, 4);
    const rate = win.copyRate(&e, &s, true);
    try std.testing.expectApproxEqAbs(cfg.copy_rate, rate, 1e-12);
    try std.testing.expectApproxEqAbs(drafts[1], out[1], 1e-12);
    try std.testing.expectApproxEqAbs(rate * drafts[1], out[2], 1e-12);
    try std.testing.expectApproxEqAbs(rate * rate * drafts[1], out[3], 1e-12);
    // copies that land whole raise the rate toward 1; copies that break lower it below the prior
    for (0..30) |_| win.observeCopy(&e, &s, true, 12, 12);
    try std.testing.expect(win.copyRate(&e, &s, true) > 0.99);
    for (0..80) |_| win.observeCopy(&e, &s, true, 1, 12); // one row landed, then a miss: toward 1 / (1 + 1)
    try std.testing.expect(win.copyRate(&e, &s, true) < 0.55);
    // an untrusted copy starts from short_rate; trusted copies teach both histories, untrusted ones only their own
    var fresh = try Stream.init(a, .{ .id = "short", .prompt = &.{ 1, 2 }, .max_new = 64 });
    defer fresh.deinit(a);
    const short: win.Plan = .{ .stream = &fresh, .position = 2, .kind = .head, .tokens = &ids, .parents = &parents, .branch_from = 3, .context_copy = true, .copy_trusted = false };
    try std.testing.expectApproxEqAbs(short_rate * drafts[1], (try planChances(&e, short, 4))[2], 1e-12);
    for (0..3) |_| win.observeCopy(&e, &fresh, true, 12, 12);
    try std.testing.expect(win.copyRate(&e, &fresh, false) > 0.6);
    for (0..60) |_| win.observeCopy(&e, &fresh, false, 0, 12);
    try std.testing.expect(win.copyRate(&e, &fresh, false) < 0.1 and win.copyRate(&e, &fresh, true) > 0.95);
    // a window without a copy keeps the head's own chances
    const plain: win.Plan = .{ .stream = &s, .position = 2, .kind = .head, .tokens = ids[0..2] };
    try std.testing.expectEqualSlices(f64, try win.nodeChances(&e, &s, 2), try planChances(&e, plain, 2));
}
