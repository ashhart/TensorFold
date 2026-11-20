//! Block passes, chained heads and copied rows retain their distinct costs through calibration.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const Config = @import("config.zig").Config;
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const fake = @import("fake.zig");
const win = @import("windows.zig");
const draft = @import("shared_drafts.zig");
const costs = [_]@import("allocate.zig").Cost{ .{ .width = 1, .ms = 10 }, .{ .width = 64, .ms = 10 } };
const chances = [_]f64{ 0.4, 0.04, 0.004, 0.0004 };

fn mtp(n: usize, sampled: bool, limited: bool, old_depth: u32) !void {
    var cfg = try Config.init(a, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4, .draft_prior = &.{ 0.4, 0.1, 0.1, 0.1 }, .mtp_step_ms = 2, .draft_step_ms = 1, .draft_streams = true, .copy_drafts = false, .allocate_shared_drafts = true, .limit_shared_drafts = limited, .plain_guard = true, .hidden_rows = true, .batch_rows = 64, .window_costs = &costs }, 16, 15);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var streams: [8]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var plans: [8]win.Plan = undefined;
    var heads: [8]usize = undefined;
    var budgets: [8]i64 = undefined;
    for (streams[0..n], plans[0..n], heads[0..n], 0..) |*s, *p, *h, i| {
        s.* = try Stream.init(a, .{ .id = "mtp", .prompt = &.{ 1, 2, 3 }, .max_new = 1000, .sampling = if (sampled) .{ .seed = 19, .temperature = 1 } else null });
        made += 1;
        s.rounds = 1;
        try s.draft_chances.appendSlice(a, &chances);
        p.* = .{ .stream = s, .position = 1, .kind = .none };
        h.* = i;
    }
    for (1..129) |round| {
        const trial = try @import("shared_price.zig").calibrate(&e, plans[0..n]);
        try t.expectEqual(round == 64 or round == 128, trial);
        const head = @as(f64, @floatFromInt(old_depth)) * 2 + @as(f64, @floatFromInt(old_depth - 1));
        try e.rule.observeShared(@intCast(n), if (trial) n else n * (old_depth + 1), if (trial) 12 else 12 + head, sampled, if (trial) 0 else head, trial, trial);
        if (round == 63 or round == 64 or round == 128) {
            try draft.budgets(&e, plans[0..n], heads[0..n], budgets[0..n]);
            const want: i64 = if (round < 128 and !limited) 4 else 1;
            for (budgets[0..n], 0..) |got, i| {
                errdefer std.debug.print("mtp n={d} sampled={any} limited={any} old_depth={d} round={d} i={d} budgets={any} price={any}\n", .{ n, sampled, limited, old_depth, round, i, budgets[0..n], e.rule.sharedPrice(n, sampled) });
                try t.expectEqual(want, got);
            }
        }
    }
    for (0..16) |_| try e.rule.observeShared(@intCast(n), n, 12, sampled, 0, true, false);
    try draft.budgets(&e, plans[0..n], heads[0..n], budgets[0..n]);
    for (budgets[0..n]) |got| try t.expectEqual(@as(i64, 1), got);
    try t.expect(e.rule.sharedPrice(1, sampled) == null);
}

test "shared price modes: MTP at two four eight streams pays steps before and after calibration" {
    for ([_]usize{ 2, 4, 8 }) |n| for ([_]bool{ false, true }) |sampled| {
        for ([_]bool{ false, true }) |limited| for ([_]u32{ 1, 3 }) |old_depth| try mtp(n, sampled, limited, old_depth);
    };
}

const Copier = struct {
    fn propose(_: *anyopaque, _: []const u32, _: i64) ![]const u32 {
        return &.{ 7, 7, 7, 7 };
    }
    fn match(_: *anyopaque) i64 {
        return 100;
    }
    const vtable: @import("proposer.zig").Proposer.VTable = .{ .propose = propose, .last_match = match, .priced = true };
};

test "shared price modes: mixed copied rows never pay a neural step or start a block pass" {
    for ([_]bool{ false, true }) |block| {
        var cfg = try Config.init(a, .{ .fixed_draft_block = block, .exact_width = 16, .mtp = true, .drafts = 4, .draft_streams = true, .mtp_step_ms = 4, .plain_guard = true, .hidden_rows = true, .batch_rows = 64, .allocate_shared_drafts = true, .window_costs = &costs }, 16, 15);
        defer cfg.deinit(a);
        cfg.copy_rate = 1;
        var target: fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        e.alone = false;
        try e.rule.observeShared(2, 2, 12, false, 0, true, false);
        try e.rule.observeShared(2, 4, 16, false, 4, false, false);
        var copy = try Stream.init(a, .{ .id = "copy", .prompt = &.{ 1, 2, 3 }, .max_new = 100, .proposer = .{ .ptr = &target, .vtable = &Copier.vtable } });
        defer copy.deinit(a);
        var head = try Stream.init(a, .{ .id = "head", .prompt = &.{ 1, 2, 3 }, .max_new = 100 });
        defer head.deinit(a);
        try head.draft_chances.appendSlice(a, &.{ 0.9, 0.01, 0.001, 0.0001 });
        const plans = [_]win.Plan{ .{ .stream = &copy, .position = 1, .kind = .none }, .{ .stream = &head, .position = 1, .kind = .none } };
        var budgets: [1]i64 = undefined;
        try draft.budgets(&e, &plans, &.{1}, &budgets);
        try t.expectEqual(@as(i64, 0), budgets[0]);
        if (!block) {
            copy.proposer = null;
            copy.drafts = false;
            head.draft_chances.clearRetainingCapacity();
            try head.draft_chances.appendSlice(a, &.{ 0.8, 0.7, 0.01, 0.001 });
            cfg.mtp_step_ms = 1;
            for ([_]f64{ 0, 10 }, [_]i64{ 2, 1 }) |extra, want| {
                cfg.draft_step_ms = extra;
                try draft.budgets(&e, &plans, &.{1}, &budgets);
                try t.expectEqual(want, budgets[0]);
            }
        }
    }
}

test "shared price modes: copy-only shared windows retain the measured row feedback" {
    var cfg = try Config.init(a, .{ .exact_width = 16, .plain_guard = true, .hidden_rows = true, .batch_rows = 64, .window_costs = &costs }, 16, 15);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var e = Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    try e.rule.observeShared(2, 2, 12, false, 0, true, false);
    try e.rule.observeShared(2, 10, 52, false, 0, false, false);
    try t.expectApproxEqAbs(@as(f64, 5), e.rule.sharedPrice(2, false).?.row, 1e-9);
    var one = try Stream.init(a, .{ .id = "copy-a", .prompt = &.{1}, .max_new = 100 });
    defer one.deinit(a);
    var two = try Stream.init(a, .{ .id = "copy-b", .prompt = &.{1}, .max_new = 100 });
    defer two.deinit(a);
    var plans = [_]win.Plan{ .{ .stream = &one, .position = 1, .kind = .copy, .tokens = &.{ 7, 7, 7, 7 } }, .{ .stream = &two, .position = 1, .kind = .copy, .tokens = &.{ 7, 7, 7, 7 } } };
    try win.allocate(&e, &plans);
    try t.expectEqual(@as(usize, 2), plans[0].count());
    try t.expectEqual(@as(usize, 2), plans[1].count());
}

test "shared price modes: the wrapper carries a drafter's explicit fixed-block fact" {
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var drafter: @import("drafted_test.zig").FakeDrafter = .{};
    defer drafter.deinit();
    var wrapper = try @import("drafted.zig").Drafted.init(a, target.backend(), drafter.drafter());
    defer wrapper.deinit();
    try t.expect(!wrapper.facts(.{ .exact_width = 16 }).fixed_draft_block);
    drafter.facts_.fixed_block = true;
    const m = wrapper.facts(.{ .exact_width = 16 });
    try t.expect(m.fixed_draft_block);
    var cfg = try Config.init(a, m, 16, 15);
    defer cfg.deinit(a);
    try t.expect(cfg.k.fixed_draft_block);
}

test "shared price modes: a stale expensive block can be measured again without retaining the spike" {
    const prices = @import("shared_price.zig");
    var p: prices.Prices = .{};
    defer p.deinit(a);
    try p.observeHead(a, 4, 4, 2, 0, false, true, false);
    try p.observeHead(a, 4, 8, 200, 4, false, false, false);
    for (0..prices.stale - 1) |_| try p.observeHead(a, 4, 4, 2, 0, false, true, false);
    try t.expectEqual(@as(f64, 200), p.blockPrice(4, false, 2, 0, 0).?.fixed);
    try p.observeHead(a, 4, 4, 2, 0, false, true, false);
    try t.expect(p.blockPrice(4, false, 2, 0, 0) == null);
    try p.observeHead(a, 4, 8, 10, 2, false, false, false);
    try t.expectEqual(@as(f64, 10), p.blockPrice(4, false, 2, 0, 0).?.fixed);
    try t.expectEqual(@as(f64, 2), p.blockPrice(4, false, 2, 0, 0).?.head);
}
