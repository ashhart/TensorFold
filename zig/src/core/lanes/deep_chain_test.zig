//! A lone stream chains up to a model's most drafts while shared rounds keep the model's shared cap.
const std = @import("std");
const depth = @import("depth.zig");

const empty: @import("table.zig").Table = .{};
const high: [7]f64 = @splat(0.95);
const low: [7]f64 = @splat(0.5);

const NoCopy = struct {
    fn copy(_: *anyopaque, _: *const depth.Who) bool {
        return false;
    }
};

fn rule(prior: []const f64, shared_cap: u32) depth.Rule {
    return .{
        .gpa = std.testing.allocator,
        .prior = prior,
        .most_drafts = 7,
        .shared_cap = shared_cap,
        .family_costs = &empty,
        .shared_costs = &empty,
        .mtp_step_ms = 0,
        .plain_guard = false,
        .node_probabilities = false,
        .batch_rows = 16,
    };
}

fn who(state: *?depth.State) depth.Who {
    return .{ .state = state, .draft_room = 100, .finished = false, .forced = false };
}

test "a lone stream asks for most_drafts on a high first-draft acceptance and stays low on a low one" {
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = NoCopy.copy };
    var r = rule(&high, 3);
    defer r.deinit();
    var s: ?depth.State = null;
    defer if (s) |*x| x.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 7), try r.headDepth(&who(&s), null, probe));
    r.prior = &low;
    var t: ?depth.State = null;
    defer if (t) |*x| x.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 1), try r.headDepth(&who(&t), null, probe));
}

test "streams that share a round chain at most the shared cap each" {
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = NoCopy.copy };
    var r = rule(&high, 3);
    defer r.deinit();
    var a: ?depth.State = null;
    var b: ?depth.State = null;
    defer if (a) |*x| x.deinit(std.testing.allocator);
    defer if (b) |*x| x.deinit(std.testing.allocator);
    var whos = [_]depth.Who{ who(&a), who(&b) };
    var out: [2]i64 = undefined;
    try r.budgets(&whos, probe, false, &out);
    try std.testing.expect(out[0] > 0 and out[1] > 0);
    try std.testing.expect(out[0] <= 3 and out[1] <= 3);
}

test "a zero shared cap keeps the ladder's top of three for a lone stream and most_drafts for shared budgets" {
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = NoCopy.copy };
    var r = rule(&high, 0);
    defer r.deinit();
    var a: ?depth.State = null;
    var b: ?depth.State = null;
    defer if (a) |*x| x.deinit(std.testing.allocator);
    defer if (b) |*x| x.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 3), try r.headDepth(&who(&a), null, probe));
    var whos = [_]depth.Who{ who(&a), who(&b) };
    var out: [2]i64 = undefined;
    try r.budgets(&whos, probe, false, &out);
    try std.testing.expect(out[0] > 3 or out[1] > 3);
}

test "under a confidence floor or node probabilities a shared round chains at most the cap; a lone stream chains seven" {
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = NoCopy.copy };
    inline for (.{ .{ 0.5, false }, .{ null, false }, .{ null, true } }) |c| {
        for ([_]u32{ 3, 0 }) |cap| {
            var r = rule(&high, cap);
            defer r.deinit();
            r.node_probabilities = true;
            r.confidence_floor = c[0];
            r.granted_depth = c[1];
            var a: ?depth.State = null;
            var b: ?depth.State = null;
            defer if (a) |*x| x.deinit(std.testing.allocator);
            defer if (b) |*x| x.deinit(std.testing.allocator);
            try std.testing.expectEqual(@as(i64, 7), try r.headDepth(&who(&a), null, probe));
            var whos = [_]depth.Who{ who(&a), who(&b) };
            var out: [2]i64 = undefined;
            try r.budgets(&whos, probe, false, &out);
            const want: i64 = if (cap == 0) 7 else 3;
            try std.testing.expectEqualSlices(i64, &.{ want, want }, &out);
        }
    }
}

test "allocated shared rows, granted or not, reserve at most the shared cap a stream" {
    const a = std.testing.allocator;
    const Cfg = @import("config.zig");
    const fake = @import("fake.zig");
    const S = @import("stream.zig").Stream;
    const win = @import("windows.zig");
    for ([_]bool{ false, true }) |granted| {
        for ([_]u32{ 3, 0 }) |cap| {
            var cfg = try Cfg.Config.init(a, .{ .exact_width = 8, .drafts = 7, .shared_mtp_cap = cap, .allocate_shared_drafts = true, .draft_probabilities = granted, .granted_depth = granted, .batch_rows = 16 }, 8, 7);
            defer cfg.deinit(a);
            var back: fake.Fake = .{ .gpa = a };
            defer back.deinit();
            var clock: fake.FixedClock = .{};
            var e = @import("engine.zig").Engine.init(a, &cfg, back.backend(), clock.clock());
            defer e.deinit();
            var one = try S.init(a, .{ .id = "one", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
            defer one.deinit(a);
            var two = try S.init(a, .{ .id = "two", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
            defer two.deinit(a);
            try one.draft_chances.appendNTimes(a, 1, 7);
            try two.draft_chances.appendNTimes(a, 1, 7);
            const plans = [_]win.Plan{ .{ .stream = &one, .position = 0, .kind = .none }, .{ .stream = &two, .position = 0, .kind = .none } };
            var out: [2]i64 = undefined;
            try @import("shared_drafts.zig").budgets(&e, &plans, &.{ 0, 1 }, &out);
            const want: i64 = if (cap == 0) 7 else 3;
            try std.testing.expectEqualSlices(i64, &.{ want, want }, &out);
        }
    }
}

test "a stream that opens beside a live one takes at most the shared cap as its first drafts" {
    const a = std.testing.allocator;
    const Cfg = @import("config.zig");
    const fake = @import("fake.zig");
    const S = @import("stream.zig").Stream;
    for ([_]u32{ 3, 0 }) |cap| {
        var cfg = try Cfg.Config.init(a, .{ .exact_width = 8, .drafts = 7, .shared_mtp_cap = cap, .draft_probabilities = true, .batch_rows = 16 }, 8, 7);
        defer cfg.deinit(a);
        var back: fake.Fake = .{ .gpa = a };
        defer back.deinit();
        var clock: fake.FixedClock = .{};
        var e = @import("engine.zig").Engine.init(a, &cfg, back.backend(), clock.clock());
        defer e.deinit();
        var live = try S.init(a, .{ .id = "live", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
        defer live.deinit(a);
        var s = try S.init(a, .{ .id = "new", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
        defer s.deinit(a);
        try std.testing.expectEqual(@as(u32, 7), try @import("open.zig").firstDepth(&e, &s, .head));
        try e.live.append(a, &live);
        defer e.live.clearRetainingCapacity();
        try std.testing.expectEqual(@as(u32, if (cap == 0) 7 else 3), try @import("open.zig").firstDepth(&e, &s, .head));
    }
}
