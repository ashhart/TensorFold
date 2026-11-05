//! Marginal host draws: live-row allocation, actual overhead attribution and failed-batch isolation.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const depth = @import("depth.zig");
const alloc = @import("allocate.zig");
const Rule = depth.Rule;
const State = depth.State;
const Who = depth.Who;
const gpa = std.testing.allocator;

/// Synthetic host-draw accounting around the real fake target's verify/commit loop, not device timing.
const HostDraws = struct {
    target: fake.Fake = .{ .gpa = gpa },
    vtable: be.Backend.VTable = undefined,
    inner_verify: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined,
    inner_first: *const fn (*anyopaque, *sm.Stream, u64) anyerror!u64 = undefined,
    host: f64 = 0,
    row_ms: f64 = 3,
    quoted: bool = true,
    fail: bool = false,

    fn of(ptr: *anyopaque) *HostDraws {
        return @fieldParentPtr("target", @as(*fake.Fake, @ptrCast(@alignCast(ptr))));
    }

    fn first(ptr: *anyopaque, s: *sm.Stream, position: u64) !u64 {
        s.verify_host_ms = 700; // a stale prefill/first-draw span must not enter a verify interval
        return of(ptr).inner_first(ptr, s, position);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const x = of(ptr);
        if (x.fail) {
            windows[0].stream.verify_host_ms = 500; // completed work before a later window fails
            return error.HostWindowFailed;
        }
        try x.inner_verify(ptr, windows, out);
        for (windows) |w| {
            const ms = x.row_ms * @as(f64, @floatFromInt(w.rows()));
            x.host += ms;
            if (ms > 0) w.stream.verify_host_ms = ms;
            // Deliberately different from actual elapsed: overhead subtraction must never use this prediction.
            w.stream.verify_row_ms = if (!x.quoted) null else if (ms > 0) 1000 else 0;
        }
    }

    fn backend(x: *HostDraws) be.Backend {
        var inner = x.target.backend();
        x.vtable = inner.vtable.*;
        x.inner_verify = inner.vtable.verify;
        x.inner_first = inner.vtable.first;
        x.vtable.verify = verify;
        x.vtable.first = first;
        inner.vtable = &x.vtable;
        return inner;
    }

    fn start(ptr: *anyopaque) void {
        const x: *HostDraws = @ptrCast(@alignCast(ptr));
        x.host = 0;
    }

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const x: *HostDraws = @ptrCast(@alignCast(ptr));
        return 10 + 4 + x.host; // forward + true fixed overhead + actual host draws
    }

    fn clock(x: *HostDraws) be.Clock {
        return .{ .ptr = x, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }
};

test "host accounting removes only quotable draws and excludes failed stale and first work across widths" {
    const win = @import("windows.zig");
    for ([_]bool{ false, true }) |quoted| for ([_]usize{ 1, 2 }) |live| {
        var costs: [8]Cost = undefined;
        for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 10 };
        var cfg = try Config.init(gpa, .{ .exact_width = 8, .mtp = true, .speculate = true, .speculate_early = false, .hidden_rows = true, .drafts = 3, .draft_prior = &.{ 1, 1, 1 }, .draft_probabilities = true, .window_costs = &costs, .mtp_step_ms = 0, .batch_rows = 8 }, 8, 7);
        defer cfg.deinit(gpa);
        cfg.k.depth_rate = 0;
        var target: HostDraws = .{ .quoted = quoted };
        defer target.target.deinit();
        var engine = Engine.init(gpa, &cfg, target.backend(), target.clock());
        defer engine.deinit();
        var streams: [2]sm.Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(gpa);
        for (streams[0..live]) |*s| {
            s.* = try sm.Stream.init(gpa, .{ .id = "host-accounting", .prompt = &.{ 2, 7, 13 }, .max_new = 256, .sampling = .{ .seed = 42, .temperature = 1.0, .top_p = 0.95 } });
            made += 1;
            try engine.addStream(s);
        }
        var narrow: usize = std.math.maxInt(usize);
        var wide: usize = 0;
        var fixed: f64 = 0;
        for (0..6) |round| {
            target.row_ms = if (round == 3) 0 else 3;
            for (streams[0..live], 0..) |*s, i| {
                s.dropHeld(gpa);
                const n = 1 + (round + i) % 3;
                const tokens = try gpa.alloc(u32, n);
                @memset(tokens, 11);
                s.next = .{ .count = @intCast(n), .tokens = tokens };
                if (round == 3) s.verify_host_ms = 900; // a zero-host verify must not reuse outside work
            }
            try engine.step();
            narrow = @min(narrow, target.target.last_rows);
            wide = @max(wide, target.target.last_rows);
            const observed = 4 + if (quoted) @as(f64, 0) else target.host;
            fixed = if (round == 0) observed else fixed + 0.2 * (observed - fixed);
            try std.testing.expectApproxEqAbs(fixed, engine.rule.fixedMs(live, true), 1e-9);
        }
        try std.testing.expect(wide > narrow);
        target.fail = true;
        try std.testing.expectError(error.HostWindowFailed, engine.step());
        try std.testing.expectApproxEqAbs(fixed, engine.rule.fixedMs(live, true), 1e-9);
        target.fail = false;
        target.row_ms = 0;
        for (streams[0..live]) |*s| {
            s.dropHeld(gpa);
            s.next = .{ .count = 1, .tokens = try gpa.dupe(u32, &.{11}) };
        }
        try engine.step();
        fixed += 0.2 * (4 - fixed);
        try std.testing.expectApproxEqAbs(fixed, engine.rule.fixedMs(live, true), 1e-9);
        // Depth selection still uses complete measured lone rounds, not an additional row quote.
        if (live == 1) {
            const s = &streams[0];
            for (0..4) |drafts| try engine.rule.observeCost(@intCast(drafts), 20, false, win.who(s));
            const before = try engine.rule.depth(win.who(s));
            s.verify_row_ms = 1_000_000;
            try std.testing.expectEqual(before, try engine.rule.depth(win.who(s)));
        }
    };
}

test "generic budgets price live host rows without borrowing finished streams' baseline work" {
    var shared = try alloc.extendCosts(gpa, &.{.{ .width = 8, .ms = 10 }}, 8);
    defer shared.deinit(gpa);
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.5, 0.2 }, .most_drafts = 3, .family_costs = &shared, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = true, .node_probabilities = false, .batch_rows = 8 };
    defer rule.deinit();
    try rule.observeOverhead(1, 1, 10, true, 0);
    var states: [2]?State = @splat(null);
    defer for (&states) |*state| if (state.*) |*s| s.deinit(gpa);
    var whos = [_]Who{
        .{ .state = &states[0], .draft_room = 64, .finished = false, .forced = false, .sampled = true },
        .{ .state = &states[1], .draft_room = 64, .finished = true, .forced = false, .row_ms = 1000 },
    };
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    var out: [2]i64 = undefined;
    for ([_]f64{ 0, 5, 0 }) |row_ms| {
        whos[0].row_ms = row_ms;
        try rule.budgets(&whos, .{ .ptr = &unused, .copy = No.copy }, true, &out);
        try std.testing.expectEqualSlices(i64, if (row_ms == 0) &.{ 3, 0 } else &.{ 1, 0 }, &out);
    }
}

test "a profitable cheap hold survives an unprofitable higher-gain companion" {
    var flat = try alloc.extendCosts(gpa, &.{.{ .width = 8, .ms = 10 }}, 8);
    defer flat.deinit(gpa);
    const holds = [_]depth.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 10 }, .{ .streams = 2, .depth = 1, .ms = 20 } };
    var rule: Rule = .{ .gpa = gpa, .prior = &.{1}, .most_drafts = 1, .family_costs = &flat, .shared_costs = &flat, .mtp_step_ms = 0, .granted_depth = true, .plain_guard = true, .node_probabilities = true, .batch_rows = 8, .k = .{ .hold_costs = &holds } };
    defer rule.deinit();
    var states: [2]?State = @splat(null);
    defer for (&states) |*state| if (state.*) |*s| s.deinit(gpa);
    const whos = [_]Who{
        .{ .state = &states[0], .draft_room = 64, .finished = false, .forced = false, .sampled = true, .granted = 1, .gain = 0.9, .prices = &.{0.9}, .row_ms = 20 },
        .{ .state = &states[1], .draft_room = 64, .finished = false, .forced = false, .granted = 1, .gain = 0.8, .prices = &.{0.8}, .row_ms = 0 },
    };
    var out = [_]i64{ 1, 1 };
    var base = @import("hold_price.zig").Base.pending(2);
    base.host_ms = 20;
    try rule.payHolds(&whos, true, &out, base);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1 }, &out);
}

test "live host quotes change recovery funding across the strict boundary and back" {
    var flat = try alloc.extendCosts(gpa, &.{.{ .width = 4, .ms = 10 }}, 4);
    defer flat.deinit(gpa);
    const holds = [_]depth.HoldCost{.{ .streams = 1, .depth = 1, .ms = 4 }};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{1}, .most_drafts = 1, .family_costs = &flat, .shared_costs = &flat, .mtp_step_ms = 0, .granted_depth = true, .plain_guard = true, .node_probabilities = true, .batch_rows = 4, .k = .{ .hold_costs = &holds } };
    defer rule.deinit();
    var states: [2]?State = @splat(null);
    defer for (&states) |*state| if (state.*) |*s| s.deinit(gpa);
    var whos = [_]Who{
        .{ .state = &states[0], .draft_room = 2, .finished = false, .forced = false, .sampled = true, .granted = 0, .prices = &.{0.95} },
        .{ .state = &states[1], .draft_room = 2, .finished = false, .forced = true, .row_ms = 1000 },
    };
    for ([_]f64{ 0, 1.5, 2, 2.5, 0, 1 }, [_]i64{ 1, 1, 0, 0, 1, 1 }) |row_ms, want| {
        whos[0].row_ms = row_ms;
        var base = @import("hold_price.zig").Base.pending(2);
        base.host_ms = row_ms; // one host-drawn pending row and one GPU-only forced companion
        var out = [_]i64{ 1, 0 };
        try rule.payHolds(&whos, true, &out, base);
        try std.testing.expectEqualSlices(i64, &.{ want, 0 }, &out);
    }
}
