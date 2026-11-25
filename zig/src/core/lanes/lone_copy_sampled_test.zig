//! A seeded sampled lone stream's copies priced short, off, and from kept prices: its reply is drafts-off decode's.
const std = @import("std");
const cfgs = @import("config.zig");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const Proposer = @import("proposer.zig").Proposer;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

/// The forwards as timed at load, on a short scratch context: 5 ms plus 0.2 ms a row.
const startup = blk: {
    var c: [16]cfgs.Cost = undefined;
    for (&c, 1..) |*x, w| x.* = .{ .width = w, .ms = 5 + 0.2 * @as(f64, @floatFromInt(w)) };
    break :blk c;
};

/// A dense model's forwards: a row costs 11.6 ms past a 65 ms round, so a 15-row copy costs about three plain rounds.
const dense = blk: {
    var c: [16]cfgs.Cost = undefined;
    for (&c, 1..) |*x, w| x.* = .{ .width = w, .ms = 65 + 11.6 * @as(f64, @floatFromInt(w)) };
    break :blk c;
};

/// A round's ms: a fixed part plus each verified row's.
const long_context = [2]f64{ 50, 40 };
const dense_rounds = [2]f64{ 65, 11.6 };

const Clock = struct {
    target: *const fake.Fake,
    times: [2]f64,

    fn clock(c: *Clock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *Clock = @ptrCast(@alignCast(ptr));
        return c.times[0] + c.times[1] * @as(f64, @floatFromInt(c.target.last_rows));
    }
};

/// The target's next 15 keyed draws at every ask: from context length `land_from` on they land, else the first misses.
const Copier = struct {
    out: [15]u32 = undefined,
    sampling: ?Sampling,
    land_from: usize = 0,

    fn propose(ptr: *anyopaque, context: []const u32, most: i64) anyerror![]const u32 {
        const c: *Copier = @ptrCast(@alignCast(ptr));
        if (most < 2) return &.{};
        const history = try gpa.alloc(u32, context.len + c.out.len);
        defer gpa.free(history);
        @memcpy(history[0..context.len], context);
        const n: usize = @intCast(@min(c.out.len, most));
        for (c.out[0..n], 0..) |*t, k| {
            t.* = fake.next(history[0 .. context.len + k], c.sampling, context.len + k);
            history[context.len + k] = t.*;
        }
        if (context.len < c.land_from) c.out[0] = (c.out[0] + 1) % fake.vocab;
        return c.out[0..n];
    }

    fn lastMatch(_: *anyopaque) i64 {
        return 100;
    }

    const vtable: Proposer.VTable = .{ .propose = propose, .last_match = lastMatch };
};

/// Copy windows the target verified (no held drafts): how many, their rows, and the first one's rows.
const Watch = struct {
    var inner: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
    var copies: u64 = 0;
    var copy_rows: u64 = 0;
    var first: ?usize = null;

    fn verify(ptr: *anyopaque, ws: []const be.Window, out: []be.Verified) anyerror!void {
        for (ws) |w| {
            if (w.held > 0 or w.tokens.len == 0) continue;
            if (first == null) first = w.tokens.len;
            copies += 1;
            copy_rows += w.tokens.len;
        }
        return inner(ptr, ws, out);
    }
};

/// A request's reply, its first copy window's rows, and its copy windows and rows.
const Reply = struct { tokens: [160]u32 = undefined, len: usize = 0, first: ?usize = null, copies: u64 = 0, copy_rows: u64 = 0 };

/// A lone request: its prompt, sampling, whether it drafts, its reply's length and where its copier's copies land.
const Ask = struct { prompt: []const u32, sampling: ?Sampling, drafts: bool = true, max_new: u32, land_from: usize = 0 };

const prompts = [_][]const u32{ &.{ 2, 7, 1, 8, 2, 8, 1, 8 }, &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5 } };
const ids = [_][]const u8{ "a", "b" };

/// One engine serving `asks` as lone requests in turn; `kept`: whether band 0 kept prices, greedy then sampled.
fn serve(m: cfgs.Model, times: [2]f64, asks: []const Ask, out: []Reply, kept: ?*[2]bool) !void {
    var cfg = try cfgs.Config.init(gpa, m, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var backend = target.backend();
    var table = backend.vtable.*;
    Watch.inner = table.verify;
    table.verify = Watch.verify;
    backend.vtable = &table;
    var clock: Clock = .{ .target = &target, .times = times };
    var e = Engine.init(gpa, &cfg, backend, clock.clock());
    defer e.deinit();
    for (asks, out, ids[0..asks.len]) |ask, *r, id| {
        Watch.copies = 0;
        Watch.copy_rows = 0;
        Watch.first = null;
        var c: Copier = .{ .sampling = ask.sampling, .land_from = ask.land_from };
        const proposer: Proposer = .{ .ptr = &c, .vtable = &Copier.vtable };
        var s = try sm.Stream.init(gpa, .{ .id = id, .prompt = ask.prompt, .max_new = ask.max_new, .sampling = ask.sampling, .drafts = ask.drafts, .proposer = proposer });
        defer s.deinit(gpa);
        try e.addStream(&s);
        while (e.activeCount() > 0) try e.step();
        r.len = s.emitted().len;
        @memcpy(r.tokens[0..r.len], s.emitted());
        r.first = Watch.first;
        r.copies = Watch.copies;
        r.copy_rows = Watch.copy_rows;
    }
    if (kept) |k| k.* = .{ e.lone_kept[0][0] != null, e.lone_kept[1][0] != null };
}

/// hip_lanes.zig facts in small (as lone_copy_price_test.zig's): timed plain rounds and copies priced.
fn hip(head: bool, forwards: []const cfgs.Cost) cfgs.Model {
    return .{ .exact_width = 16, .mtp = head, .speculate = head, .speculate_early = false, .drafts = 4, .plain_guard = true, .copy_guard = true, .window_costs = forwards, .mtp_step_ms = if (head) 0.5 else 0, .hidden_rows = true, .batch_rows = 32, .max_streams = 4 };
}

/// `asks` served drafts-off and with copies unpriced (main's path) give `got`'s replies, token for token.
fn expectSame(m: cfgs.Model, times: [2]f64, asks: []const Ask, got: []const Reply, unpriced: []Reply) !void {
    var plain_asks: [prompts.len]Ask = undefined;
    for (plain_asks[0..asks.len], asks) |*p, a| p.* = a;
    for (plain_asks[0..asks.len]) |*p| p.drafts = false;
    var plain: [prompts.len]Reply = @splat(.{});
    try serve(m, times, plain_asks[0..asks.len], plain[0..asks.len], null);
    var off = m;
    off.copy_guard = false;
    try serve(off, times, asks, unpriced, null);
    for (got, plain[0..asks.len], unpriced) |g, p, u| {
        try std.testing.expectEqualSlices(u32, p.tokens[0..p.len], g.tokens[0..g.len]);
        try std.testing.expectEqualSlices(u32, p.tokens[0..p.len], u.tokens[0..u.len]);
    }
}

test "sampled lone copy: shortened copies give the seeded reply of drafts-off decode" {
    for ([_]bool{ false, true }) |head| {
        const m = hip(head, &dense);
        const asks = [_]Ask{.{ .prompt = prompts[0], .sampling = .{ .seed = 11, .temperature = 0.8 }, .max_new = 64 }};
        var got: [1]Reply = @splat(.{});
        try serve(m, dense_rounds, &asks, &got, null);
        var unpriced: [1]Reply = @splat(.{});
        try expectSame(m, dense_rounds, &asks, &got, &unpriced);
        try std.testing.expect(got[0].first.? < 15); // the untimed price at the prior's rate cuts it
        try std.testing.expectEqual(@as(?usize, 15), unpriced[0].first); // unpriced, it is the whole proposal
        var greedy: [1]Reply = @splat(.{});
        try serve(m, dense_rounds, &.{.{ .prompt = prompts[0], .sampling = null, .max_new = 64 }}, &greedy, null);
        try std.testing.expect(!std.mem.eql(u32, greedy[0].tokens[0..greedy[0].len], got[0].tokens[0..got[0].len]));
    }
}

test "sampled lone copy: copies priced off give the seeded reply of drafts-off decode" {
    for ([_]bool{ false, true }) |head| {
        const m = hip(head, &startup);
        const asks = [_]Ask{.{ .prompt = prompts[0], .sampling = .{ .seed = 23, .temperature = 0.8 }, .max_new = 96, .land_from = std.math.maxInt(usize) }};
        var got: [1]Reply = @splat(.{});
        try serve(m, long_context, &asks, &got, null);
        var unpriced: [1]Reply = @splat(.{});
        try expectSame(m, long_context, &asks, &got, &unpriced);
        try std.testing.expect(got[0].copy_rows < got[0].len); // a few copies measure them, then a probe now and then
        try std.testing.expect(got[0].copy_rows * 4 < unpriced[0].copy_rows);
    }
}

test "sampled lone copy: a request's first copy reads the prices the sampled request before it kept" {
    for ([_]bool{ false, true }) |head| {
        const m = hip(head, &dense);
        const asks = [_]Ask{
            .{ .prompt = prompts[0], .sampling = .{ .seed = 5, .temperature = 0.8 }, .max_new = 64 },
            .{ .prompt = prompts[1], .sampling = .{ .seed = 9, .temperature = 0.8 }, .max_new = 64 },
        };
        var got: [2]Reply = @splat(.{});
        var kept: [2]bool = undefined;
        try serve(m, dense_rounds, &asks, &got, &kept);
        var unpriced: [2]Reply = @splat(.{});
        try expectSame(m, dense_rounds, &asks, &got, &unpriced);
        try std.testing.expectEqual([2]bool{ false, true }, kept); // filed under the sampled mode, not the greedy one
        try std.testing.expect(got[0].first.? < 15);
        try std.testing.expectEqual(@as(?usize, 15), got[1].first); // the first request's landed copies taught the rate
    }
}
