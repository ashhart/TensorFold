//! The single-flight join: identical concurrent prompts join a filling leader (fake backend).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const prefix = @import("lane_host_prefix.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;

const system_len = 1030; // the shared system prompt
const cut = 1024; // its last start on the store's 16-row grid, the cut the leader's fill keeps
const tail_len = 512; // the prompts' own tail: the rows past the cut a released waiter would prefill beside the leader
const prompt_len = system_len + tail_len;
const requests = 4; // the issue's shape: several identical prompts arrive together

const Snaps = struct {
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores; // lane-core backends restore inside their own prompt pass
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// Each prompt chunk sleeps `slow_ms` and records the call: the stream, and how many streams are open.
const Hook = struct {
    fake: *lanes.fake.Fake,
    slow_ms: i64 = 0,
    cancel: ?struct { engine: api.Engine, id: Id, at: usize } = null,
    streams: std.ArrayList(*lanes.Stream) = .empty,
    open: std.ArrayList(usize) = .empty,

    fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const h: *Hook = @ptrCast(@alignCast(ctx));
        if (h.slow_ms > 0) std.Io.sleep(std.testing.io, .fromMilliseconds(h.slow_ms), .awake) catch {};
        h.streams.append(gpa, s) catch {};
        h.open.append(gpa, h.fake.lanes.count()) catch {};
        if (h.cancel) |c| if (chunk == c.at) c.engine.cancel(c.id);
    }
};

const Run = struct {
    boxes: [8]Box = @splat(.{}),
    reasons: [8]Reason = undefined,
    cached: [8]?u32 = @splat(null),
    waits: u64 = 0,
    joins: u64 = 0,
    timeouts: u64 = 0,
    queued_after: usize = 0,
    leader_open_most: usize = 0, // the most streams open at any prompt call of the first filling stream
    misses: u64 = 0,

    fn deinit(r: *Run) void {
        for (&r.boxes) |*b| b.tokens.deinit(gpa);
    }
};

/// The identical prompt every request names: the system prompt, then the same tail.
fn promptOf(buf: *[prompt_len]u32) []const u32 {
    for (buf[0..system_len], 0..) |*t, k| t.* = 100 + @as(u32, @intCast(k % 97));
    for (buf[system_len..], 0..) |*t, k| t.* = 300 + @as(u32, @intCast(k));
    return buf;
}

/// Identical requests queued before the host starts (together); `budget` 0 is a cold cache; `cancel_at` cancels.
fn fanout(wait_ms: ?u32, slow_ms: i64, cancel_at: ?usize, budget: u64, out: *Run) !void {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 13 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .planned = true, .grid = 16, .chunk = 16 };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, rules, budget);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 8 });
    host.cache = &store;
    if (wait_ms) |ms| host.prefix.ns = @as(i96, ms) * std.time.ns_per_ms;
    const e = host.engine();
    var hook = Hook{ .fake = &target, .slow_ms = slow_ms };
    if (cancel_at) |at| hook.cancel = .{ .engine = e, .id = 1, .at = at };
    defer {
        hook.streams.deinit(gpa);
        hook.open.deinit(gpa);
    }
    target.prefill_hook = Hook.call;
    target.prefill_hook_ctx = &hook;
    var buf: [prompt_len]u32 = undefined;
    const prompt = promptOf(&buf);
    var reqs: [requests]Request = undefined;
    const shared = [_]u32{system_len};
    for (0..requests) |i| {
        reqs[i] = .{ .prompt = prompt, .max_tokens = 6, .shared_prefixes = &shared };
        try e.submit(@intCast(i + 1), &reqs[i], .{ .ctx = &out.boxes[i], .event = Box.event });
    }
    try host.start();
    defer host.stop();
    for (0..requests) |i| {
        out.reasons[i] = out.boxes[i].wait();
        out.cached[i] = out.boxes[i].cached;
    }
    host.lock();
    out.queued_after = host.queued.items.len;
    host.unlock();
    out.waits = host.prefix.waits;
    out.joins = host.prefix.joins;
    out.timeouts = host.prefix.timeouts;
    out.misses = store.counts.misses;
    if (hook.streams.items.len > 0) {
        const leader = hook.streams.items[0];
        for (hook.streams.items, hook.open.items) |s, n| {
            if (s == leader) out.leader_open_most = @max(out.leader_open_most, n);
        }
    }
}

test "single-flight: identical prompts join the filling leader, whose whole fill runs alone" {
    var on: Run = .{};
    defer on.deinit();
    try fanout(null, 1, null, 1 << 24, &on);
    var off: Run = .{};
    defer off.deinit();
    try fanout(0, 1, null, 1 << 24, &off);
    for (0..requests) |i| {
        try std.testing.expectEqual(Reason.length, on.reasons[i]);
        try std.testing.expectEqual(Reason.length, off.reasons[i]);
        try std.testing.expectEqualSlices(u32, off.boxes[i].tokens.items, on.boxes[i].tokens.items); // joined == solo
    }
    // the leader prefills whole; each joiner waits for the whole fill and resumes the leader's kept cut
    try std.testing.expectEqualSlices(?u32, &.{ 0, cut, cut, cut }, on.cached[0..requests]);
    try std.testing.expectEqual(@as(u64, requests - 1), on.joins);
    try std.testing.expectEqual(@as(u64, requests - 1), on.waits);
    try std.testing.expectEqual(@as(u64, 0), on.timeouts);
    try std.testing.expectEqual(@as(usize, 0), on.queued_after);
    // the leader's whole fill ran with no joiner's stream open beside it: one identical prompt computes at a time
    try std.testing.expectEqual(@as(usize, 1), on.leader_open_most);
    // off (TENSORFOLD_PREFIX_WAIT_MS=0): every prompt fills side by side, its stream open beside the leader's
    try std.testing.expectEqualSlices(?u32, &.{ 0, 0, 0, 0 }, off.cached[0..requests]);
    try std.testing.expectEqual(@as(u64, 0), off.waits);
    try std.testing.expect(off.leader_open_most > 1); // the check detects the side-by-side prefill this joins
    try std.testing.expectEqual(@as(u64, requests), off.misses); // cold: every prompt missed, none waited
    try std.testing.expectEqual(@as(u64, 1), on.misses); // only the leader's lookup missed; the joiners hit its cut
}

test "single-flight: cold cache — nothing will be kept, so the join does not arm and the prompts run side by side" {
    var on: Run = .{};
    defer on.deinit();
    try fanout(null, 1, null, 0, &on);
    var off: Run = .{};
    defer off.deinit();
    try fanout(0, 1, null, 0, &off);
    for (0..requests) |i| {
        try std.testing.expectEqual(Reason.length, on.reasons[i]);
        try std.testing.expectEqual(Reason.length, off.reasons[i]);
        try std.testing.expectEqualSlices(u32, off.boxes[i].tokens.items, on.boxes[i].tokens.items); // == solo
    }
    // a zero budget plans no mark and keeps nothing: every lookup missed, nothing is resumable
    try std.testing.expectEqual(@as(u64, requests), on.misses);
    try std.testing.expectEqualSlices(?u32, &.{ 0, 0, 0, 0 }, on.cached[0..requests]);
    // the join did not arm: with nothing kept a joiner would wait out the whole fill, then prefill from the start
    try std.testing.expectEqual(@as(u64, 0), on.joins);
    try std.testing.expectEqual(@as(u64, 0), on.waits);
    try std.testing.expect(on.leader_open_most > 1); // every prompt's stream open beside the leader's
}

test "single-flight: fail-open, the leader cancelled — every waiter prefills its own prompt and matches a solo run" {
    const cancel_at = 2; // the leader dies on its second prompt chunk
    var on: Run = .{};
    defer on.deinit();
    try fanout(null, 1, cancel_at, 1 << 24, &on);
    var off: Run = .{};
    defer off.deinit();
    try fanout(0, 1, cancel_at, 1 << 24, &off);
    // the leader's own reply is the cancellation in both runs
    try std.testing.expectEqual(Reason.cancelled, on.reasons[0]);
    try std.testing.expectEqual(Reason.cancelled, off.reasons[0]);
    for (1..requests) |i| {
        // each released waiter finished whole, token for token a no-join run's
        try std.testing.expectEqual(Reason.length, on.reasons[i]);
        try std.testing.expectEqual(Reason.length, off.reasons[i]);
        try std.testing.expectEqualSlices(u32, off.boxes[i].tokens.items, on.boxes[i].tokens.items);
    }
    // the joiners armed on identity, then the leader's cancellation released them: no bound passed
    try std.testing.expectEqual(@as(u64, requests - 1), on.joins);
    try std.testing.expectEqual(@as(u64, requests - 1), on.waits);
    try std.testing.expectEqual(@as(u64, 0), on.timeouts);
    try std.testing.expectEqual(@as(u64, 0), off.joins);
}

test "single-flight: fail-open, the bound passes — every waiter is released and prefills beside the leader" {
    // a 1 ms bound against a leader whose chunks each take 30 ms: no waiter survives to the whole fill
    var on: Run = .{};
    defer on.deinit();
    try fanout(1, 30, null, 1 << 24, &on);
    var off: Run = .{};
    defer off.deinit();
    try fanout(0, 30, null, 1 << 24, &off);
    for (0..requests) |i| {
        try std.testing.expectEqual(Reason.length, on.reasons[i]);
        try std.testing.expectEqual(Reason.length, off.reasons[i]);
        try std.testing.expectEqualSlices(u32, off.boxes[i].tokens.items, on.boxes[i].tokens.items); // released == solo
    }
    // each joiner armed, then the bound released it: the timeouts counter carries the releases
    try std.testing.expectEqual(@as(u64, requests - 1), on.joins);
    try std.testing.expectEqual(@as(u64, requests - 1), on.timeouts);
    // each released waiter left before the leader's first keep and does not re-lookup: it prefilled from zero
    try std.testing.expectEqual(@as(u64, requests), on.misses);
    try std.testing.expectEqualSlices(?u32, &.{ 0, 0, 0, 0 }, on.cached[0..requests]);
    // the released waiters filled side by side: the bound gives back the throughput the join would have held
    try std.testing.expect(on.leader_open_most > 1);
}
