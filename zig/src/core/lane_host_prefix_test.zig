//! The prefix wait: prompts sharing a cut another prompt is filling compute it once (fake backend).
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
const cut = 1024; // its last start on the store's 16-row grid, the cut its fill keeps
const prompt_len = system_len + 16;
const unrelated = 4; // the request at this index starts with another system prompt

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
    remote: ?u32 = null, // the finished stats' remote_prefix: tokens resumed from a state built on a pulled prefix
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.remote = f.stats.remote_prefix;
                b.done = f.reason;
            },
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

/// Each prompt chunk sleeps `slow_ms`; request `id` is cancelled as the first request's fill starts its chunk `at`.
const Hook = struct {
    engine: ?api.Engine = null,
    slow_ms: i64 = 0,
    at: usize = 0,
    id: Id = 0, // 0: none
    sent: bool = false,
    refuse_at: u32 = 0, // the restore of a state kept this far fails at the first chunk, as qwen27_slots.opening can
    fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const c: *Hook = @ptrCast(@alignCast(ctx));
        if (c.refuse_at > 0 and chunk == 0 and s.reuse.saved != null and s.reuse.at == c.refuse_at) {
            s.reuse_failed = true; // the pass prefills from 0 with reuse.remote still set, as on the Mac
            s.reuse.saved = null;
        }
        if (c.slow_ms > 0) std.Io.sleep(std.testing.io, .fromMilliseconds(c.slow_ms), .awake) catch {};
        if (c.sent or c.id == 0 or chunk != c.at or s.prompt()[system_len] != 300) return;
        c.sent = true;
        c.engine.?.cancel(c.id);
    }
};

/// A producer's exporter over the fake target: the tokens a handoff exported, its copy dropped.
const Exports = struct {
    target: *lanes.fake.Fake,
    tokens: ?usize = null,
    fn exported(ptr: *anyopaque, _: LaneHost.Export.Handoff, tokens: []const u32, saved: pc.Saved) void {
        const x: *Exports = @ptrCast(@alignCast(ptr));
        x.tokens = tokens.len;
        x.target.drop(saved);
    }
    fn failed(_: *anyopaque, _: LaneHost.Export.Handoff, _: []const u8) void {}
};

const Setup = struct {
    wait_ms: ?u32 = null, // null: the default bound
    stepped: bool = true, // prompts fill a chunk a step (prefill_step)
    requests: usize = 5,
    shared: u32 = system_len, // the shared block the requests name
    hook: Hook = .{},
    prefill_chunks: usize = 13, // fake prompt chunks: 1 means one call would fill the whole prompt
    decode_share: ?f64 = null, // --decode-share: on a fake that opts in, it caps fill chunks at 512 rows
    handoff: ?usize = null, // this request is a handoff, exported at its prompt's last token
    handoff_len: usize = prompt_len, // the handoff's prompt length
    twin: ?usize = null, // this request repeats request 0's prompt
    twin_chunks: []const u32 = &.{}, // the twin's own chunk starts
    remote_at: u32 = 0, // TF_PREFILL_KEEP: the system prompt kept this far as a remote state, as a pulled turn leaves it
};

const Run = struct {
    boxes: [8]Box = @splat(.{}),
    reasons: [8]Reason = undefined,
    cached: [8]?u32 = @splat(null),
    remote: [8]?u32 = @splat(null),
    waits: u64 = 0,
    timeouts: u64 = 0,
    queued_after: usize = 0,
    exported: ?usize = null, // the tokens the handoff exported

    fn deinit(r: *Run) void {
        for (&r.boxes) |*b| b.tokens.deinit(gpa);
    }
};

/// The prompt of request `i`: a system prompt (another for `unrelated`), then 16 tokens of its own.
fn promptOf(buf: *[prompt_len]u32, i: usize) []const u32 {
    const seed: u32 = if (i == unrelated) 500 else 100;
    for (buf[0..system_len], 0..) |*t, k| t.* = seed + @as(u32, @intCast(k % 97));
    for (buf[system_len..], 0..) |*t, k| t.* = 300 + @as(u32, @intCast(i * 16 + k));
    return buf;
}

/// The system prompt's state at `at`, kept marked remote: the fake fills a stream that far and the store keeps it.
fn keepRemote(target: *lanes.fake.Fake, store: *pc.Store, at: u32) !void {
    var buf: [prompt_len]u32 = undefined;
    const p = promptOf(&buf, 0);
    var s = try lanes.Stream.init(gpa, .{ .id = "pulled", .prompt = p[0..at], .max_new = 1 });
    defer s.deinit(gpa);
    try target.backend().prefill(&s);
    defer target.backend().release(&s);
    store.keep_remote = true;
    defer store.keep_remote = false;
    try std.testing.expect(store.keep(p, at, &s, &.{}, &.{}));
}

/// Every request queued before the host starts, so they arrive together, then each answered in full.
fn fanout(su: Setup, out: *Run) !void {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = su.prefill_chunks };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (su.stepped) target.stepped() else target.backend(), clock.clock());
    defer core.deinit();
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .planned = true, .grid = 16, .chunk = 16 };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, rules, 1 << 24);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 8, .decode_share = su.decode_share });
    host.cache = &store;
    if (su.wait_ms) |ms| host.prefix.ns = @as(i96, ms) * std.time.ns_per_ms;
    var exports: Exports = .{ .target = &target };
    if (su.handoff != null) host.exporter = .{ .ptr = &exports, .snapshots = store.family, .exported = Exports.exported, .failed = Exports.failed };
    if (su.remote_at > 0) try keepRemote(&target, &store, su.remote_at);
    host.keep_remote = su.remote_at > 0;
    const e = host.engine();
    var hook = su.hook;
    hook.engine = e;
    target.prefill_hook = Hook.call;
    target.prefill_hook_ctx = &hook;
    var bufs: [8][prompt_len]u32 = undefined;
    var reqs: [8]Request = undefined;
    const shared = [_]u32{su.shared};
    for (0..su.requests) |i| {
        reqs[i] = .{ .prompt = promptOf(&bufs[i], i), .max_tokens = 6, .shared_prefixes = &shared };
        if (su.handoff) |k| if (k == i) {
            reqs[i].prompt = reqs[i].prompt[0..su.handoff_len];
            reqs[i].handoff = .{ .id = @splat(0x2b) };
        };
        if (su.twin) |k| if (k == i) {
            reqs[i].prompt = reqs[0].prompt;
            reqs[i].chunks = su.twin_chunks;
        };
        try e.submit(@intCast(i + 1), &reqs[i], .{ .ctx = &out.boxes[i], .event = Box.event });
    }
    try host.start();
    defer host.stop();
    for (0..su.requests) |i| {
        out.reasons[i] = out.boxes[i].wait();
        out.cached[i] = out.boxes[i].cached;
        out.remote[i] = out.boxes[i].remote;
    }
    host.lock();
    out.queued_after = host.queued.items.len;
    host.unlock();
    out.waits = host.prefix.waits;
    out.timeouts = host.prefix.timeouts;
    out.exported = exports.tokens;
}

test "prefix wait: requests on a system prompt another prompt is filling wait for its cut and resume it" {
    var on: Run = .{};
    defer on.deinit();
    try fanout(.{}, &on);
    var off: Run = .{};
    defer off.deinit();
    try fanout(.{ .wait_ms = 0 }, &off);
    for (0..5) |i| {
        try std.testing.expectEqual(Reason.length, on.reasons[i]);
        try std.testing.expectEqualSlices(u32, off.boxes[i].tokens.items, on.boxes[i].tokens.items); // resumed == fresh
    }
    // the first computes the system prompt and three resume its cut; the unrelated one neither waits nor is waited for
    try std.testing.expectEqualSlices(?u32, &.{ 0, cut, cut, cut, 0 }, on.cached[0..5]);
    try std.testing.expectEqual(@as(u64, 3), on.waits);
    try std.testing.expectEqual(@as(u64, 0), on.timeouts);
    try std.testing.expectEqual(@as(usize, 0), on.queued_after);
    // off (TENSORFOLD_PREFIX_WAIT_MS=0): every prompt filling side by side computes the system prompt itself
    try std.testing.expectEqualSlices(?u32, &.{ 0, 0, 0, 0, 0 }, off.cached[0..5]);
    try std.testing.expectEqual(@as(u64, 0), off.waits);
}

test "prefix wait: with TF_PREFILL_KEEP a fill resumed from a kept remote state is waited for, and its waiters resume its cut" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .remote_at = 512 }, &r);
    for (0..5) |i| try std.testing.expectEqual(Reason.length, r.reasons[i]);
    // the first resumes the remote state and keeps the cut, remote too; three wait for it rather than fill beside it
    try std.testing.expectEqualSlices(?u32, &.{ 512, cut, cut, cut, 0 }, r.cached[0..5]);
    try std.testing.expectEqualSlices(?u32, &.{ 512, cut, cut, cut, 0 }, r.remote[0..5]); // remote, never exact
    try std.testing.expectEqual(@as(u64, 3), r.waits);
}

test "prefix wait: a cancelled leader releases its waiter, which computes the prefix itself" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .hook = .{ .at = 3, .id = 1 } }, &r);
    try std.testing.expectEqual(Reason.cancelled, r.reasons[0]);
    try std.testing.expectEqual(Reason.length, r.reasons[1]);
    try std.testing.expectEqual(@as(?u32, 0), r.cached[1]);
    try std.testing.expectEqual(@as(u64, 1), r.waits);
    try std.testing.expectEqual(@as(u64, 0), r.timeouts);
}

test "prefix wait: a waiter cancelled while it waits leaves the queue once; the fill runs on" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .hook = .{ .at = 3, .id = 2 } }, &r);
    try std.testing.expectEqual(Reason.length, r.reasons[0]);
    try std.testing.expectEqual(Reason.cancelled, r.reasons[1]);
    try std.testing.expectEqual(@as(u64, 1), r.waits);
    try std.testing.expectEqual(@as(usize, 0), r.queued_after);
}

test "prefix wait: at the bound a waiter is released, counted, and fills beside the leader" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .wait_ms = 1, .hook = .{ .slow_ms = 3 } }, &r);
    try std.testing.expectEqual(Reason.length, r.reasons[1]);
    try std.testing.expectEqual(@as(u64, 1), r.waits);
    try std.testing.expectEqual(@as(u64, 1), r.timeouts);
}

test "prefix wait: a handoff's fill leads the wait for the shared cut it keeps before its own cut, and the waiters resume there" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .handoff = 0 }, &r);
    for (0..5) |i| try std.testing.expectEqual(Reason.length, r.reasons[i]);
    try std.testing.expectEqualSlices(?u32, &.{ 0, cut, cut, cut, 0 }, r.cached[0..5]);
    try std.testing.expectEqual(@as(u64, 3), r.waits);
    try std.testing.expectEqual(@as(?usize, prompt_len - 1), r.exported); // and it still exports at its own cut
}

test "prefix wait: a request whose own chunks start at a handoff's off-grid cut does not wait for it, as the handoff never keeps it" {
    var r: Run = .{};
    defer r.deinit();
    const starts = [_]u32{prompt_len - 1}; // the handoff's cut: a start for its twin, off the leader's 16-row grid
    try fanout(.{ .requests = 2, .handoff = 0, .twin = 1, .twin_chunks = &starts }, &r);
    for (0..2) |i| try std.testing.expectEqual(Reason.length, r.reasons[i]);
    try std.testing.expectEqual(@as(u64, 0), r.waits);
    try std.testing.expectEqual(@as(?usize, prompt_len - 1), r.exported);
}

test "prefix wait: a handoff waits for no cut at or past its own, which it could not resume and still export" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .handoff = 1, .handoff_len = cut + 1 }, &r); // the leader keeps the handoff's cut
    try std.testing.expectEqual(@as(u64, 0), r.waits);
    try std.testing.expectEqual(@as(?u32, 0), r.cached[1]);
    try std.testing.expectEqual(@as(?usize, cut), r.exported);
}

test "prefix wait: whole prompt passes never wait, and the kept cut still serves the later requests" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .stepped = false }, &r);
    try std.testing.expectEqualSlices(?u32, &.{ 0, cut, cut, cut, 0 }, r.cached[0..5]);
    try std.testing.expectEqual(@as(u64, 0), r.waits);
}

test "prefix wait: a shared cut under the store's min_gap is not waited for" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .shared = 130 }, &r); // its cut, 128, adds under 256 tokens past the waiter's hit
    try std.testing.expectEqual(@as(u64, 0), r.waits);
}

test "prefix wait: TENSORFOLD_PREFIX_WAIT_MS is whole milliseconds, 0 off, anything else the default" {
    const default_ns = @as(i96, prefix.Wait.default_ms) * std.time.ns_per_ms;
    try std.testing.expectEqual(default_ns, prefix.Wait.fromText("").ns);
    try std.testing.expectEqual(default_ns, prefix.Wait.fromText("x").ns);
    try std.testing.expectEqual(default_ns, prefix.Wait.fromText("-5").ns);
    try std.testing.expectEqual(default_ns, prefix.Wait.fromText("1.5").ns);
    try std.testing.expectEqual(@as(i96, 0), prefix.Wait.fromText("0").ns);
    try std.testing.expectEqual(@as(i96, 250) * std.time.ns_per_ms, prefix.Wait.fromText("250").ns);
}

test "prefix wait: with an active share and the cap on, a waiter is released within the bound and replies equal the no-share run" {
    _ = @import("sys").env.unsetenv("TENSORFOLD_PREFILL_SHARE"); // the flag carries the share here; the env must not leak in
    var on: Run = .{};
    defer on.deinit();
    try fanout(.{ .requests = 2, .wait_ms = 25, .prefill_chunks = 1, .decode_share = 0.5 }, &on);
    var off: Run = .{};
    defer off.deinit();
    try fanout(.{ .requests = 2, .wait_ms = 25, .prefill_chunks = 1 }, &off);
    for (0..2) |i| {
        try std.testing.expectEqual(Reason.length, on.reasons[i]);
        try std.testing.expectEqualSlices(u32, off.boxes[i].tokens.items, on.boxes[i].tokens.items); // capped chunks: resumed == fresh
    }
    try std.testing.expectEqual(@as(u64, 1), on.waits); // the leader's cut released the waiter
    try std.testing.expectEqual(@as(u64, 0), on.timeouts); // within the 25 ms bound, not at it
}

test "prefix wait: a kept remote leader whose copy is refused prefills from 0, so its cut is exact and so are its waiters" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .remote_at = 512, .hook = .{ .refuse_at = 512 } }, &r);
    for (0..5) |i| try std.testing.expectEqual(Reason.length, r.reasons[i]);
    try std.testing.expectEqualSlices(?u32, &.{ 0, cut, cut, cut, 0 }, r.cached[0..5]); // the leader ran from token 0
    try std.testing.expectEqualSlices(?u32, &.{ 0, 0, 0, 0, 0 }, r.remote[0..5]); // so the cut it kept is the Mac's own
    try std.testing.expectEqual(@as(u64, 3), r.waits);
}
