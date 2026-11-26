//! A prompt-cache flush drops every kept state once the requests in flight end; a later turn prefills cold.
const std = @import("std");
const fixture = @import("rows_async_fixture.zig");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;
const io = std.testing.io;

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

/// A request's events; whether `flush` had answered when its prompt pass reported and when it ended, and `peer` ended.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    logprobs: std.ArrayList(api.LogprobRow) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,
    ended: std.atomic.Value(bool) = .init(false),
    flush: ?*const api.ImprintAsk = null,
    peer: ?*const Box = null,
    flushed_at_prefill: ?bool = null,
    flushed_at_end: ?bool = null,
    peer_ended_at_prefill: ?bool = null,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        const answered = if (b.flush) |f| f.done.load(.acquire) else null;
        switch (e.*) {
            .prefilled => |c| {
                b.cached = c;
                b.flushed_at_prefill = answered;
                b.peer_ended_at_prefill = if (b.peer) |p| p.ended.load(.acquire) else null;
            },
            .logprobs => |rows| b.logprobs.appendSlice(gpa, rows) catch @panic("OOM"),
            .prompt_logprobs, .audio => {},
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.flushed_at_end = answered;
                b.ended.store(true, .release);
            },
        }
    }
    fn finished(b: *@This()) ?Reason {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.done;
    }
    fn wait(b: *@This()) !Reason {
        for (0..5000) |_| {
            if (b.finished()) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.RequestTimedOut;
    }
};

/// Holds the engine thread inside the first prompt pass after `armed`, until `open`.
const Gate = struct {
    armed: std.atomic.Value(bool) = .init(false),
    inside: std.atomic.Value(bool) = .init(false),
    open: std.atomic.Value(bool) = .init(false),
    fn chunk(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
        const g: *@This() = @ptrCast(@alignCast(ctx));
        if (!g.armed.swap(false, .acq_rel)) return;
        g.inside.store(true, .release);
        while (!g.open.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    fn entered(g: *@This()) !void {
        for (0..5000) |_| {
            if (g.inside.load(.acquire)) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.GateNotReached;
    }
};

/// A lane host over the fake target with a prompt cache (`cached`), two lanes; `gate` holds a prompt pass.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake = .{ .gpa = gpa },
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    store: pc.Store = undefined,
    host: LaneHost = undefined,
    gate: Gate = .{},

    fn start(r: *Rig, cached: bool) !void {
        r.target.prefill_hook = Gate.chunk;
        r.target.prefill_hook_ctx = &r.gate;
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        const family: pc.Snapshots = .{ .ptr = &r.target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } };
        r.store = pc.Store.init(gpa, family, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        r.host = LaneHost.init(gpa, io, &r.core, .{ .lanes = 2 });
        if (cached) r.host.cache = &r.store;
        try r.host.start();
    }
    fn deinit(r: *Rig) void {
        r.host.stop();
        r.store.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
    }
    fn engine(r: *Rig) api.Engine {
        return r.host.engine();
    }
};

fn rig() !Rig {
    return .{ .cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7) };
}

/// One request run to its end; its tokens (the caller frees them) and where it resumed.
fn run(e: api.Engine, id: Id, r: *const Request) !struct { tokens: std.ArrayList(u32), cached: ?u32 } {
    var b: Box = .{};
    try e.submit(id, r, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.length, try b.wait());
    return .{ .tokens = b.tokens, .cached = b.cached };
}

const Exact = struct {
    tokens: std.ArrayList(u32),
    logprobs: std.ArrayList(api.LogprobRow),
    cached: ?u32,

    fn deinit(x: *Exact) void {
        x.tokens.deinit(gpa);
        x.logprobs.deinit(gpa);
    }
};

fn exact(e: api.Engine, id: Id, r: *const Request) !Exact {
    var b: Box = .{};
    var request = r.*;
    request.logprobs = 3;
    try e.submit(id, &request, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.length, try b.wait());
    return .{ .tokens = b.tokens, .logprobs = b.logprobs, .cached = b.cached };
}

const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };

/// The next turn after the initial prompt and its reply `first`.
fn nextTurn(first: []const u32, base: []const u32) ![]u32 {
    return std.mem.concat(gpa, u32, &.{ base, first[0..4], &.{ 7, 7, 5, 3 } });
}

/// Two turns of one conversation, the second resumed from the first's kept state; its request and tokens.
const Warm = struct {
    first: std.ArrayList(u32),
    prompt: []u32,
    second: std.ArrayList(u32),
    fn request(w: *const Warm) Request {
        return .{ .prompt = w.prompt, .max_tokens = 12, .history_len = @intCast(w.prompt.len - 2) };
    }
    fn deinit(w: *Warm) void {
        w.first.deinit(gpa);
        gpa.free(w.prompt);
        w.second.deinit(gpa);
    }
};

fn warmAt(e: api.Engine, salt: u32) !Warm {
    var base = t1;
    base[0] += salt;
    var first = try run(e, 1, &.{ .prompt = &base, .max_tokens = 6, .history_len = 8 });
    errdefer first.tokens.deinit(gpa);
    const prompt = try nextTurn(first.tokens.items, &base);
    errdefer gpa.free(prompt);
    const r: Request = .{ .prompt = prompt, .max_tokens = 12, .history_len = @intCast(prompt.len - 2) };
    const second = try run(e, 2, &r);
    try std.testing.expectEqual(@as(?u32, 8), second.cached);
    return .{ .first = first.tokens, .prompt = prompt, .second = second.tokens };
}

fn warm(e: api.Engine) !Warm {
    return warmAt(e, 0);
}

test "prompt cache flush: every kept state goes, and the resumed turn then prefills cold with the same tokens" {
    var r = try rig();
    try r.start(true);
    defer r.deinit();
    const e = r.engine();
    var w = try warm(e);
    defer w.deinit();
    const kept = r.store.entries.items.len;
    const held = r.store.held;
    try std.testing.expect(kept >= 2);
    var ask: api.ImprintAsk = .{ .op = .flush };
    try std.testing.expect(e.imprintAsk(&ask));
    try std.testing.expect(ask.wait(io, 5000));
    try std.testing.expectEqual(@as(u32, @intCast(kept)), ask.dropped);
    try std.testing.expectEqual(held, ask.ram_freed);
    try std.testing.expectEqual(@as(u64, 0), ask.freed); // no disk store here
    try std.testing.expect(!ask.disk);
    try std.testing.expectEqual(@as(usize, 0), r.store.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), r.store.held);
    const req = w.request();
    var cold = try run(e, 3, &req);
    defer cold.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 0), cold.cached); // nothing to resume: the whole prompt prefilled
    try std.testing.expectEqualSlices(u32, w.second.items, cold.tokens.items);
}

fn expectReplay(w: *const Warm, want: *const Exact, got: *const Exact, cached: ?u32) !void {
    try std.testing.expectEqual(cached, got.cached);
    try std.testing.expectEqualSlices(u32, w.second.items, got.tokens.items);
    try std.testing.expectEqual(want.logprobs.items.len, got.logprobs.items.len);
    for (want.logprobs.items, got.logprobs.items) |a, b| {
        try std.testing.expectEqual(a.token, b.token);
        try std.testing.expectEqual(a.logprob, b.logprob);
        try std.testing.expectEqualSlices(u32, a.ids[0..a.count], b.ids[0..b.count]);
        try std.testing.expectEqualSlices(f32, a.logprobs[0..a.count], b.logprobs[0..b.count]);
    }
}

test "prompt cache flush: retain one resumes exactly and drops the other RAM state" {
    var r = try rig();
    try r.start(true);
    defer r.deinit();
    const e = r.engine();
    var w0 = try warmAt(e, 0);
    defer w0.deinit();
    var w1 = try warmAt(e, 20);
    defer w1.deinit();
    var w2 = try warmAt(e, 40);
    defer w2.deinit();
    while (r.store.entries.items.len > 3) {
        var removed = false;
        for (r.store.entries.items, 0..) |entry, i| {
            if (entry.at == 8) {
                r.store.remove(i);
                removed = true;
                break;
            }
        }
        try std.testing.expect(removed);
    }
    try std.testing.expectEqual(@as(usize, 3), r.store.entries.items.len);
    var b0 = try exact(e, 10, &w0.request());
    defer b0.deinit();
    var b1 = try exact(e, 11, &w1.request());
    defer b1.deinit();
    var b2 = try exact(e, 12, &w2.request());
    defer b2.deinit();
    const retained = r.store.find(w2.prompt, &.{}, &.{}) orelse return error.MissingState;
    try std.testing.expectEqual(@as(usize, 3), r.store.entries.items.len);
    const held = r.store.held;
    var ask: api.ImprintAsk = .{ .op = .flush, .retain = 1 };
    ask.answer(&r.store, .{});
    try std.testing.expectEqual(@as(usize, 2), ask.dropped);
    try std.testing.expectEqual(@as(usize, 1), r.store.entries.items.len);
    try std.testing.expectEqual(retained.bytes, r.store.held);
    try std.testing.expectEqual(held - retained.bytes, ask.ram_freed);
    try std.testing.expect(r.store.find(w0.prompt, &.{}, &.{}) == null);
    try std.testing.expect(r.store.find(w1.prompt, &.{}, &.{}) == null);
    try std.testing.expect(r.store.find(w2.prompt, &.{}, &.{}) == retained);
    var hit = try exact(e, 20, &w2.request());
    defer hit.deinit();
    try expectReplay(&w2, &b2, &hit, retained.at);
    var cold0 = try exact(e, 21, &w0.request());
    defer cold0.deinit();
    try expectReplay(&w0, &b0, &cold0, 0);
    var cold1 = try exact(e, 22, &w1.request());
    defer cold1.deinit();
    try expectReplay(&w1, &b1, &cold1, 0);
}

test "prompt cache flush: retain one removes only the other two disk states" {
    const x = try fixture.Fixture.open("flush-retain", .off, true, 1 << 20);
    defer x.close();
    const p0 = fixture.prompt(70);
    const p1 = fixture.prompt(71);
    const p2 = fixture.prompt(72);
    for ([_]*const [1100]u32{ &p0, &p1, &p2 }, 0..) |p, i| try x.keep(p, @intCast(300 + i * 300));
    try x.idle(3);
    try std.testing.expectEqual(@as(usize, 3), x.s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 3), x.r.states.items.len);
    const kept_disk = x.r.states.items[x.r.best(&p2, &.{}, &.{}, 0).?].own + x.r.states.items[x.r.best(&p2, &.{}, &.{}, 0).?].rows;
    const held_ram = x.s.held;
    const held_disk = x.r.held();
    const kept_ram = x.s.entries.items[2].bytes;
    var ask: api.ImprintAsk = .{ .op = .flush, .retain = 1 };
    ask.answer(&x.s, .{});
    try std.testing.expectEqual(@as(usize, 2), ask.dropped);
    try std.testing.expectEqual(held_ram - kept_ram, ask.ram_freed);
    try std.testing.expectEqual(kept_ram, x.s.held);
    try std.testing.expectEqual(held_disk - kept_disk, ask.freed);
    try std.testing.expectEqual(kept_disk, x.r.held());
    try std.testing.expectEqual(@as(usize, 1), x.s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
    try std.testing.expect(x.r.best(&p0, &.{}, &.{}, 0) == null);
    try std.testing.expect(x.r.best(&p1, &.{}, &.{}, 0) == null);
    try std.testing.expect(x.r.best(&p2, &.{}, &.{}, 0) != null);
}

test "prompt cache flush: a request in flight finishes unchanged and the flush answers after it; arrivals wait" {
    var r = try rig();
    try r.start(true);
    defer r.deinit();
    const e = r.engine();
    var w = try warm(e);
    defer w.deinit();
    const req = w.request();
    // the same turn again, held inside its prompt pass while two flushes are asked
    var ask: api.ImprintAsk = .{ .op = .flush };
    var again: api.ImprintAsk = .{ .op = .flush };
    r.gate.armed.store(true, .release);
    var b: Box = .{ .flush = &ask };
    defer b.tokens.deinit(gpa);
    try e.submit(3, &req, .{ .ctx = &b, .event = Box.event });
    try r.gate.entered();
    try std.testing.expect(e.imprintAsk(&ask));
    try std.testing.expect(e.imprintAsk(&again));
    // a request submitted behind the flushes waits for them, though a lane is free: it starts after, and prefills cold
    var later: Box = .{ .flush = &ask };
    defer later.tokens.deinit(gpa);
    try e.submit(4, &req, .{ .ctx = &later, .event = Box.event });
    r.gate.open.store(true, .release);
    try std.testing.expect(ask.wait(io, 5000));
    try std.testing.expect(again.wait(io, 5000));
    try std.testing.expectEqual(Reason.length, try b.wait());
    try std.testing.expectEqual(@as(?bool, false), b.flushed_at_end); // it ended before the flush ran
    try std.testing.expect(b.cached.? > 0); // it resumed from a kept state, which the flush left alone while it ran
    try std.testing.expectEqualSlices(u32, w.second.items, b.tokens.items);
    try std.testing.expect(ask.dropped >= 2);
    try std.testing.expectEqual(@as(u32, 0), again.dropped); // the second flush found nothing left
    try std.testing.expectEqual(Reason.length, try later.wait());
    try std.testing.expectEqual(@as(?bool, true), later.flushed_at_prefill);
    try std.testing.expectEqual(@as(?u32, 0), later.cached);
    try std.testing.expectEqualSlices(u32, w.second.items, later.tokens.items);
}

test "prompt cache flush: retain one while the newest conversation is in flight" {
    var r = try rig();
    try r.start(true);
    defer r.deinit();
    const e = r.engine();
    var w = try warm(e);
    defer w.deinit();
    var ask: api.ImprintAsk = .{ .op = .flush, .retain = 1 };
    const req = w.request();
    r.gate.armed.store(true, .release);
    var b: Box = .{ .flush = &ask };
    defer b.tokens.deinit(gpa);
    try e.submit(3, &req, .{ .ctx = &b, .event = Box.event });
    try r.gate.entered();
    try std.testing.expect(e.imprintAsk(&ask));
    var later: Box = .{ .flush = &ask };
    defer later.tokens.deinit(gpa);
    try e.submit(4, &req, .{ .ctx = &later, .event = Box.event });
    r.gate.open.store(true, .release);
    try std.testing.expect(ask.wait(io, 5000));
    try std.testing.expectEqual(Reason.length, try b.wait());
    try std.testing.expectEqual(@as(?bool, false), b.flushed_at_end);
    try std.testing.expect(b.cached.? > 0);
    try std.testing.expectEqualSlices(u32, w.second.items, b.tokens.items);
    try std.testing.expectEqual(@as(u32, 1), ask.dropped);
    try std.testing.expectEqual(Reason.length, try later.wait());
    try std.testing.expectEqual(@as(?bool, true), later.flushed_at_prefill);
    try std.testing.expect(later.cached.? > 0);
    try std.testing.expectEqualSlices(u32, w.second.items, later.tokens.items);
    try std.testing.expectEqual(@as(usize, 1), r.store.entries.items.len);
}

test "prompt cache flush: a withdrawn flush releases the hold at once and drops nothing" {
    var r = try rig();
    try r.start(true);
    defer r.deinit();
    const e = r.engine();
    var w = try warm(e);
    defer w.deinit();
    const req = w.request();
    var ask: api.ImprintAsk = .{ .op = .flush };
    r.gate.armed.store(true, .release);
    var b: Box = .{};
    defer b.tokens.deinit(gpa);
    try e.submit(3, &req, .{ .ctx = &b, .event = Box.event });
    try r.gate.entered();
    try std.testing.expect(e.imprintAsk(&ask));
    var later: Box = .{ .peer = &b };
    defer later.tokens.deinit(gpa);
    try e.submit(4, &req, .{ .ctx = &later, .event = Box.event });
    try std.testing.expect(ask.withdraw()); // its caller's wait ran out before any request ended
    try std.testing.expect(!ask.withdraw());
    r.gate.open.store(true, .release);
    try std.testing.expectEqual(Reason.length, try b.wait());
    try std.testing.expectEqual(Reason.length, try later.wait());
    try std.testing.expectEqual(@as(?bool, false), later.peer_ended_at_prefill); // admitted beside it: nothing held
    try std.testing.expect(later.cached.? > 0); // it resumed from a kept state
    try std.testing.expectEqualSlices(u32, w.second.items, later.tokens.items);
    try std.testing.expect(!ask.done.load(.acquire));
    try std.testing.expect(r.store.entries.items.len >= 2);
    var next = try run(e, 5, &req); // the host goes on admitting once it is idle and has dropped the withdrawn ask
    defer next.tokens.deinit(gpa);
    try std.testing.expect(next.cached.? > 0);
    try std.testing.expect(!ask.done.load(.acquire));
}

test "prompt cache flush: a lane host without a prompt cache refuses it at once, holding nothing" {
    var r = try rig();
    try r.start(false);
    defer r.deinit();
    const e = r.engine();
    var ask: api.ImprintAsk = .{ .op = .flush };
    try std.testing.expect(!e.imprintAsk(&ask));
    var first = try run(e, 1, &.{ .prompt = &t1, .max_tokens = 6, .history_len = 8 });
    defer first.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 0), first.cached);
}
