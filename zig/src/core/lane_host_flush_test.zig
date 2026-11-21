//! A prompt-cache flush drops every kept state once the requests in flight end; a later turn prefills cold.
const std = @import("std");
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
            .logprobs, .prompt_logprobs, .audio => {},
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

const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };

/// The next turn after `t1` and its reply `first`: the history, four reply tokens, a tool result, a new prompt.
fn nextTurn(first: []const u32) ![]u32 {
    return std.mem.concat(gpa, u32, &.{ &t1, first[0..4], &.{ 7, 7, 5, 3 } });
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

fn warm(e: api.Engine) !Warm {
    var first = try run(e, 1, &.{ .prompt = &t1, .max_tokens = 6, .history_len = 8 });
    errdefer first.tokens.deinit(gpa);
    const prompt = try nextTurn(first.tokens.items);
    errdefer gpa.free(prompt);
    const r: Request = .{ .prompt = prompt, .max_tokens = 12, .history_len = @intCast(prompt.len - 2) };
    const second = try run(e, 2, &r);
    try std.testing.expectEqual(@as(?u32, 8), second.cached); // resumed from the first turn's kept state
    return .{ .first = first.tokens, .prompt = prompt, .second = second.tokens };
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
