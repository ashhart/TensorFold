//! Forgetting one conversation on the lane host: its next turn prefills cold with the same tokens, others still resume.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
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
    const vtable: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop };
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,
    ask: ?*const api.ImprintAsk = null, // whether it was answered as this request prefilled and as it ended
    asked_at_prefill: ?bool = null,
    asked_at_end: ?bool = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        const asked = if (b.ask) |a| a.done.load(.acquire) else null;
        switch (e.*) {
            .prefilled => |c| {
                b.cached = c;
                b.asked_at_prefill = asked;
            },
            .logprobs, .prompt_logprobs, .audio => {},
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.asked_at_end = asked;
                b.done = f.reason;
            },
        }
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
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
};

/// A lane host on the fake backend with a prompt cache, its parts owned here.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake = .{ .gpa = gpa },
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    store: pc.Store = undefined,
    host: LaneHost = undefined,

    fn init(r: *Rig, gate: ?*Gate, step: u32) !void {
        const cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.* = .{ .cfg = cfg };
        if (gate) |g| {
            r.target.prefill_hook = Gate.chunk;
            r.target.prefill_hook_ctx = g;
        }
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        const family: pc.Snapshots = .{ .ptr = &r.target, .vtable = &Snaps.vtable };
        r.store = pc.Store.init(gpa, family, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        r.host = LaneHost.init(gpa, io, &r.core, .{ .lanes = 2, .prefill_step = step });
        r.host.cache = &r.store;
    }

    fn deinit(r: *Rig) void {
        r.host.stop();
        r.store.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
    }
};

const Turn = struct { tokens: std.ArrayList(u32), cached: ?u32 };

fn run(e: api.Engine, id: api.Id, r: Request) !Turn {
    var b: Box = .{};
    try e.submit(id, &r, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b.wait());
    return .{ .tokens = b.tokens, .cached = b.cached };
}

fn turn(e: api.Engine, id: api.Id, prompt: []const u32, history: u32, max: u32) !Turn {
    return run(e, id, .{ .prompt = prompt, .max_tokens = max, .history_len = history });
}

fn forget(e: api.Engine, ask: *api.ImprintAsk) !void {
    try std.testing.expect(e.imprintAsk(ask));
    try std.testing.expect(ask.wait(io, 5000));
}

/// A conversation's first turn and its second (the history, four reply tokens, a tool result, a new prompt).
const Talk = struct {
    second: []u32,
    first_reply: std.ArrayList(u32),
    fn start(e: api.Engine, id: api.Id, first: []const u32) !Talk {
        var t1 = try turn(e, id, first, 8, 6);
        errdefer t1.tokens.deinit(gpa);
        const second = try std.mem.concat(gpa, u32, &.{ first, t1.tokens.items[0..4], &.{ 7, 7, 5, 3 } });
        return .{ .first_reply = t1.tokens, .second = second };
    }
    fn deinit(t: *Talk) void {
        t.first_reply.deinit(gpa);
        gpa.free(t.second);
    }
    fn history(t: *const Talk) u32 {
        return @intCast(t.second.len - 2);
    }
};

const x1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
const y1 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8 };

test "forget one conversation: its next turn starts cold with the same tokens, the other resumes, unknown drops none" {
    var rig: Rig = undefined;
    try rig.init(null, 0);
    defer rig.deinit();
    try rig.host.start();
    const e = rig.host.engine();
    var x = try Talk.start(e, 1, &x1);
    defer x.deinit();
    var y = try Talk.start(e, 2, &y1);
    defer y.deinit();
    var xw = try turn(e, 3, x.second, x.history(), 12);
    defer xw.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 8), xw.cached);
    var yw = try turn(e, 4, y.second, y.history(), 12);
    defer yw.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 8), yw.cached);
    // a conversation the server never saw: nothing goes
    const before = rig.store.entries.items.len;
    const z = [_]u32{ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 };
    var none: api.ImprintAsk = .{ .op = .drop, .prompt = &z };
    try forget(e, &none);
    try std.testing.expectEqual(@as(u32, 0), none.dropped + none.shared);
    try std.testing.expectEqual(before, rig.store.entries.items.len);
    // x named by its second turn's prompt: its states go, the second turn's reply among them
    var gone: api.ImprintAsk = .{ .op = .drop, .prompt = x.second };
    try forget(e, &gone);
    try std.testing.expect(gone.dropped >= 2 and gone.ram_freed > 0);
    try std.testing.expectEqual(@as(u32, 0), gone.out.ram); // nothing of x left to resume
    var xc = try turn(e, 5, x.second, x.history(), 12);
    defer xc.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 0), xc.cached);
    try std.testing.expectEqualSlices(u32, xw.tokens.items, xc.tokens.items);
    // y still resumes, with the same tokens
    var yr = try turn(e, 6, y.second, y.history(), 12);
    defer yr.tokens.deinit(gpa);
    try std.testing.expect(yr.cached.? >= 8);
    try std.testing.expectEqualSlices(u32, yw.tokens.items, yr.tokens.items);
}

test "forget one conversation: a turn in flight ends unchanged first, then its states go; arrivals wait, start cold" {
    var gate: Gate = .{};
    var rig: Rig = undefined;
    try rig.init(&gate, 4); // prompts in chunks of 4
    defer rig.deinit();
    try rig.host.start();
    const e = rig.host.engine();
    var x = try Talk.start(e, 1, &x1);
    defer x.deinit();
    var second = try turn(e, 2, x.second, x.history(), 12);
    defer second.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 8), second.cached);
    // x's third turn, held inside its first prompt chunk while x (named by its second) is forgotten
    const third = try std.mem.concat(gpa, u32, &.{ x.second, second.tokens.items[0..4], &.{ 7, 7, 5, 3 } });
    defer gpa.free(third);
    const r: Request = .{ .prompt = third, .max_tokens = 12, .history_len = @intCast(third.len - 2) };
    var ask: api.ImprintAsk = .{ .op = .drop, .prompt = x.second };
    gate.armed.store(true, .release);
    var b: Box = .{ .ask = &ask };
    defer b.tokens.deinit(gpa);
    try e.submit(3, &r, .{ .ctx = &b, .event = Box.event });
    while (!gate.inside.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expect(e.imprintAsk(&ask));
    // the same turn asked behind the forget, though a lane is free: it starts after it, and prefills cold
    var later: Box = .{ .ask = &ask };
    defer later.tokens.deinit(gpa);
    try e.submit(4, &r, .{ .ctx = &later, .event = Box.event });
    gate.open.store(true, .release);
    try std.testing.expect(ask.wait(io, 5000));
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?bool, false), b.asked_at_end); // it ended before the forget ran
    try std.testing.expectEqual(@as(?u32, x.history()), b.cached); // it resumed the second turn's state
    try std.testing.expect(ask.dropped >= 3); // the first two turns' states and the one it kept as it ran
    try std.testing.expectEqual(Reason.length, later.wait());
    try std.testing.expectEqual(@as(?bool, true), later.asked_at_prefill);
    try std.testing.expectEqual(@as(?u32, 0), later.cached); // nothing of x left, the third turn's own state included
    try std.testing.expectEqualSlices(u32, b.tokens.items, later.tokens.items); // resumed or cold, the same reply
}

test "forget one conversation: a framed family's image request is named by its framed key; other frames stay" {
    var rig: Rig = undefined;
    try rig.init(null, 0);
    defer rig.deinit();
    rig.host.frame_key = struct {
        fn key(images: []const api.Image) u32 {
            return images[0].gw;
        }
    }.key;
    try rig.host.start();
    const e = rig.host.engine();
    // prompt ids 2..5 are one image's placeholders (token 9); cache_key holds ids derived from its content
    const t1 = [_]u32{ 3, 1, 9, 9, 9, 9, 2, 6, 5, 3 };
    var key = t1;
    for (2..6) |k| key[k] = 0x8000_0000 + @as(u32, @intCast(k));
    const a = [_]api.Image{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 2, .tokens = 4, .hash = 42 }};
    const b = [_]api.Image{.{ .pixels = &.{}, .gh = 1, .gw = 2, .at = 2, .tokens = 4, .hash = 42 }};
    var ra = try run(e, 1, .{ .prompt = &t1, .max_tokens = 6, .history_len = 8, .cache_key = &key, .images = &a });
    defer ra.tokens.deinit(gpa);
    var rb = try run(e, 2, .{ .prompt = &t1, .max_tokens = 6, .history_len = 8, .cache_key = &key, .images = &b });
    defer rb.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), rig.store.entries.items.len); // one state a frame, at 8
    var ask: api.ImprintAsk = .{ .op = .drop, .prompt = &key, .images = &a };
    try forget(e, &ask);
    try std.testing.expectEqual(@as(u32, 1), ask.dropped);
    try std.testing.expectEqual(@as(usize, 1), rig.store.entries.items.len);
    var again = try run(e, 3, .{ .prompt = &t1, .max_tokens = 6, .history_len = 8, .cache_key = &key, .images = &a });
    defer again.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 0), again.cached);
    try std.testing.expectEqualSlices(u32, ra.tokens.items, again.tokens.items);
    var other = try run(e, 4, .{ .prompt = &t1, .max_tokens = 6, .history_len = 8, .cache_key = &key, .images = &b });
    defer other.tokens.deinit(gpa);
    try std.testing.expectEqual(@as(?u32, 8), other.cached); // the other frame's conversation still resumes
}
