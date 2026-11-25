//! Folded memory through the actual lane host: routed only when serving learning, never a prompt of placeholders.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const t = std.testing;

/// A request's tokens and its finished event's reason.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    failed: bool = false,
    reason: ?api.Reason = null,
    memory_version: ?u64 = null,
    message: [160]u8 = @splat(0), // the finish's message, cut to fit
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(t.io);
        defer b.mutex.unlock(t.io);
        switch (e.*) {
            .tokens => |got| b.tokens.appendSlice(t.allocator, got) catch {
                b.failed = true;
            },
            .finished => |f| {
                b.reason = f.reason;
                b.memory_version = f.stats.memory_version;
                const n = @min(f.message.len, b.message.len);
                @memcpy(b.message[0..n], f.message[0..n]);
            },
            else => {},
        }
    }
    fn wait(b: *@This()) !void {
        for (0..10_000) |_| { // 10 s
            b.mutex.lockUncancelable(t.io);
            const done = b.reason != null;
            const failed = b.failed;
            b.mutex.unlock(t.io);
            if (failed) return error.OutOfMemory;
            if (done) return;
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
    fn deinit(b: *@This()) void {
        b.tokens.deinit(t.allocator);
    }
};

/// A bank with units: counts its routes, and answers each with `handle` (null: a miss).
const Ctrl = struct {
    routed: u32 = 0,
    dropped: u32 = 0,
    execs: u32 = 0,
    handle: ?api.MemorySnapHandle = null,
    version: u64 = 7,
    fn of(ctx: *anyopaque) *Ctrl {
        return @ptrCast(@alignCast(ctx));
    }
    pub fn controller(c: *Ctrl) api.MemoryController {
        return .{ .ctx = c, .hasUnits = has, .routeAndCompose = route, .dropSnap = drop, .exec = exec, .version = versionOf };
    }
    fn versionOf(ctx: *anyopaque) u64 {
        return of(ctx).version;
    }
    fn has(_: *anyopaque) bool {
        return true;
    }
    fn route(ctx: *anyopaque, _: []const u32) anyerror!?api.MemorySnapHandle {
        of(ctx).routed += 1;
        return of(ctx).handle;
    }
    fn drop(ctx: *anyopaque, _: *anyopaque) void {
        of(ctx).dropped += 1;
    }
    fn exec(ctx: *anyopaque, _: *const api.MemoryOp) api.MemoryResult {
        of(ctx).execs += 1;
        return .{};
    }
};

const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const query = [_]u32{ 5, 9 };
const request: api.Request = .{ .prompt = &prompt, .max_tokens = 12, .memory_query = &query };

/// One request through a lane host over the fake target, `ctrl` attached when given; `serving`: Info.folded_memory.
fn reply(ctrl: ?*Ctrl, serving: bool, box: *Box, prepare: ?*const fn (*lanes.Engine) void) !void {
    return replyPriced(ctrl, serving, box, prepare, null);
}

/// `reply` on a host whose ledger holds `bytes`, a request costing its prompt rows and reply tokens (null: none).
fn replyPriced(ctrl: ?*Ctrl, serving: bool, box: *Box, prepare: ?*const fn (*lanes.Engine) void, bytes: ?u64) !void {
    const Cost = struct {
        fn cost(_: *anyopaque, prompt_rows: u32, max_tokens: u32) u64 {
            return @as(u64, prompt_rows) + max_tokens;
        }
    };
    var unused: u8 = 0;
    const gpa = t.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    if (prepare) |p| p(&core);
    var host = LaneHost.init(gpa, t.io, &core, .{ .lanes = 2, .folded_memory = serving, .ledger = if (bytes) |n| .{ .ctx = &unused, .total = n, .cost = Cost.cost } else null });
    if (ctrl) |c| host.folded_memory = c.controller();
    try host.start();
    defer host.stop();
    try host.engine().submit(1, &request, .{ .ctx = box, .event = Box.event });
    try box.wait();
}

test "folded memory: an attached bank routes only when the engine serves learning, and off the reply is main's" {
    var plain: Box = .{};
    defer plain.deinit();
    try reply(null, false, &plain, null);
    // a bank attached without --slide (as a saved file once did): never asked, the reply unchanged
    var c_off: Ctrl = .{};
    var off: Box = .{};
    defer off.deinit();
    try reply(&c_off, false, &off, null);
    try t.expectEqual(@as(u32, 0), c_off.routed);
    try t.expectEqualSlices(u32, plain.tokens.items, off.tokens.items);
    try t.expectEqual(plain.reason, off.reason);
    // serving learning: asked once; a miss changes nothing
    var c_on: Ctrl = .{};
    var on: Box = .{};
    defer on.deinit();
    try reply(&c_on, true, &on, null);
    try t.expectEqual(@as(u32, 1), c_on.routed);
    try t.expectEqualSlices(u32, plain.tokens.items, on.tokens.items);
}

test "folded memory: without a bank the in-flight answer takes no host lock between rounds" {
    const gpa = t.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, t.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Answer = struct {
        done: std.atomic.Value(bool) = .init(false),
        fn run(a: *@This(), h: *LaneHost) void {
            @import("lane_host_memory.zig").answerOneInFlight(h);
            a.done.store(true, .release);
        }
    };
    var answer: Answer = .{};
    host.lock(); // as a request's admission holds it: the engine thread's call must not wait for it
    const thread = std.Thread.spawn(.{}, Answer.run, .{ &answer, &host }) catch |e| {
        host.unlock();
        return e;
    };
    var waited: usize = 0;
    while (!answer.done.load(.acquire) and waited < 2_000) : (waited += 1) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    const unblocked = answer.done.load(.acquire);
    host.unlock();
    thread.join();
    try t.expect(unblocked);
}

/// The fake as a family whose memory restore fails: it refuses, and notes a prompt reaching prefill zeros first.
const Unrestorable = struct {
    var base: *const lanes.backend.Backend.VTable = undefined;
    var vt: lanes.backend.Backend.VTable = undefined;
    var snap: u8 = 0;
    var fed_placeholders = false;
    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        if (s.reuse.saved == @as(?*anyopaque, @ptrCast(&snap))) {
            s.reuse_failed = true; // as a refused snapshot restore sets it
            try lanes.memory_restore.refuseFallback(s);
            s.reuse = .{};
        }
        if (s.prompt().len > 0 and s.prompt()[0] == 0) fed_placeholders = true;
        return base.prefill(ptr, s);
    }
    fn install(core: *lanes.Engine) void {
        base = core.backend.vtable;
        vt = base.*;
        vt.prefill = prefill;
        core.backend.vtable = &vt;
    }
};

test "folded memory: a prefix the backend cannot restore runs the request's own prompt, never placeholders as text" {
    var plain: Box = .{};
    defer plain.deinit();
    try reply(null, false, &plain, null);
    Unrestorable.fed_placeholders = false;
    var c: Ctrl = .{ .handle = .{ .saved = @ptrCast(&Unrestorable.snap), .at = 4 } };
    var got: Box = .{};
    defer got.deinit();
    try reply(&c, true, &got, Unrestorable.install);
    try t.expect(!Unrestorable.fed_placeholders);
    try t.expectEqual(@as(u32, 1), c.routed); // the retry runs without memory: routed once
    try t.expectEqual(@as(u32, 1), c.dropped); // its snap given back once
    try t.expectEqualSlices(u32, plain.tokens.items, got.tokens.items);
    try t.expectEqual(plain.reason, got.reason);
}

test "folded memory: a memory prefix is priced like any prompt, and a request it cannot fit is refused by name" {
    // the plain request costs 8 + 12 = 20 of 40 bytes; with a 30-row prefix it would cost 30 + 8 + 12 = 50
    var plain: Box = .{};
    defer plain.deinit();
    try replyPriced(null, false, &plain, null, 40);
    try t.expectEqual(api.Reason.length, plain.reason.?);
    var c: Ctrl = .{ .handle = .{ .saved = @ptrCast(&Unrestorable.snap), .at = 30 } };
    var got: Box = .{};
    defer got.deinit();
    try replyPriced(&c, true, &got, Unrestorable.install, 40);
    try t.expectEqual(api.Reason.failed, got.reason.?);
    try t.expectStringStartsWith(&got.message, "the memory prefix this request would run does not fit");
    try t.expectEqual(@as(u32, 1), c.dropped);
    try t.expectEqual(@as(usize, 0), got.tokens.items.len);
}

test "folded memory: a reply reports the memory version its routing saw, and none when learning is off" {
    var c: Ctrl = .{ .version = 41 };
    var on: Box = .{};
    defer on.deinit();
    try reply(&c, true, &on, null);
    try t.expectEqual(@as(?u64, 41), on.memory_version);
    var off: Box = .{};
    defer off.deinit();
    try reply(&c, false, &off, null);
    try t.expectEqual(@as(?u64, null), off.memory_version);
}

test "folded memory: without --slide no holder is built, and the engine thread never polls a memory op" {
    const memory = @import("lane_host_memory.zig");
    try t.expect(!memory.wanted("/models/qwen27", false)); // a saved bank alone
    try t.expect(!memory.wanted("", true));
    try t.expect(memory.wanted("/models/qwen27", true));
    // the loop is chosen when the host starts: a bank attached later, with an op waiting, is never asked off
    for ([_]bool{ false, true }) |slide| {
        const gpa = t.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var ctrl: Ctrl = .{};
        var host = LaneHost.init(gpa, t.io, &core, .{ .lanes = 2 });
        if (slide) host.folded_memory = ctrl.controller();
        try host.start();
        defer host.stop();
        // a whole request first: the engine thread has chosen its loop by the time it replies
        var box: Box = .{};
        defer box.deinit();
        try host.engine().submit(1, &.{ .prompt = &prompt, .max_tokens = 12 }, .{ .ctx = &box, .event = Box.event });
        try box.wait();
        const op: api.MemoryOp = .stats;
        var ask: memory.Ask = .{ .op = &op };
        host.lock();
        host.folded_memory = ctrl.controller();
        try host.mem_ops.asks.append(gpa, &ask);
        host.wake.broadcast(t.io);
        host.unlock();
        var again: Box = .{};
        defer again.deinit();
        try host.engine().submit(2, &.{ .prompt = &prompt, .max_tokens = 12 }, .{ .ctx = &again, .event = Box.event });
        try again.wait();
        for (0..300) |_| { // idle turns too (the loop waits up to 100 ms between them)
            host.lock();
            const done = ask.done;
            host.unlock();
            if (done) break;
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        host.lock();
        const done = ask.done;
        if (!done) _ = host.mem_ops.asks.orderedRemove(0); // the plain loop never took it
        host.unlock();
        try t.expectEqual(slide, done);
        try t.expectEqual(@as(u32, @intFromBool(slide)), ctrl.execs);
    }
}
