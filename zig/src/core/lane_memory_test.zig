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
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(t.io);
        defer b.mutex.unlock(t.io);
        switch (e.*) {
            .tokens => |got| b.tokens.appendSlice(t.allocator, got) catch {
                b.failed = true;
            },
            .finished => |f| b.reason = f.reason,
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
    handle: ?api.MemorySnapHandle = null,
    fn of(ctx: *anyopaque) *Ctrl {
        return @ptrCast(@alignCast(ctx));
    }
    pub fn controller(c: *Ctrl) api.MemoryController {
        return .{ .ctx = c, .hasUnits = has, .routeAndCompose = route, .dropSnap = drop, .exec = exec };
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
    fn exec(_: *anyopaque, _: *const api.MemoryOp) api.MemoryResult {
        return .{};
    }
};

const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const query = [_]u32{ 5, 9 };
const request: api.Request = .{ .prompt = &prompt, .max_tokens = 12, .memory_query = &query };

/// One request through a lane host over the fake target, `ctrl` attached when given; `serving`: Info.folded_memory.
fn reply(ctrl: ?*Ctrl, serving: bool, box: *Box, prepare: ?*const fn (*lanes.Engine) void) !void {
    const gpa = t.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    if (prepare) |p| p(&core);
    var host = LaneHost.init(gpa, t.io, &core, .{ .lanes = 2, .folded_memory = serving });
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
