//! A background warm pass and the real request that arrives while it runs, through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Engine = api.Engine;

/// A request's tokens and how it finished, as the host reports them.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    tokens: usize = 0,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// Chunks the fake ran for prompt-only streams.
const WarmChunks = struct {
    n: std.atomic.Value(usize) = .init(0),
    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        if (s.max_new == 0) _ = w.n.fetchAdd(1, .monotonic);
    }
};

test "a background warm pass steps aside for a request that arrives during it, and the request runs" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    var real: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 48 };
    const Arrival = struct { // the real request arrives during the warm pass's first chunk
        engine: Engine,
        request: *const Request,
        box: *Box,
        sent: bool = false,
        fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (a.sent or s.max_new != 0 or chunk != 0) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.box, .event = Box.event }) catch {};
        }
    };
    var arrival: Arrival = .{ .engine = host.engine(), .request = &request, .box = &real };
    target.prefill_hook = Arrival.hook;
    target.prefill_hook_ctx = &arrival;
    try host.engine().submit(1, &warm_request, .{ .ctx = &warm, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, warm.wait()); // it stepped aside at its next chunk
    try std.testing.expectEqual(Reason.length, real.wait());
    try std.testing.expectEqual(@as(usize, 48), real.tokens);
}

test "a warm pass waits for a quiet engine: a request sent right after the reply goes first and the pass runs no chunk" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var chunks: WarmChunks = .{};
    target.prefill_hook = WarmChunks.hook;
    target.prefill_hook_ctx = &chunks;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.warm_quiet = 30 * std.time.ns_per_s; // far past the next request: it must step aside, not time out
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    var real: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 48 };
    try host.engine().submit(1, &warm_request, .{ .ctx = &warm, .event = Box.event });
    while (true) { // the engine thread has taken the pass: the request comes after it, as a client's next one does
        host.lock();
        const taken = host.queued.items.len == 0;
        host.unlock();
        if (taken) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try host.engine().submit(2, &request, .{ .ctx = &real, .event = Box.event });
    try std.testing.expectEqual(Reason.length, real.wait());
    try std.testing.expectEqual(@as(usize, 48), real.tokens);
    try std.testing.expectEqual(Reason.cancelled, warm.wait());
    try std.testing.expectEqual(@as(usize, 0), chunks.n.load(.monotonic)); // the GPU never ran a chunk of it
}

test "a warm pass with no request after it runs once the engine has been quiet" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var chunks: WarmChunks = .{};
    target.prefill_hook = WarmChunks.hook;
    target.prefill_hook_ctx = &chunks;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.warm_quiet = 20 * std.time.ns_per_ms;
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var warm: Box = .{};
    const warm_request: Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true };
    const t0 = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    try host.engine().submit(1, &warm_request, .{ .ctx = &warm, .event = Box.event });
    try std.testing.expectEqual(Reason.length, warm.wait());
    try std.testing.expect(std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - t0 >= 20 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(usize, 3), chunks.n.load(.monotonic)); // every chunk ran
}
