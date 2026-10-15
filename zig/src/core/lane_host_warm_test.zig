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

/// A paused turn on a warm host (bytes 64 + 4 a token, room for four 8-token states): the regenerate's cached tokens.
fn pausedTurn(lookahead: u32) !struct { cached: ?u32, same: bool } {
    const gpa = std.testing.allocator;
    const pc = api.prompt_cache;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .lookahead = lookahead, .warm = if (lookahead == 0) .on else .waits, .min_prompt = 0, .min_gap = 4 }, 400);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Run = struct {
        mutex: std.Io.Mutex = .init,
        cached: ?u32 = null,
        tokens: [8]u32 = undefined,
        n: usize = 0,
        done: bool = false,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| for (t) |x| if (b.n < b.tokens.len) {
                    b.tokens[b.n] = x;
                    b.n += 1;
                },
                .finished => b.done = true,
                else => {},
            }
        }
        fn go(b: *@This(), engine: Engine, id: Id, r: *const Request) !void {
            try engine.submit(id, r, .{ .ctx = b, .event = event });
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) return;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const x = [_]u32{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 };
    const z = [_]u32{ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209 };
    const y = [_]u32{ 300, 301, 302, 303, 304, 305, 306, 307, 308, 309 };
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; // history 8, then a two-token generation prompt
    const warm = t1[0..9].* ++ [_]u32{ 50, 51, 52, 53, 54, 55 }; // the reply as sent and the next turn's opening
    var runs: [6]Run = @splat(.{});
    for ([_][]const u32{ &x, &z }, 0..) |p, i| try runs[i].go(e, i + 1, &.{ .prompt = p, .max_tokens = 2, .history_len = 8 });
    try runs[2].go(e, 3, &.{ .prompt = &t1, .max_tokens = 2, .history_len = 8 });
    try runs[3].go(e, 4, &.{ .prompt = &warm, .max_tokens = 0, .background = true, .history_len = warm.len, .born = t1.len });
    try runs[4].go(e, 5, &.{ .prompt = &y, .max_tokens = 2, .history_len = 8 }); // its keep needs room too
    try runs[5].go(e, 6, &.{ .prompt = &t1, .max_tokens = 2, .history_len = 8 }); // regenerated
    try std.testing.expect(store.held <= store.budget);
    return .{ .cached = runs[5].cached, .same = runs[5].n == 2 and std.mem.eql(u32, runs[5].tokens[0..2], runs[2].tokens[0..2]) };
}

test "a finished warm pass keeps its states with its request's length: the history it extends outlives its keep and later ones" {
    for ([_]u32{ 0, 1 }) |lookahead| { // the 27B keeps the pass's end; Nemotron on CUDA a mark one token before it
        const got = try pausedTurn(lookahead);
        try std.testing.expectEqual(@as(?u32, 8), got.cached);
        try std.testing.expect(got.same);
    }
}
