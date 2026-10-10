//! The lane host's halt: a stop's drain deadline ends the rounds at a boundary, what is left fails with the reason.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "halt (a stop's drain deadline): the rounds end at a round boundary, every request queued or running fails with the reason" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // one prompt prefilling at a time (the next waits queued), 3 rows a round
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: u64 = 0, decoding: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        rounds: u32 = 0,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            const p = self(ctx);
            for (p.seqs.items) |x| if (!x.decoding) return .wait;
            p.seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            const p = self(ctx);
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (p.seqs.items) |*x| if (!x.decoding) {
                if (x.done < x.n - 1) {
                    const end = @min(x.done + 3, x.n - 1);
                    try pieces.append(gpa, .{ .key = x.key, .start = x.done, .end = end, .save = false });
                    x.done = end;
                }
                if (x.done == x.n - 1) {
                    try finals.append(gpa, x.key);
                    x.decoding = true;
                }
            };
            p.rounds += 1;
        }
        fn left(ctx: *anyopaque, key: usize) void {
            const p = self(ctx);
            for (p.seqs.items, 0..) |x, i| if (x.key == key) {
                _ = p.seqs.orderedRemove(i);
                return;
            };
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl: Plan = .{};
    defer pl.seqs.deinit(gpa);
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        message: []const u8 = "",
        /// the planner's rounds when the finish arrived (on the engine thread, between rounds)
        rounds_at_end: u32 = 0,
        plan: *Plan,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                    b.rounds_at_end = b.plan.rounds;
                },
                else => {},
            }
        }
        fn count(b: *@This()) usize {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return b.tokens.items.len;
        }
    };
    var long_prompt: [600]u32 = undefined;
    for (&long_prompt, 0..) |*t, i| t.* = @intCast(1 + i % 13);
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var boxes: [3]Box = .{ .{ .plan = &pl }, .{ .plan = &pl }, .{ .plan = &pl } };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    // 1: decoding (no end of its own), 2: prefilling a long prompt in pieces, 3: queued behind 2
    const r1: Request = .{ .prompt = &p1, .max_tokens = 1 << 30 };
    const r2: Request = .{ .prompt = &long_prompt, .max_tokens = 1 << 30 };
    const r3: Request = .{ .prompt = &p1, .max_tokens = 8 };
    const e = host.engine();
    try host.start();
    try e.submit(1, &r1, .{ .ctx = &boxes[0], .event = Box.event });
    while (boxes[0].count() < 4) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try e.submit(2, &r2, .{ .ctx = &boxes[1], .event = Box.event });
    try e.submit(3, &r3, .{ .ctx = &boxes[2], .event = Box.event });
    while (boxes[0].count() < 20) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    host.halt("server restarting: retry shortly");
    const rounds = pl.rounds;
    for (&boxes) |*b| {
        try std.testing.expectEqual(@as(?Reason, .failed), b.done);
        try std.testing.expectEqualStrings("server restarting: retry shortly", b.message);
        // no round was planned after the finishes: the loop stopped at the boundary they were sent at
        try std.testing.expectEqual(rounds, b.rounds_at_end);
    }
    // the decoding stream's tokens are whole rounds' (the fake's own sequence), the prefilling one had none yet
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &p1);
    for (boxes[0].tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 0), boxes[1].tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), boxes[2].tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), pl.seqs.items.len);
    // nothing runs after it; the stop that follows (a close) finds the thread gone
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    try std.testing.expectEqual(rounds, pl.rounds);
    host.stop();
}

test "halt fails a request still filling its prompt with the reason and frees its lane; a decoding one fails too" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 2 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Chunks = struct {
        n: std.atomic.Value(usize) = .init(0), // chunks the second request's pass ran
        fn call(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            _ = c.n.fetchAdd(1, .monotonic);
        }
    };
    var chunks: Chunks = .{};
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        message: []const u8 = "",
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                },
                else => {},
            }
        }
        fn count(b: *@This()) usize {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return b.tokens.items.len;
        }
    };
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var boxes: [3]Box = .{ .{}, .{}, .{} };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    // 1: decoding (no end of its own), 2: filling a prompt that never ends, 3: queued behind both lanes
    const r1: Request = .{ .prompt = &p1, .max_tokens = 1 << 30 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 8 };
    const r3: Request = .{ .prompt = &p2, .max_tokens = 8 };
    const e = host.engine();
    try host.start();
    try e.submit(1, &r1, .{ .ctx = &boxes[0], .event = Box.event });
    while (boxes[0].count() < 4) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // nothing fills now: the next pass takes chunks until the halt
    target.prefill_chunks = std.math.maxInt(usize);
    target.prefill_hook = Chunks.call;
    target.prefill_hook_ctx = &chunks;
    try e.submit(2, &r2, .{ .ctx = &boxes[1], .event = Box.event });
    try e.submit(3, &r3, .{ .ctx = &boxes[2], .event = Box.event });
    while (chunks.n.load(.monotonic) < 3) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    host.halt("server restarting: retry shortly");
    for (&boxes) |*b| {
        try std.testing.expectEqual(@as(?Reason, .failed), b.done);
        try std.testing.expectEqualStrings("server restarting: retry shortly", b.message);
    }
    try std.testing.expect(boxes[0].tokens.items.len >= 4);
    try std.testing.expectEqual(@as(usize, 0), boxes[1].tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), boxes[2].tokens.items.len);
    // every lane freed: the decoding stream's and the filling one's
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expectEqual(@as(usize, 0), host.filling.items.len);
    host.stop();
}
