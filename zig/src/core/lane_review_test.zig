//! The lane host under a family round planner: prompt pieces, failed piece runs, learning, refusal.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Memory = api.Memory;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Engine = api.Engine;
const FakeLearner = @import("lane_host_test.zig").FakeLearner;
const LearnBox = @import("lane_host_test.zig").LearnBox;

test "more than sixteen planned pieces retain their job until its final" {
    const gpa = std.testing.allocator;
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .join_tail = true, .piece_runs = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.runBackend(), clock.clock());
    defer core.deinit();
    // jobs on the page allocator: a freed job is unmapped, so reading one faults (the testing allocator hides it)
    var host = LaneHost.init(std.heap.page_allocator, std.testing.io, &core, .{ .lanes = 4 });
    // every admitted prompt's rows in one piece and its final in the same round (as for short prompts)
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            self(ctx).seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (self(ctx).seqs.items) |*x| if (!x.done) {
                // the host's lists: its allocator (it frees them)
                for (0..x.n - 1) |row| try pieces.append(std.heap.page_allocator, .{ .key = x.key, .start = row, .end = row + 1, .save = false });
                try finals.append(std.heap.page_allocator, x.key);
                x.done = true;
            };
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
        tokens: usize = 0,
        done: ?Reason = null,
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
    const p1: [18]u32 = @splat(3);
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var b1: Box = .{};
    var b2: Box = .{};
    var b3: Box = .{};
    const r1: Request = .{ .prompt = &p1, .max_tokens = 12 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    // The first request alone needs seventeen pieces before its final.
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try host.start();
    defer host.stop();
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(usize, 2), target.joins);
    // the host serves the next request
    try e.submit(3, &r1, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(usize, 12), b3.tokens);
    try std.testing.expectEqual(@as(usize, 4), target.runs);
}

test "halt transfers queued and admitted jobs when every subsequent allocation fails" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var failing = std.testing.FailingAllocator.init(gpa, .{});
    var host = LaneHost.init(failing.allocator(), std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    const Box = struct {
        finished: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            if (e.* == .finished and e.finished.reason == .failed) b.finished += 1;
        }
    };
    var box: Box = .{};
    const request: Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 2 };
    const engine = host.engine();
    try engine.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try engine.submit(2, &request, .{ .ctx = &box, .event = Box.event });
    try host.admitted.append(host.gpa, host.queued.pop().?);
    failing.fail_index = failing.alloc_index;
    host.closing = true;
    @import("lane_halt.zig").haltAll(&host, "drain deadline");
    try std.testing.expectEqual(@as(usize, 2), box.finished);
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len + host.admitted.items.len);
}

test "failed repeated pieces drop their job once and skip its remaining batches" {
    const gpa = std.testing.allocator;
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .join_tail = true, .piece_runs = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .fail_runs = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.runBackend(), clock.clock());
    defer core.deinit();
    // jobs on the page allocator: a freed job is unmapped, so reading one faults (the testing allocator hides it)
    var host = LaneHost.init(std.heap.page_allocator, std.testing.io, &core, .{ .lanes = 4 });
    // every admitted prompt's rows in one piece and its final in the same round (as for short prompts)
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            self(ctx).seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (self(ctx).seqs.items) |*x| if (!x.done) {
                // the host's lists: its allocator (it frees them)
                for (0..x.n - 1) |row| try pieces.append(std.heap.page_allocator, .{ .key = x.key, .start = row, .end = row + 1, .save = false });
                try finals.append(std.heap.page_allocator, x.key);
                x.done = true;
            };
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
        tokens: usize = 0,
        done: ?Reason = null,
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
    const p1: [18]u32 = @splat(3);
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var b1: Box = .{};
    var b2: Box = .{};
    var b3: Box = .{};
    const r1: Request = .{ .prompt = &p1, .max_tokens = 12 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    // The first request alone needs seventeen pieces before its final.
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try host.start();
    defer host.stop();
    try std.testing.expectEqual(Reason.failed, b1.wait());
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(usize, 1), target.joins);
    // the host serves the next request
    try e.submit(3, &r1, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(usize, 12), b3.tokens);
    try std.testing.expectEqual(@as(usize, 3), target.runs);
}
