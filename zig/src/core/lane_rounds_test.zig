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

test "a round planner: prompts in pieces between the other streams' decode rounds, queued admissions, the same tokens" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // one prompt prefilling at a time (the next waits queued), 3 rows a round, a final when its rows are in
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: u64 = 0, decoding: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        waits: u32 = 0,
        rounds: u32 = 0,
        mixed: u32 = 0, // rounds that ran a piece while another stream decoded
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            const p = self(ctx);
            for (p.seqs.items) |x| if (!x.decoding) {
                p.waits += 1;
                return .wait;
            };
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
            var decoding = false;
            for (p.seqs.items) |x| decoding = decoding or x.decoding;
            for (p.seqs.items) |*x| if (!x.decoding) {
                if (x.done < x.n - 1) {
                    const end = @min(x.done + 3, x.n - 1);
                    try pieces.append(gpa, .{ .key = x.key, .start = x.done, .end = end, .save = false });
                    x.done = end;
                    if (decoding) p.mixed += 1;
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
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) !Reason {
            for (0..10_000) |_| { // 10 s
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
            return error.Timeout;
        }
    };
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };
    var b1: Box = .{};
    var b2: Box = .{};
    defer b1.tokens.deinit(gpa);
    defer b2.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &p1, .max_tokens = 40 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    // both queued before the first round, so the second waits whatever the scheduler does
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try host.start();
    {
        errdefer host.stop(); // a timed-out wait stops the host thread before this frame goes
        try std.testing.expectEqual(Reason.length, try b1.wait());
        try std.testing.expectEqual(Reason.length, try b2.wait());
    }
    host.stop();
    // the replies are the whole-prompt path's tokens
    for ([_]struct { p: []const u32, b: *Box }{ .{ .p = &p1, .b = &b1 }, .{ .p = &p2, .b = &b2 } }) |c| {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, c.p);
        for (c.b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
    }
    try std.testing.expectEqual(@as(usize, 40), b1.tokens.items.len);
    try std.testing.expectEqual(@as(usize, 12), b2.tokens.items.len);
    // the second prompt waited queued, then ran its pieces while the first decoded
    try std.testing.expect(pl.waits > 0 and pl.mixed >= 3);
    try std.testing.expectEqual(@as(usize, 3 + 4), target.pieces_run);
    try std.testing.expectEqual(@as(usize, 0), pl.seqs.items.len);
}

test "a failed piece run fails its requests alone: their finals of the same round do not run, the host keeps serving" {
    // a run's error drops its jobs, whose finals are listed in the same round; runFinals must not read them
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
                try pieces.append(std.heap.page_allocator, .{ .key = x.key, .start = 0, .end = x.n - 1, .save = false });
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
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var b1: Box = .{};
    var b2: Box = .{};
    var b3: Box = .{};
    const r1: Request = .{ .prompt = &p1, .max_tokens = 12 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    // both admitted before the host's first round: one run of their pieces, which fails
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try host.start();
    defer host.stop();
    try std.testing.expectEqual(Reason.failed, b1.wait());
    try std.testing.expectEqual(Reason.failed, b2.wait());
    try std.testing.expectEqual(@as(usize, 0), target.joins); // no final of theirs ran
    // the host serves the next request
    try e.submit(3, &r1, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(usize, 12), b3.tokens);
    try std.testing.expectEqual(@as(usize, 1), target.runs);
}

test "under a round planner a learn request still steps while no request waits or runs" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    const Idle = struct {
        fn none(_: *anyopaque) void {}
        fn admit(_: *anyopaque, _: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            return .wait;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn key(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(_: *anyopaque, _: *std.ArrayList(api.Rounds.Piece), _: *std.ArrayList(usize)) anyerror!void {}
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var ctx: u8 = 0;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    host.rounds = .{ .ctx = &ctx, .vtable = &.{ .begin_admit = Idle.none, .admit = Idle.admit, .admitted = Idle.admitted, .arm = Idle.key, .began = Idle.began, .plan = Idle.plan, .left = Idle.key, .after = Idle.after } };
    var fake: FakeLearner = .{};
    host.learner = .{ .ctx = &fake, .begin = FakeLearner.begin, .step = FakeLearner.step, .abort = FakeLearner.abort };
    try host.start();
    defer host.stop();
    const example: api.Example = .{ .ids = &.{ 1, 2, 3 }, .start = 2, .decide = 0 };
    const learn: api.LearnRequest = .{ .train = &.{example} };
    var box: LearnBox = .{};
    defer box.tags.deinit(gpa);
    try host.engine().learn(&learn, box.sink());
    box.wait();
    try std.testing.expectEqualSlices(std.meta.Tag(api.LearnEvent), &.{ .learned, .done }, box.tags.items);
    try std.testing.expectEqual(@as(u32, 3), box.steps);
}

test "a round planner over a backend without prompt pieces is refused at start" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var whole = target.backend();
    var vt = whole.vtable.*;
    vt.begin = null;
    whole.vtable = &vt;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, whole, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var ctx: u8 = 0;
    host.rounds = .{ .ctx = &ctx, .vtable = undefined };
    try std.testing.expectError(error.PiecesUnsupported, host.start());
}

test "a failed piece fails its request alone: its final of the same round does not run, the host keeps serving" {
    // a job dropped for a failed piece must leave the round's finals: `drop` frees it, runFinals must not read it
    const gpa = std.testing.allocator;
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .join_tail = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .fail_pieces = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.joinBackend(), clock.clock());
    defer core.deinit();
    // jobs on the page allocator: a freed job is unmapped, so reading one faults (the testing allocator hides it)
    const pa = std.heap.page_allocator;
    var host = LaneHost.init(pa, std.testing.io, &core, .{ .lanes = 4 });
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
                try pieces.append(pa, .{ .key = x.key, .start = 0, .end = x.n - 1, .save = false });
                try finals.append(pa, x.key);
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
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var b1: Box = .{};
    var b2: Box = .{};
    var b3: Box = .{};
    const r1: Request = .{ .prompt = &p1, .max_tokens = 12 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    // both admitted before the host's first round: the first piece fails, the second's request is served
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try host.start();
    defer host.stop();
    try std.testing.expectEqual(Reason.failed, b1.wait());
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(usize, 12), b2.tokens);
    // the host serves the next request
    try e.submit(3, &r1, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(usize, 12), b3.tokens);
}
