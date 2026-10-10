//! LaneHost with the prompt cache: a conversation's next turn resumes from its kept prompt state (the fake backend).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;

test "a lane host resumes a conversation from its kept prompt state and reports the cached tokens" {
    const gpa = std.testing.allocator;
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
            return error.BackendRestores; // lane-core backends restore inside their own prompt pass
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .logprobs => {},
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
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
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(?u32, 0), b1.cached);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    var t2: std.ArrayList(u32) = .empty; // the next turn: the history, the reply, a tool result, a new generation prompt
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached);
    var history: std.ArrayList(u32) = .empty; // the fake's tokens read the whole history: any slip in it changes them
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
}

test "an image's placeholders are cached by its content: another image under the same ids resumes none of it" {
    const gpa = std.testing.allocator;
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
            return error.BackendRestores; // lane-core backends restore inside their own prompt pass
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .logprobs => {},
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
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
    const e = host.engine();
    // prompt ids 3..5 are one image's placeholders (token 9); cache_key holds ids derived from each image's content
    const t1 = [_]u32{ 3, 1, 9, 9, 9, 9, 2, 6, 5, 3 };
    var key_a = t1;
    var key_b = t1;
    for (2..6) |k| {
        key_a[k] = 0x8000_0000 + @as(u32, @intCast(k));
        key_b[k] = 0x9000_0000 + @as(u32, @intCast(k));
    }
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8, .cache_key = &key_a };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len); // image A's state at 8, past its image
    // the same text and placeholder ids with image B: the kept state is A's, so B resumes none of it
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8, .cache_key = &key_b };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 0), b2.cached);
    // image A again resumes its own state
    var b3: Box = .{};
    defer b3.tokens.deinit(gpa);
    const r3: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8, .cache_key = &key_a };
    try e.submit(3, &r3, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(?u32, 8), b3.cached);
}

test "a turn's mark evicting the state it resumed from: the hit is reported first and the entry freed once" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct { // a budget of one state: each keep records the hits the store had counted by then
        f: *lanes.fake.Fake,
        store: ?*pc.Store = null,
        hits_at_save: [4]u64 = @splat(0),
        saves: usize = 0,
        live: usize = 0,
        fn of(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const x = of(ptr);
            x.hits_at_save[x.saves] = x.store.?.counts.hits;
            x.saves += 1;
            x.live += 1;
            return x.f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            of(ptr).live -= 1;
            of(ptr).f.drop(saved);
        }
    };
    var snaps: Snaps = .{ .f = &target };
    var store = pc.Store.init(gpa, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 150);
    defer store.deinit();
    snaps.store = &store;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .logprobs => {},
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
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
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    var t2: std.ArrayList(u32) = .empty; // 18 tokens: its history's state (128 bytes) and turn 1's (96) pass the budget together
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached);
    try std.testing.expectEqual(@as(usize, 2), snaps.saves);
    try std.testing.expectEqual(@as(u64, 1), snaps.hits_at_save[1]); // the resume was reported before the keep evicted it
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expectEqual(@as(u32, 16), store.entries.items[0].at);
    try std.testing.expectEqual(@as(usize, 1), snaps.live);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

test "a learn step that moves the weights drops every kept state: the next turn prefills from the start" {
    const gpa = std.testing.allocator;
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
    const Moves = struct { // one step that changes the weights and ends the lesson
        sink: ?api.LearnSink = null,
        fn begin(ctx: *anyopaque, _: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            l.sink = sink;
        }
        fn step(ctx: *anyopaque) api.Learner.Step {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            const done: api.LearnEvent = .{ .done = .{} };
            l.sink.?.event(l.sink.?.ctx, &done);
            return .{ .done = true, .changed = true };
        }
        fn abort(_: *anyopaque) void {}
    };
    const Learned = struct {
        mutex: std.Io.Mutex = .init,
        done: bool = false,
        fn event(ctx: *anyopaque, e: *const api.LearnEvent) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            if (e.* == .done) b.done = true;
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var moves: Moves = .{};
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    host.learner = .{ .ctx = &moves, .begin = Moves.begin, .step = Moves.step, .abort = Moves.abort };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .logprobs => {},
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
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
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    var learned: Learned = .{};
    const example: api.Example = .{ .ids = &.{ 1, 2, 3 }, .start = 2 };
    const lesson: api.LearnRequest = .{ .train = &.{example} };
    try e.learn(&lesson, .{ .ctx = &learned, .event = Learned.event });
    while (true) {
        learned.mutex.lockUncancelable(std.testing.io);
        const d = learned.done;
        learned.mutex.unlock(std.testing.io);
        if (d) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    var t2: std.ArrayList(u32) = .empty; // the same next turn that resumes at 8 when the weights stay
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 0), b2.cached);
    try std.testing.expectEqual(@as(u64, 0), store.counts.hits);
}
