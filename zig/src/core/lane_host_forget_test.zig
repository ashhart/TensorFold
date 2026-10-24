//! Threaded prompt erasure preserves unrelated reuse and reports unsupported or stopped engines.
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
            .logprobs, .prompt_logprobs, .audio => {},
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

/// One request of `prompt` (its history all but the last 2 tokens); the tokens it restored from the cache.
fn turn(e: api.Engine, id: Id, prompt: []const u32) !u32 {
    var b: Box = .{};
    defer b.tokens.deinit(gpa);
    const r: Request = .{ .prompt = prompt, .max_tokens = 3, .history_len = @intCast(prompt.len - 2) };
    try e.submit(id, &r, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b.wait());
    return b.cached.?;
}

fn holds(store: *const pc.Store, span: []const u32) bool {
    for (store.entries.items) |x| if (std.mem.indexOf(u32, x.tokens, span) != null) return true;
    return false;
}

const note = [_]u32{ 50, 51, 52, 53 };
const with_note = [_]u32{ 1, 2, 3 } ++ note ++ [_]u32{ 9, 9, 4, 4 };
const without = [_]u32{ 1, 2, 3, 60, 61, 62, 63, 9, 9, 4, 4 };

test "forget_prompts on a lane host: the note's conversation no longer resumes past it, an unrelated one still does" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const e = host.engine();
    try std.testing.expectEqual(@as(u32, 0), try turn(e, 1, &with_note));
    try std.testing.expectEqual(@as(u32, 0), try turn(e, 2, &without));
    // the planted state resumes: without the forget the note's conversation restores past its block
    try std.testing.expectEqual(@as(u32, 9), try turn(e, 3, &with_note));
    try std.testing.expect(holds(&store, &note));

    const out = try e.forgetPrompts(&note);
    try std.testing.expect(out.states >= 1);
    try std.testing.expect(out.complete());
    try std.testing.expect(!holds(&store, &note));
    try std.testing.expectEqual(@as(u32, 0), try turn(e, 4, &with_note)); // read again from the start
    try std.testing.expectEqual(@as(u32, 9), try turn(e, 5, &without)); // the unrelated state survived
}

test "forget_prompts on a lane host without a prompt cache forgets nothing and says so; a stopped host refuses" {
    try emptyHost(false);
    try emptyHost(true);
}

// No requests: every planner callback is unreachable; the engine thread must still answer forgets.
const IdlePlanner = struct {
    fn begin(_: *anyopaque) void {
        unreachable;
    }
    fn admit(_: *anyopaque, _: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
        unreachable;
    }
    fn admitted(_: *anyopaque) anyerror!void {
        unreachable;
    }
    fn arm(_: *anyopaque, _: usize) void {
        unreachable;
    }
    fn began(_: *anyopaque, _: usize, _: bool) void {
        unreachable;
    }
    fn plan(_: *anyopaque, _: *std.ArrayList(api.Rounds.Piece), _: *std.ArrayList(usize)) anyerror!void {
        unreachable;
    }
    fn after(_: *anyopaque, _: f64, _: f64) void {
        unreachable;
    }
    const table: api.Rounds.VTable = .{ .begin_admit = begin, .admit = admit, .admitted = admitted, .arm = arm, .began = began, .plan = plan, .left = arm, .after = after };
};

fn emptyHost(planned: bool) !void {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (planned) target.pieceBackend() else target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var planner_ctx: u8 = 0;
    if (planned) host.rounds = .{ .ctx = &planner_ctx, .vtable = &IdlePlanner.table };
    try host.start();
    const e = host.engine();
    try std.testing.expectEqual(pc.Forgotten{}, try e.forgetPrompts(&note));
    host.stop();
    try std.testing.expectError(error.Closed, e.forgetPrompts(&note));
}

test "an engine without the forget hook answers Unsupported" {
    const Bare = struct {
        fn info(_: *anyopaque) api.Info {
            return .{};
        }
        fn submit(_: *anyopaque, _: Id, _: *const Request, _: api.Sink) api.SubmitError!void {}
        fn cancel(_: *anyopaque, _: Id) void {}
        fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
            out.* = .{};
        }
        fn memory(_: *anyopaque, _: bool) ?api.Memory {
            return null;
        }
    };
    var dummy: u8 = 0;
    const e: api.Engine = .{ .ctx = &dummy, .vtable = &.{ .info = Bare.info, .submit = Bare.submit, .cancel = Bare.cancel, .status = Bare.status, .memory = Bare.memory } };
    try std.testing.expectError(error.Unsupported, e.forgetPrompts(&note));
}
