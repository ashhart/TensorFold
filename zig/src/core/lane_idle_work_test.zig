//! Idle weight release restores scoring and folded-memory work on both real lane loops.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const t = std.testing;

const State = struct {
    released: std.atomic.Value(bool) = .init(true),
    idle_calls: std.atomic.Value(u32) = .init(0),
    restores: std.atomic.Value(u32) = .init(0),
    active: std.atomic.Value(u32) = .init(0),
    fail: bool = false,
    var scoring: *State = undefined;
    fn of(ctx: *anyopaque) *State {
        return @ptrCast(@alignCast(ctx));
    }
    fn restore(ctx: *anyopaque) !void {
        const s = of(ctx);
        if (s.fail) return error.RestoreFailed;
        _ = s.restores.fetchAdd(1, .acq_rel);
        s.released.store(false, .release);
    }
    fn idle(ctx: *anyopaque) void {
        const s = of(ctx);
        std.debug.assert(s.active.load(.acquire) == 0);
        s.released.store(true, .release);
        _ = s.idle_calls.fetchAdd(1, .acq_rel);
    }
    fn begin(ctx: *anyopaque) void {
        _ = of(ctx).active.fetchAdd(1, .acq_rel);
    }
    fn end(ctx: *anyopaque) void {
        std.debug.assert(of(ctx).active.fetchSub(1, .acq_rel) == 1);
    }
    fn score(_: *anyopaque, _: []const u32, _: []const u32, logits: []f64, _: bool) !f64 {
        try t.expect(!scoring.released.load(.acquire));
        try t.expectEqual(@as(u32, 1), scoring.active.load(.acquire));
        @memset(logits, 3);
        return 7.5;
    }
    fn has(_: *anyopaque) bool {
        return false;
    }
    fn route(_: *anyopaque, _: []const u32) !?api.MemorySnapHandle {
        return null;
    }
    fn drop(_: *anyopaque, _: *anyopaque) void {}
    fn exec(ctx: *anyopaque, op: *const api.MemoryOp) api.MemoryResult {
        const s = of(ctx);
        if (op.* == .init and s.released.load(.acquire)) return .{ .err = "weights still released" };
        return .{ .units = 9 };
    }
};

const IdlePlan = struct {
    fn none(_: *anyopaque) void {}
    fn admit(_: *anyopaque, _: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
        return .{ .wait = .engine_busy };
    }
    fn admitted(_: *anyopaque) !void {}
    fn key(_: *anyopaque, _: usize) void {}
    fn began(_: *anyopaque, _: usize, _: bool) void {}
    fn plan(_: *anyopaque, _: *std.ArrayList(api.Rounds.Piece), _: *std.ArrayList(usize)) !void {}
    fn after(_: *anyopaque, _: f64, _: f64) void {}
    const vt: api.Rounds.VTable = .{ .begin_admit = none, .admit = admit, .admitted = admitted, .arm = key, .began = began, .plan = plan, .left = key, .after = after };
};

test "weight release: both loops restore scores and memory work, preserve CPU ops, refuse failed restores" {
    const level = t.log_level;
    t.log_level = .err;
    defer t.log_level = level;
    for ([_]bool{ false, true }) |planned| for ([_]bool{ false, true }) |enabled| {
        var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
        defer cfg.deinit(t.allocator);
        var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
        defer fake.deinit();
        const plain = if (planned) fake.pieceBackend() else fake.backend();
        var vt = plain.vtable.*;
        vt.score = State.score;
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(t.allocator, &cfg, .{ .ptr = plain.ptr, .vtable = &vt }, clock.clock());
        defer core.deinit();
        var state: State = .{};
        State.scoring = &state;
        if (!enabled) state.released.store(false, .release);
        var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
        host.activity = .{ .ctx = &state, .begin = State.begin, .end = State.end };
        host.folded_memory = .{ .ctx = &state, .hasUnits = State.has, .routeAndCompose = State.route, .dropSnap = State.drop, .exec = State.exec };
        if (enabled) host.weight_release = .{ .ctx = &state, .restore = State.restore, .idle = State.idle };
        if (planned) host.rounds = .{ .ctx = &state, .vtable = &IdlePlan.vt };
        const helpers = @import("lane_host_test.zig");
        var learner: helpers.FakeLearner = .{};
        host.learner = .{ .ctx = &learner, .begin = helpers.FakeLearner.begin, .step = helpers.FakeLearner.step, .abort = helpers.FakeLearner.abort };
        try host.start();
        defer host.stop();
        if (enabled) {
            for (0..5000) |_| {
                if (state.idle_calls.load(.acquire) > 0) break;
                try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
            }
            try t.expect(state.idle_calls.load(.acquire) > 0);
        }
        var logits: [1]f64 = undefined;
        try t.expectEqual(@as(f64, 7.5), try host.engine().score(&.{1}, &.{2}, &logits, false));
        try t.expectEqual(@as(f64, 3), logits[0]);
        const init: api.MemoryOp = .{ .init = .{ .cal_ids = &.{}, .prefix_ids = &.{}, .suffix_ids = &.{} } };
        const result = host.engine().memoryOp(&init);
        try t.expect(result.err == null);
        try t.expectEqual(@as(u32, 9), result.units);
        host.lock();
        const before = state.restores.load(.acquire);
        state.fail = true;
        host.unlock();
        const stats: api.MemoryOp = .stats;
        try t.expectEqual(@as(u32, 9), host.engine().memoryOp(&stats).units);
        try t.expectEqual(before, state.restores.load(.acquire));
        if (enabled) {
            try t.expectError(error.Failed, host.engine().score(&.{1}, &.{2}, &logits, false));
            try t.expectEqualStrings("weight restoration failed", host.engine().memoryOp(&init).err.?);
            const lesson: api.LearnRequest = .{ .train = &.{} };
            var learned: helpers.LearnBox = .{};
            defer learned.tags.deinit(t.allocator);
            try t.expectError(error.Busy, host.engine().learn(&lesson, learned.sink()));
        } else try t.expectEqual(@as(u32, 0), state.idle_calls.load(.acquire));
    };
}
