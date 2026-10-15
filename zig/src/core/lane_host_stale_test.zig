//! A learner whose put-back the device lacks: replies refused, learn requests taken, replies back once it lands.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

/// A learner whose every step reports `stale`, ending each lesson at once; `hold` keeps a step waiting inside.
const Learner = struct {
    sink: ?api.LearnSink = null,
    stale: bool = true,
    hold: std.atomic.Value(bool) = .init(false),
    entered: std.atomic.Value(bool) = .init(false),

    fn begin(ctx: *anyopaque, _: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
        const l: *Learner = @ptrCast(@alignCast(ctx));
        l.sink = sink;
    }
    fn step(ctx: *anyopaque) api.Learner.Step {
        const l: *Learner = @ptrCast(@alignCast(ctx));
        l.entered.store(true, .release);
        while (l.hold.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        l.sink.?.event(l.sink.?.ctx, &.{ .done = .{} });
        return .{ .done = true, .changed = false, .stale = l.stale };
    }
    fn abort(_: *anyopaque) void {}
};

/// Whether a lesson ended, and a reply finished, set from the engine's thread.
const Seen = struct {
    mutex: std.Io.Mutex = .init,
    done: bool = false,
    reason: ?Reason = null,
    said: [256]u8 = undefined, // the finished reply's message
    said_len: usize = 0,

    fn learnSink(s: *Seen) api.LearnSink {
        return .{ .ctx = s, .event = learned };
    }
    fn learned(ctx: *anyopaque, e: *const api.LearnEvent) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.mutex.lockUncancelable(std.testing.io);
        defer s.mutex.unlock(std.testing.io);
        if (e.* == .done) s.done = true;
    }
    fn replied(ctx: *anyopaque, _: Id, e: *const Event) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.mutex.lockUncancelable(std.testing.io);
        defer s.mutex.unlock(std.testing.io);
        if (e.* != .finished) return;
        s.said_len = @min(s.said.len, e.finished.message.len);
        @memcpy(s.said[0..s.said_len], e.finished.message[0..s.said_len]);
        s.reason = e.finished.reason;
    }
    /// Waits up to 5 s for the lesson to end or the reply to finish.
    fn wait(s: *Seen, comptime what: enum { lesson, reply }) !void {
        for (0..5000) |_| {
            s.mutex.lockUncancelable(std.testing.io);
            const got = if (what == .lesson) s.done else s.reason != null;
            s.mutex.unlock(std.testing.io);
            if (got) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.NeverSeen;
    }
};

/// Waits up to 5 s for the host's stale flag to read `want` (the flag follows the step, after its events).
fn waitStale(host: *LaneHost, want: bool) !void {
    for (0..5000) |_| {
        host.mutex.lockUncancelable(std.testing.io);
        const s = host.stale;
        host.mutex.unlock(std.testing.io);
        if (s == want) return;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.StaleFlagNeverChanged;
}

test "a learner whose put-back the device lacks refuses replies, takes learn requests, and clears once it lands" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var learner: Learner = .{};
    host.learner = .{ .ctx = &learner, .begin = Learner.begin, .step = Learner.step, .abort = Learner.abort };
    try host.start();
    defer host.stop();
    const learn: api.LearnRequest = .{};
    var first: Seen = .{};
    try host.engine().learn(&learn, first.learnSink());
    try first.wait(.lesson);
    try waitStale(&host, true);
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 8 };
    var refused: Seen = .{};
    try std.testing.expectError(error.Stale, host.engine().submit(1, &request, .{ .ctx = &refused, .event = Seen.replied }));
    // a learn request still goes in, so the learner can try the put-back again; once it lands, replies are taken
    learner.stale = false;
    var second: Seen = .{};
    try host.engine().learn(&learn, second.learnSink());
    try second.wait(.lesson);
    try waitStale(&host, false);
    var served: Seen = .{};
    try host.engine().submit(2, &request, .{ .ctx = &served, .event = Seen.replied });
    try served.wait(.reply);
    try std.testing.expectEqual(Reason.length, served.reason.?);
}

test "a reply queued while a learner's step runs ends failed once the step says stale, never admitted; a heal serves the next" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var learner: Learner = .{};
    learner.hold.store(true, .release);
    host.learner = .{ .ctx = &learner, .begin = Learner.begin, .step = Learner.step, .abort = Learner.abort };
    try host.start();
    defer host.stop();
    // runs before `host.stop`, which joins the engine thread: a failure or timeout while it is held must not hang that
    defer learner.hold.store(false, .release);
    const learn: api.LearnRequest = .{};
    var first: Seen = .{};
    try host.engine().learn(&learn, first.learnSink());
    // the step is under way and waits: the device is not yet known stale, so a reply is taken and queued
    for (0..5000) |_| if (!learner.entered.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expect(learner.entered.load(.acquire));
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 8 };
    var queued: Seen = .{};
    try host.engine().submit(1, &request, .{ .ctx = &queued, .event = Seen.replied });
    learner.hold.store(false, .release);
    try first.wait(.lesson);
    // the step said stale: the queued reply ends failed with the put-back's words, and the backend never saw it
    try queued.wait(.reply);
    try std.testing.expectEqual(Reason.failed, queued.reason.?);
    try std.testing.expectEqualStrings(api.stale_message, queued.said[0..queued.said_len]);
    try std.testing.expectEqual(@as(usize, 0), target.prefill_count);
    try std.testing.expectEqual(@as(u64, 0), target.rounds);
    try std.testing.expectEqual(@as(u32, 0), target.lanes.count());
    try waitStale(&host, true);
    // the queue is empty, so the learn request that heals is not held behind it; then a reply is served
    learner.stale = false;
    var second: Seen = .{};
    try host.engine().learn(&learn, second.learnSink());
    try second.wait(.lesson);
    try waitStale(&host, false);
    var served: Seen = .{};
    try host.engine().submit(2, &request, .{ .ctx = &served, .event = Seen.replied });
    try served.wait(.reply);
    try std.testing.expectEqual(Reason.length, served.reason.?);
    try std.testing.expect(target.prefill_count > 0);
}
