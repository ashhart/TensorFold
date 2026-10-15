//! A learner whose put-back the device lacks: replies refused, learn requests taken, replies back once it lands.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

/// A learner whose every step reports `stale`, ending each lesson at once.
const Learner = struct {
    sink: ?api.LearnSink = null,
    stale: bool = true,

    fn begin(ctx: *anyopaque, _: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
        const l: *Learner = @ptrCast(@alignCast(ctx));
        l.sink = sink;
    }
    fn step(ctx: *anyopaque) api.Learner.Step {
        const l: *Learner = @ptrCast(@alignCast(ctx));
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
        if (e.* == .finished) s.reason = e.finished.reason;
    }
    fn wait(s: *Seen, comptime what: enum { lesson, reply }) void {
        while (true) {
            s.mutex.lockUncancelable(std.testing.io);
            const got = if (what == .lesson) s.done else s.reason != null;
            s.mutex.unlock(std.testing.io);
            if (got) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
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
    first.wait(.lesson);
    try waitStale(&host, true);
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 8 };
    var refused: Seen = .{};
    try std.testing.expectError(error.Stale, host.engine().submit(1, &request, .{ .ctx = &refused, .event = Seen.replied }));
    // a learn request still goes in, so the learner can try the put-back again; once it lands, replies are taken
    learner.stale = false;
    var second: Seen = .{};
    try host.engine().learn(&learn, second.learnSink());
    second.wait(.lesson);
    try waitStale(&host, false);
    var served: Seen = .{};
    try host.engine().submit(2, &request, .{ .ctx = &served, .event = Seen.replied });
    served.wait(.reply);
    try std.testing.expectEqual(Reason.length, served.reason.?);
}
