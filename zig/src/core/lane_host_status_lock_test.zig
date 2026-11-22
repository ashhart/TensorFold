//! A status read must never wait for another status read (issue 3200): a monitoring client
//! parked inside a slow source read (health, fsr) must not hold the host's lock, or the next
//! reader and the round that shares that lock wait for it.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Event = api.Event;
const Request = api.Request;

/// A health source the test parks the reader inside: `read` signals `entered`, then waits
/// for `release`, exactly the shape of a slow backend probe a dashboard read makes.
const Gate = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    calls: std.atomic.Value(u32) = .init(0),

    fn read(ctx: *anyopaque) bool {
        const g: *Gate = @ptrCast(@alignCast(ctx));
        if (g.calls.fetchAdd(1, .monotonic) != 0) return true; // only the first read parks
        g.entered.store(true, .release);
        // auto-release: under the old code the caller holds the host's lock here, so nothing
        // else can set `release`; the cap keeps that run failing, not hanging
        var ms: usize = 0;
        while (!g.release.load(.acquire) and ms < 1500) : (ms += 1)
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        return true;
    }
};

const Tally = struct {
    tokens: std.atomic.Value(usize) = .init(0),
    done: std.atomic.Value(bool) = .init(false),

    fn event(ctx: *anyopaque, id: api.Id, e: *const Event) void {
        _ = id;
        const t: *@This() = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .tokens => |toks| _ = t.tokens.fetchAdd(toks.len, .monotonic),
            .finished => t.done.store(true, .release),
            else => {},
        }
    }
};

test "status: a read parked in a slow source read does not block the next read" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var gate: Gate = .{};
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.health = .{ .ctx = &gate, .read = Gate.read };
    try host.start();
    defer host.stop();

    // The reader's first status read parks inside health.read and stays there until the
    // release: the shape of a dashboard polling a slow backend counter.
    const Reader = struct {
        fn run(engine: api.Engine, gated: *Gate) void {
            _ = gated;
            var status: api.Status = .{};
            var stream_tokens: [8]u32 = undefined;
            engine.status(&status, &stream_tokens); // parks inside the gate's read
            engine.status(&status, &stream_tokens); // one more read after the release
        }
    };
    var reader = try std.Thread.spawn(.{}, Reader.run, .{ host.engine(), &gate });
    defer reader.join();

    var waited: usize = 0;
    while (!gate.entered.load(.acquire) and waited < 2000) : (waited += 1)
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expect(gate.entered.load(.acquire)); // the reader is parked inside health.read

    // The next read must not wait for it: the reader holds no host lock.
    var status: api.Status = .{};
    var stream_tokens: [8]u32 = undefined;
    const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    host.engine().status(&status, &stream_tokens);
    const waited_ns = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began;
    try std.testing.expect(waited_ns < 200 * std.time.ns_per_ms); // it returned while the reader was parked
    try std.testing.expect(status.healthy); // and it read a whole snapshot

    gate.release.store(true, .release);

    // The request still runs and finishes normally.
    var tally: Tally = .{};
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var request: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    _ = try host.engine().submit(1, &request, .{ .ctx = &tally, .event = Tally.event });
    var deadline: usize = 0;
    while (!tally.done.load(.acquire) and deadline < 8000) : (deadline += 1)
        std.Io.sleep(std.testing.io, .fromMilliseconds(2), .awake) catch {};
    try std.testing.expect(tally.done.load(.acquire)); // the request finishes normally
    try std.testing.expect(tally.tokens.load(.monotonic) >= 8); // with its tokens
}

/// A health source every read pays for: the shape of a backend counter that costs the reader time.
const Slow = struct {
    fn read(ctx: *anyopaque) bool {
        _ = ctx;
        std.Io.sleep(std.testing.io, .fromMilliseconds(2), .awake) catch {};
        return true;
    }
};

test "status: a reader hammering the status during rounds never makes the round wait" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.health = .{ .ctx = &host, .read = Slow.read };
    try host.start();
    defer host.stop();

    var tally: Tally = .{};
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var request: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    _ = try host.engine().submit(1, &request, .{ .ctx = &tally, .event = Tally.event });

    // Wait for the request to be running: the baseline counts only the readers' waits.
    var status: api.Status = .{};
    var stream_tokens: [8]u32 = undefined;
    var waited: usize = 0;
    while (waited < 4000) : (waited += 1) {
        host.engine().status(&status, &stream_tokens);
        if (status.running > 0) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(2), .awake) catch {};
    }
    const baseline = host.round_waits.load(.monotonic);

    // Four monitoring clients poll the status as fast as they can while the round runs.
    const Hammer = struct {
        fn run(engine: api.Engine, stop: *std.atomic.Value(bool)) void {
            var s: api.Status = .{};
            var buf: [8]u32 = undefined;
            while (!stop.load(.acquire)) engine.status(&s, &buf);
        }
    };
    var stop = std.atomic.Value(bool).init(false);
    var hammers: [4]std.Thread = undefined;
    for (&hammers) |*t| t.* = try std.Thread.spawn(.{}, Hammer.run, .{ host.engine(), &stop });
    defer {
        stop.store(true, .release);
        for (hammers) |t| t.join();
    }

    var deadline: usize = 0;
    while (!tally.done.load(.acquire) and deadline < 10000) : (deadline += 1)
        std.Io.sleep(std.testing.io, .fromMilliseconds(2), .awake) catch {};
    try std.testing.expect(tally.done.load(.acquire)); // the request finished under the hammering
    try std.testing.expectEqual(baseline, host.round_waits.load(.monotonic)); // and no round waited for a reader
}
