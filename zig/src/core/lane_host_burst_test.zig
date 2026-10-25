//! Short prompts beside decoding streams: a burst of them pauses decoding a bounded while (lane_host_fill.zig).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

/// Prompt passes against the decoder's rounds (its token deliveries), counted as their events arrive.
const Log = struct {
    mutex: std.Io.Mutex = .init,
    run: usize = 0, // prompt passes since the decoder's last round
    most: usize = 0, // the most passes in a row with no round between them
    rounds: usize = 0, // decoder rounds since the last short prompt's pass
    wait: usize = 0, // the most rounds between two short prompts' passes
    shorts: usize = 0,
    done: usize = 0,
    fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
        const o: *@This() = @ptrCast(@alignCast(ctx));
        o.mutex.lockUncancelable(std.testing.io);
        defer o.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => {
                o.run += 1;
                o.most = @max(o.most, o.run);
                if (id < first_short) return;
                if (o.shorts > 0) o.wait = @max(o.wait, o.rounds);
                o.shorts += 1;
                o.rounds = 0;
            },
            .tokens => if (id == decoder) {
                o.run = 0;
                o.rounds += 1;
            },
            .finished => o.done += 1,
            else => {},
        }
    }
    fn until(o: *@This(), n: usize) void {
        while (true) {
            o.mutex.lockUncancelable(std.testing.io);
            const d = o.done;
            o.mutex.unlock(std.testing.io);
            if (d >= n) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

const first_short: Id = 10;
const decoder: Id = 1;

test "one-token replies arriving without end pause the decoder for at most a lane count of prompts" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.share = .{ .p = 0.5 };
    var log: Log = .{};
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const long: Request = .{ .prompt = &prompt, .max_tokens = 1 << 16 };
    const one: Request = .{ .prompt = &prompt, .max_tokens = 1 };
    const e = host.engine();
    // a reply of one token frees its lane at its prompt's end: the next waits in the queue, as under steady arrivals
    try e.submit(decoder, &long, .{ .ctx = &log, .event = Log.event });
    for (0..40) |i| try e.submit(@intCast(first_short + i), &one, .{ .ctx = &log, .event = Log.event });
    try host.start();
    defer host.stop();
    log.until(40);
    try std.testing.expect(log.most <= 2);
}
