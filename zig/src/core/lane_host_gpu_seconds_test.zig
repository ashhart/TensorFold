//! A reply's gpu_seconds through the actual lane host: absent unless the backend times its passes (TF_Q27_PROFILE).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const t = std.testing;

/// A request's tokens, and its finished event's reason and counts.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    failed: bool = false,
    reason: ?api.Reason = null,
    stats: api.Stats = .{},
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(t.io);
        defer b.mutex.unlock(t.io);
        switch (e.*) {
            .tokens => |got| b.tokens.appendSlice(t.allocator, got) catch {
                b.failed = true;
            },
            .finished => |f| {
                b.stats = f.stats;
                b.stats.prefill_widths = &.{}; // the event's slices are the host's: the counts are what stay
                b.stats.prefill_raised = &.{};
                b.stats.telemetry_json = "";
                b.reason = f.reason;
            },
            else => {},
        }
    }
    fn wait(b: *@This()) !void {
        for (0..10_000) |_| { // 10 s
            b.mutex.lockUncancelable(t.io);
            const done = b.reason != null;
            const failed = b.failed;
            b.mutex.unlock(t.io);
            if (failed) return error.OutOfMemory;
            if (done) return;
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

const prompt = [_]u32{ 3, 1, 4, 1, 5 };
const request: api.Request = .{ .prompt = &prompt, .max_tokens = 24 };

/// One request through a lane host over the fake target, which charges `charge` GPU seconds a pass (null: untimed).
fn reply(charge: ?f64, box: *Box) !void {
    const gpa = t.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .gpu_charge = charge };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, t.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    try host.engine().submit(1, &request, .{ .ctx = box, .event = Box.event });
    try box.wait();
}

test "gpu_seconds: an untimed backend's reply carries none, and timing changes no token or count" {
    var off: Box = .{};
    defer off.tokens.deinit(t.allocator);
    try reply(null, &off);
    var on: Box = .{};
    defer on.tokens.deinit(t.allocator);
    try reply(0.25, &on);
    try t.expect(off.stats.gpu_seconds == null);
    try t.expect(on.stats.rounds > 0);
    try t.expect(on.stats.gpu_seconds.? >= 0.25 * @as(f64, @floatFromInt(on.stats.rounds))); // every round it ran in
    try t.expectEqual(api.Reason.length, off.reason.?);
    try t.expectEqual(off.reason, on.reason);
    try t.expectEqualSlices(u32, off.tokens.items, on.tokens.items);
    try t.expectEqual(off.stats.rounds, on.stats.rounds);
    try t.expectEqual(off.stats.drafted, on.stats.drafted);
    try t.expectEqual(off.stats.accepted, on.stats.accepted);
    try t.expectEqual(off.stats.min_rows, on.stats.min_rows);
    try t.expectEqual(off.stats.loop_period, on.stats.loop_period);
}
