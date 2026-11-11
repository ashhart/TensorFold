//! A handoff request cancelled by its cut captures no export: the producer holds no state for a decoder that is gone.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const rt = @import("lane_remote_test.zig");
const t = std.testing;

/// A cancel check that turns on at its second look: the fake's pass check is the first, the export's the second.
const Late = struct {
    looks: u32 = 0,
    done: std.atomic.Value(bool) = .init(false),
    reason: ?api.Reason = null,

    fn check(ptr: *anyopaque) bool {
        const l: *Late = @ptrCast(@alignCast(ptr));
        l.looks += 1;
        return l.looks > 1;
    }

    fn hook(ctx: *anyopaque, s: *lanes.Stream, n: usize) void {
        if (n == 0) s.cancel_check = .{ .ptr = ctx, .check = check };
    }

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const l: *Late = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .finished => |f| {
                l.reason = f.reason;
                l.done.store(true, .release);
            },
            else => {},
        }
    }
};

fn handoff(late: ?*Late, e: *rt.Exp) !void {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var seen: Late = .{};
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator, .prefill_hook = if (late != null) Late.hook else null, .prefill_hook_ctx = &seen };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    e.fake = &fake;
    host.exporter = e.exporter();
    try host.start();
    defer host.stop();
    try host.engine().submit(1, &.{ .prompt = &rt.six, .max_tokens = 8, .handoff = rt.id }, .{ .ctx = &seen, .event = Late.event });
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!seen.done.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.ExportFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    if (late) |l| l.* = seen;
}

test "handoff: a request cancelled by its cut takes no snapshot and exports nothing" {
    var live: rt.Exp = .{};
    try handoff(null, &live);
    try t.expectEqual(@as(?usize, 5), live.tokens); // the control: not cancelled, exported at its cut

    var gone: rt.Exp = .{};
    var late: Late = .{};
    try handoff(&late, &gone);
    try t.expectEqual(api.Reason.cancelled, late.reason.?);
    try t.expectEqual(@as(?usize, null), gone.tokens); // no snapshot was taken, none handed to the exporter
    try t.expectEqual(@as(?usize, null), gone.kept);
}
