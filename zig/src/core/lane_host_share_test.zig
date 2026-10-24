//! A long prompt's step is owed no decode time by streams that cannot decode (lane_host_fill.zig).
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;
const fill = @import("lane_host_fill.zig");

test "paused streams are not decoders: a long prompt's step is due when every unfinished stream is paused" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.share = .{ .p = 0.5, .debt_ns = 1000 };
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 8, .eos = &.{96} });
    defer s.deinit(gpa);
    try core.addStream(&s);
    defer core.backend.release(&s);
    try std.testing.expect(core.activeCount() == 1 and !fill.due(&host)); // a runnable decoder is owed the debt
    s.paused = true;
    try std.testing.expect(core.activeCount() == 1 and fill.due(&host)); // paused: counted active, nothing to pay with
    s.paused = false;
    s.finished = true;
    try std.testing.expect(fill.due(&host));
}
