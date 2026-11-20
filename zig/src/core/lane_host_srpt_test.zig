//! A terminal prompt's step must not charge its freed job's lent time (issue #3709).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const fill = @import("lane_host_fill.zig");
const ends = @import("lane_host_end.zig");
const Host = @import("lane_host.zig").LaneHost;

const Sink = struct {
    fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
};

test "srpt: a finished prompt's lend sample lands only while its job still fills" {
    const gpa = std.heap.page_allocator; // a job's own pages are unmapped at destroy, so a stale write faults
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = Host.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.srpt = .{ .on = true, .fill_ns_row = 1e9, .round_ns = 1, .round_tokens = 1 };
    defer {
        ends.closeAll(&host);
        host.stop();
    }
    const decoder: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 10000, .drafts = false };
    try host.engine().submit(1, &decoder, .{ .ctx = &host, .event = Sink.event });
    try std.testing.expect(host.admitOne());
    fill.run(&host);
    try std.testing.expectEqual(@as(usize, 0), host.filling.items.len);
    try std.testing.expectEqual(@as(usize, 1), host.core.live.items.len); // the decoder keeps `decoding` on
    const prompt: [2048]u32 = @splat(3);
    const terminal: api.Request = .{ .prompt = &prompt, .max_tokens = 1, .drafts = false };
    try host.engine().submit(2, &terminal, .{ .ctx = &host, .event = Sink.event });
    try std.testing.expect(host.admitOne());
    try std.testing.expectEqual(@as(usize, 1), host.filling.items.len);
    fill.run(&host); // the step's last rows deliver and destroy the job; its lent time has nowhere to land
    try std.testing.expectEqual(@as(usize, 0), host.filling.items.len);
    try std.testing.expectEqual(@as(usize, 1), host.core.live.items.len);
}
