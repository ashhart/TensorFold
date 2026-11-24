const std = @import("std");
const clock = @import("clock.zig");
const server = @import("server.zig");

test "queue phase: server notes measured waits only when explicitly enabled" {
    const defaults: server.Config = .{ .served_name = "test", .model_ids = &.{"test"} };
    try std.testing.expect(!defaults.phase_metrics);
    for ([_]bool{ false, true }) |enabled| {
        var srv: server.Server = undefined;
        srv.io = std.testing.io;
        srv.metrics = .{ .gpa = std.testing.allocator, .phases = enabled };
        defer srv.metrics.deinit();
        srv.noteRequest(10, 0, 3, 0, 0, 2, clock.nowNs(srv.io), null, null, 0.25, 0.3);
        try std.testing.expectEqual(@as(u64, if (enabled) 1 else 0), srv.metrics.queue.n);
        srv.noteRequest(10, 0, 3, 0, 0, 2, clock.nowNs(srv.io), null, null, null, null);
        try std.testing.expectEqual(@as(u64, if (enabled) 1 else 0), srv.metrics.queue.n);
        try std.testing.expectEqual(@as(u64, 6), srv.metrics.generation);
    }
}
