//! Drain prefetched work before returning a reply or releasing its buffers.
const std = @import("std");

pub fn drain(cbs: anytype, tp: anytype) !void {
    var failed: ?anyerror = null;
    for (cbs) |*slot| if (slot.*) |cb| {
        cb.wait();
        slot.* = null;
        if (tp != null and cb.failure() != null and failed == null) failed = error.GpuFailed;
    };
    if (tp) |peer| if (peer.failed.load(.acquire) and failed == null) {
        failed = error.TpLinkFailed;
    };
    if (failed) |err| return err;
}

const Probe = struct {
    failed: std.atomic.Value(bool) = .init(false),
    waits: usize = 0,
    status_queries: usize = 0,

    const Buffer = struct {
        probe: *Probe,
        terminal: bool = false,
        gpu_error: bool = false,

        fn wait(cb: Buffer) void {
            cb.probe.waits += 1;
            if (cb.terminal) cb.probe.failed.store(true, .release);
        }

        fn failure(cb: Buffer) ?[]const u8 {
            cb.probe.status_queries += 1;
            return if (cb.gpu_error) "planted failure" else null;
        }
    };
};

test "terminal release during the final prefetched drain cannot return success" {
    var peer: Probe = .{};
    var cbs = [_]?Probe.Buffer{ .{ .probe = &peer, .terminal = true }, .{ .probe = &peer } };
    try std.testing.expectError(error.TpLinkFailed, drain(&cbs, @as(?*Probe, &peer)));
    try std.testing.expectEqual(@as(usize, 2), peer.waits);
    for (cbs) |cb| try std.testing.expect(cb == null);
}

test "a failed link is reported even when no prefetched work remains" {
    var peer: Probe = .{};
    peer.failed.store(true, .release);
    var cbs = [_]?Probe.Buffer{null};
    try std.testing.expectError(error.TpLinkFailed, drain(&cbs, @as(?*Probe, &peer)));
}

test "a GPU failure drains every remaining buffer before returning its error" {
    var probe: Probe = .{};
    var cbs = [_]?Probe.Buffer{ .{ .probe = &probe, .gpu_error = true }, .{ .probe = &probe } };
    try std.testing.expectError(error.GpuFailed, drain(&cbs, @as(?*Probe, &probe)));
    try std.testing.expectEqual(@as(usize, 2), probe.waits);
    for (cbs) |cb| try std.testing.expect(cb == null);
}

test "one-Mac final drain only waits and clears without querying GPU status" {
    var probe: Probe = .{};
    var cbs = [_]?Probe.Buffer{ .{ .probe = &probe, .gpu_error = true }, .{ .probe = &probe } };
    try drain(&cbs, @as(?*Probe, null));
    try std.testing.expectEqual(@as(usize, 2), probe.waits);
    try std.testing.expectEqual(@as(usize, 0), probe.status_queries);
    for (cbs) |cb| try std.testing.expect(cb == null);
}

test "a healthy final drain completes all buffers" {
    var peer: Probe = .{};
    var cbs = [_]?Probe.Buffer{ .{ .probe = &peer }, null, .{ .probe = &peer } };
    try drain(&cbs, @as(?*Probe, &peer));
    try std.testing.expectEqual(@as(usize, 2), peer.waits);
}
