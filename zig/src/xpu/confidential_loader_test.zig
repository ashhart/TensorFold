//! --confidential's loader policy over Level Zero: discovery and initialization are refused before any open.
const std = @import("std");
const ze = @import("ze");
const policy = @import("sys").load_policy;
const device_id = @import("device_id.zig");
const context = @import("context.zig");
const t = std.testing;

fn refused() u32 {
    return policy.refusals.load(.monotonic);
}

/// XPU_DEVICE_ID skips discovery, so it is blank for the test (unsetenv would break std's environment scan).
const NoOverride = struct {
    saved: ?[:0]u8 = null,
    fn take() !NoOverride {
        const sys = @import("sys");
        const was = sys.getenv("XPU_DEVICE_ID") orelse return .{};
        const saved = try t.allocator.dupeSentinel(u8, was, 0);
        _ = sys.env.setenv("XPU_DEVICE_ID", "", 1);
        return .{ .saved = saved };
    }
    fn restore(n: NoOverride) void {
        const v = n.saved orelse return;
        defer t.allocator.free(v);
        _ = @import("sys").env.setenv("XPU_DEVICE_ID", v, 1);
    }
};

test "confidential: XPU discovery asks the loader policy and is refused, whatever XPU_DEVICE_ID the host sets" {
    const kept = try NoOverride.take();
    defer kept.restore();
    policy.confine();
    defer policy.release();
    const before = refused();
    try t.expectError(error.LoaderUnavailable, device_id.detectDeviceId());
    try t.expectEqual(before + 1, refused());
}

test "confidential: XPU initialization asks the loader policy and is refused, by context and by device" {
    policy.confine();
    defer policy.release();
    const before = refused();
    try t.expectError(error.LoaderUnavailable, context.Driver.open());
    try t.expectEqual(before + 1, refused());
    var out: [1]ze.Gpu = undefined;
    try t.expectError(error.LoaderUnavailable, ze.Gpu.openPeers(0xe211, &out));
    try t.expectEqual(before + 2, refused());
}

test "confidential: without the flag, XPU loads are the loader's, as before" {
    const before = refused();
    if (context.Driver.open()) |opened| {
        var d = opened;
        d.close();
    } else |_| {}
    try t.expectEqual(before, refused());
}
