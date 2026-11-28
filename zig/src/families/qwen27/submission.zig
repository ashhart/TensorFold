//! Command queue selection for the native single-device Qwen27 server.
const std = @import("std");
const mtl = @import("metal");
const environment = @import("sys").environment;
const ane = @import("ane_policy.zig");

pub fn open(device: mtl.Device, served: bool) !mtl.Queue {
    const root = @import("root");
    if (comptime @hasDecl(root, "tf_metal4") and root.tf_metal4) {
        const requested = served and try environment.read(bool, "TF_METAL4", false);
        const columns = if (requested) try ane.columns((environment.Process{}).get("TF_ANE_COLUMNS")) else 0;
        return select(device, requested, requested and ane.requestedOnChip(columns, device.tensorUnits()));
    }
    return device.queue();
}

fn select(device: anytype, requested: bool, ane_requested: bool) !@typeInfo(@TypeOf(device.queue())).error_union.payload {
    if (requested and !ane_requested) {
        if (device.queue4()) |q| {
            std.log.info("qwen27: Metal 4 command queue", .{});
            return q;
        } else |_| std.log.warn("qwen27: Metal 4 queue unavailable; using Metal 3", .{});
    }
    if (requested and ane_requested) std.log.info("qwen27: using Metal 3 for the Neural Engine prompt share", .{});
    return device.queue();
}

test "Qwen27 selects Metal 4 only when requested and supported" {
    const Fake = struct {
        available: bool,
        attempted: *bool,

        pub fn queue(_: @This()) error{}!u8 {
            return 3;
        }

        pub fn queue4(d: @This()) error{Unsupported}!u8 {
            d.attempted.* = true;
            if (!d.available) return error.Unsupported;
            return 4;
        }
    };
    const level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = level;
    var attempted = false;
    const available: Fake = .{ .available = true, .attempted = &attempted };
    try std.testing.expectEqual(3, try select(available, false, false));
    try std.testing.expectEqual(3, try select(available, true, true));
    try std.testing.expect(!attempted);
    try std.testing.expectEqual(4, try select(available, true, false));
    try std.testing.expect(attempted);
    attempted = false;
    try std.testing.expectEqual(3, try select(Fake{ .available = false, .attempted = &attempted }, true, false));
    try std.testing.expect(attempted);
}

test "Qwen27 submission exposes its Metal queue API on a host" {
    _ = open;
}
