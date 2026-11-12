//! A ring omission returns before reading poisoned buffers or issuing a device or NCCL call.
const std = @import("std");
const tp = @import("tp");
const Ctx = @import("gpu_checks.zig").Ctx;

test "relay suite: omitted send/recv avoids poisoned buffers and counts the comparison" {
    var raw: [4][16]u8 = @splat(@splat(0));
    var views: [4][]u8 = undefined;
    for (&raw, &views) |*bytes, *view| view.* = bytes;
    var c: Ctx = undefined;
    c.rank = 0;
    c.world = 4;
    c.ins = &views;
    c.in = .{ .d = undefined, .ptr = 0, .len = 0 };
    c.stream = .{ .d = undefined, .handle = null };
    c.omit_p2p = true;
    c.omitted = 0;
    const comm: tp.Collective = .{ .ptr = @ptrFromInt(1), .vtable = undefined };
    try c.pointToPoint(comm, "nccl", 16, 7);
    try std.testing.expectEqual(@as(u32, 1), c.omitted);
    for (raw) |bytes| try std.testing.expect(std.mem.allEqual(u8, &bytes, 0));
    c.omit_p2p = false;
    try std.testing.expectError(error.Invalid, c.pointToPoint(comm, "nccl", 16, 7));
    try std.testing.expect(!std.mem.allEqual(u8, &raw[0], 0));
    try std.testing.expectEqual(@as(u32, 1), c.omitted);
}
