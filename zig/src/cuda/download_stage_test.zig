//! The synchronous download's staging path, with a driver that only writes pinned destinations.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const stage = @import("download_stage.zig");

var pinned: [64 << 10]u8 align(16) = undefined;
var destination: usize = 0;

fn pinnedRange(ptr: usize) bool {
    const base = @intFromPtr(&pinned[0]);
    return ptr >= base and ptr < base + pinned.len;
}

fn hostAlloc(out: ?*?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
    if (len > pinned.len) return 1;
    out.?.* = &pinned[0];
    return 0;
}

fn hostFree(_: ?*anyopaque) callconv(.c) abi.Result {
    return 0;
}

/// Writes only a pinned destination, as the GB10 fault leaves pageable memory untouched.
fn copyToHost(dst: ?*anyopaque, _: abi.DevicePtr, len: usize) callconv(.c) abi.Result {
    destination = @intFromPtr(dst.?);
    if (pinnedRange(destination)) @memset(@as([*]u8, @ptrCast(dst.?))[0..len], 0x5a);
    return 0;
}

test "a synchronous download lands in pinned memory, not the caller's pageable slice" {
    var d: Driver = undefined;
    d.api.cuMemHostAlloc = hostAlloc;
    d.api.cuMemFreeHost = hostFree;
    d.api.cuMemcpyDtoH_v2 = copyToHost;
    stage.drop();
    defer stage.drop();
    var out: [64]u8 = @splat(0xaa);
    const src: memory.DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = out.len, .borrowed = true };
    try src.download(0, &out);
    try std.testing.expect(std.mem.allEqual(u8, &out, 0x5a));
    try std.testing.expect(pinnedRange(destination));
    try std.testing.expect(destination != @intFromPtr(&out));
}
