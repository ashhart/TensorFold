//! The download staging path, with a fake driver that writes only pinned destinations.
const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const stage = @import("download_stage.zig");

var pinned: [64 << 10]u8 align(16) = undefined;
var destination: usize = 0;
var allocations: usize = 0;
var integrated_flag: c_int = 1;

fn pinnedRange(ptr: usize) bool {
    const base = @intFromPtr(&pinned[0]);
    return ptr >= base and ptr < base + pinned.len;
}

fn hostAlloc(out: ?*?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
    if (len > pinned.len) return 1;
    allocations += 1;
    out.?.* = &pinned[0];
    return 0;
}

fn hostFree(_: ?*anyopaque) callconv(.c) abi.Result {
    return 0;
}

fn contextDevice(out: *abi.Device) callconv(.c) abi.Result {
    out.* = 0;
    return 0;
}

fn deviceAttribute(out: *c_int, attribute: abi.DeviceAttribute, _: abi.Device) callconv(.c) abi.Result {
    out.* = if (attribute == .integrated) integrated_flag else 0;
    return 0;
}

fn copyToHost(dst: ?*anyopaque, _: abi.DevicePtr, len: usize) callconv(.c) abi.Result {
    destination = @intFromPtr(dst.?);
    if (pinnedRange(destination) or integrated_flag == 0) @memset(@as([*]u8, @ptrCast(dst.?))[0..len], 0x5a);
    return 0;
}

fn driver() Driver {
    var d: Driver = undefined;
    d.api.cuMemHostAlloc = hostAlloc;
    d.api.cuMemFreeHost = hostFree;
    d.api.cuCtxGetDevice = contextDevice;
    d.api.cuDeviceGetAttribute = deviceAttribute;
    d.api.cuMemcpyDtoH_v2 = copyToHost;
    return d;
}

/// A real library handle: a fake driver's close closes one, as Driver.close does for an opened driver.
fn library() !@import("sys").DynLib {
    if (builtin.os.tag == .windows) return @import("sys").DynLib.open("kernel32.dll");
    if (builtin.os.tag == .macos) return @import("sys").DynLib.open("/usr/lib/libSystem.B.dylib");
    return @import("sys").DynLib.open("libc.so.6");
}

test "an integrated GPU's download lands in pinned memory, not the caller's pageable slice" {
    integrated_flag = 1;
    allocations = 0;
    stage.forget();
    stage.drop();
    defer stage.forget();
    defer stage.drop();
    const before = memory.usage(false).host;
    var d = driver();
    var out: [64]u8 = @splat(0xaa);
    const src: memory.DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = out.len, .borrowed = true };
    try src.download(0, &out);
    try std.testing.expect(std.mem.allEqual(u8, &out, 0x5a));
    try std.testing.expect(pinnedRange(destination));
    try std.testing.expect(destination != @intFromPtr(&out));
    try std.testing.expectEqual(@as(usize, 1), allocations);
    try std.testing.expect(memory.usage(false).host >= before + (64 << 10));
}

test "a discrete GPU's download copies straight into the caller's slice" {
    integrated_flag = 0;
    allocations = 0;
    stage.forget();
    stage.drop();
    defer stage.forget();
    defer stage.drop();
    var d = driver();
    var out: [64]u8 = @splat(0xaa);
    const src: memory.DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = out.len, .borrowed = true };
    try src.download(0, &out);
    try std.testing.expect(std.mem.allEqual(u8, &out, 0x5a));
    try std.testing.expectEqual(@intFromPtr(&out), destination);
    try std.testing.expectEqual(@as(usize, 0), allocations);
}

test "a driver closing the stage frees this thread's buffer, so no freed driver is ever freed through" {
    integrated_flag = 1;
    allocations = 0;
    stage.forget();
    stage.drop();
    defer stage.forget();
    defer stage.drop();
    const before = memory.usage(false).host;
    var one = driver();
    one.lib = try library();
    var open = true;
    defer if (open) one.lib.close(); // a failed check still lets go of the handle
    var out: [64]u8 = @splat(0xaa);
    const src: memory.DeviceBuffer = .{ .d = &one, .ptr = 0x1000, .len = out.len, .borrowed = true };
    try src.download(0, &out);
    try std.testing.expectEqual(@as(usize, 1), allocations);
    one.close();
    open = false;
    try std.testing.expectEqual(before, memory.usage(false).host);
    // the next driver stages its own buffer instead of freeing through the one that closed
    var two = driver();
    const other: memory.DeviceBuffer = .{ .d = &two, .ptr = 0x2000, .len = out.len, .borrowed = true };
    try other.download(0, &out);
    try std.testing.expectEqual(@as(usize, 2), allocations);
    try std.testing.expect(std.mem.allEqual(u8, &out, 0x5a));
}

test "a driver closing the stage forgets the device it cached, so a reused address is asked again" {
    integrated_flag = 1;
    stage.forget();
    stage.drop();
    defer stage.forget();
    var one = driver();
    one.lib = try library();
    var open = true;
    defer if (open) one.lib.close();
    try std.testing.expect(try stage.integrated(&one));
    integrated_flag = 0;
    one.close();
    open = false;
    // a replacement driver at that address must be asked: the reading holds for the driver that gave it
    try std.testing.expect(!try stage.integrated(&one));
}
