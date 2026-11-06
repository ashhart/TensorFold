//! The download staging path, with a fake driver that writes only pinned destinations.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const stage = @import("download_stage.zig");

var pinned: [stage.most]u8 align(16) = undefined;
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

fn copySequence(dst: ?*anyopaque, src: abi.DevicePtr, len: usize) callconv(.c) abi.Result {
    if (!pinnedRange(@intFromPtr(dst.?))) return abi.error_not_ready;
    const bytes = @as([*]u8, @ptrCast(dst.?))[0..len];
    for (bytes, 0..) |*b, i| b.* = @truncate((src + i) ^ ((src + i) >> 21));
    return 0;
}

test "an unowned device span retains chunk offsets, exact tails and destination guards" {
    integrated_flag = 1;
    allocations = 0;
    stage.forget();
    stage.drop();
    defer stage.forget();
    defer stage.drop();
    var d = driver();
    d.api.cuMemcpyDtoH_v2 = copySequence;
    const n = stage.most + 257;
    const bytes = try std.testing.allocator.alloc(u8, n + 32);
    defer std.testing.allocator.free(bytes);
    @memset(bytes, 0xaa);
    const src: u64 = 0x103b;
    try memory.download(&d, src, bytes[16..][0..n]);
    for (bytes[16..][0..n], 0..) |b, i| try std.testing.expectEqual(@as(u8, @truncate((src + i) ^ ((src + i) >> 21))), b);
    try std.testing.expect(std.mem.allEqual(u8, bytes[0..16], 0xaa));
    try std.testing.expect(std.mem.allEqual(u8, bytes[n + 16 ..], 0xaa));
    try std.testing.expectEqual(@as(usize, 1), allocations);
}

test "an empty unowned readback needs no driver or allocation" {
    const d: Driver = undefined;
    try memory.download(&d, 0, &.{});
}

test "closing a driver drops its thread's bounce and forgets its integrated reading" {
    integrated_flag = 1;
    allocations = 0;
    stage.forget();
    stage.drop();
    defer stage.forget();
    defer stage.drop();
    const before = memory.usage(false).host;
    var d = driver();
    const library = switch (@import("builtin").os.tag) {
        .macos => "/usr/lib/libSystem.B.dylib",
        .windows => "kernel32.dll",
        else => "libc.so.6",
    };
    d.lib = try @import("sys").DynLib.open(library);
    var closed = false;
    defer if (!closed) d.close();
    var out: [64]u8 = @splat(0xaa);
    try memory.download(&d, 0x1000, &out);
    var other = driver();
    other.lib = try @import("sys").DynLib.open(library);
    other.close();
    try std.testing.expectEqual(before + (64 << 10), memory.usage(false).host);
    d.close();
    closed = true;
    try std.testing.expectEqual(before, memory.usage(false).host);
    integrated_flag = 0;
    d = driver();
    d.lib = try @import("sys").DynLib.open(library);
    closed = false;
    @memset(&out, 0xaa);
    try memory.download(&d, 0x1000, &out);
    try std.testing.expectEqual(@intFromPtr(&out), destination);
    try std.testing.expectEqual(@as(usize, 1), allocations);
    try std.testing.expectEqual(before, memory.usage(false).host);
}
