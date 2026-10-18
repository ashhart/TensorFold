//! Inert CUDA storage and pending-use sentinels for the production H3 entry points.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const abi = cuda.abi;
pub var memory: [4 << 20]u8 align(256) = undefined;
pub var at: usize = 0;
pub var pending = false;
pub var wait_fail = false;
pub var alloc_fail = false;
pub var map_fail = false;
pub var frees: usize = 0;
pub var unsafe_frees: usize = 0;
pub var unloads: usize = 0;
pub var hosts: usize = 0;
pub var last_freed: u64 = 0;
pub var host_guard: ?*const fn () bool = null;
fn current(_: abi.Context) callconv(.c) c_int {
    return 0;
}
fn sync() callconv(.c) c_int {
    if (pending and wait_fail) return abi.error_not_ready;
    pending = false;
    return 0;
}
fn streamSync(_: abi.Stream) callconv(.c) c_int {
    return sync();
}
fn alloc(out: *u64, n: usize) callconv(.c) c_int {
    if (alloc_fail) return abi.error_not_ready;
    at = (at + 255) / 256 * 256;
    if (n > memory.len - at) return abi.error_not_ready;
    out.* = @intFromPtr(memory[at..].ptr);
    at += n;
    return 0;
}
fn free(ptr: u64) callconv(.c) c_int {
    frees += 1;
    last_freed = ptr;
    if (pending) unsafe_frees += 1;
    return 0;
}
fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) c_int {
    var ptr: u64 = 0;
    const status = alloc(&ptr, n);
    if (status != 0) return status;
    out.* = @ptrFromInt(ptr);
    hosts += 1;
    return 0;
}
fn hostFree(_: ?*anyopaque) callconv(.c) c_int {
    if (host_guard) |guard| if (!guard()) {
        unsafe_frees += 1;
    };
    if (pending) unsafe_frees += 1;
    hosts -= 1;
    return 0;
}
fn hostDevice(out: *u64, p: ?*anyopaque, _: c_uint) callconv(.c) c_int {
    if (map_fail) return abi.error_not_ready;
    out.* = @intFromPtr(p.?);
    return 0;
}
fn htod(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) c_int {
    @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
    return 0;
}
fn dtoh(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) c_int {
    @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
    return 0;
}
fn fill(dst: u64, v: u8, n: usize) callconv(.c) c_int {
    @memset(@as([*]u8, @ptrFromInt(dst))[0..n], v);
    return 0;
}
fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
    pending = true;
    return 0;
}
fn unload(_: abi.Module) callconv(.c) c_int {
    unloads += 1;
    if (pending) unsafe_frees += 1;
    return 0;
}
fn release(_: abi.Device) callconv(.c) c_int {
    if (pending) unsafe_frees += 1;
    return 0;
}
pub fn model() !*h3.Model {
    at = 0;
    pending = false;
    wait_fail = false;
    alloc_fail = false;
    map_fail = false;
    frees = 0;
    unsafe_frees = 0;
    unloads = 0;
    hosts = 0;
    last_freed = 0;
    host_guard = null;
    const m = try std.testing.allocator.create(h3.Model);
    m.* = undefined;
    m.gpa = std.testing.allocator;
    m.driver.lib = try std.DynLib.open(if (@import("builtin").os.tag == .macos) "/usr/lib/libSystem.B.dylib" else "libc.so.6");
    m.driver.api.cuCtxSetCurrent = current;
    m.driver.api.cuCtxSynchronize = sync;
    m.driver.api.cuStreamSynchronize = streamSync;
    m.driver.api.cuMemAlloc_v2 = alloc;
    m.driver.api.cuMemFree_v2 = free;
    m.driver.api.cuMemHostAlloc = hostAlloc;
    m.driver.api.cuMemFreeHost = hostFree;
    m.driver.api.cuMemHostGetDevicePointer_v2 = hostDevice;
    m.driver.api.cuMemcpyHtoD_v2 = htod;
    m.driver.api.cuMemcpyDtoH_v2 = dtoh;
    m.driver.api.cuMemsetD8_v2 = fill;
    m.driver.api.cuLaunchKernel = launch;
    m.driver.api.cuModuleUnload = unload;
    m.driver.api.cuDevicePrimaryCtxRelease_v2 = release;
    m.ctx = .{ .d = &m.driver, .device = 3, .handle = @ptrFromInt(0x3000) };
    m.stream = .{ .d = &m.driver, .handle = null };
    m.module = .{ .d = &m.driver, .handle = @ptrFromInt(1) };
    for (&m.functions) |*f| f.* = .{ .d = &m.driver, .handle = @ptrFromInt(1) };
    const empty: cuda.DeviceBuffer = .{ .d = &m.driver, .ptr = 0, .len = 0, .borrowed = true };
    m.blocks = try m.gpa.alloc(h3.Block, 1);
    m.blocks[0] = .{ .qkvc = empty, .out = empty, .fc1 = empty, .fc2 = empty, .norms = empty };
    m.gemms = .{ .gpa = m.gpa, .lt = undefined, .workspace = empty };
    m.gemms.lt.lib = try std.DynLib.open(if (@import("builtin").os.tag == .macos) "/usr/lib/libSystem.B.dylib" else "libc.so.6");
    m.hidden = 128;
    m.inner = 128;
    m.heads = 1;
    m.mlp = 128;
    m.geometry = .{};
    m.load_stage = .{};
    m.load_failed = false;
    m.retained_next = null;
    m.inputs = @splat(null);
    m.profile = true;
    m.spent = @splat(0);
    if (@hasField(h3.Model, "hosts")) m.hosts = .empty;
    if (@hasField(h3.Model, "tp")) m.tp = null;
    return m;
}
pub fn clean(m: *h3.Model) void {
    pending = false;
    wait_fail = false;
    m.deinit();
}
pub fn input() h3.Model.Inputs {
    return .{ .x = @intFromPtr(&x), .tables = @intFromPtr(&tables), .line = @intFromPtr(&line), .cos = 0, .sin = 0, .slot = @intFromPtr(&slot), .sizes = @intFromPtr(&sizes), .rows = 2, .tiles = 1, .lines = 1, .rot = 0, .prefix_tiles = 0, .keep = 0 };
}
pub var x: [256]u16 = @splat(0);
pub var tables: [768]f32 = @splat(0);
pub var line: [2]i32 = .{ 0, 0 };
pub var slot: [2]i32 = .{ 0, 1 };
pub var sizes: [1]i32 = .{2};
