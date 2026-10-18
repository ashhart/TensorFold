//! Fault-injected driver for ownership tests. It allocates no device memory and executes no GPU kernel.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
pub var modules: usize = 0;
pub var unloaded: usize = 0;
pub var allocated: usize = 0;
pub var lengths: [1024]usize = @splat(0);
pub var fail_event = false;
pub var freed: usize = 0;
pub var double_free: usize = 0;
pub var early_free: usize = 0;
pub var pending: bool = false;
pub var fail_alloc = false;
pub var fail_copy = false;
pub var fail_sync = false;
pub var fail_record = false;
pub var live: [1024]bool = @splat(false);
pub var memory: [1024][1024]u8 = @splat(@splat(0));
pub var pinned: ?[]align(16) u8 = null;
pub fn init() cuda.Driver {
    modules = 0;
    fail_event = false;
    unloaded = 0;
    allocated = 0;
    freed = 0;
    double_free = 0;
    early_free = 0;
    pending = false;
    fail_alloc = false;
    fail_copy = false;
    fail_sync = false;
    fail_record = false;
    @memset(&live, false);
    var d: cuda.Driver = undefined;
    d.api.cuModuleLoadData = moduleLoad;
    d.api.cuModuleUnload = moduleUnload;
    d.api.cuModuleGetFunction = moduleFunction;
    d.api.cuMemAlloc_v2 = alloc;
    d.api.cuMemsetD8_v2 = memset8;
    d.api.cuMemFree_v2 = free;
    d.api.cuMemcpyHtoD_v2 = copy;
    d.api.cuMemcpyDtoH_v2 = download;
    d.api.cuMemcpyHtoDAsync_v2 = copyAsync;
    d.api.cuMemcpyDtoDAsync_v2 = deviceCopy;
    d.api.cuCtxSynchronize = ctxSync;
    d.api.cuStreamSynchronize = streamSync;
    d.api.cuMemHostAlloc = hostAlloc;
    d.api.cuMemFreeHost = hostFree;
    d.api.cuEventCreate = eventCreate;
    d.api.cuEventRecord = eventRecord;
    d.api.cuEventSynchronize = eventSync;
    d.api.cuEventQuery = query;
    d.api.cuEventDestroy_v2 = eventDestroy;
    return d;
}
fn alloc(out: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    if (fail_alloc) return abi.error_not_ready;
    allocated += 1;
    live[allocated] = true;
    lengths[allocated] = n;
    out.* = allocated * 256;
    return 0;
}
fn free(ptr: abi.DevicePtr) callconv(.c) abi.Result {
    const id = ptr / 256;
    if (!live[id]) double_free += 1 else {
        live[id] = false;
        freed += 1;
    }
    if (pending) early_free += 1;
    return 0;
}
fn copy(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
    if (n <= 1024) @memcpy(memory[dst / 256][0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
    pending = true;
    return if (fail_copy) abi.error_not_ready else 0;
}
fn copyAsync(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize, _: abi.Stream) callconv(.c) abi.Result {
    return copy(dst, src, n);
}
fn ctxSync() callconv(.c) abi.Result {
    if (fail_sync) return abi.error_not_ready;
    pending = false;
    return 0;
}
fn streamSync(_: abi.Stream) callconv(.c) abi.Result {
    return ctxSync();
}
fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) abi.Result {
    const bytes = std.testing.allocator.alignedAlloc(u8, .@"16", n + 16) catch return abi.error_not_ready;
    std.mem.writeInt(usize, bytes[0..@sizeOf(usize)], n + 16, .little);
    pinned = @alignCast(bytes[16..]);
    out.* = bytes.ptr + 16;
    return 0;
}
fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
    if (pending) early_free += 1;
    const base: [*]align(16) u8 = @alignCast(@as([*]u8, @ptrCast(ptr.?)) - 16);
    const n = std.mem.readInt(usize, base[0..@sizeOf(usize)], .little);
    if (pinned) |last| if (last.ptr == @as([*]u8, @ptrCast(ptr.?))) {
        pinned = null;
    };
    std.testing.allocator.free(base[0..n]);
    return 0;
}
fn query(_: abi.Event) callconv(.c) abi.Result {
    return abi.error_not_ready;
}
fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
    if (fail_event) return abi.error_not_ready;
    out.* = @ptrFromInt(1);
    return 0;
}
fn eventRecord(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
    return if (fail_record) abi.error_not_ready else 0;
}
fn eventSync(_: abi.Event) callconv(.c) abi.Result {
    return ctxSync();
}
fn eventDestroy(_: abi.Event) callconv(.c) abi.Result {
    return 0;
}

fn download(dst: ?*anyopaque, src: abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    if (n > 1024) return abi.error_not_ready;
    @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], memory[src / 256][0..n]);
    return 0;
}

fn moduleLoad(out: *abi.Module, _: ?*const anyopaque) callconv(.c) abi.Result {
    modules += 1;
    out.* = @ptrFromInt(modules);
    return 0;
}
fn moduleUnload(_: abi.Module) callconv(.c) abi.Result {
    unloaded += 1;
    return 0;
}
fn moduleFunction(out: *abi.Function, _: abi.Module, _: [*:0]const u8) callconv(.c) abi.Result {
    out.* = @ptrFromInt(1);
    return 0;
}

fn memset8(_: abi.DevicePtr, _: u8, _: usize) callconv(.c) abi.Result {
    pending = true;
    return 0;
}

fn deviceCopy(_: abi.DevicePtr, _: abi.DevicePtr, _: usize, _: abi.Stream) callconv(.c) abi.Result {
    pending = true;
    return if (fail_copy) abi.error_not_ready else 0;
}
