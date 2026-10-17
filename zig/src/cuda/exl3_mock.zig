//! Host-only driver fault injection for EXL3 ownership tests.
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
pub var frees: usize = 0;
pub var allocations: usize = 0;
pub var fail: bool = false;
pub var copied: [512]u8 = @splat(0);
pub var copies: [32][512]u8 = @splat(@splat(0));
pub var copy_count: usize = 0;
pub var copy_len: usize = 0;
pub var next: u64 = 256;
pub var fail_copy: usize = 0; // the copy (1 for the first) whose waits fail, 0 for none
pub var fail_waits: usize = 0; // how many of its waits fail before one completes
pub var pending: bool = false; // a copy is accepted and no wait has completed it
pub var bad_frees: usize = 0; // device frees while a copy was pending
pub fn init() Driver {
    copy_count = 0;
    frees = 0;
    allocations = 0;
    fail = false;
    next = 256;
    fail_copy = 0;
    fail_waits = 0;
    pending = false;
    bad_frees = 0;
    var d: Driver = undefined;
    d.api.cuMemAlloc_v2 = alloc;
    d.api.cuMemFree_v2 = free;
    d.api.cuMemcpyHtoD_v2 = copy;
    d.api.cuStreamSynchronize = sync;
    d.api.cuCtxSynchronize = context;
    d.api.cuGetErrorName = name;
    d.api.cuGetErrorString = name;
    return d;
}
fn alloc(p: *abi.DevicePtr, _: usize) callconv(.c) abi.Result {
    if (fail) return abi.error_not_ready;
    p.* = next;
    next += 256;
    allocations += 1;
    return 0;
}
fn free(_: abi.DevicePtr) callconv(.c) abi.Result {
    if (pending) bad_frees += 1;
    frees += 1;
    return 0;
}
fn copy(_: abi.DevicePtr, src: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
    copy_len = n;
    const count = @min(n, copied.len);
    @memcpy(copied[0..count], @as([*]const u8, @ptrCast(src.?))[0..count]);
    if (copy_count < copies.len) copies[copy_count] = copied;
    copy_count += 1;
    pending = true;
    return 0;
}
fn sync(_: abi.Stream) callconv(.c) abi.Result {
    if (pending and copy_count == fail_copy and fail_waits > 0) {
        fail_waits -= 1;
        return abi.error_not_ready;
    }
    pending = false;
    return 0;
}
fn context() callconv(.c) abi.Result {
    pending = false;
    return 0;
}
fn name(_: abi.Result, out: *?[*:0]const u8) callconv(.c) abi.Result {
    out.* = "mock";
    return 0;
}
