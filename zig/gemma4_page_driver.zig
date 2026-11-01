//! Inert CUDA allocations and ordered copies; no GPU or kernel arithmetic is executed.
const std = @import("std");
const cuda = @import("cuda");
const a = std.testing.allocator;
const Entry = struct { ptr: u64 = 0, len: usize = 0 };
var blocks: [512]Entry = @splat(.{});
pub var skip_copy = false;
pub var allocations: usize = 0;
pub var graphs_dropped: usize = 0;

fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
    const mem = a.alignedAlloc(u8, .fromByteUnits(256), n) catch return 2;
    @memset(mem, 0);
    for (&blocks) |*b| if (b.ptr == 0) {
        b.* = .{ .ptr = @intFromPtr(mem.ptr), .len = n };
        allocations += 1;
        out.* = b.ptr;
        return 0;
    };
    a.free(mem);
    return 2;
}
fn free(ptr: u64) callconv(.c) cuda.abi.Result {
    for (&blocks) |*b| if (b.ptr == ptr) {
        const mem: [*]align(256) u8 = @ptrFromInt(ptr);
        a.free(mem[0..b.len]);
        b.* = .{};
        return 0;
    };
    return 1;
}
fn h2d(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    const out: [*]u8 = @ptrFromInt(dst);
    const in: [*]const u8 = @ptrCast(src.?);
    @memcpy(out[0..n], in[0..n]);
    return 0;
}
fn d2d(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    if (!skip_copy) {
        const out: [*]u8 = @ptrFromInt(dst);
        const in: [*]const u8 = @ptrFromInt(src);
        @memcpy(out[0..n], in[0..n]);
    }
    return 0;
}
fn fill(dst: u64, value: u8, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    const out: [*]u8 = @ptrFromInt(dst);
    @memset(out[0..n], value);
    return 0;
}
fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    return 0;
}
fn contextSync() callconv(.c) cuda.abi.Result {
    return 0;
}
fn dropGraph(_: cuda.abi.GraphExec) callconv(.c) cuda.abi.Result {
    graphs_dropped += 1;
    return 0;
}

pub fn driver() cuda.Driver {
    var api: cuda.abi.Api = undefined;
    api.cuMemAlloc_v2 = alloc;
    api.cuMemFree_v2 = free;
    api.cuMemcpyHtoDAsync_v2 = h2d;
    api.cuMemcpyDtoDAsync_v2 = d2d;
    api.cuMemsetD8Async = fill;
    api.cuStreamSynchronize = sync;
    api.cuCtxSynchronize = contextSync;
    api.cuGraphExecDestroy = dropGraph;
    return .{ .lib = undefined, .api = api };
}

pub fn empty() bool {
    for (blocks) |b| if (b.ptr != 0) return false;
    return true;
}
