//! Addressable host buffers exercise the real CUDA snapshot copies without a GPU or model.
const std = @import("std");
const cuda = @import("cuda");
const t = std.testing;
const Entry = struct { bytes: []align(256) u8, pinned: bool };
pub const Device = struct {
    var buffers: [64]?Entry = @splat(null);
    pub var copies: usize = 0;
    pub var fail_copy_in: usize = 0;
    pub var fail_wait = false;
    pub var fail_wait_on_copy_error = false;
    pub var pending = false;
    pub var early_frees: usize = 0;
    pub var guards: usize = 0;

    pub fn live() usize {
        var n: usize = 0;
        for (buffers) |b| n += @intFromBool(b != null);
        return n;
    }
    fn allocate(n: usize, pinned: bool) !u64 {
        const b = try t.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(256), n + 512);
        @memset(b, 0xa5);
        for (&buffers) |*entry| if (entry.* == null) {
            entry.* = .{ .bytes = b, .pinned = pinned };
            return @intFromPtr(b.ptr + 256);
        };
        t.allocator.free(b);
        return error.OutOfMemory;
    }
    fn dispose(ptr: u64) cuda.abi.Result {
        early_frees += @intFromBool(pending);
        for (&buffers) |*entry| if (entry.*) |b| {
            if (ptr != @intFromPtr(b.bytes.ptr + 256)) continue;
            guards += @intFromBool(!std.mem.allEqual(u8, b.bytes[0..256], 0xa5));
            guards += @intFromBool(!std.mem.allEqual(u8, b.bytes[b.bytes.len - 256 ..], 0xa5));
            t.allocator.free(b.bytes);
            entry.* = null;
            return 0;
        };
        return 1;
    }
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = allocate(n, false) catch return 2;
        return 0;
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        return dispose(ptr);
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(allocate(n, true) catch return 2);
        return 0;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) cuda.abi.Result {
        return dispose(@intFromPtr(ptr));
    }
    fn valid(ptr: u64, n: usize) bool {
        for (buffers) |entry| if (entry) |b| {
            const start = @intFromPtr(b.bytes.ptr + 256);
            if (ptr >= start and ptr - start <= b.bytes.len - 512 and n <= b.bytes.len - 512 - (ptr - start)) return true;
        };
        return false;
    }
    fn copy(dst: u64, src: u64, n: usize, check_src: bool, check_dst: bool) cuda.abi.Result {
        if (fail_copy_in > 0) {
            fail_copy_in -= 1;
            if (fail_copy_in == 0) {
                if (fail_wait_on_copy_error) fail_wait = true;
                return cuda.abi.error_not_ready;
            }
        }
        if ((check_dst and !valid(dst, n)) or (check_src and !valid(src, n))) return 1;
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        copies += 1;
        pending = true;
        return 0;
    }
    fn dtod(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return copy(dst, src, n, true, true);
    }
    fn dtoh(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return copy(@intFromPtr(dst), src, n, true, false);
    }
    fn htod(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return copy(dst, @intFromPtr(src), n, false, true);
    }
    fn fill(dst: u64, v: u8, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (!valid(dst, n)) return 1;
        @memset(@as([*]u8, @ptrFromInt(dst))[0..n], v);
        pending = true;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_wait) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) cuda.abi.Result {
        return sync(null);
    }
    fn event(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn query(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return if (pending or fail_wait) cuda.abi.error_not_ready else 0;
    }
    fn wait(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return sync(null);
    }
    fn drop(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    pub fn driver() cuda.Driver {
        std.debug.assert(live() == 0);
        copies = 0;
        fail_copy_in = 0;
        fail_wait = false;
        fail_wait_on_copy_error = false;
        pending = false;
        early_frees = 0;
        guards = 0;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemcpyDtoDAsync_v2 = dtod;
        d.api.cuMemcpyDtoHAsync_v2 = dtoh;
        d.api.cuMemcpyHtoDAsync_v2 = htod;
        d.api.cuMemsetD8Async = fill;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = context;
        d.api.cuEventCreate = event;
        d.api.cuEventRecord = record;
        d.api.cuEventQuery = query;
        d.api.cuEventSynchronize = wait;
        d.api.cuEventDestroy_v2 = drop;
        return d;
    }
};
