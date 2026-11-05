//! A CPU recording driver retains DMA pointers until explicit completion.
const std = @import("std");
const cuda = @import("cuda");
const t = std.testing;
pub const Fake = struct {
    pub var pins: [16]?[]align(4096) u8 = @splat(null);
    pub var pending: ?struct { bytes: []const u8, dst: usize } = null;
    pub var gpu: [8192]u8 = undefined;
    pub var last_source: usize = 0;
    pub var copies: usize = 0;
    pub var allocs: usize = 0;
    pub var frees: usize = 0;
    pub var early: usize = 0;
    pub var fail = false;
    pub var fail_record = false;
    pub var fail_alloc_at: usize = 0;
    fn alloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        allocs += 1;
        if (allocs == fail_alloc_at) return cuda.abi.error_not_ready;
        for (&pins) |*p| if (p.* == null) {
            p.* = t.allocator.alignedAlloc(u8, .fromByteUnits(4096), n) catch return 1;
            out.* = p.*.?.ptr;
            return 0;
        };
        return 1;
    }
    fn free(ptr: ?*anyopaque) callconv(.c) cuda.abi.Result {
        for (&pins) |*p| if (p.*) |b| {
            if (@intFromPtr(b.ptr) != @intFromPtr(ptr)) continue;
            if (pending) |work| if (@intFromPtr(work.bytes.ptr) >= @intFromPtr(b.ptr) and @intFromPtr(work.bytes.ptr) < @intFromPtr(b.ptr) + b.len) {
                early += 1;
            };
            t.allocator.free(b);
            p.* = null;
            frees += 1;
            return 0;
        };
        return 1;
    }
    fn upload(dst: u64, ptr: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (pending != null) return 1;
        last_source = @intFromPtr(ptr);
        pending = .{ .bytes = @as([*]const u8, @ptrCast(ptr.?))[0..n], .dst = @intCast(dst - 0x1000) };
        copies += 1;
        return 0;
    }
    fn syncCopy(dst: u64, ptr: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        const code = upload(dst, ptr, n, null);
        return if (code != 0) code else sync();
    }
    fn sync() callconv(.c) cuda.abi.Result {
        if (fail) return cuda.abi.error_not_ready;
        if (pending) |p| @memcpy(gpu[p.dst..][0..p.bytes.len], p.bytes);
        pending = null;
        return 0;
    }
    fn streamSync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return sync();
    }
    fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (fail_record) cuda.abi.error_not_ready else 0;
    }
    fn eventSync(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return sync();
    }
    fn drop(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    pub fn driver() cuda.Driver {
        t.expect(pending == null) catch unreachable;
        for (pins) |p| t.expect(p == null) catch unreachable;
        allocs = 0;
        frees = 0;
        early = 0;
        copies = 0;
        fail = false;
        fail_record = false;
        fail_alloc_at = 0;
        @memset(&gpu, 0);
        var d: cuda.Driver = .{ .api = undefined, .lib = undefined };
        d.api.cuMemHostAlloc = alloc;
        d.api.cuMemFreeHost = free;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuMemcpyHtoD_v2 = syncCopy;
        d.api.cuCtxSynchronize = sync;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuEventCreate = create;
        d.api.cuEventRecord = record;
        d.api.cuEventSynchronize = eventSync;
        d.api.cuEventQuery = eventSync;
        d.api.cuEventDestroy_v2 = drop;
        return d;
    }
};
