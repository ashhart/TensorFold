//! A fake queue observes leases at completion proof without launching device work.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const t = std.testing;
pub const Device = struct {
    pub const stream: abi.Stream = @ptrFromInt(17);
    pub var observe: ?*const fn () bool = null;
    pub var expected: usize = 0;
    pub var early = false;
    pub var proofs: usize = 0;
    pub var pending = false;
    pub var fail_wait = false;
    pub var fail_fill = false;
    pub var freed_pending: usize = 0;
    pub var fail_wait_in: usize = 0; // > 0: the stream wait this many waits on fails, once
    pub var hosts: usize = 0; // pinned buffers held
    var next: u64 = 0x10000;
    fn alloc(out: *u64, len: usize) callconv(.c) c_int {
        out.* = next;
        next += len + 256;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        freed_pending += @intFromBool(pending);
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize, s: abi.Stream) callconv(.c) c_int {
        _ = s;
        pending = true;
        if (fail_fill) {
            fail_fill = false;
            fail_wait = true;
            return abi.error_not_ready;
        }
        return 0;
    }
    fn legacyFill(dst: u64, v: u8, n: usize) callconv(.c) c_int {
        return fill(dst, v, n, null);
    }
    fn proof() void {
        if (pending) {
            if (observe) |held| early = early or !held();
            proofs += 1;
        }
        pending = false;
    }
    fn wait(_: abi.Stream) callconv(.c) c_int {
        if (fail_wait) return abi.error_not_ready;
        if (fail_wait_in > 0) {
            fail_wait_in -= 1;
            if (fail_wait_in == 0) return abi.error_not_ready;
        }
        proof();
        return 0;
    }
    fn context() callconv(.c) c_int {
        proof();
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize, _: abi.Stream) callconv(.c) c_int {
        pending = true;
        return 0;
    }
    fn copyOut(_: ?*anyopaque, _: u64, _: usize, _: abi.Stream) callconv(.c) c_int {
        pending = true;
        return 0;
    }
    // each buffer's length in the 16 bytes before it
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) c_int {
        const all = t.allocator.alignedAlloc(u8, .@"16", len + 16) catch return abi.error_not_ready;
        std.mem.writeInt(usize, all[0..@sizeOf(usize)], len, .little);
        out.* = all.ptr + 16;
        hosts += 1;
        return 0;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) c_int {
        freed_pending += @intFromBool(pending);
        const at: [*]align(16) u8 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(ptr.?)) - 16));
        t.allocator.free(at[0 .. std.mem.readInt(usize, at[0..@sizeOf(usize)], .little) + 16]);
        hosts -= 1;
        return 0;
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn eventRecord(_: abi.Event, _: abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn eventQuery(_: abi.Event) callconv(.c) c_int {
        return if (pending) abi.error_not_ready else 0;
    }
    fn eventDrop(_: abi.Event) callconv(.c) c_int {
        return 0;
    }
    pub fn driver() Driver {
        observe = null;
        expected = 0;
        early = false;
        proofs = 0;
        pending = false;
        fail_wait = false;
        fail_fill = false;
        freed_pending = 0;
        fail_wait_in = 0;
        hosts = 0;
        next = 0x10000;
        var d: Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = fill;
        d.api.cuMemsetD8_v2 = legacyFill;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuMemcpyDtoHAsync_v2 = copyOut;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventQuery;
        d.api.cuEventDestroy_v2 = eventDrop;
        return d;
    }
};
