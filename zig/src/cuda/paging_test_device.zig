//! A fake queue observes leases at completion proof without launching device work.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const t = std.testing;
pub const Device = struct {
    pub const stream: abi.Stream = @ptrFromInt(17);
    pub const Copy = struct { dst: u64, src: u64, len: usize };
    pub fn bytes(ptr: u64, len: usize) []u8 {
        return @as([*]u8, @ptrFromInt(ptr))[0..len];
    }
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
    pub var devices: usize = 0;
    pub var fail_alloc = false;
    pub var fail_dtod = false;
    pub var dtods: usize = 0;
    pub var last_dtod: Copy = .{ .dst = 0, .src = 0, .len = 0 };
    fn alloc(out: *u64, len: usize) callconv(.c) c_int {
        if (fail_alloc) {
            fail_alloc = false;
            return abi.error_not_ready;
        }
        const all = t.allocator.alignedAlloc(u8, .@"16", len + 16) catch return abi.error_not_ready;
        std.mem.writeInt(usize, all[0..@sizeOf(usize)], len, .little);
        out.* = @intFromPtr(all.ptr + 16);
        devices += 1;
        return 0;
    }
    fn free(ptr: u64) callconv(.c) c_int {
        freed_pending += @intFromBool(pending);
        const at: [*]align(16) u8 = @ptrCast(@alignCast(@as([*]u8, @ptrFromInt(ptr)) - 16));
        t.allocator.free(at[0 .. std.mem.readInt(usize, at[0..@sizeOf(usize)], .little) + 16]);
        devices -= 1;
        return 0;
    }
    fn fill(dst: u64, v: u8, n: usize, s: abi.Stream) callconv(.c) c_int {
        _ = s;
        pending = true;
        if (fail_fill) {
            fail_fill = false;
            fail_wait = true;
            return abi.error_not_ready;
        }
        @memset(bytes(dst, n), v);
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
    fn copy(dst: u64, src: ?*const anyopaque, len: usize, _: abi.Stream) callconv(.c) c_int {
        pending = true;
        if (len > 0) @memcpy(bytes(dst, len), @as([*]const u8, @ptrCast(src.?))[0..len]);
        return 0;
    }
    fn copyOut(dst: ?*anyopaque, src: u64, len: usize, _: abi.Stream) callconv(.c) c_int {
        pending = true;
        if (len > 0) @memcpy(@as([*]u8, @ptrCast(dst.?))[0..len], bytes(src, len));
        return 0;
    }
    fn dtod(dst: u64, src: u64, len: usize, _: abi.Stream) callconv(.c) c_int {
        pending = true;
        dtods += 1;
        last_dtod = .{ .dst = dst, .src = src, .len = len };
        if (fail_dtod) {
            fail_dtod = false;
            return abi.error_not_ready;
        }
        @memcpy(bytes(dst, len), bytes(src, len));
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
        devices = 0;
        fail_alloc = false;
        fail_dtod = false;
        dtods = 0;
        last_dtod = .{ .dst = 0, .src = 0, .len = 0 };
        var d: Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = fill;
        d.api.cuMemsetD8_v2 = legacyFill;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuMemcpyDtoHAsync_v2 = copyOut;
        d.api.cuMemcpyDtoDAsync_v2 = dtod;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventQuery;
        d.api.cuEventDestroy_v2 = eventDrop;
        return d;
    }
};
