const std = @import("std");
const t = std.testing;
const source = @import("copy_source.zig");
const abi = @import("abi.zig");
pub const Fake = struct {
    pub var fail = true;
    pub var graph_count: usize = 0;
    pub var exec_count: usize = 0;
    pub var replays: usize = 0;
    pub var fail_begin = false;
    pub var fail_end = false;
    var in_capture = false;
    var captured: ?[]const u8 = null;
    var copies: usize = 0;
    pub var fail_event = false;
    pub var fail_record = false;
    var interleave = false;
    var driver: ?*@import("driver.zig").Driver = null;
    pub var early: usize = 0;
    var destroyed: usize = 0;
    var pins: [32]?[]align(16) u8 = @splat(null);
    var host: ?[]align(16) u8 = null;
    pub var pending: ?[]const u8 = null;
    pub var result: [16]u8 = undefined;
    fn copy(_: u64, ptr: ?*const anyopaque, len: usize) callconv(.c) abi.Result {
        copies += 1;
        pending = @as([*]const u8, @ptrCast(ptr.?))[0..len];
        return 0;
    }
    fn streamSync(_: abi.Stream) callconv(.c) abi.Result {
        return sync();
    }
    fn sync() callconv(.c) abi.Result {
        if (interleave) {
            interleave = false;
            fail = true;
            source.synchronousCopy(driver.?, 512, &@as([16]u8, @splat(0x72)), t.allocator) catch {};
            fail = false;
            return 0;
        }
        if (fail) return abi.error_not_ready;
        if (pending) |bytes| @memcpy(result[0..@min(bytes.len, result.len)], bytes[0..@min(bytes.len, result.len)]);
        pending = null;
        return 0;
    }
    fn asyncCopy(dst: u64, ptr: ?*const anyopaque, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (in_capture) {
            captured = @as([*]const u8, @ptrCast(ptr.?))[0..len];
            return 0;
        }
        return copy(dst, ptr, len);
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        host = t.allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        for (&pins) |*pin| if (pin.* == null) {
            pin.* = host;
            break;
        };
        out.* = host.?.ptr;
        return 0;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
        if (graph_count > 0 or exec_count > 0) early += 1;
        for (&pins) |*pin| if (pin.*) |bytes| {
            if (@intFromPtr(ptr) != @intFromPtr(bytes.ptr)) continue;
            if (pending) |p| if (@intFromPtr(p.ptr) == @intFromPtr(ptr)) {
                early += 1;
            };
            t.allocator.free(bytes);
            pin.* = null;
            host = null;
            return 0;
        };
        unreachable;
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        if (fail_event) return abi.error_not_ready;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return if (fail_record) abi.error_not_ready else 0;
    }
    fn eventSync(_: abi.Event) callconv(.c) abi.Result {
        return sync();
    }
    fn eventDestroy(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn streamDestroy(_: abi.Stream) callconv(.c) abi.Result {
        destroyed += 1;
        return 0;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        if (fail_begin) return abi.error_not_ready;
        in_capture = true;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        if (fail_end) return abi.error_not_ready;
        in_capture = false;
        graph_count += 1;
        out.* = @ptrFromInt(graph_count);
        return 0;
    }
    fn instantiate(out: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        exec_count += 1;
        out.* = @ptrFromInt(exec_count);
        return 0;
    }
    fn graphDestroy(_: abi.Graph) callconv(.c) abi.Result {
        graph_count -= 1;
        return 0;
    }
    fn execDestroy(_: abi.GraphExec) callconv(.c) abi.Result {
        exec_count -= 1;
        return 0;
    }
    fn launch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        replays += 1;
        pending = captured;
        return 0;
    }
    pub fn init() @import("driver.zig").Driver {
        fail_begin = false;
        fail_end = false;
        graph_count = 0;
        exec_count = 0;
        replays = 0;
        in_capture = false;
        captured = null;
        fail = true;
        fail_event = false;
        fail_record = false;
        interleave = false;
        pending = null;
        early = 0;
        destroyed = 0;
        var d: @import("driver.zig").Driver = undefined;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuMemcpyHtoDAsync_v2 = asyncCopy;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuCtxSynchronize = sync;
        d.api.cuStreamDestroy_v2 = streamDestroy;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphDestroy = graphDestroy;
        d.api.cuGraphExecDestroy = execDestroy;
        d.api.cuGraphLaunch = launch;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = record;
        d.api.cuEventSynchronize = eventSync;
        d.api.cuEventQuery = eventSync;
        d.api.cuEventDestroy_v2 = eventDestroy;
        return d;
    }
};
