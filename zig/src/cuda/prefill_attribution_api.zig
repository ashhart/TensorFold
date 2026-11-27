//! Attribution forwarding wrappers, installed once only when the prompt ledger is explicitly enabled.
const abi = @import("abi.zig");
const ledger = @import("prefill_ledger.zig");
const Ptr = ?*anyopaque;
const CPtr = ?*const anyopaque;
const Params = ?[*]?*anyopaque;
pub fn API(comptime Trace: type) type {
    return struct {
        var owner: *Trace = undefined;
        pub fn install(t: *Trace) void {
            owner = t;
            inline for (.{ "cuLaunchKernel", "cuLaunchKernelEx", "cuStreamBeginCapture_v2", "cuStreamEndCapture", "cuGraphLaunch", "cuStreamSynchronize", "cuCtxSynchronize", "cuEventSynchronize", "cuMemcpy2DAsync_v2", "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoD_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoH_v2", "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoD_v2", "cuMemsetD8Async", "cuMemsetD8_v2", "cuMemsetD32Async", "cuMemsetD32_v2" }) |name| @field(t.driver.api, name) = @field(@This(), name);
        }
        fn copy(stage: ledger.Stage, bytes: usize, stream: abi.Stream, async: bool) ?usize {
            if (!owner.recording()) return null;
            return owner.begin(.{ .stage = stage, .bytes = bytes, .operation = if (async) "async" else "blocking" }, stream, true);
        }
        fn done(ticket: ?usize, stream: abi.Stream, result: abi.Result) void {
            owner.work(stream, result);
            owner.end(ticket, result);
        }
        fn tap(f: abi.Function, rows: u32, stream: abi.Stream) ?usize {
            if (!owner.recording()) return null;
            for (owner.tap_functions) |function| if (function != null and function == f)
                return owner.begin(.{ .stage = .tap_upload, .rows = rows, .operation = "gather_taps" }, stream, true);
            return null;
        }
        fn cuLaunchKernel(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: abi.Stream, args: Params, extra: Params) callconv(.c) abi.Result {
            const ticket = tap(f, gx, stream);
            const result = owner.original.cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
            done(ticket, stream, result);
            return result;
        }
        fn cuLaunchKernelEx(config: *const abi.LaunchConfig, f: abi.Function, args: Params, extra: Params) callconv(.c) abi.Result {
            const result = owner.original.cuLaunchKernelEx(config, f, args, extra);
            owner.work(config.stream, result);
            return result;
        }
        fn cuStreamBeginCapture_v2(stream: abi.Stream, mode: abi.CaptureMode) callconv(.c) abi.Result {
            const result = owner.original.cuStreamBeginCapture_v2(stream, mode);
            if (owner.owns() and result == 0) {
                owner.graph(true);
                owner.capturing = true;
            }
            return result;
        }
        fn cuStreamEndCapture(stream: abi.Stream, graph: *abi.Graph) callconv(.c) abi.Result {
            const result = owner.original.cuStreamEndCapture(stream, graph);
            if (owner.owns()) {
                owner.capturing = false;
                owner.collect(false);
            }
            return result;
        }
        fn cuGraphLaunch(exec: abi.GraphExec, stream: abi.Stream) callconv(.c) abi.Result {
            const result = owner.original.cuGraphLaunch(exec, stream);
            if (result == 0) owner.graph(false);
            owner.work(stream, result);
            return result;
        }
        fn cuStreamSynchronize(stream: abi.Stream) callconv(.c) abi.Result {
            owner.beforeWait(@as(?abi.Stream, stream));
            const ticket = if (owner.recording()) owner.begin(.{ .stage = .host_sync, .operation = "cuStreamSynchronize" }, stream, false) else null;
            const result = owner.original.cuStreamSynchronize(stream);
            owner.end(ticket, result);
            if (result == 0) owner.collect(false);
            return result;
        }
        fn cuCtxSynchronize() callconv(.c) abi.Result {
            owner.beforeWait(null);
            const ticket = if (owner.recording()) owner.begin(.{ .stage = .host_sync, .operation = "cuCtxSynchronize" }, owner.stream, false) else null;
            const result = owner.original.cuCtxSynchronize();
            owner.end(ticket, result);
            if (result == 0) owner.collect(false);
            return result;
        }
        fn cuEventSynchronize(event: abi.Event) callconv(.c) abi.Result {
            const ticket = if (owner.recording()) owner.begin(.{ .stage = .host_sync, .operation = "cuEventSynchronize" }, owner.stream, false) else null;
            const result = owner.original.cuEventSynchronize(event);
            owner.end(ticket, result);
            if (result == 0) owner.collect(false);
            return result;
        }
        fn cuMemcpy2DAsync_v2(c: *const abi.Memcpy2D, stream: abi.Stream) callconv(.c) abi.Result {
            const ticket = copy(if (c.dst_type == abi.memory_type_host) .d2h else if (c.src_type == abi.memory_type_host) .h2d else .d2d, @import("std").math.mul(usize, c.width, c.height) catch 0, stream, true);
            const result = owner.original.cuMemcpy2DAsync_v2(c, stream);
            done(ticket, stream, result);
            return result;
        }
        fn cuMemcpyHtoDAsync_v2(dst: abi.DevicePtr, src: CPtr, bytes: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const ticket = copy(.h2d, bytes, stream, true);
            const result = owner.original.cuMemcpyHtoDAsync_v2(dst, src, bytes, stream);
            done(ticket, stream, result);
            return result;
        }
        fn cuMemcpyHtoD_v2(dst: abi.DevicePtr, src: CPtr, bytes: usize) callconv(.c) abi.Result {
            const stream: abi.Stream = null;
            const ticket = copy(.h2d, bytes, stream, false);
            const result = owner.original.cuMemcpyHtoD_v2(dst, src, bytes);
            done(ticket, stream, result);
            return result;
        }
        fn cuMemcpyDtoHAsync_v2(dst: Ptr, src: abi.DevicePtr, bytes: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const ticket = copy(.d2h, bytes, stream, true);
            const result = owner.original.cuMemcpyDtoHAsync_v2(dst, src, bytes, stream);
            done(ticket, stream, result);
            return result;
        }
        fn cuMemcpyDtoH_v2(dst: Ptr, src: abi.DevicePtr, bytes: usize) callconv(.c) abi.Result {
            const stream: abi.Stream = null;
            const ticket = copy(.d2h, bytes, stream, false);
            const result = owner.original.cuMemcpyDtoH_v2(dst, src, bytes);
            done(ticket, stream, result);
            return result;
        }
        fn cuMemcpyDtoDAsync_v2(dst: abi.DevicePtr, src: abi.DevicePtr, bytes: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const ticket = copy(.d2d, bytes, stream, true);
            const result = owner.original.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream);
            done(ticket, stream, result);
            return result;
        }
        fn cuMemcpyDtoD_v2(dst: abi.DevicePtr, src: abi.DevicePtr, bytes: usize) callconv(.c) abi.Result {
            const stream: abi.Stream = null;
            const ticket = copy(.d2d, bytes, stream, false);
            const result = owner.original.cuMemcpyDtoD_v2(dst, src, bytes);
            done(ticket, stream, result);
            return result;
        }
        fn cuMemsetD8Async(dst: abi.DevicePtr, value: u8, count: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const result = owner.original.cuMemsetD8Async(dst, value, count, stream);
            owner.work(stream, result);
            return result;
        }
        fn cuMemsetD8_v2(dst: abi.DevicePtr, value: u8, count: usize) callconv(.c) abi.Result {
            const result = owner.original.cuMemsetD8_v2(dst, value, count);
            owner.work(owner.stream, result);
            return result;
        }
        fn cuMemsetD32Async(dst: abi.DevicePtr, value: c_uint, count: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const result = owner.original.cuMemsetD32Async(dst, value, count, stream);
            owner.work(stream, result);
            return result;
        }
        fn cuMemsetD32_v2(dst: abi.DevicePtr, value: c_uint, count: usize) callconv(.c) abi.Result {
            const result = owner.original.cuMemsetD32_v2(dst, value, count);
            owner.work(owner.stream, result);
            return result;
        }
    };
}
