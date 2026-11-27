//! Driver-table wrappers installed only while the explicit attribution mode owns one CUDA engine.
const std = @import("std");
const abi = @import("abi.zig");
const meta = @import("round_attribution_meta.zig");
const model = @import("profile").attribution_model;
const Params = meta.Params;
const Ptr = ?*anyopaque;
const CPtr = ?*const anyopaque;
pub fn API(comptime Trace: type) type {
    return struct {
        pub var owner: ?*Trace = null;
        fn t() *Trace {
            return owner.?;
        }
        pub fn install(trace: *Trace) void {
            owner = trace;
            inline for (.{ "cuLaunchKernel", "cuLaunchKernelEx", "cuModuleGetFunction", "cuStreamBeginCapture_v2", "cuStreamEndCapture", "cuGraphInstantiateWithFlags", "cuGraphLaunch", "cuGraphDestroy", "cuGraphExecDestroy", "cuGraphExecUpdate_v2", "cuStreamSynchronize", "cuCtxSynchronize", "cuEventSynchronize", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoDAsync_v2", "cuMemcpy2DAsync_v2", "cuMemsetD8Async", "cuMemsetD32Async", "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuMemcpyDtoD_v2", "cuMemsetD8_v2", "cuMemsetD32_v2", "cuGraphExecKernelNodeSetParams_v2" }) |name|
                @field(trace.driver.api, name) = @field(@This(), name);
        }
        fn host(trace: *Trace, begin: u64, id: u64, wait: bool) void {
            if (trace.active) trace.addHost(.{ .correlation = id, .parent_scope = trace.scope_id, .phase = if (wait and trace.phase == .land) trace.readback else trace.phase, .kind = if (wait) .wait else .span, .begin_ns = begin, .end_ns = trace.now() });
        }
        fn kernel(f: abi.Function, grid: [3]u32, block: [3]u32, cluster: [3]u32, stream: abi.Stream, args: Params) ?usize {
            const trace = t();
            const desc = meta.kernel(trace.name(f), grid, block, cluster, args, trace.previous);
            trace.previous = desc;
            return trace.before(stream, desc);
        }
        fn cuLaunchKernel(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: abi.Stream, args: Params, extra: Params) callconv(.c) abi.Result {
            const trace = t();
            const event = kernel(f, .{ gx, gy, gz }, .{ bx, by, bz }, .{ 1, 1, 1 }, stream, args);
            const result = trace.original.cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
            trace.after(event, result);
            return result;
        }
        fn cuLaunchKernelEx(config: *const abi.LaunchConfig, f: abi.Function, args: Params, extra: Params) callconv(.c) abi.Result {
            const trace = t();
            var cluster: [3]u32 = .{ 1, 1, 1 };
            if (config.attrs) |attrs| for (attrs[0..config.num_attrs]) |a| if (a.id == .cluster_dimension) {
                cluster = .{ a.value.cluster_dim.x, a.value.cluster_dim.y, a.value.cluster_dim.z };
            };
            const event = kernel(f, .{ config.grid_x, config.grid_y, config.grid_z }, .{ config.block_x, config.block_y, config.block_z }, cluster, config.stream, args);
            const result = trace.original.cuLaunchKernelEx(config, f, args, extra);
            trace.after(event, result);
            return result;
        }
        fn cuModuleGetFunction(out: *abi.Function, module: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
            const trace = t();
            const result = trace.original.cuModuleGetFunction(out, module, name);
            if (result == 0) {
                if (trace.functions.get(@intFromPtr(out.*))) |cached|
                    if (std.mem.eql(u8, cached, std.mem.span(name))) return result;
                const owned = trace.permanent.allocator().dupe(u8, std.mem.span(name)) catch {
                    trace.fail("kernel name allocation");
                    return result;
                };
                trace.functions.put(trace.a, @intFromPtr(out.*), owned) catch trace.fail("kernel name map allocation");
            }
            return result;
        }
        fn cuStreamBeginCapture_v2(stream: abi.Stream, mode: abi.CaptureMode) callconv(.c) abi.Result {
            const trace = t();
            const result = trace.original.cuStreamBeginCapture_v2(stream, mode);
            if (result == 0) {
                if (trace.capturing) trace.fail("nested capture unsupported");
                trace.capturing = true;
                trace.capture.clearRetainingCapacity();
            }
            return result;
        }
        fn cuStreamEndCapture(stream: abi.Stream, graph: *abi.Graph) callconv(.c) abi.Result {
            const trace = t();
            const result = trace.original.cuStreamEndCapture(stream, graph);
            if (result == 0) {
                trace.captured(graph.*, true);
                return result;
            }
            var status: abi.CaptureStatus = .none;
            const status_result = trace.original.cuStreamIsCapturing(stream, &status);
            if (status_result != 0 or status != .none) {
                trace.quarantine(trace.capture.items, "capture end did not prove termination");
            } else {
                trace.captured(null, false);
            }
            return result;
        }
        fn cuGraphInstantiateWithFlags(out: *abi.GraphExec, graph: abi.Graph, flags: u64) callconv(.c) abi.Result {
            const trace = t();
            const result = trace.original.cuGraphInstantiateWithFlags(out, graph, flags);
            if (result == 0) if (trace.graphs.get(@intFromPtr(graph))) |g| {
                g.refs += 1;
                trace.execs.put(trace.a, @intFromPtr(out.*), g) catch {
                    trace.quarantine(g.pairs, "executable graph map allocation");
                    trace.release(g);
                };
            };
            return result;
        }
        fn cuGraphLaunch(exec: abi.GraphExec, stream: abi.Stream) callconv(.c) abi.Result {
            const trace = t();
            const id = trace.replay(exec) orelse trace.next();
            const begin = trace.now();
            if (trace.active and stream != trace.stream) trace.fail("graph replay on secondary stream");
            const result = trace.original.cuGraphLaunch(exec, stream);
            host(trace, begin, id, false);
            if (result != 0) trace.fail("graph replay failed");
            return result;
        }
        fn cuGraphDestroy(graph: abi.Graph) callconv(.c) abi.Result {
            const trace = t();
            const result = trace.original.cuGraphDestroy(graph);
            if (result == 0) if (trace.graphs.fetchRemove(@intFromPtr(graph))) |g| trace.release(g.value);
            return result;
        }
        fn cuGraphExecDestroy(exec: abi.GraphExec) callconv(.c) abi.Result {
            const trace = t();
            const result = trace.original.cuGraphExecDestroy(exec);
            if (result == 0) if (trace.execs.fetchRemove(@intFromPtr(exec))) |g| trace.release(g.value);
            return result;
        }
        fn cuGraphExecUpdate_v2(exec: abi.GraphExec, graph: abi.Graph, info: *abi.ExecUpdateResultInfo) callconv(.c) abi.Result {
            const trace = t();
            const result = trace.original.cuGraphExecUpdate_v2(exec, graph, info);
            if (result == 0 and info.result == .success) {
                if (trace.execs.fetchRemove(@intFromPtr(exec))) |g| trace.release(g.value);
                if (trace.graphs.get(@intFromPtr(graph))) |g| {
                    g.refs += 1;
                    trace.execs.put(trace.a, @intFromPtr(exec), g) catch {
                        trace.quarantine(g.pairs, "updated graph map allocation");
                        trace.release(g);
                    };
                } else trace.fail("untracked graph update");
            }
            return result;
        }
        fn cuStreamSynchronize(stream: abi.Stream) callconv(.c) abi.Result {
            const trace = t();
            const begin = trace.now();
            const id = trace.next();
            const result = trace.original.cuStreamSynchronize(stream);
            host(trace, begin, id, true);
            if (result == 0) trace.collect(stream);
            return result;
        }
        fn cuCtxSynchronize() callconv(.c) abi.Result {
            const trace = t();
            const begin = trace.now();
            const id = trace.next();
            const result = trace.original.cuCtxSynchronize();
            host(trace, begin, id, true);
            if (result == 0) trace.collect(null);
            return result;
        }
        fn cuEventSynchronize(event: abi.Event) callconv(.c) abi.Result {
            const trace = t();
            const begin = trace.now();
            const id = trace.next();
            const result = trace.original.cuEventSynchronize(event);
            host(trace, begin, id, true);
            return result;
        }
        fn copied(stream: abi.Stream, op: model.Op, bytes: usize) ?usize {
            return t().before(stream, .{ .symbol = @tagName(op), .op = op, .bytes = bytes });
        }
        fn cuMemcpyHtoDAsync_v2(dst: abi.DevicePtr, src: CPtr, bytes: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const event = copied(stream, .h2d, bytes);
            const result = t().original.cuMemcpyHtoDAsync_v2(dst, src, bytes, stream);
            t().after(event, result);
            return result;
        }
        fn cuMemcpyDtoHAsync_v2(dst: Ptr, src: abi.DevicePtr, bytes: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const event = copied(stream, .d2h, bytes);
            const result = t().original.cuMemcpyDtoHAsync_v2(dst, src, bytes, stream);
            t().after(event, result);
            return result;
        }
        fn cuMemcpyDtoDAsync_v2(dst: abi.DevicePtr, src: abi.DevicePtr, bytes: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const event = copied(stream, .d2d, bytes);
            const result = t().original.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream);
            t().after(event, result);
            return result;
        }
        fn cuMemcpy2DAsync_v2(copy: *const abi.Memcpy2D, stream: abi.Stream) callconv(.c) abi.Result {
            const bytes = std.math.mul(usize, copy.width, copy.height) catch {
                t().fail("copy size overflow");
                return t().original.cuMemcpy2DAsync_v2(copy, stream);
            };
            const event = copied(stream, if (copy.dst_type == abi.memory_type_host) .d2h else if (copy.src_type == abi.memory_type_host) .h2d else .d2d, bytes);
            const result = t().original.cuMemcpy2DAsync_v2(copy, stream);
            t().after(event, result);
            return result;
        }
        fn cuMemsetD8Async(dst: abi.DevicePtr, value: u8, bytes: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const event = copied(stream, .memset, bytes);
            const result = t().original.cuMemsetD8Async(dst, value, bytes, stream);
            t().after(event, result);
            return result;
        }
        fn cuMemsetD32Async(dst: abi.DevicePtr, value: c_uint, words: usize, stream: abi.Stream) callconv(.c) abi.Result {
            const bytes = std.math.mul(usize, words, 4) catch {
                t().fail("memset size overflow");
                return t().original.cuMemsetD32Async(dst, value, words, stream);
            };
            const event = copied(stream, .memset, bytes);
            const result = t().original.cuMemsetD32Async(dst, value, words, stream);
            t().after(event, result);
            return result;
        }

        fn unsupportedSync() void {
            if (t().active) t().fail("synchronous default-stream operation outside supported event timeline");
        }
        fn cuMemcpyHtoD_v2(dst: abi.DevicePtr, src: CPtr, bytes: usize) callconv(.c) abi.Result {
            unsupportedSync();
            return t().original.cuMemcpyHtoD_v2(dst, src, bytes);
        }
        fn cuMemcpyDtoH_v2(dst: Ptr, src: abi.DevicePtr, bytes: usize) callconv(.c) abi.Result {
            unsupportedSync();
            return t().original.cuMemcpyDtoH_v2(dst, src, bytes);
        }
        fn cuMemcpyDtoD_v2(dst: abi.DevicePtr, src: abi.DevicePtr, bytes: usize) callconv(.c) abi.Result {
            unsupportedSync();
            return t().original.cuMemcpyDtoD_v2(dst, src, bytes);
        }
        fn cuMemsetD8_v2(dst: abi.DevicePtr, value: u8, bytes: usize) callconv(.c) abi.Result {
            unsupportedSync();
            return t().original.cuMemsetD8_v2(dst, value, bytes);
        }
        fn cuMemsetD32_v2(dst: abi.DevicePtr, value: c_uint, words: usize) callconv(.c) abi.Result {
            unsupportedSync();
            return t().original.cuMemsetD32_v2(dst, value, words);
        }
        fn cuGraphExecKernelNodeSetParams_v2(exec: abi.GraphExec, node: abi.GraphNode, params: *const abi.KernelNodeParams) callconv(.c) abi.Result {
            const trace = t();
            // DFlash updates live gather/rope/ring arguments, whose captured grid and classification stay fixed.
            {
                const name = trace.name(params.func);
                var found = false;
                if (trace.execs.get(@intFromPtr(exec))) |g| for (g.pairs) |i| {
                    const d = trace.pairs.items[i].desc;
                    if (std.mem.eql(u8, name, d.symbol)) {
                        found = true;
                        if (d.op == .qmmf or d.op == .affine_g64 or !std.meta.eql(d.grid, .{ params.grid_x, params.grid_y, params.grid_z }) or !std.meta.eql(d.block, .{ params.block_x, params.block_y, params.block_z })) trace.fail("mutated graph projection or geometry needs fresh descriptors");
                    }
                };
                if (!found) trace.fail("untracked graph kernel mutation");
            }
            return trace.original.cuGraphExecKernelNodeSetParams_v2(exec, node, params);
        }
    };
}
