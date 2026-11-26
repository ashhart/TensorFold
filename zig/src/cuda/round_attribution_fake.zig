//! Recording driver for graph event ownership and the actual attribution wrappers.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
const trace = cuda.round_attribution;
const a = std.testing.allocator;
var stamps: [256]u64 = @splat(0);
var count: usize = 0;
var destroyed: usize = 0;
var tick: u64 = 0;
var capturing = false;
var captured: std.ArrayList(abi.Event) = .empty;
var launches: usize = 0;
var waits: usize = 0;
var bound_context: abi.Context = null;
fn bindContext(context: abi.Context) callconv(.c) abi.Result {
    bound_context = context;
    return 0;
}
fn create(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
    count += 1;
    out.* = @ptrFromInt(count);
    return 0;
}
fn destroy(_: abi.Event) callconv(.c) abi.Result {
    destroyed += 1;
    return 0;
}
fn record(event: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
    if (capturing) captured.append(a, event) catch return 2 else {
        tick += 1000;
        stamps[@intFromPtr(event)] = tick;
    }
    return 0;
}
fn external(event: abi.Event, stream: abi.Stream, flags: c_uint) callconv(.c) abi.Result {
    std.debug.assert(flags == abi.event_record_external);
    return record(event, stream);
}
fn elapsed(out: *f32, first: abi.Event, last: abi.Event) callconv(.c) abi.Result {
    out.* = @as(f32, @floatFromInt(stamps[@intFromPtr(last)] - stamps[@intFromPtr(first)])) / 1e6;
    return 0;
}
fn sync() callconv(.c) abi.Result {
    waits += 1;
    return 0;
}
fn streamSync(_: abi.Stream) callconv(.c) abi.Result {
    return sync();
}
fn start(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
    capturing = true;
    captured.clearRetainingCapacity();
    return 0;
}
fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
    capturing = false;
    out.* = @ptrFromInt(300);
    return 0;
}
fn instantiate(out: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
    out.* = @ptrFromInt(400);
    return 0;
}
fn replay(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
    for (captured.items) |e| {
        tick += 1000;
        stamps[@intFromPtr(e)] = tick;
    }
    return 0;
}
fn graphDestroy(_: abi.Graph) callconv(.c) abi.Result {
    return 0;
}
fn execDestroy(_: abi.GraphExec) callconv(.c) abi.Result {
    return 0;
}
fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: trace.meta.Params, _: trace.meta.Params) callconv(.c) abi.Result {
    if (!capturing) {
        launches += 1;
        tick += 1000;
    }
    return 0;
}
fn lookup(out: *abi.Function, _: abi.Module, _: [*:0]const u8) callconv(.c) abi.Result {
    out.* = @ptrFromInt(80);
    return 0;
}
fn version(out: *c_int) callconv(.c) abi.Result {
    out.* = 13000;
    return 0;
}
pub const Rig = struct {
    driver: *cuda.Driver,
    timing: *trace.Trace,
    pub fn init() !Rig {
        count = 0;
        destroyed = 0;
        tick = 0;
        capturing = false;
        launches = 0;
        waits = 0;
        const d = try a.create(cuda.Driver);
        errdefer a.destroy(d);
        d.* = undefined;
        d.func_name = null;
        d.api.cuModuleGetFunction = lookup;
        d.api.cuDriverGetVersion = version;
        d.api.cuCtxSetCurrent = bindContext;
        d.api.cuEventCreate = create;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuEventRecord = record;
        d.api.cuEventRecordWithFlags = external;
        d.api.cuEventElapsedTime = elapsed;
        d.api.cuCtxSynchronize = sync;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuStreamBeginCapture_v2 = start;
        d.api.cuStreamEndCapture = end;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphLaunch = replay;
        d.api.cuGraphDestroy = graphDestroy;
        d.api.cuGraphExecDestroy = execDestroy;
        d.api.cuLaunchKernel = launch;
        const t = try a.create(trace.Trace);
        errdefer a.destroy(t);
        t.* = .{ .a = a, .io = std.testing.io, .driver = d, .original = d.api, .permanent = .init(a), .round_arena = .init(a) };
        _ = create(&t.begin_event, 0);
        _ = create(&t.end_event, 0);
        t.context = @ptrFromInt(900);
        bound_context = t.context;
        trace.owner = t;
        trace.wrappers.install(t);
        return .{ .driver = d, .timing = t };
    }
    pub fn deinit(r: Rig) void {
        r.timing.deinit();
        a.destroy(r.driver);
        captured.deinit(a);
        captured = .empty;
        std.debug.assert(count == destroyed);
    }
    pub fn kernel(r: Rig) !void {
        try std.testing.expectEqual(@as(abi.Result, 0), r.driver.api.cuLaunchKernel(@ptrFromInt(80), 1, 1, 1, 32, 1, 1, 0, null, null, null));
    }
    pub fn graph(r: Rig) !abi.GraphExec {
        try std.testing.expectEqual(@as(abi.Result, 0), r.driver.api.cuStreamBeginCapture_v2(null, .thread_local));
        try r.kernel();
        var g: abi.Graph = null;
        _ = r.driver.api.cuStreamEndCapture(null, &g);
        var exec: abi.GraphExec = null;
        _ = r.driver.api.cuGraphInstantiateWithFlags(&exec, g, 0);
        _ = r.driver.api.cuGraphDestroy(g);
        return exec;
    }
    pub fn unbind(_: Rig) void {
        bound_context = null;
    }
    pub fn bound(r: Rig) bool {
        return bound_context == r.timing.context;
    }
    pub fn launchCount(_: Rig) usize {
        return launches;
    }
    pub fn waitCount(_: Rig) usize {
        return waits;
    }
};
