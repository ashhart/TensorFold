//! REV295 graph behavior without CUDA hardware: capture, refresh, replay, and teardown.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const land = @import("cuda_land.zig");
const keep = @import("cuda_keep_capture.zig");
const Seg = @import("cuda_cache.zig").Seg;
const Path = @import("cuda_cache.zig").Path;

const Graph = struct {
    var capturing = false;
    var records: usize = 0;
    var instantiates: usize = 0;
    var updates: usize = 0;
    var launches: usize = 0;
    var destroyed: usize = 0;
    var verdict: abi.ExecUpdateResult = .success;

    fn reset() void {
        records = 0;
        instantiates = 0;
        updates = 0;
        launches = 0;
        destroyed = 0;
        verdict = .success;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        capturing = true;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        capturing = false;
        records += 1;
        out.* = @ptrFromInt(records);
        return 0;
    }
    fn status(_: abi.Stream, out: *abi.CaptureStatus) callconv(.c) abi.Result {
        out.* = if (capturing) .active else .none;
        return 0;
    }
    fn instantiate(out: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        instantiates += 1;
        out.* = @ptrFromInt(instantiates);
        return 0;
    }
    fn update(_: abi.GraphExec, _: abi.Graph, info: *abi.ExecUpdateResultInfo) callconv(.c) abi.Result {
        updates += 1;
        info.result = verdict;
        return if (verdict == .success) 0 else 910;
    }
    fn graphDestroy(_: abi.Graph) callconv(.c) abi.Result {
        return 0;
    }
    fn launch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        launches += 1;
        return 0;
    }
    fn execDestroy(_: abi.GraphExec) callconv(.c) abi.Result {
        destroyed += 1;
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuStreamIsCapturing = status;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphExecUpdate_v2 = update;
        d.api.cuGraphDestroy = graphDestroy;
        d.api.cuGraphLaunch = launch;
        d.api.cuGraphExecDestroy = execDestroy;
        return d;
    }
};

const Fn = enum { argmax_slices, argmax_merge, apply, conv };
const Draw = struct { rule: ?u8 };
const Draws = struct {
    calls: usize = 0,
    pub fn drawRows(d: *Draws, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {
        d.calls += 1;
    }
};
const LandEngine = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    lever7: @import("cuda_lever7.zig").Options,
    land_graphs: land.Graphs = .empty,
    logits: u64 = 0x1000,
    tables: cuda.DeviceBuffer,
    amax: u64 = 0x2000,
    picks: u64 = 0x3000,
    g: struct { vocab: usize } = .{ .vocab = 128 },
    fns: struct { argmax_slices: Fn, argmax_merge: Fn } = .{ .argmax_slices = .argmax_slices, .argmax_merge = .argmax_merge },
    draws: Draws = .{},
    kernel_calls: usize = 0,

    pub fn run(e: *LandEngine, _: Fn, _: [3]usize, _: [3]usize, _: anytype) !void {
        e.kernel_calls += 1;
    }
};

test "REV295 land replays exact keys, recaptures moved pointers, and leaves mixed runs eager" {
    Graph.reset();
    var d = Graph.driver();
    const previous = cuda.copy_source.current();
    cuda.copy_source.setCurrent(@ptrFromInt(0x3518));
    defer cuda.copy_source.setCurrent(previous);
    var e: LandEngine = .{
        .gpa = t.allocator,
        .stream = .{ .d = &d, .handle = @ptrFromInt(1) },
        .lever7 = .{ .capture = true },
        .tables = .{ .d = &d, .ptr = 0x4000, .len = 1024, .borrowed = true },
    };
    defer land.deinit(&e);
    var pk = [_]Draw{ .{ .rule = null }, .{ .rule = null } };
    try land.run(&e, &pk, 2);
    try land.run(&e, &pk, 2);
    try t.expectEqual([_]usize{ 1, 1, 2, 2 }, .{ Graph.records, Graph.instantiates, Graph.launches, e.kernel_calls });
    e.logits += 0x1000;
    try land.run(&e, &pk, 2);
    try t.expectEqual([_]usize{ 2, 2, 3, 4 }, .{ Graph.records, Graph.instantiates, Graph.launches, e.kernel_calls });
    pk[1].rule = 1;
    try land.run(&e, &pk, 2);
    try t.expectEqual([_]usize{ 2, 6, 1 }, .{ Graph.records, e.kernel_calls, e.draws.calls });
}

const KeepGeometry = struct {
    v_heads: usize = 2,
    pub fn deltas(_: KeepGeometry) usize {
        return 1;
    }
    pub fn convBytes(_: KeepGeometry) usize {
        return 512;
    }
};
const KeepEngine = struct {
    stream: cuda.Stream,
    g: KeepGeometry = .{},
    geo: c_int = 0,
    keeps: struct { apply: Fn, conv: Fn } = .{ .apply = .apply, .conv = .conv },
    kernel_calls: usize = 0,
    last_seg: ?Seg = null,
    last_path: ?Path = null,

    pub fn run(e: *KeepEngine, _: Fn, _: [3]usize, _: [3]usize, args: anytype) !void {
        e.kernel_calls += 1;
        inline for (args) |arg| {
            if (@TypeOf(arg) == Seg) e.last_seg = arg;
            if (@TypeOf(arg) == Path) e.last_path = arg;
        }
    }
};

test "REV295 keep refreshes Seg and Path, updates in place, and replaces changed topology" {
    Graph.reset();
    var d = Graph.driver();
    const previous = cuda.copy_source.current();
    cuda.copy_source.setCurrent(@ptrFromInt(0x3519));
    defer cuda.copy_source.setCurrent(previous);
    var e: KeepEngine = .{ .stream = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var capture: keep.Capture = .{};
    defer capture.deinit();
    var sg: Seg = std.mem.zeroes(Seg);
    sg.len = 10;
    var p: Path = .{ .n = 2, .rows = @splat(0) };
    p.rows[1] = 1;
    try capture.run(&e, sg, p);
    sg.len = 11;
    p.n = 3;
    p.rows[2] = 2;
    try capture.run(&e, sg, p);
    try t.expectEqual([_]usize{ 2, 1, 1, 2, 4 }, .{ Graph.records, Graph.instantiates, Graph.updates, Graph.launches, e.kernel_calls });
    try t.expectEqual(@as(c_int, 11), e.last_seg.?.len);
    try t.expectEqual(@as(c_int, 3), e.last_path.?.n);
    Graph.verdict = .topology_changed;
    sg.len = 12;
    try capture.run(&e, sg, p);
    try t.expectEqual([_]usize{ 3, 2, 2, 3, 6, 1 }, .{ Graph.records, Graph.instantiates, Graph.updates, Graph.launches, e.kernel_calls, Graph.destroyed });
}

const Norm = struct {
    var values: [6]u64 = undefined;
    var calls: usize = 0;
    fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        values[calls] = @as(*align(1) u64, @ptrCast(args.?[7].?)).*;
        calls += 1;
        return abi.success;
    }
};

test "REV295 actual norm launch suppresses xs only for the opted ModelOpt target" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Norm.launch;
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.g.bf16 = false;
    e.g.eps = 1e-6;
    e.g.hidden = 256;
    e.fns.norm = .{ .d = &d, .handle = @ptrFromInt(1) };
    e.fns.norm_shift = e.fns.norm;
    e.h = 0x1000;
    e.r = 0x2000;
    e.x = 0x3000;
    e.xs = 0x4000;
    e.tables = .{ .d = &d, .ptr = 0x5000, .len = 1024, .borrowed = true };
    e.learning = null;
    e.lever7 = .{};
    e.mo = null;
    Norm.calls = 0;
    try e.norm(true, 0x6000, 1);
    e.mo = @as(@import("cuda_modelopt.zig").Kernels, undefined);
    try e.norm(true, 0x6000, 1);
    e.lever7.skip_xs = true;
    for ([_]usize{ 1, 8, 17 }) |rows| try e.norm(true, 0x6000, rows);
    var hook: @import("cuda_learn_hooks.zig").Keep = undefined;
    e.learning = &hook;
    try e.norm(true, 0x6000, 17);
    try t.expectEqual([6]u64{ 0x4000, 0x4000, 0, 0, 0, 0x4000 }, Norm.values);
}
