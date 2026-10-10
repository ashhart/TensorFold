//! HIP graphs: captured from a stream or built node by node, instantiated once, replayed and updated in place.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const std = @import("std");
const Stream = @import("stream.zig").Stream;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");

pub const Node = abi.GraphNode;

/// Every launch on `stream` is recorded until `endCapture`; nothing runs.
pub fn beginCapture(stream: Stream, mode: abi.CaptureMode) runtime.Error!void {
    try runtime.check(stream.r.api.hipStreamBeginCapture(stream.handle, mode));
}

/// An invalidated capture returns an error and no graph.
pub fn endCapture(stream: Stream) runtime.Error!Graph {
    var g: abi.Graph = null;
    try runtime.check(stream.r.api.hipStreamEndCapture(stream.handle, &g));
    if (g == null) return error.Invalid;
    return .{ .r = stream.r, .handle = g };
}

pub fn captureStatus(stream: Stream) runtime.Error!abi.CaptureStatus {
    var s: abi.CaptureStatus = .none;
    try runtime.check(stream.r.api.hipStreamIsCapturing(stream.handle, &s));
    return s;
}

/// A cooperative launch has no graph node form.
fn nodeParams(r: *const runtime.Runtime, f: Function, cfg: launch.Config, args: *launch.Args) runtime.Error!abi.KernelNodeParams {
    try cfg.validate();
    if (cfg.cooperative or f.r != r) return error.Invalid;
    return .{
        .block = .{ .x = cfg.block.x, .y = cfg.block.y, .z = cfg.block.z },
        .extra = null,
        .func = f.handle,
        .grid = .{ .x = cfg.grid.x, .y = cfg.grid.y, .z = cfg.grid.z },
        .params = args.pointers(),
        .shared_bytes = cfg.shared,
    };
}

pub const Graph = struct {
    r: *const runtime.Runtime,
    handle: abi.Graph,

    pub fn init(r: *const runtime.Runtime) runtime.Error!Graph {
        var g: abi.Graph = null;
        try runtime.check(r.api.hipGraphCreate(&g, 0));
        if (g == null) return error.Invalid;
        return .{ .r = r, .handle = g };
    }

    pub fn deinit(self: *Graph) void {
        _ = self.r.api.hipGraphDestroy(self.handle);
        self.* = undefined;
    }

    /// A kernel node after `deps`; HIP copies the argument values, so `args` may change afterwards.
    pub fn addKernel(self: Graph, deps: []const Node, f: Function, cfg: launch.Config, args: *launch.Args) runtime.Error!Node {
        const p = try nodeParams(self.r, f, cfg, args);
        var n: Node = null;
        try runtime.check(self.r.api.hipGraphAddKernelNode(&n, self.handle, if (deps.len > 0) deps.ptr else null, deps.len, &p));
        return n;
    }

    /// New arguments or geometry for a node of this graph, before it is instantiated or an exec is updated from it.
    pub fn setKernel(self: Graph, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) runtime.Error!void {
        const p = try nodeParams(self.r, f, cfg, args);
        try runtime.check(self.r.api.hipGraphKernelNodeSetParams(node, &p));
    }

    pub fn depend(self: Graph, from: Node, to: Node) runtime.Error!void {
        const a = [1]Node{from};
        const b = [1]Node{to};
        try runtime.check(self.r.api.hipGraphAddDependencies(self.handle, &a, &b, 1));
    }

    /// Fills `out` with the graph's nodes (capture order for a captured graph) and returns them.
    pub fn nodes(self: Graph, out: []Node) runtime.Error![]Node {
        var n: usize = out.len;
        try runtime.check(self.r.api.hipGraphGetNodes(self.handle, out.ptr, &n));
        if (n > out.len) return error.Invalid;
        return out[0..n];
    }

    pub fn instantiate(self: Graph) runtime.Error!Exec {
        var e: abi.GraphExec = null;
        try runtime.check(self.r.api.hipGraphInstantiateWithFlags(&e, self.handle, 0));
        if (e == null) return error.Invalid;
        return .{ .r = self.r, .handle = e };
    }
};

pub const Exec = struct {
    r: *const runtime.Runtime,
    handle: abi.GraphExec,

    pub fn deinit(self: *Exec) void {
        _ = self.r.api.hipGraphExecDestroy(self.handle);
        self.* = undefined;
    }

    /// Moves the graph's work to the device ahead of the first launch, so that launch pays no setup.
    pub fn upload(self: Exec, stream: Stream) runtime.Error!void {
        if (self.r != stream.r) return error.Invalid;
        try runtime.check(self.r.api.hipGraphUpload(self.handle, stream.handle));
    }

    pub fn launchOn(self: Exec, stream: Stream) runtime.Error!void {
        if (self.r != stream.r) return error.Invalid;
        try runtime.check(self.r.api.hipGraphLaunch(self.handle, stream.handle));
    }

    /// One kernel node's arguments or geometry, changed in the executable graph without rebuilding it.
    pub fn setKernel(self: Exec, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) runtime.Error!void {
        const p = try nodeParams(self.r, f, cfg, args);
        try runtime.check(self.r.api.hipGraphExecKernelNodeSetParams(self.handle, node, &p));
    }

    /// Every node's parameters from `g`, which must keep the topology; a refused update leaves the exec unchanged.
    pub fn update(self: Exec, g: Graph) runtime.Error!abi.ExecUpdateResult {
        var node: Node = null;
        var result: abi.ExecUpdateResult = .success;
        const res = self.r.api.hipGraphExecUpdate(self.handle, g.handle, &node, &result);
        if (result != .success) return result;
        try runtime.check(res);
        return .success;
    }
};

test "kernel nodes refuse cooperative launches, bad geometry and foreign functions before HIP" {
    var r: runtime.Runtime = undefined;
    var other: runtime.Runtime = undefined;
    var args: launch.Args = .{};
    const f: Function = .{ .r = &r, .handle = @ptrFromInt(64) };
    const ok: launch.Config = .{ .grid = .{ .x = 2 }, .block = .{ .x = 64, .y = 2 }, .shared = 128 };
    const p = try nodeParams(&r, f, ok, &args);
    try std.testing.expectEqual(@as(c_uint, 2), p.grid.x);
    try std.testing.expectEqual(@as(c_uint, 2), p.block.y);
    try std.testing.expectEqual(@as(c_uint, 128), p.shared_bytes);
    try std.testing.expectError(error.Invalid, nodeParams(&r, f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 64 }, .cooperative = true }, &args));
    try std.testing.expectError(error.Invalid, nodeParams(&r, f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 2048 } }, &args));
    try std.testing.expectError(error.Invalid, nodeParams(&other, f, ok, &args));
}

test "a refused exec update reports HIP's verdict, not an error" {
    const Mock = struct {
        fn refuse(_: abi.GraphExec, _: abi.Graph, node: *abi.GraphNode, result: *abi.ExecUpdateResult) callconv(.c) abi.Result {
            node.* = null;
            result.* = .topology_changed;
            return 1;
        }
        fn fail(_: abi.GraphExec, _: abi.Graph, _: *abi.GraphNode, _: *abi.ExecUpdateResult) callconv(.c) abi.Result {
            return 1;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipGraphExecUpdate = Mock.refuse;
    const e: Exec = .{ .r = &r, .handle = @ptrFromInt(8) };
    const g: Graph = .{ .r = &r, .handle = @ptrFromInt(16) };
    try std.testing.expectEqual(abi.ExecUpdateResult.topology_changed, try e.update(g));
    r.api.hipGraphExecUpdate = Mock.fail;
    try std.testing.expectError(error.HipFailed, e.update(g));
}
