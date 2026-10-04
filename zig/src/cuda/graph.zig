//! CUDA graphs: captured from a stream or built node by node, instantiated once, replayed and updated in place.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const Stream = @import("stream.zig").Stream;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");

pub const Node = abi.GraphNode;

/// Starts capturing `stream`; every launch on it is recorded until `endCapture`, nothing runs.
pub fn beginCapture(stream: Stream, mode: abi.CaptureMode) Error!void {
    try stream.d.check(stream.d.api.cuStreamBeginCapture_v2(stream.handle, mode), "cuStreamBeginCapture");
}

/// Ends the capture; an invalidated capture returns an error and no graph.
pub fn endCapture(stream: Stream) Error!Graph {
    var g: abi.Graph = null;
    try stream.d.check(stream.d.api.cuStreamEndCapture(stream.handle, &g), "cuStreamEndCapture");
    return .{ .d = stream.d, .handle = g };
}

pub fn captureStatus(stream: Stream) Error!abi.CaptureStatus {
    var s: abi.CaptureStatus = .none;
    try stream.d.check(stream.d.api.cuStreamIsCapturing(stream.handle, &s), "cuStreamIsCapturing");
    return s;
}

fn nodeParams(f: Function, cfg: launch.Config, args: *launch.Args) Error!abi.KernelNodeParams {
    try cfg.validate();
    if (cfg.cluster != null or cfg.pdl or cfg.cooperative) return error.Invalid;
    return .{
        .func = f.handle,
        .grid_x = cfg.grid.x,
        .grid_y = cfg.grid.y,
        .grid_z = cfg.grid.z,
        .block_x = cfg.block.x,
        .block_y = cfg.block.y,
        .block_z = cfg.block.z,
        .shared_bytes = cfg.shared,
        .params = args.pointers(),
        .extra = null,
    };
}

pub const Graph = struct {
    d: *const Driver,
    handle: abi.Graph,

    pub fn init(d: *const Driver) Error!Graph {
        var g: abi.Graph = null;
        try d.check(d.api.cuGraphCreate(&g, 0), "cuGraphCreate");
        return .{ .d = d, .handle = g };
    }

    pub fn deinit(self: *Graph) void {
        _ = self.d.api.cuGraphDestroy(self.handle);
        self.* = undefined;
    }

    /// A kernel node after `deps`; the driver copies the argument values, so `args` may change afterwards.
    pub fn addKernel(self: Graph, deps: []const Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!Node {
        const p = try nodeParams(f, cfg, args);
        var n: Node = null;
        try self.d.check(self.d.api.cuGraphAddKernelNode_v2(&n, self.handle, if (deps.len > 0) deps.ptr else null, deps.len, &p), "cuGraphAddKernelNode");
        return n;
    }

    /// New arguments or geometry for a node of this (template) graph, before it is instantiated again or updated from.
    pub fn setKernel(self: Graph, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!void {
        const p = try nodeParams(f, cfg, args);
        try self.d.check(self.d.api.cuGraphKernelNodeSetParams_v2(node, &p), "cuGraphKernelNodeSetParams");
    }

    pub fn depend(self: Graph, from: Node, to: Node) Error!void {
        const a = [1]Node{from};
        const b = [1]Node{to};
        try self.d.check(self.d.api.cuGraphAddDependencies(self.handle, &a, &b, 1), "cuGraphAddDependencies");
    }

    pub fn nodeCount(self: Graph) Error!usize {
        var n: usize = 0;
        try self.d.check(self.d.api.cuGraphGetNodes(self.handle, null, &n), "cuGraphGetNodes");
        return n;
    }

    /// Fills `out` with the graph's nodes (capture order for captured graphs) and returns them.
    pub fn nodes(self: Graph, out: []Node) Error![]Node {
        var n: usize = out.len;
        try self.d.check(self.d.api.cuGraphGetNodes(self.handle, out.ptr, &n), "cuGraphGetNodes");
        return out[0..@min(n, out.len)];
    }

    pub fn instantiate(self: Graph) Error!Exec {
        var e: abi.GraphExec = null;
        try self.d.check(self.d.api.cuGraphInstantiateWithFlags(&e, self.handle, 0), "cuGraphInstantiateWithFlags");
        return .{ .d = self.d, .handle = e };
    }
};

pub const Exec = struct {
    d: *const Driver,
    handle: abi.GraphExec,

    pub fn deinit(self: *Exec) void {
        _ = self.d.api.cuGraphExecDestroy(self.handle);
        self.* = undefined;
    }

    /// Moves the graph's work to the device ahead of the first launch, so that launch pays no setup.
    pub fn upload(self: Exec, stream: Stream) Error!void {
        try self.d.check(self.d.api.cuGraphUpload(self.handle, stream.handle), "cuGraphUpload");
    }

    pub fn launchOn(self: Exec, stream: Stream) Error!void {
        try self.d.check(self.d.api.cuGraphLaunch(self.handle, stream.handle), "cuGraphLaunch");
    }

    /// Changes one kernel node's arguments or geometry in the executable graph without rebuilding it.
    pub fn setKernel(self: Exec, node: Node, f: Function, cfg: launch.Config, args: *launch.Args) Error!void {
        const p = try nodeParams(f, cfg, args);
        try self.d.check(self.d.api.cuGraphExecKernelNodeSetParams_v2(self.handle, node, &p), "cuGraphExecKernelNodeSetParams");
    }

    /// Takes every node's parameters from `g`, which must have the same topology; returns the driver's verdict.
    pub fn update(self: Exec, g: Graph) Error!abi.ExecUpdateResult {
        var info: abi.ExecUpdateResultInfo = .{ .result = .success, .error_node = null, .error_from_node = null };
        const res = self.d.api.cuGraphExecUpdate_v2(self.handle, g.handle, &info);
        if (res == abi.success) return info.result;
        if (info.result != .success) return info.result;
        try self.d.check(res, "cuGraphExecUpdate");
        unreachable;
    }
};
