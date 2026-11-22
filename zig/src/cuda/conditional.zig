//! Optional CUDA 12.3 conditional graphs; missing entry points preserve the caller's ordinary stream path.
const std = @import("std");
const abi = @import("abi.zig");
const graph = @import("graph.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
pub const Handle = u64;
pub const Params = extern struct {
    kind: c_int = 13,
    reserved0: [3]c_int = @splat(0),
    value: extern union {
        reserved: [29]u64,
        conditional: extern struct { handle: Handle, kind: c_int, size: c_uint, graphs: ?[*]abi.Graph, context: abi.Context },
    } = .{ .reserved = @splat(0) },
    reserved1: u64 = 0,
};
pub const Api = struct {
    create: *const fn (*Handle, abi.Graph, abi.Context, c_uint, c_uint) callconv(.c) abi.Result,
    add: *const fn (*abi.GraphNode, abi.Graph, ?[*]const abi.GraphNode, usize, *Params) callconv(.c) abi.Result,
    capture: *const fn (abi.Stream, abi.Graph, ?[*]const abi.GraphNode, ?*const anyopaque, usize, abi.CaptureMode) callconv(.c) abi.Result,
    pub fn resolve(lib: anytype) ?Api {
        return .{
            .create = lib.lookup(@FieldType(Api, "create"), "cuGraphConditionalHandleCreate") orelse return null,
            .add = lib.lookup(@FieldType(Api, "add"), "cuGraphAddNode") orelse return null,
            .capture = lib.lookup(@FieldType(Api, "capture"), "cuStreamBeginCaptureToGraph") orelse return null,
        };
    }
    fn checked(d: *const Driver, result: abi.Result, name: []const u8) !void {
        if (result == 801 or result == 800) return error.GraphNotPermitted;
        try d.check(result, name);
    }
    pub fn handle(a: Api, parent: graph.Graph, context: abi.Context) !Handle {
        var out: Handle = 0;
        try checked(parent.d, a.create(&out, parent.handle, context, 0, 1), "cuGraphConditionalHandleCreate");
        return out;
    }
    /// One WHILE body owned by the parent graph; the handle resets to zero on each launch.
    pub fn loop(a: Api, parent: graph.Graph, deps: []const graph.Node, context: abi.Context, condition: Handle) !struct { node: graph.Node, body: graph.Graph } {
        var p: Params = .{};
        p.value.conditional = .{ .handle = condition, .kind = 1, .size = 1, .graphs = null, .context = context };
        var node: graph.Node = null;
        try checked(parent.d, a.add(&node, parent.handle, if (deps.len > 0) deps.ptr else null, deps.len, &p), "cuGraphAddNode(WHILE)");
        const bodies = p.value.conditional.graphs orelse return error.Invalid;
        return .{ .node = node, .body = .{ .d = parent.d, .handle = bodies[0] } };
    }
    pub fn instantiate(g: graph.Graph) !graph.Exec {
        var exec: abi.GraphExec = null;
        try checked(g.d, g.d.api.cuGraphInstantiateWithFlags(&exec, g.handle, 0), "cuGraphInstantiateWithFlags(conditional)");
        return .{ .d = g.d, .handle = exec };
    }
    pub fn upload(exec: graph.Exec, stream: Stream) !void {
        try checked(exec.d, exec.d.api.cuGraphUpload(exec.handle, stream.handle), "cuGraphUpload(conditional)");
    }
    /// Capturing into an existing body shares the normal upload-source lifetime guard.
    pub fn begin(a: Api, s: Stream, g: graph.Graph, deps: []const graph.Node) !void {
        const source = @import("copy_source.zig");
        try source.captureBegin(s, .thread_local);
        errdefer source.captureEnd(s);
        try checked(s.d, a.capture(s.handle, g.handle, if (deps.len > 0) deps.ptr else null, null, deps.len, .thread_local), "cuStreamBeginCaptureToGraph");
    }
};
comptime {
    std.debug.assert(@sizeOf(Params) == 256 and @offsetOf(Params, "value") == 16);
}
