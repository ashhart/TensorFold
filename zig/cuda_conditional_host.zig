//! Conditional-node ABI and capability refusal through the actual production API wrapper, without a driver.
const std = @import("std");
const cuda = @import("cuda");
const a = cuda.abi;
const api = cuda.conditional;
const t = std.testing;
const Record = struct {
    var fail: a.Result = 0;
    var graph: [1]a.Graph = .{@ptrFromInt(8)};
    fn create(out: *u64, _: a.Graph, _: a.Context, initial: c_uint, flags: c_uint) callconv(.c) a.Result {
        if (initial != 0 or flags != 1) return 1;
        out.* = 9;
        return fail;
    }
    fn add(out: *a.GraphNode, _: a.Graph, deps: ?[*]const a.GraphNode, n: usize, p: *api.Params) callconv(.c) a.Result {
        if (p.kind != 13 or p.value.conditional.kind != 1 or p.value.conditional.size != 1 or p.value.conditional.handle != 9 or n != 1 or deps == null) return 1;
        p.value.conditional.graphs = &graph;
        out.* = @ptrFromInt(7);
        return fail;
    }
    fn capture(_: a.Stream, _: a.Graph, _: ?[*]const a.GraphNode, _: ?*const anyopaque, _: usize, _: a.CaptureMode) callconv(.c) a.Result {
        return 801;
    }
};
test "CUDA conditional: while ABI creates a reset handle and returns the parent-owned body" {
    Record.fail = 0;
    var d: cuda.Driver = undefined;
    const g: cuda.graph.Graph = .{ .d = &d, .handle = @ptrFromInt(6) };
    const calls: api.Api = .{ .create = Record.create, .add = Record.add, .capture = Record.capture };
    const handle = try calls.handle(g, null);
    const loop = try calls.loop(g, &.{@ptrFromInt(5)}, null, handle);
    try t.expectEqual(@as(a.GraphNode, @ptrFromInt(7)), loop.node);
    try t.expectEqual(Record.graph[0], loop.body.handle);
}
test "CUDA conditional: explicit driver unsupported and policy refusal permit the original host loop" {
    var d: cuda.Driver = undefined;
    const g: cuda.graph.Graph = .{ .d = &d, .handle = @ptrFromInt(6) };
    const calls: api.Api = .{ .create = Record.create, .add = Record.add, .capture = Record.capture };
    for ([_]a.Result{ 800, 801 }) |result| {
        Record.fail = result;
        try t.expectError(error.GraphNotPermitted, calls.handle(g, null));
        try t.expectError(error.GraphNotPermitted, calls.loop(g, &.{@ptrFromInt(5)}, null, 9));
    }
    Record.fail = 0;
}

const Instantiate = struct {
    fn upload(_: a.GraphExec, _: a.Stream) callconv(.c) a.Result {
        return result;
    }
    var result: a.Result = 0;
    fn call(out: *a.GraphExec, _: a.Graph, _: u64) callconv(.c) a.Result {
        out.* = @ptrFromInt(12);
        return result;
    }
};
test "CUDA conditional: instantiation unsupported and policy refusal allow fallback before replay" {
    var d: cuda.Driver = undefined;
    d.api.cuGraphInstantiateWithFlags = Instantiate.call;
    d.api.cuGraphUpload = Instantiate.upload;
    const graph: cuda.graph.Graph = .{ .d = &d, .handle = @ptrFromInt(6) };
    for ([_]a.Result{ 800, 801 }) |result| {
        Instantiate.result = result;
        try t.expectError(error.GraphNotPermitted, api.Api.instantiate(graph));
        try t.expectError(error.GraphNotPermitted, api.Api.upload(.{ .d = &d, .handle = @ptrFromInt(12) }, .{ .d = &d, .handle = null }));
    }
    Instantiate.result = 0;
    try t.expectEqual(@as(a.GraphExec, @ptrFromInt(12)), (try api.Api.instantiate(graph)).handle);
}
