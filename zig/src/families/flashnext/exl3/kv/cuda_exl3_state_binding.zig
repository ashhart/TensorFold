const std = @import("std");
const cuda = @import("cuda");
const binding = @import("exl3_state_binding.zig");

pub const NodeType = *const fn (cuda.graph.Node, *c_uint) callconv(.c) cuda.abi.Result;

pub fn nodeType(driver: *const cuda.Driver) ?NodeType {
    var library = driver.lib;
    return library.lookup(NodeType, "cuGraphNodeGetType");
}

pub const Backend = struct {
    graph: cuda.graph.Graph,
    exec: cuda.graph.Exec,
    node_type: NodeType,

    pub fn init(graph: cuda.graph.Graph, exec: cuda.graph.Exec) ?Backend {
        return initWithNodeType(graph, exec, nodeType(graph.d));
    }

    pub fn initWithNodeType(graph: cuda.graph.Graph, exec: cuda.graph.Exec, node_type: ?NodeType) ?Backend {
        return .{ .graph = graph, .exec = exec, .node_type = node_type orelse return null };
    }

    pub fn nodes(self: Backend, template: usize) ![]usize {
        if (template != @intFromPtr(self.graph.handle)) return error.UnsupportedBinding;
        const count = try self.graph.nodeCount();
        const storage = try std.heap.c_allocator.alloc(cuda.graph.Node, count);
        defer std.heap.c_allocator.free(storage);
        const actual = try self.graph.nodes(storage);
        if (actual.len != count) return error.UnsupportedBinding;
        const identifiers = try std.heap.c_allocator.alloc(usize, count);
        for (actual, identifiers) |node, *identifier| identifier.* = @intFromPtr(node);
        return identifiers;
    }

    pub fn freeNodes(_: Backend, nodes_list: []usize) void {
        std.heap.c_allocator.free(nodes_list);
    }

    pub fn kernel(self: Backend, node: usize) !?binding.Header {
        var kind: c_uint = undefined;
        try self.graph.d.check(self.node_type(@ptrFromInt(node), &kind), "cuGraphNodeGetType(state binding)");
        if (kind == 4) return error.UnsupportedBinding;
        if (kind != 0) return null;
        var params: cuda.abi.KernelNodeParams = undefined;
        try self.graph.d.check(self.graph.d.api.cuGraphKernelNodeGetParams_v2(@ptrFromInt(node), &params), "cuGraphKernelNodeGetParams(state binding)");
        return .{
            .function = @intFromPtr(params.func),
            .geometry = .{ .grid = .{ params.grid_x, params.grid_y, params.grid_z }, .block = .{ params.block_x, params.block_y, params.block_z }, .shared = params.shared_bytes },
            .params = params.params,
            .extra = params.extra != null,
            .alternate_kernel = params.func == null,
        };
    }

    pub fn set(self: Backend, node: usize, function: usize, geometry: binding.Geometry, parameters: binding.Parameters) !void {
        var args: cuda.Args = .{};
        args.add(parameters.nonstate.gq);
        args.add(parameters.nonstate.gk);
        args.add(parameters.nonstate.gv);
        args.add(parameters.nonstate.ggb);
        args.add(parameters.state_in);
        if (parameters.state_out != 0) args.add(parameters.state_out);
        args.add(parameters.nonstate.rows);
        args.add(parameters.nonstate.ys);
        try self.exec.setKernel(@ptrFromInt(node), .{ .d = self.exec.d, .handle = @ptrFromInt(function) }, .{
            .grid = .{ .x = geometry.grid[0], .y = geometry.grid[1], .z = geometry.grid[2] },
            .block = .{ .x = geometry.block[0], .y = geometry.block[1], .z = geometry.block[2] },
            .shared = geometry.shared,
        }, &args);
    }
};

test "exl3: state binding adapter declines absent type query and propagates query errors" {
    std.testing.refAllDecls(Backend);
    const Fake = struct {
        var getter_calls: usize = 0;
        var kind: c_uint = 1;
        var result: cuda.abi.Result = 0;

        fn nodeType(_: cuda.graph.Node, output: *c_uint) callconv(.c) cuda.abi.Result {
            output.* = kind;
            return result;
        }

        fn get(_: cuda.graph.Node, _: *cuda.abi.KernelNodeParams) callconv(.c) cuda.abi.Result {
            getter_calls += 1;
            return cuda.abi.error_not_ready;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuGraphKernelNodeGetParams_v2 = Fake.get;
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = @ptrFromInt(42) };
    const exec: cuda.graph.Exec = .{ .d = &driver, .handle = @ptrFromInt(43) };
    try std.testing.expect(Backend.initWithNodeType(graph, exec, null) == null);
    const backend = Backend.initWithNodeType(graph, exec, Fake.nodeType).?;
    Fake.getter_calls = 0;
    Fake.kind = 1;
    Fake.result = 0;
    try std.testing.expect(try backend.kernel(1) == null);
    try std.testing.expectEqual(@as(usize, 0), Fake.getter_calls);
    Fake.result = cuda.abi.error_not_ready;
    try std.testing.expectError(error.NotReady, backend.kernel(1));
    try std.testing.expectEqual(@as(usize, 0), Fake.getter_calls);
    Fake.result = 0;
    Fake.kind = 0;
    try std.testing.expectError(error.NotReady, backend.kernel(1));
    try std.testing.expectEqual(@as(usize, 1), Fake.getter_calls);
    Fake.kind = 4;
    try std.testing.expectError(error.UnsupportedBinding, backend.kernel(1));
}

test "exl3: state binding adapter setters retain exact schema7 and schema8 types" {
    const Fake = struct {
        var output: bool = false;
        var captured: binding.Parameters = undefined;
        var geometry: binding.Geometry = undefined;
        var function: usize = 0;
        var node: usize = 0;

        fn value(comptime Value: type, params: [*]?*anyopaque, index: usize) Value {
            var copied: Value = undefined;
            @memcpy(std.mem.asBytes(&copied), @as([*]const u8, @ptrCast(params[index].?))[0..@sizeOf(Value)]);
            return copied;
        }

        fn set(_: cuda.abi.GraphExec, target: cuda.graph.Node, params: *const cuda.abi.KernelNodeParams) callconv(.c) cuda.abi.Result {
            const args = params.params.?;
            captured = .{
                .nonstate = .{
                    .gq = value(u64, args, 0),
                    .gk = value(u64, args, 1),
                    .gv = value(u64, args, 2),
                    .ggb = value(u64, args, 3),
                    .rows = value(c_int, args, if (output) 6 else 5),
                    .ys = value(u64, args, if (output) 7 else 6),
                },
                .state_in = value(u64, args, 4),
                .state_out = if (output) value(u64, args, 5) else 0,
            };
            geometry = .{ .grid = .{ params.grid_x, params.grid_y, params.grid_z }, .block = .{ params.block_x, params.block_y, params.block_z }, .shared = params.shared_bytes };
            function = @intFromPtr(params.func);
            node = @intFromPtr(target);
            return 0;
        }

        fn nodeType(_: cuda.graph.Node, _: *c_uint) callconv(.c) cuda.abi.Result {
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuGraphExecKernelNodeSetParams_v2 = Fake.set;
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = @ptrFromInt(42) };
    const exec: cuda.graph.Exec = .{ .d = &driver, .handle = @ptrFromInt(43) };
    const backend = Backend.initWithNodeType(graph, exec, Fake.nodeType).?;
    const geometry: binding.Geometry = .{ .grid = .{ 16, 8, 1 }, .block = .{ 128, 1, 1 } };
    for (1..binding.max_width + 1) |width| for ([_]bool{ false, true }) |output| for ([_]bool{ false, true }) |flipped| {
        Fake.output = output;
        const parameters: binding.Parameters = .{
            .nonstate = .{ .gq = 0x100000001, .gk = 0x200000002, .gv = 0x300000003, .ggb = 0x400000004, .rows = @intCast(width), .ys = 0x500000005 },
            .state_in = if (flipped) 2000 else 1000,
            .state_out = if (!output) 0 else if (flipped) 1000 else 2000,
        };
        try backend.set(9, if (output) 11 else 10, geometry, parameters);
        try std.testing.expectEqualDeep(parameters, Fake.captured);
        try std.testing.expectEqualDeep(geometry, Fake.geometry);
        try std.testing.expectEqual(@as(usize, if (output) 11 else 10), Fake.function);
        try std.testing.expectEqual(@as(usize, 9), Fake.node);
    };
}
