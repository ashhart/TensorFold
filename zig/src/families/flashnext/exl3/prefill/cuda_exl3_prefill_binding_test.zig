const std = @import("std");
const cuda = @import("cuda");
const binding = @import("cuda_exl3_prefill_binding.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const parts = @import("../../cuda_batch_state.zig");

fn request(shared: bool) binding.Request {
    return .{
        .functions = .{ .prep = 10, .scores = 11, .chunks = 12, .ahead = 13, .plan = 14, .remap = 15 },
        .rows = 512,
        .sub = 256,
        .pos0 = 4096,
        .positions = 0x1000000010,
        .cap = 32768,
        .origin = 1024,
        .page_rows = 256,
        .shared = shared,
        .generation = if (shared) 0x1234567800000001 else 0,
        .keys = 0x2000000020,
        .values = 0x3000000030,
    };
}

fn recordFor(req: binding.Request, kind: binding.Kind, block: usize) binding.Record {
    const function = switch (kind) {
        .prep => req.functions.prep,
        .scores => req.functions.scores,
        .chunks => req.functions.chunks,
        .plan => req.functions.plan,
        .remap => req.functions.remap,
    };
    var record: binding.Record = .{
        .node = @ptrFromInt(1),
        .params = .{ .func = @ptrFromInt(function), .grid_x = 256, .grid_y = 1, .grid_z = 1, .block_x = 256, .block_y = 1, .block_z = 1, .shared_bytes = 0, .params = null, .extra = null },
        .kind = kind,
        .block = block,
    };
    for (binding.schema(kind, req.shared), 0..) |size, index| {
        record.values[index] = if (size == 8) 0x8000000000 + index else 100 + index;
    }
    const position = req.positions + block * 16;
    switch (kind) {
        .prep => {
            record.params.grid_y = wt.q_heads + wt.kv_heads + wt.index_heads + 1;
            record.values[1] = position;
            record.values[7] = req.keys -% (req.origin * wt.kv_heads * wt.head_dim * 2);
            record.values[8] = req.values -% (req.origin * wt.kv_heads * wt.head_dim * 2);
        },
        .scores => {
            record.values[2] = position;
            record.values[4] = parts.scoreStride(req.cap);
            record.params.grid_y = req.scores(block);
        },
        .chunks => {
            record.params.grid_y = wt.kv_heads;
            record.params.block_x = 128;
            record.values[3] = position;
            record.params.grid_z = req.chunks(block);
        },
        .plan => {
            record.values[3] = 256;
            record.values[7] = req.origin;
            record.params.grid_x = 1;
            record.params.block_x = 1024;
            if (req.shared) record.values[22] = req.generation;
        },
        .remap => record.values[4] = req.origin,
    }
    return record;
}

fn recordsFor(req: binding.Request) [10]binding.Record {
    var records: [10]binding.Record = undefined;
    for (0..2) |block| for (std.enums.values(binding.Kind), 0..) |kind, index| {
        records[block * 5 + index] = recordFor(req, kind, block);
        records[block * 5 + index].node = @ptrFromInt(block * 5 + index + 1);
    };
    return records;
}

test "exl3: prefill origin binding keeps dense-sparse and paging-split topology explicit" {
    const base = request(true);
    try base.validate();
    var target = base;
    target.pos0 = 8192;
    target.origin = 4096;
    target.generation += 1;
    try std.testing.expect(base.compatible(target));
    target.page_rows = 32;
    try std.testing.expect(!base.compatible(target));
    target = base;
    target.pos0 = 1536;
    try std.testing.expect(!base.compatible(target));
    target = base;
    target.positions += 16;
    try std.testing.expect(!base.compatible(target));
    target = base;
    target.keys += 1024;
    try std.testing.expect(!base.compatible(target));
    target = base;
    target.functions.plan += 10;
    try std.testing.expect(!base.compatible(target));
    target = base;
    target.sub = 128;
    try std.testing.expectError(error.UnsupportedPrefillBinding, target.validate());
    target = base;
    target.pos0 = target.cap;
    try std.testing.expectError(error.UnsupportedPrefillBinding, target.validate());
}

const Capture = struct {
    var records: [10]binding.Record = undefined;
    var args: cuda.Args = .{};
    var kind: c_uint = 0;
    var count: usize = 10;
    var destroyed: usize = 0;
    var shared: bool = true;

    fn nodes(_: cuda.abi.Graph, output: ?[*]cuda.graph.Node, size: *usize) callconv(.c) cuda.abi.Result {
        if (output) |storage| for (0..@min(size.*, count)) |index| {
            storage[index] = @ptrFromInt(index + 1);
        };
        size.* = count;
        return 0;
    }

    fn nodeType(_: cuda.graph.Node, output: *c_uint) callconv(.c) cuda.abi.Result {
        output.* = kind;
        return 0;
    }

    fn params(node: cuda.graph.Node, output: *cuda.abi.KernelNodeParams) callconv(.c) cuda.abi.Result {
        const record = records[@intFromPtr(node) - 1];
        args = .{};
        for (binding.schema(record.kind, shared), 0..) |size, index| {
            if (size == 8) args.add(record.values[index]) else args.add(@as(u32, @truncate(record.values[index])));
        }
        output.* = record.params;
        output.params = args.pointers();
        return 0;
    }

    fn destroy(_: cuda.abi.Graph) callconv(.c) cuda.abi.Result {
        destroyed += 1;
        return 0;
    }

    fn driver() cuda.Driver {
        var result: cuda.Driver = undefined;
        result.api.cuGraphGetNodes = nodes;
        result.api.cuGraphKernelNodeGetParams_v2 = params;
        result.api.cuGraphDestroy = destroy;
        return result;
    }
};

test "exl3: prefill binding validates captured schemas, missing nodes, duplicates and child graphs" {
    var driver = Capture.driver();
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = @ptrFromInt(99) };
    for ([_]bool{ false, true }) |shared| {
        const req = request(shared);
        Capture.shared = shared;
        Capture.records = recordsFor(req);
        Capture.count = 10;
        Capture.kind = 0;
        const loaded = try binding.Binding.createWithNodeType(std.testing.allocator, graph, req, Capture.nodeType);
        try std.testing.expectEqual(@as(usize, 10), loaded.records.len);
        for (loaded.records, Capture.records) |actual, expected| try std.testing.expectEqualDeep(expected.values, actual.values);
        loaded.deinit(std.testing.allocator);
        Capture.count = 9;
        try std.testing.expectError(error.UnsupportedPrefillBinding, binding.Binding.createWithNodeType(std.testing.allocator, graph, req, Capture.nodeType));
        Capture.count = 10;
        Capture.records[9] = Capture.records[0];
        try std.testing.expectError(error.UnsupportedPrefillBinding, binding.Binding.createWithNodeType(std.testing.allocator, graph, req, Capture.nodeType));
        Capture.records = recordsFor(req);
        Capture.records[0].values[7] += 1;
        try std.testing.expectError(error.UnsupportedPrefillBinding, binding.Binding.createWithNodeType(std.testing.allocator, graph, req, Capture.nodeType));
        Capture.kind = 4;
        try std.testing.expectError(error.UnsupportedPrefillBinding, binding.Binding.createWithNodeType(std.testing.allocator, graph, req, Capture.nodeType));
        try std.testing.expectError(error.UnsupportedPrefillBinding, binding.Binding.createWithNodeType(std.testing.allocator, graph, req, null));
    }
}

const Setter = struct {
    calls: usize = 0,
    fail_at: ?usize = null,
    records: [10]binding.Record = undefined,

    pub fn set(self: *@This(), record: binding.Record, _: bool) !void {
        if (self.fail_at == self.calls) return error.InjectedSetterFailure;
        self.calls += 1;
        self.records[@intFromPtr(record.node) - 1] = record;
    }
};

test "exl3: prefill rebind updates all origin arguments and grids without stale scalars" {
    for ([_]bool{ false, true }) |shared| {
        const req = request(shared);
        var records = recordsFor(req);
        var state: binding.Binding = .{ .graph = undefined, .request = req, .records = &records };
        var setter: Setter = .{};
        var target = req;
        target.pos0 = 8192;
        target.origin = 4096;
        try state.apply(&setter, target);
        try std.testing.expectEqual(@as(usize, 8), setter.calls);
        for (records) |record| {
            const before = recordFor(req, record.kind, record.block);
            const expected = before.rebound(target);
            try std.testing.expectEqualDeep(expected.values, record.values);
            try std.testing.expectEqual(expected.params.grid_y, record.params.grid_y);
            try std.testing.expectEqual(expected.params.grid_z, record.params.grid_z);
            if (record.kind == .plan and shared) try std.testing.expectEqual(target.generation, record.values[22]);
        }
        setter.calls = 0;
        try state.apply(&setter, target);
        try std.testing.expectEqual(@as(usize, 0), setter.calls);
        try state.apply(&setter, req);
        try std.testing.expectEqual(@as(usize, 8), setter.calls);
        for (records) |record| try std.testing.expectEqualDeep(recordFor(req, record.kind, record.block).values, record.values);
    }
}

test "exl3: prefill partial setter failures poison replay and topology declines do not mutate" {
    const req = request(true);
    var records = recordsFor(req);
    var state: binding.Binding = .{ .graph = undefined, .request = req, .records = &records };
    var setter: Setter = .{ .fail_at = 1 };
    var target = req;
    target.pos0 = 8192;
    target.origin = 4096;
    try std.testing.expectError(error.InjectedSetterFailure, state.apply(&setter, target));
    try std.testing.expect(state.poisoned);
    try std.testing.expectEqual(req, state.request);
    try std.testing.expectError(error.UnsupportedPrefillBinding, state.apply(&setter, req));
    state.poisoned = false;
    target.pos0 = 1536;
    setter.calls = 0;
    try std.testing.expectError(error.UnsupportedPrefillBinding, state.apply(&setter, target));
    try std.testing.expectEqual(@as(usize, 0), setter.calls);
}

test "exl3: prefill CUDA setter preserves scalar widths and 64-bit generations" {
    const Fake = struct {
        var expected: binding.Record = undefined;
        var shared: bool = false;
        var matches: bool = false;

        fn set(_: cuda.abi.GraphExec, node: cuda.graph.Node, params: *const cuda.abi.KernelNodeParams) callconv(.c) cuda.abi.Result {
            matches = node == expected.node and params.func == expected.params.func and params.grid_y == expected.params.grid_y and params.grid_z == expected.params.grid_z and params.kern == null and params.ctx == null and params.extra == null;
            for (binding.schema(expected.kind, shared), 0..) |size, index| {
                const value = @as([*]const u8, @ptrCast(params.params.?[index].?))[0..size];
                matches = matches and std.mem.eql(u8, value, std.mem.asBytes(&expected.values[index])[0..size]);
            }
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuGraphExecKernelNodeSetParams_v2 = Fake.set;
    const backend: binding.Backend = .{ .exec = .{ .d = &driver, .handle = @ptrFromInt(99) } };
    for ([_]bool{ false, true }) |shared| for (std.enums.values(binding.Kind)) |kind| {
        Fake.expected = recordFor(request(shared), kind, 0);
        Fake.expected.params.kern = @ptrFromInt(123);
        Fake.expected.params.ctx = @ptrFromInt(456);
        Fake.shared = shared;
        try backend.set(Fake.expected, shared);
        try std.testing.expect(Fake.matches);
    };
}
