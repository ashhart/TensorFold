//! Actual prompt dispatch and occupancy configuration under a recording driver; no CUDA device.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const ql = @import("qlinear.zig");
const A = ql.Affine4Kernels;
const t = std.testing;

const Mock = struct {
    var calls: usize = 0;
    var attributes: usize = 0;
    var blocks: c_int = 2;
    var function: usize = 0;
    var shared: u32 = 0;
    var threads: u32 = 0;
    var grid: u32 = 0;
    var values: [11]u64 = undefined;

    fn get(out: *abi.Function, _: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
        t.expectEqualStrings(ql.symbols.prefill2, std.mem.span(name)) catch unreachable;
        out.* = @ptrFromInt(2);
        return 0;
    }
    fn set(f: abi.Function, attr: abi.FunctionAttribute, value: c_int) callconv(.c) abi.Result {
        t.expectEqual(@as(usize, 2), @intFromPtr(f)) catch unreachable;
        const expected = if (attributes % 2 == 0) abi.FunctionAttribute.max_dynamic_shared_size_bytes else .preferred_shared_memory_carveout;
        t.expectEqual(expected, attr) catch unreachable;
        t.expectEqual(@as(c_int, if (attributes % 2 == 0) 41984 else 100), value) catch unreachable;
        attributes += 1;
        return 0;
    }
    fn occupancy(out: *c_int, _: abi.Function, count: c_int, bytes: usize) callconv(.c) abi.Result {
        t.expectEqual(@as(c_int, 256), count) catch unreachable;
        t.expectEqual(@as(usize, 41984), bytes) catch unreachable;
        out.* = blocks;
        return 0;
    }
    fn attribute(out: *c_int, attr: abi.FunctionAttribute, _: abi.Function) callconv(.c) abi.Result {
        out.* = if (attr == .num_regs) 128 else 0;
        return 0;
    }
    fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, smem: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        t.expectEqual(@as(c_uint, 1), gy) catch unreachable;
        t.expectEqual(@as(c_uint, 1), gz) catch unreachable;
        t.expectEqual(@as(c_uint, 1), by) catch unreachable;
        t.expectEqual(@as(c_uint, 1), bz) catch unreachable;
        function = @intFromPtr(f);
        grid = gx;
        threads = bx;
        shared = smem;
        for (0..5) |i| values[i] = @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).*;
        for (5..11) |i| values[i] = @intCast(@as(*const c_int, @ptrCast(@alignCast(args.?[i].?))).*);
        calls += 1;
        return 0;
    }
};

test "prompt stages2: exact selection, default launch bytes and unchanged half/reference tiles" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    d.api.cuModuleGetFunction = Mock.get;
    d.api.cuFuncSetAttribute = Mock.set;
    d.api.cuFuncGetAttribute = Mock.attribute;
    d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor = Mock.occupancy;
    var a: A = undefined;
    a.prefill_fn = .{ .d = &d, .handle = @ptrFromInt(3) };
    a.prefill22_fn = .{ .d = &d, .handle = @ptrFromInt(22) };
    a.prefill2_fn = null;
    a.half_prompt = false;
    a.pdl = true;
    const stream: @import("stream.zig").Stream = .{ .d = &d, .handle = null };
    const mod: Module = .{ .d = &d, .handle = @ptrFromInt(1) };
    Mock.attributes = 0;
    Mock.blocks = 2;
    for ([_]usize{ 17, 127, 128, 129, 2048, 4096, 8192 }) |rows| {
        for ([_]usize{ 6144, 17408 }) |k| {
            const q: ql.Affine4 = .{ .w = 10, .s = 20, .b = 30, .n = 5120, .k = k, .npad = 5120 };
            a.prefill2_fn = null;
            try a.prompt(stream, 40, q, 50, rows);
            try t.expectEqual(@as(usize, 3), Mock.function);
            try t.expectEqual(@as(u32, 62976), Mock.shared);
            const baseline = Mock.values;
            const grid = Mock.grid;
            try a.enablePromptStages2(mod);
            try a.prompt(stream, 40, q, 50, rows);
            try t.expectEqual(@as(usize, 2), Mock.function);
            try t.expectEqual(@as(u32, 41984), Mock.shared);
            try t.expectEqual(@as(u32, 256), Mock.threads);
            try t.expectEqual(grid, Mock.grid);
            try t.expectEqualSlices(u64, &baseline, &Mock.values);
            try a.promptThreeStage(stream, 40, q, 50, rows);
            try t.expectEqual(@as(usize, 3), Mock.function);
            try a.promptHalf(stream, 40, q, 50, rows);
            try t.expectEqual(@as(usize, 22), Mock.function);
            try t.expectEqual(@as(u32, 128), Mock.threads);
        }
    }
    for ([_][3]usize{ .{ 5120, 17408, 16 }, .{ 17408, 5120, 4096 }, .{ 5120, 5120, 4096 }, .{ 5119, 6144, 4096 }, .{ 5120, 6144, 8193 } }) |dims| {
        const q: ql.Affine4 = .{ .w = 10, .s = 20, .b = 30, .n = dims[0], .k = dims[1], .npad = 5120 };
        try a.prompt(stream, 40, q, 50, dims[2]);
        try t.expectEqual(@as(usize, 3), Mock.function);
        if (dims[2] == 16) {
            try a.promptStages2(stream, 40, q, 50, dims[2]);
            try t.expectEqual(@as(usize, 2), Mock.function);
        } else try t.expectError(error.UnsupportedPromptStages2, a.promptStages2(stream, 40, q, 50, dims[2]));
    }
    a.prefill2_fn = null;
    Mock.blocks = 1;
    try t.expectError(error.PromptStages2Occupancy, a.enablePromptStages2(mod));
    try t.expect(a.prefill2_fn == null);
    a.pdl = false;
    const before = Mock.attributes;
    try t.expectError(error.PromptStages2RequiresGb10, a.enablePromptStages2(mod));
    try t.expectEqual(before, Mock.attributes);
}
