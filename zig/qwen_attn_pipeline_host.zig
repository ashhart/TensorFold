//! The actual prompt launcher and resource admission through the CUDA ABI, without a GPU.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const prompt = qwen.prompt;
const t = std.testing;

const Mock = struct {
    var attr: [3]c_int = .{ 255, 0, 0 };
    var blocks: c_int = 1;
    var selected: usize = 0;
    var fp8 = false;
    var paged = false;
    var args: [16]u64 = @splat(0);
    var shared: u32 = 0;
    var grid: [3]u32 = undefined;
    fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        t.expectEqualStrings(prompt.symbols.pipeline, std.mem.span(name)) catch unreachable;
        out.* = @ptrFromInt(2);
        return 0;
    }
    fn set(f: cuda.abi.Function, name: cuda.abi.FunctionAttribute, bytes: c_int) callconv(.c) c_int {
        t.expectEqual(@as(usize, 2), @intFromPtr(f)) catch unreachable;
        t.expectEqual(cuda.abi.FunctionAttribute.max_dynamic_shared_size_bytes, name) catch unreachable;
        t.expectEqual(@as(c_int, 98304), bytes) catch unreachable;
        return 0;
    }
    fn attribute(out: *c_int, name: cuda.abi.FunctionAttribute, f: cuda.abi.Function) callconv(.c) c_int {
        const i: usize = switch (name) {
            .num_regs => 0,
            .shared_size_bytes => 1,
            .local_size_bytes => 2,
            else => unreachable,
        };
        out.* = if (@intFromPtr(f) == 2) attr[i] else if (i == 0) 255 else 0;
        return 0;
    }
    fn occupancy(out: *c_int, f: cuda.abi.Function, threads: c_int, bytes: usize) callconv(.c) c_int {
        t.expectEqual(@as(c_int, 256), threads) catch unreachable;
        t.expectEqual(@as(usize, 98304), bytes) catch unreachable;
        out.* = if (@intFromPtr(f) == 2) blocks else 1;
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, bytes: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        selected = @intFromPtr(f);
        shared = bytes;
        grid = .{ gx, gy, gz };
        t.expectEqual(@as(c_uint, 256), bx) catch unreachable;
        t.expectEqual(@as(c_uint, 1), by) catch unreachable;
        t.expectEqual(@as(c_uint, 1), bz) catch unreachable;
        const np: usize = if (fp8) 6 else 4;
        args = @splat(0);
        for (0..np) |i| args[i] = @as(*const u64, @ptrCast(@alignCast(params.?[i].?))).*;
        for (np..np + 7) |i| args[i] = @as(*const u32, @ptrCast(@alignCast(params.?[i].?))).*;
        if (paged) {
            for (np + 7..np + 9) |i| args[i] = @as(*const u64, @ptrCast(@alignCast(params.?[i].?))).*;
            args[np + 9] = @as(*const u32, @ptrCast(@alignCast(params.?[np + 9].?))).*;
        }
        return 0;
    }
};

fn function(d: *cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn driver() cuda.Driver {
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Mock.get;
    d.api.cuFuncSetAttribute = Mock.set;
    d.api.cuFuncGetAttribute = Mock.attribute;
    d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor = Mock.occupancy;
    d.api.cuLaunchKernel = Mock.launch;
    return d;
}
fn functions(d: *cuda.Driver) prompt.Fns {
    return .{ .bf16 = function(d, 1), .fp8 = function(d, 3), .bf16_whole = function(d, 4), .fp8_whole = function(d, 5), .bf16_p256 = function(d, 6), .bf16_p256_ids = function(d, 7), .chain = function(d, 8) };
}

test "Qwen attention pipeline: actual selection and arguments preserve default, FP8, pages and suffix rows" {
    var d = driver();
    var e: qwen.engine.Engine = undefined;
    e.g.q_heads = 24;
    e.g.kv_heads = 4;
    e.qn = 0x10000;
    e.y = 0x20000;
    e.stream = .{ .d = &d, .handle = null };
    var stage: [qwen.engine.Engine.segs_at + @sizeOf(qwen.cache.Seg)]u8 align(16) = @splat(0);
    e.staged.bytes = &stage;
    var sg = std.mem.zeroes(qwen.cache.Seg);
    sg.kv = 0x400000;
    sg.cap = 1 << 20;
    for ([_]usize{ 1, 63, 64, 65, 4096, 8192 }) |rows| for ([_]usize{ 0, 28672, 126976 }) |at| {
        for ([_]bool{ false, true }) |fp8| for ([_]u32{ 0, 48, 256 }) |ptok| {
            e.kv8 = fp8;
            Mock.fp8 = fp8;
            Mock.paged = ptok != 0;
            sg.pt = if (ptok != 0) 0x90000 else 0;
            sg.ptok = @intCast(ptok);
            sg.ps = 0x100000;
            sg.len = @intCast(at);
            @memcpy(stage[qwen.engine.Engine.segs_at..][0..@sizeOf(qwen.cache.Seg)], std.mem.asBytes(&sg));
            for ([_]usize{ 0, 3 }) |from| {
                e.prompt_attn = functions(&d);
                try prompt.runFrom(&e, 1, from, from + rows);
                const before = Mock.args;
                const id = Mock.selected;
                const shared = Mock.shared;
                const grid = Mock.grid;
                try prompt.enablePipeline(&e.prompt_attn, .{ .d = &d, .handle = null }, true);
                try prompt.runFrom(&e, 1, from, from + rows);
                try t.expectEqual(@as(usize, if (!fp8 and ptok == 0) 2 else id), Mock.selected);
                try t.expectEqualSlices(u64, &before, &Mock.args);
                try t.expectEqualSlices(u32, &grid, &Mock.grid);
                try t.expectEqual(shared, Mock.shared);
            }
        };
    };
}

test "Qwen attention pipeline: resource or occupancy regression and unsupported chip refuse before enabling" {
    var d = driver();
    const m: cuda.Module = .{ .d = &d, .handle = null };
    var f = functions(&d);
    try t.expectError(error.QwenAttentionPipelineNeedsGb10, prompt.enablePipeline(&f, m, false));
    for ([_][3]c_int{ .{ 256, 0, 0 }, .{ 255, 1024, 0 }, .{ 255, 0, 4 } }) |bad| {
        Mock.attr = bad;
        try t.expectError(error.QwenAttentionPipelineResources, prompt.enablePipeline(&f, m, true));
        try t.expect(f.pipeline == null);
    }
    Mock.attr = .{ 255, 0, 0 };
    Mock.blocks = 0;
    try t.expectError(error.QwenAttentionPipelineOccupancy, prompt.enablePipeline(&f, m, true));
    try t.expect(f.pipeline == null);
    Mock.blocks = 1;
}

test "Qwen attention pipeline: registered switch keeps defaults and accepts only strict flags" {
    const env = @import("sys").environment;
    const profile = @import("core").profile;
    const name = "TF_CUDA_QWEN_ATTN_PIPELINE";
    for ([_]?[]const u8{ null, "0" }) |v| try t.expect(!try env.value(bool, name, v, false));
    try t.expect(try env.value(bool, name, "1", false));
    for ([_][]const u8{ "", " ", "on", "true", "2", " 1", "1 " }) |v| try t.expectError(error.InvalidEnvironment, env.value(bool, name, v, false));
    const row = profile.entry(.TF_CUDA_QWEN_ATTN_PIPELINE);
    try t.expect(row.applies("qwen3_5", "cuda") and row.applies("qwen3_5_moe", "cuda"));
    try t.expect(!row.applies("glm5_next", "cuda"));
}
