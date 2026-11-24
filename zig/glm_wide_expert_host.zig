//! The actual GLM expert dispatch at a recording CUDA boundary, with no device arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const cell = @import("tests/cuda/contract_glm_wide_expert.zig");
const t = std.testing;
const Rec = struct {
    var id: usize = 0;
    var grid: u32 = 0;
    var threads: u32 = 0;
    var shared: u32 = 0;
    var values: [14]u64 = undefined;
    fn arg(comptime T: type, p: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(p.?[i].?))).*;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, smem: c_uint, _: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        id = @intFromPtr(f);
        grid = gx;
        threads = bx;
        shared = smem;
        for (0..14) |i| values[i] = switch (i) {
            0, 3, 4, 7, 8, 9, 10 => arg(u64, p, i),
            12 => @as(u32, @bitCast(arg(f32, p, i))),
            13 => @as(u32, @bitCast(arg(i32, p, i))),
            else => @intCast(arg(i32, p, i)),
        };
        return 0;
    }
    fn function(d: *cuda.Driver, n: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(n) };
    }
};
fn kernels(d: *cuda.Driver) glm.experts.Kernels {
    var k: glm.experts.Kernels = undefined;
    k.b32 = null;
    k.wide_tile = false;
    k.nvfp4.warps = 4;
    for (&k.nvfp4.fns, 0..) |*f, i| f.* = Rec.function(d, i + 1);
    for (&k.nvfp4.fact_fns, 0..) |*f, i| f.* = Rec.function(d, 20 + i);
    for (&k.nvfp4.prompt_fns, 0..) |*f, i| f.* = Rec.function(d, 30 + i);
    k.nvfp4.resident = @splat(100000);
    return k;
}
fn dispatch(rows: usize, on: bool, fact: bool, prompt: bool, which: glm.experts.Which) !void {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.launch;
    var k = kernels(&d);
    k.wide_tile = on;
    const plan: glm.experts.Plan = .{ .route = .{ .items = 60, .counts = 70, .members = 80, .rank = 90, .hist = 100 }, .slots = 2, .experts = 2, .prompt = prompt, .fact = fact };
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 20, .down = 30, .up_scale = 40, .down_scale = 50, .width = 512, .dims = 4096, .experts = 1 };
    try glm.experts.run(&k, .{ .d = &d, .handle = null }, which, layer, plan, 10, if (which == .gate_up) 4096 else 512, rows, 110, 1);
    const wide = on and rows > 16 and fact and !prompt and which == .down_f32;
    const want: usize = if (prompt) (if (which == .gate_up) 31 else 32) else if (fact and which == .gate_up) 20 else if (wide) 21 else 2;
    try t.expectEqual(want, Rec.id);
    if (which == .down_f32 and !prompt) {
        try t.expectEqual([14]u64{ 10, 512, 0, 30, 50, 16, 128, 60, 70, 80, 110, 4096, 0, 1 }, Rec.values);
        try t.expectEqual(@as(u32, if (wide) 256 else 128), Rec.threads);
        try t.expectEqual(@as(u32, if (wide) 34816 else 0), Rec.shared);
        const items = glm.experts.maxItems(rows * 2, 2, if (wide) 64 else 16);
        try t.expectEqual(@as(u32, @intCast(if (wide) items * 16 else (items * 128 + 3) / 4)), Rec.grid);
    }
}
test "GLM wide expert: sixteen keeps old down, seventeen24and32 take matching fact down only when enabled" {
    for ([_]usize{ 16, 17, 24, 32 }) |rows| for ([_]bool{ false, true }) |on| try dispatch(rows, on, true, false, .down_f32);
}
test "GLM wide expert: classic down, fact gateup and prompt keep their original dispatch" {
    for ([_]usize{ 16, 17, 24, 32 }) |rows| for ([_]bool{ false, true }) |on| {
        try dispatch(rows, on, false, false, .down_f32);
        try dispatch(rows, on, true, false, .gate_up);
        try dispatch(rows, on, true, true, .down);
    };
}
test "GLM wide expert: byte cells include finite and forced nonfinite everywidth plus corruption control" {
    const spec = cell.spec();
    try t.expectEqual(@as(usize, 8), spec.served.len);
    var suite = cell.Suite{ .ks = undefined, .cells = undefined };
    for (&suite.cells, 0..) |*p, i| {
        p.* = .{ .rig = undefined, .ks = &suite.ks, .nonfinite = i == 1, .corrupt = i == 2 };
    }
    for (suite.cells[0..2]) |*p| {
        const k = p.kernel();
        try t.expect(k.control == null and k.spec.replaces != null);
    }
    try t.expect(suite.cells[1].nonfinite and suite.cells[2].kernel().control == .shapes);
}
