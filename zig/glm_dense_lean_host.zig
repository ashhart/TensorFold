//! Actual GLM dense prompt launch descriptors and the per-output fragment mapping, without a GPU.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const t = std.testing;
const Rec = struct {
    var id: usize = 0;
    var grid: u32 = 0;
    var threads: u32 = 0;
    var shared: u32 = 0;
    var args: [15]u64 = undefined;
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        id = @intFromPtr(f);
        grid = gx;
        threads = bx;
        shared = sm;
        for ("888484444448848", 0..) |size, i| args[i] = if (size == '8') @as(*const u64, @ptrCast(@alignCast(p.?[i].?))).* else @as(*const u32, @ptrCast(@alignCast(p.?[i].?))).*;
        return 0;
    }
};
test "GLM dense lean: enabled actual MLP helper selects two-block geometry with identical argument bytes" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var p: cuda.nvfp4.Prompt = undefined;
    p.sms = 48;
    p.lean = .{ .d = &d, .handle = @ptrFromInt(10) };
    p.lean_f32 = .{ .d = &d, .handle = @ptrFromInt(11) };
    for (&p.dense_lean, 0..) |*f, i| f.* = .{ .d = &d, .handle = @ptrFromInt(20 + i) };
    var k: glm.mlp.Kernels = .{ .set = undefined, .linear = .{ .nvfp4_prompt = &p }, .copy_rows = undefined };
    for ([_]usize{ 1, 17, 63, 64, 65, 127, 128, 129, 2048, 8192, 32768 }) |rows| for ([_]bool{ false, true }) |wide| {
        const n: usize = if (wide) 4096 else 12288;
        const width: usize = if (wide) 12288 else 4096;
        const w = cuda.nvfp4.weight(0x1000, 0x2000, 0.0123, n, width);
        k.dense_lean = false;
        try glm.mlp.mm(k, s, w, 0x3000, width + 8, rows, 0x4000, wide, true);
        const old = Rec.args;
        try t.expectEqual(@as(usize, if (wide) 11 else 10), Rec.id);
        try t.expectEqual(@as(u32, @intCast(@min(48, ((rows + 127) / 128) * ((n + 255) / 256)))), Rec.grid);
        try t.expectEqual(@as(u32, 76848), Rec.shared);
        k.dense_lean = true;
        try glm.mlp.mm(k, s, w, 0x3000, width + 8, rows, 0x4000, wide, true);
        try t.expectEqual(@as(usize, if (wide) 21 else 20), Rec.id);
        try t.expectEqual(@as(u32, @intCast(@min(96, ((rows + 63) / 64) * ((n + 127) / 128)))), Rec.grid);
        try t.expectEqual(@as(u32, 256), Rec.threads);
        try t.expectEqual(@as(u32, 38448), Rec.shared);
        // Only the M-row cache band changes with tile height; it has no arithmetic role.
        for (old, Rec.args, 0..) |a, b, i| if (i != 10) try t.expectEqual(a, b);
    };
}
fn owner(row: usize, col: usize, bm: usize, bn: usize, wn: usize) [4]usize {
    const wm = (row % bm) / (bm / (8 / wn));
    const warp_n = (col % bn) / (bn / wn);
    const lane = ((row % 16) % 8) * 4 + (col % 8) / 2;
    return .{ wm, warp_n, lane, (row % 16) / 8 * 2 + col % 2 };
}
test "GLM dense lean: every output keeps its MMA lane accumulator component and absolute K order" {
    for (0..257) |r| for (0..513) |c| {
        const old = owner(r, c, 128, 256, 8);
        const new = owner(r, c, 64, 128, 4);
        try t.expectEqual(old[2], new[2]);
        try t.expectEqual(old[3], new[3]);
    };
    try t.expectEqual(@as(u32, 38448), cuda.nvfp4.Prompt.denseLeanSharedBytes());
    const source = @embedFile("kernels/cuda/prompt16_lean.cuh");
    try t.expect(std.mem.indexOf(u8, source, "for (int g = 0; g < KG; ++g, ++used)") != null);
    try t.expect(std.mem.indexOf(u8, source, "for (int kt = 0; kt < GS / 16; ++kt)") != null);
    try t.expect(std.mem.indexOf(u8, source, "mma(acc[i][j], a[i], b0, b1)") != null);
}
