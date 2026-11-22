//! Record the real hcPre launch boundary and exercise the contract fixture without CUDA.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const cell = @import("tests/cuda/contract_glm_hc_shared.zig");
const t = std.testing;
const Rec = struct {
    var ids: [2]usize = undefined;
    var grids: [2]usize = undefined;
    var blocks: [2]usize = undefined;
    var used: usize = 0;
    var rows_arg: i64 = 0;
    var wide: i64 = 0;
    var nb: c_int = 0;
    fn arg(comptime T: type, p: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(p.?[i].?))).*;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        ids[used] = @intFromPtr(f);
        grids[used] = gx;
        blocks[used] = bx;
        if (used == 0) {
            std.debug.assert(gy == 16);
            wide = arg(i64, args, 3);
            nb = arg(c_int, args, 4);
            rows_arg = if (ids[0] == 1) -1 else arg(i64, args, 5);
        }
        used += 1;
        return 0;
    }
    fn function(d: *cuda.Driver, id: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
};
fn dispatch(rows: usize, on: bool) !void {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.launch;
    d.api.cuLaunchKernelEx = glm.pdl.forwardEx(Rec.launch);
    var k: glm.glue.Kernels = undefined;
    k.hc_partial8 = Rec.function(&d, 1);
    k.hc_partial8_shared = Rec.function(&d, 2);
    k.hc_partial = Rec.function(&d, 3);
    k.hc_finish = Rec.function(&d, 4);
    k.hc_finish_f32w = Rec.function(&d, 5);
    Rec.used = 0;
    try glm.glue.hcPre(.{ .glue = &k, .hc_shared_fn = on }, .{ .d = &d, .handle = null }, 1, 2, 3, 4, .{ .bf16 = 5 }, 6, 7, 8, 9, 10, 1e-6, 1e-6, 20, rows, 4096, 4);
    const shared = rows < 64 and on;
    try t.expectEqual(@as(usize, 2), Rec.used);
    try t.expectEqual(@as(usize, if (rows >= 64) 3 else if (shared) 2 else 1), Rec.ids[0]);
    try t.expectEqual(@as(usize, 4), Rec.ids[1]);
    try t.expectEqual(if (shared or rows >= 64) (rows + 3) / 4 else rows, Rec.grids[0]);
    try t.expectEqual(rows, Rec.grids[1]);
    try t.expectEqual(@as(usize, if (rows >= 64) 128 else 1024), Rec.blocks[0]);
    try t.expectEqual(@as(i64, 16384), Rec.wide);
    try t.expectEqual(@as(c_int, 16), Rec.nb);
    try t.expectEqual(@as(i64, if (shared or rows >= 64) @intCast(rows) else -1), Rec.rows_arg);
}
test "GLM shared HC: every served decode width selects shared only when enabled" {
    for (cell.rows) |rows| for ([_]bool{ false, true }) |on| try dispatch(rows, on);
}
test "GLM shared HC: prompt and ragged prompt retain the original 128-thread kernel" {
    for ([_]usize{ 64, 65, 2048 }) |rows| for ([_]bool{ false, true }) |on| try dispatch(rows, on);
}
test "GLM shared HC: contract covers every requested width and byte mutation is detected" {
    const spec = cell.spec();
    try t.expectEqualStrings("tf_glm_hc_partial8", spec.replaces.?);
    try t.expectEqual(@as(usize, 7), spec.served.len);
    for (cell.rows, spec.served) |rows, shape| {
        try t.expectEqual(rows, shape.rows);
        try t.expectEqual(@as(usize, 16384), shape.k);
    }
    var a = [_]u8{ 0xff, 0x00, 0x7f, 0x80 };
    var b = a;
    try t.expect(cell.sameBytes(&a, &b));
    b[0] ^= 1;
    try t.expect(!cell.sameBytes(&a, &b));
}
test "GLM shared HC: fixture non-finites are planted in both BF16 inputs and replay exactly" {
    const x = try t.allocator.alloc(u16, 16384);
    defer t.allocator.free(x);
    const w = try t.allocator.alloc(u16, 24 * 16384);
    defer t.allocator.free(w);
    cell.fillInputs(x, w, .nonfinite, 51);
    for ([_]u16{ 0x7fc1, 0xffc2, 0x7f80, 0xff80 }, 0..) |h, i| {
        try t.expectEqual(h, x[x.len - 1 - i * 128]);
        try t.expectEqual(h, w[(i * 5 + 1) * 16384 + 17 + i * 128]);
    }
    const bytes = try t.allocator.dupe(u16, x);
    defer t.allocator.free(bytes);
    cell.fillInputs(x, w, .nonfinite, 51);
    try t.expectEqualSlices(u16, bytes, x);
}
test "GLM shared HC: new resource records have no fabricated compiler measurements" {
    const records = @import("core").contract.registry.landed_cuda;
    for (records) |r| if (std.mem.eql(u8, r.launch.symbol, "tf_glm_hc_partial8_shared")) {
        try t.expectEqual(@as(u32, 1024), r.launch.threads);
        for (r.usage) |u| try t.expectEqual(@as(u32, 0), u.regs);
        return;
    };
    return error.TestUnexpectedResult;
}

test "GLM shared HC: finite and forced nonfinite replacements both cover all widths, with a red byte control" {
    var pair = cell.Pair.init(undefined, undefined, undefined);
    try t.expectEqual(@as(usize, 3), pair.cells.len);
    try t.expect(!pair.cells[0].nonfinite and pair.cells[1].nonfinite);
    try t.expect(pair.cells[2].corrupt);
    for (pair.cells[0..2]) |*x| {
        const k = x.kernel();
        try t.expectEqual(@as(usize, 7), k.spec.served.len);
        try t.expect(k.control == null and k.spec.replaces != null);
    }
    try t.expect(pair.cells[2].kernel().control == .shapes);
}
