//! The production backward launchers carry each family's arithmetic through single and packed passes.
const std = @import("std");
const gt = @import("gdn_train.zig");
const cuda = @import("root.zig");
const t = std.testing;
const Mock = struct {
    var mode: gt.Mode = .flash_next;
    var seen: u8 = 0;
    var bad: usize = 0;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        const at: usize = switch (id) {
            1 => 10,
            2 => 8,
            3 => 6,
            4 => 13,
            5 => 4,
            6 => 7,
            7 => 5,
            else => return 0,
        };
        const s = @as(*const gt.Shape, @ptrCast(@alignCast(args.?[at].?))).*;
        if (s.mode != mode or s.nk != 1 or s.nv != 2 or s.rows < 1 or s.rows > 8) bad += 1;
        seen |= @as(u8, 1) << @intCast(id - 1);
        return 0;
    }
};
test "dense gradient: GDN modes reach every real replay launch for single and packed sequences" {
    try t.expectEqual(@as(usize, 24), @sizeOf(gt.Shape));
    try t.expectEqual(@as(usize, 20), @offsetOf(gt.Shape, "mode"));
    try t.expectEqual(gt.Mode.flash_next, gt.Shape.init(8, 1, 2).mode);
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Mock.launch;
    var k: gt.Kernels = undefined;
    for (&k.fns, 0..) |*f, i| f.* = .{ .d = &d, .handle = @ptrFromInt(i + 1) };
    for (&k.train_fns, 0..) |*f, i| f.* = .{ .d = &d, .handle = @ptrFromInt(i + 20) };
    const o: gt.Ops = .{ .k = &k, .s = .{ .d = &d, .handle = null } };
    var layer: gt.Layer = .{ .nk = 1, .nv = 2, .hidden = 64, .eps = 1e-6, .conv_w = 1, .a_log = 2, .dt_bias = 3, .norm_w = 4, .in_proj = .{ .w = 5, .s = 6, .b = 7, .n = 772, .k = 64 }, .out_proj = .{ .w = 8, .s = 9, .b = 10, .n = 64, .k = 256 } };
    const sc = gt.Scratch.carve(4096, 8, 1, 2, 64);
    const wide = gt.Wide.carve(1 << 24, 8, 1, 2);
    for ([_]gt.Mode{ .flash_next, .qwen27 }) |mode| {
        layer.mode = mode;
        Mock.mode = mode;
        Mock.seen = 0;
        Mock.bad = 0;
        try o.backward(layer, sc, 100, null, null, 200, 300, 8, false);
        try t.expectEqual(@as(u8, 0x7f), Mock.seen);
        try t.expectEqual(@as(usize, 0), Mock.bad);
        Mock.seen = 0;
        Mock.bad = 0;
        try o.backwardAll(layer, sc, wide, &.{ .{ .p = 100, .rows = 5, .at = 0 }, .{ .p = 500, .rows = 3, .at = 5 } }, 200, 300, 8, false);
        try t.expectEqual(@as(u8, 0x7f), Mock.seen);
        try t.expectEqual(@as(usize, 0), Mock.bad);
    }
}
