//! Inkling 2's glue launchers without a GPU: Mini's and Small's shapes launch, the rest are refused by name.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("ops.zig");
const t = std.testing;

const Fake = struct {
    var count: usize = 0;
    var grid: [3]u32 = undefined;
    fn launch(_: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        grid = .{ gx, gy, gz };
        count += 1;
        return 0;
    }
};

fn kernels(d: *const cuda.Driver) ops.Kernels {
    var k: ops.Kernels = undefined;
    k.mod = .{ .d = d, .handle = null };
    k.rows_mod = .{ .d = d, .handle = null };
    const info = @typeInfo(ops.Kernels).@"struct";
    inline for (info.field_names, info.field_types, 1..) |name, ty, i| {
        if (ty == cuda.Function) @field(k, name) = .{ .d = d, .handle = @ptrFromInt(i) };
    }
    return k;
}

test "inkling2 ops: Mini's widths launch (hidden 2,304, 8 key heads, 128 experts) as Small's do" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const k = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    Fake.count = 0;
    for ([_]i32{ 2304, 4608 }) |c| try ops.mix(&k, s, .{ .partial = 0, .query = 0, .out = 0, .n_blocks = 8, .partial_bf16 = 1, .nq = 3, .rows = 1, .c = c, .eps = 1e-6 });
    for ([_][2]usize{ .{ 128, 8 }, .{ 384, 8 } }) |e| try ops.topk(&k, s, 0, 0, 0, 0, 16, e[0], e[1], 8);
    for ([_][2]i32{ .{ 512, 4 }, .{ 1024, 8 } }) |w| try ops.conv(&k, s, .{ .x = 0, .state = 0, .w = 0, .dyn = 1, .heads = w[1], .c = w[0], .rows = 4, .mode = 0 }, 1);
    try t.expectEqual(@as(usize, 6), Fake.count);
    try t.expectEqual([3]u32{ 8, 1, 1 }, Fake.grid); // Mini's k/v conv: 1,024 columns in blocks of 128
}

test "inkling2 ops: widths, routers and dynamic taps the kernels cannot hold are refused by name, nothing launched" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const k = kernels(&d);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    Fake.count = 0;
    for ([_]i32{ 0, 2300, 5248, -128 }) |c| try t.expectError(error.Inkling2MixWidth, ops.mix(&k, s, .{ .partial = 0, .query = 0, .out = 0, .n_blocks = 1, .partial_bf16 = 1, .nq = 1, .rows = 1, .c = c, .eps = 1e-6 }));
    for ([_][2]usize{ .{ 4097, 8 }, .{ 128, 33 }, .{ 128, 0 }, .{ 4, 8 }, .{ 0, 0 } }) |e| try t.expectError(error.Inkling2RouterShape, ops.topk(&k, s, 0, 0, 0, 0, 1, e[0], e[1], 8));
    for ([_][2]i32{ .{ 1024, 0 }, .{ 1024, 3 }, .{ 1020, 8 } }) |w| try t.expectError(error.Inkling2ConvHeads, ops.conv(&k, s, .{ .x = 0, .state = 0, .w = 0, .dyn = 1, .heads = w[1], .c = w[0], .rows = 1, .mode = 0 }, 1));
    try t.expectEqual(@as(usize, 0), Fake.count);
    // static taps read no heads: any width launches
    try ops.conv(&k, s, .{ .x = 0, .state = 0, .w = 0, .heads = 3, .c = 1000, .rows = 1, .mode = 0 }, 1);
    try t.expectEqual(@as(usize, 1), Fake.count);
}
