//! Wrapper ABI, logical positions and per-row routing. This does not emulate CUDA numerical arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const media = @import("cuda_media.zig");
const wt = @import("cuda_prompt_wt.zig");
const Capture = struct {
    var calls: usize = 0;
    var values: [8][22]u64 = undefined;
    var configs: [8][7]u32 = undefined;
    var fns: [8]usize = undefined;
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, s: cuda.abi.Stream, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) c_int {
        std.debug.assert(extra == null and s == @as(cuda.abi.Stream, @ptrFromInt(99)));
        const id = @intFromPtr(f);
        // the paged forms (4, 5) take the flat ones' arguments and a Pg last, read here by its table's address
        const count: usize = switch (id) {
            1 => 7,
            2 => 21,
            4 => 22,
            5 => 14,
            else => 13,
        };
        for (0..count) |i| {
            const wide = switch (id) {
                1 => i < 3,
                2, 4 => i < 11 or i == 18 or i == 21,
                else => i < 5 or i == 10 or i == 13,
            };
            values[calls][i] = if (wide) @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).* else @as(*const u32, @ptrCast(@alignCast(args.?[i].?))).*;
        }
        configs[calls] = .{ gx, gy, gz, bx, by, bz, sm };
        fns[calls] = id;
        calls += 1;
        return 0;
    }
};
fn view(d: *const cuda.Driver, features: u64) media.View {
    return .{ .prep = .{ .d = d, .handle = @ptrFromInt(2) }, .pool = .{ .d = d, .handle = @ptrFromInt(3) }, .prep_paged = .{ .d = d, .handle = @ptrFromInt(4) }, .pool_paged = .{ .d = d, .handle = @ptrFromInt(5) }, .image = .{ .features = features, .src = 2000, .rope = 3000, .n = 20, .delta = -2, .width = 2560, .splice_fn = .{ .d = d, .handle = @ptrFromInt(1) } } };
}
test "the splice, attention and pooled indexer launches get every pointer, scalar and grid, mixed rows too" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(99) };
    Capture.calls = 0;
    const m = view(&d, 1000);
    try m.image.inject(s, 4000, 2, 3, 4);
    try std.testing.expectEqualSlices(u64, &.{ 1000, 2000, 4000, 2, 20, 2560, 4 }, Capture.values[0][0..7]);
    try std.testing.expectEqual([7]u32{ 3, 40, 1, 256, 1, 1, 0 }, Capture.configs[0]);
    try m.attn(s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1e-6, 3, .{});
    const eps: u32 = @bitCast(@as(f32, 1e-6));
    const delta: u32 = @bitCast(@as(i32, -2));
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, eps, wt.proj_n, wt.q_heads, wt.kv_heads, wt.index_heads, wt.index_dim, wt.half, 3000, 20, delta }, Capture.values[1][0..21]);
    try std.testing.expectEqual([7]u32{ 3, wt.q_heads + wt.kv_heads + wt.index_heads + 1, 1, 256, 1, 1, 0 }, Capture.configs[1]);
    try m.pooled(s, 1, 2, 3, 4, 5, 1e-6, 8, .{});
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3, 4, 5, eps, 8, wt.index_dim, wt.half, wt.index_ratio, 3000, 20, delta }, Capture.values[2][0..13]);
    try std.testing.expectEqual([7]u32{ 8 / wt.index_ratio + 2, 1, 1, 128, 1, 1, 0 }, Capture.configs[2]);
    try std.testing.expectEqual(@as(usize, 3), Capture.fns[2]);
    const Row = struct { seq: struct { media: ?media.View }, pos: usize, rows: usize = 1 };

    const rows = [_]Row{ .{ .seq = .{ .media = m }, .pos = 2 }, .{ .seq = .{ .media = null }, .pos = 2 }, .{ .seq = .{ .media = view(&d, 1100) }, .pos = 5 } };
    try media.inject(null, &rows, s, 9000, 0, 3);
    try std.testing.expectEqualSlices(u64, &.{ 1000, 2000, 9000, 2, 20, 2560, 4 }, Capture.values[3][0..7]);
    try std.testing.expectEqualSlices(u64, &.{ 1100, 2000, 9000 + 2 * wt.wide * 2, 5, 20, 2560, 4 }, Capture.values[4][0..7]);
    try media.injectNext(m, s, 8000, 1, 2);
    try std.testing.expectEqual(@as(u64, 2), Capture.values[5][3]);
    try media.injectNext(m, s, 8000, 19, 1);
    try std.testing.expectEqual(@as(usize, 6), Capture.calls);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 1, 1, 1 }, Capture.fns[0..6]);
}

test "a paged stream's media rows take the paged launches: the same arguments, its pages last" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(99) };
    Capture.calls = 0;
    const m = view(&d, 1000);
    const pg: @import("cuda_glue.zig").Pg = .{ .pt = 7000, .ps = 64 * 30784 / 2, .ptok = 64 };
    try m.attn(s, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1e-6, 3, pg);
    try m.pooled(s, 1, 2, 3, 4, 5, 1e-6, 8, pg);
    const eps: u32 = @bitCast(@as(f32, 1e-6));
    const delta: u32 = @bitCast(@as(i32, -2));
    try std.testing.expectEqualSlices(usize, &.{ 4, 5 }, Capture.fns[0..2]);
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, eps, wt.proj_n, wt.q_heads, wt.kv_heads, wt.index_heads, wt.index_dim, wt.half, 3000, 20, delta, 7000 }, &Capture.values[0]);
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3, 4, 5, eps, 8, wt.index_dim, wt.half, wt.index_ratio, 3000, 20, delta, 7000 }, Capture.values[1][0..14]);
    try std.testing.expectEqual([7]u32{ 3, wt.q_heads + wt.kv_heads + wt.index_heads + 1, 1, 256, 1, 1, 0 }, Capture.configs[0]);
    try std.testing.expectEqual([7]u32{ 8 / wt.index_ratio + 2, 1, 1, 128, 1, 1, 0 }, Capture.configs[1]);
}
