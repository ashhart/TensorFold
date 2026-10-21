//! Real KDA projection dispatch with a recording CUDA driver, including uneven head shares.
const std = @import("std");
const cuda = @import("cuda");
const kda = @import("cuda_kda_block.zig");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");
const shard = @import("cuda_shard.zig");
const Config = @import("config.zig").Config;
const Capture = struct {
    const Call = struct { id: usize, x: u64, w: u64, ldx: u64, n: u64, k: u64, rows: u64 };
    var calls: [32]Call = undefined;
    var count: usize = 0;
    fn arg(args: ?[*]?*anyopaque, index: usize, wide: bool) u64 {
        const p: [*]const u8 = @ptrCast(args.?[index].?);
        return if (wide) std.mem.readInt(u64, p[0..8], .little) else std.mem.readInt(u32, p[0..4], .little);
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id == 3 or id == 6) return cuda.abi.success;
        const tiled = id == 1 or id == 2;
        const row = id == 5;
        calls[count] = .{
            .id = id,
            .x = arg(args, 0, true),
            .w = arg(args, if (tiled) 2 else 1, true),
            .ldx = arg(args, if (tiled) 1 else if (row) 4 else 5, true),
            .n = arg(args, if (tiled) 8 else if (row) 5 else 6, !tiled),
            .k = arg(args, if (tiled) 9 else if (row) 6 else 7, !tiled),
            .rows = arg(args, if (tiled) 7 else if (row) 3 else 4, !tiled),
        };
        count += 1;
        return cuda.abi.success;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn layer(heads: usize, dims: usize) kda.Layer {
    return .{ .proj = 0x100000000, .fb = 0x200000000, .gb = 0x300000000, .o = 0x400000000, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = heads, .dims = dims };
}
test "GLM TP3: real KDA projections accept uneven shares and retain aligned tiled dispatch" {
    const c = Config{};
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var kernels: mm.Kernels = .{ .by_rows = @splat(function(&d, 1)), .prompt = function(&d, 2), .reduce = function(&d, 3), .pdl = false, .packed_rows = undefined, .tile = undefined, .tile_packed = undefined };
    var raw: glue.Kernels = undefined;
    raw.bmm_decode = function(&d, 4);
    raw.bmm_rows = function(&d, 5);
    raw.slice_sum = function(&d, 6);
    const scratch: kda.Scratch = .{ .p = 0x500000000, .ka = 0x600000000, .kg = 0x700000000, .mm_part = 0x800000000, .saves = undefined, .wide = undefined };
    for ([_]usize{ 1, 2, 3, 4, 5, 8 }) |world| for (0..world) |rank| {
        const share = try shard.Share.of(&c, rank, world);
        const l = layer(share.kda.len(), c.hidden);
        const width = l.width();
        const aligned = l.heads % 8 == 0;
        if (world == 3) try std.testing.expectEqual(@as(usize, if (rank == 0) 8726 else 8341), width);
        for ([_]usize{ 1, 17, 64, 65, 127, 128, 2066 }) |rows| for ([_]bool{ false, true }) |prompt| {
            Capture.count = 0;
            try kda.project(.{ .glue = &raw, .mm = &kernels, .prompt = prompt }, stream, l, scratch, 0x900000000, c.hidden, rows);
            const weights = [_]u64{ l.proj, l.fb, l.gb };
            const xs = [_]u64{ 0x900000000, scratch.p + 3 * l.heads * 128 * 2, scratch.p + (3 * l.heads * 128 + 128) * 2 };
            const strides = [_]usize{ c.hidden, width, width };
            const ns = [_]usize{ width, l.heads * 128, l.heads * 128 };
            const ks = [_]usize{ c.hidden, 128, 128 };
            const per: usize = if (aligned and !prompt and rows > 64 and rows < 128) 2 else 1;
            try std.testing.expectEqual(3 * per, Capture.count);
            for (weights, xs, strides, ns, ks) |w, x, stride, n, k| {
                var done: usize = 0;
                var found: usize = 0;
                for (Capture.calls[0..Capture.count]) |call| if (call.w == w) {
                    try std.testing.expectEqual(aligned, call.id == 1 or call.id == 2);
                    try std.testing.expectEqual(x + done * stride * 2, call.x);
                    try std.testing.expectEqual(@as(u64, stride), call.ldx);
                    try std.testing.expectEqual(@as(u64, n), call.n);
                    try std.testing.expectEqual(@as(u64, k), call.k);
                    done += @intCast(call.rows);
                    found += 1;
                };
                try std.testing.expectEqual(rows, done);
                try std.testing.expectEqual(per, found);
            }
        };
    };
}
