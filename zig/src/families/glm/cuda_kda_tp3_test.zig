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
        const row = id == 5 or id == 7 or id == 8; // the prompt kernels' argument order
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
    return .{ .proj = .{ .bf16 = 0x100000000 }, .fb = 0x200000000, .gb = 0x300000000, .o = .{ .bf16 = 0x400000000 }, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = heads, .dims = dims };
}
test "GLM TP3: real KDA projections accept uneven shares and retain aligned tiled dispatch" {
    const c = Config{};
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(Capture.launch);
    d.func_name = null;
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
            const weights = [_]u64{ l.proj.bf16, l.fb, l.gb };
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

test "GLM uneven worlds: a padded KDA stack keeps the B16 kernels, staged on its 16-byte rows; glm_mm only if tiled" {
    const c = Config{};
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Capture.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var kernels: mm.Kernels = .{ .by_rows = @splat(function(&d, 1)), .prompt = function(&d, 2), .reduce = function(&d, 3), .pdl = false, .packed_rows = undefined, .tile = undefined, .tile_packed = undefined };
    var raw: glue.Kernels = undefined;
    raw.bmm_decode = function(&d, 4);
    raw.bmm_rows = function(&d, 5);
    raw.slice_sum = function(&d, 6);
    raw.bmm_tile = function(&d, 7);
    raw.bmm_tile1 = function(&d, 8);
    inline for (.{ "bmm_d32s1", "bmm_d32s2", "bmm_d32s3", "bmm_d64s2" }, 9..) |name, id| @field(raw, name) = function(&d, id);
    const scratch: kda.Scratch = .{ .p = 0x500000000, .ka = 0x600000000, .kg = 0x700000000, .mm_part = 0x800000000, .saves = undefined, .wide = undefined };
    defer kda.pad_to = 0;
    defer kda.tile_pad = false;
    for ([_]usize{ 3, 5 }) |world| for (0..world) |rank| for ([_]bool{ false, true }) |tiled| {
        const heads = (try shard.Share.of(&c, rank, world)).kda.len();
        kda.pad_to = if (tiled) kda.padWidth(c.kda_heads, world) else kda.alignedWidth(heads); // as Engine.open
        kda.tile_pad = tiled;
        const l = layer(heads, c.hidden);
        const width = l.matWidth();
        const common: usize = if (world == 3) 8728 else 5264;
        try std.testing.expectEqual(if (tiled) common else std.mem.alignForward(usize, l.width(), 8), width);
        if (world == 3 and !tiled) try std.testing.expectEqual(@as(usize, if (rank == 0) 8728 else 8344), width);
        try std.testing.expect(width >= l.width() and width <= common and l.width() % 8 != 0);
        try std.testing.expectEqual(tiled, l.tiledInput());
        const n = l.heads * 128;
        const view: usize = if (n == 2816 or n == 2688) 11 else 4; // Triton's staged tile where it has the shape
        const counts = if (tiled) [_]usize{ 1, 17, 64, 128, 2066 } else [_]usize{ 1, 17, 127, 128, 2066 }; // glm_mm: 64 a launch
        for (counts) |rows| {
            Capture.count = 0;
            try kda.project(.{ .glue = &raw, .mm = &kernels, .prompt = rows >= 128 }, stream, l, scratch, 0x900000000, c.hidden, rows);
            try std.testing.expectEqual(@as(usize, 3), Capture.count);
            const stack = Capture.calls[0];
            try std.testing.expectEqual(l.proj.bf16, stack.w);
            try std.testing.expectEqual(@as(u64, width), stack.n);
            try std.testing.expectEqual(@as(u64, rows), stack.rows);
            try std.testing.expectEqual(@as(usize, if (tiled) (if (rows < 128) 1 else 2) else if (rows < 128) 4 else 7), stack.id);
            for (Capture.calls[1..3], [_]u64{ l.fb, l.gb }, [_]usize{ 0, 128 }) |call, w, off| {
                try std.testing.expectEqual(w, call.w);
                try std.testing.expectEqual(scratch.p + (3 * n + off) * 2, call.x);
                try std.testing.expectEqual(@as(u64, width), call.ldx);
                try std.testing.expectEqual(@as(u64, n), call.n);
                try std.testing.expectEqual(@as(usize, if (tiled) (if (rows < 128) 1 else 2) else if (rows < 128) view else 8), call.id);
            }
        }
    };
    for ([_]usize{ 1, 2, 4, 8 }) |world| { // even stacks: the engine's width is their own, and glm_mm reads them as before
        const l = layer((try shard.Share.of(&c, 0, world)).kda.len(), c.hidden);
        kda.pad_to = kda.alignedWidth(l.heads);
        kda.tile_pad = false;
        try std.testing.expectEqual(l.width(), l.matWidth());
        try std.testing.expectEqual(kda.padWidth(c.kda_heads, world), l.matWidth());
        try std.testing.expect(l.tiledInput());
    }
}
