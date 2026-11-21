//! Real lane dispatches keep another family's cluster schedule when Flash Next enables its own fused split.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
const qmmf = cuda.qmmf;

const Seen = struct {
    const Dispatch = struct { id: usize, grid_z: u32, cluster_z: u32 = 0, pdl: bool = false, slices: i32 };
    var last: Dispatch = undefined;

    fn plain(f: abi.Function, _: c_uint, _: c_uint, gz: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        last = .{ .id = @intFromPtr(f), .grid_z = gz, .slices = @as(*const i32, @ptrCast(@alignCast(args.?[9].?))).* };
        return abi.success;
    }

    fn extended(c: *const abi.LaunchConfig, f: abi.Function, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
        _ = plain(f, c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, c.stream, args, extra);
        if (c.attrs) |attrs| for (attrs[0..c.num_attrs]) |at| switch (at.id) {
            .cluster_dimension => last.cluster_z = at.value.cluster_dim.z,
            .programmatic_stream_serialization => last.pdl = at.value.int == 1,
            else => {},
        };
        return abi.success;
    }
};

test "fused split isolation: enabling one lane leaves another lane's actual dispatch unchanged" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernel = Seen.plain;
    d.api.cuLaunchKernelEx = Seen.extended;
    const base: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(10) };
    const small: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(20) };
    const wide: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(21) };
    var other: qmmf.Lane = .{
        .mod = undefined,
        .fns = @splat(@splat(base)),
        .f32_fns = @splat(@splat(base)),
        .reduce = @splat(base),
        .half = @splat(@splat(@splat(base))),
        .half_below = 0,
        .fuse_s = @splat(@splat(@splat(small))),
        .clusters = true,
    };
    for (&other.fns) |*set| set[6] = wide;
    for (&other.f32_fns) |*set| set[6] = wide;
    var flash = other;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]qmmf.Mode{ .mxfp8, .bf16 }) |mode| for ([_]qmmf.Out{ .bf16, .f32 }) |out| {
        for ([_]usize{ 1, 17, 33 }) |rows| for ([_]u32{ 2047, 2048, 2560 }) |n| for ([_]bool{ false, true }) |pdl| {
            other.pdl = pdl;
            flash.pdl = pdl;
            const w = try qmmf.pinned(.{ .mode = mode, .codes = 0, .scales = 0, .n = n, .k = 6144, .npad = 2560 }, 4);
            try std.testing.expectEqual(@as(usize, 0), other.fuse_split_n);
            try other.matmulOut(stream, 0, w.k, rows, w, 0, out, 0);
            const before = Seen.last;
            try std.testing.expectEqual(Seen.Dispatch{ .id = 10, .grid_z = 4, .cluster_z = 4, .pdl = pdl, .slices = 4 }, before);
            flash.fuse_split_n = 2048;
            try flash.matmulOut(stream, 0, w.k, rows, w, 0, out, 0);
            if (n < 2048) {
                try std.testing.expectEqual(before, Seen.last);
            } else {
                try std.testing.expectEqual(Seen.Dispatch{ .id = if (rows <= 32) 20 else 21, .grid_z = 1, .pdl = pdl, .slices = 4 }, Seen.last);
            }
            try other.matmulOut(stream, 0, w.k, rows, w, 0, out, 0);
            try std.testing.expectEqual(before, Seen.last);
        };
    };
}
