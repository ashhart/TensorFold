//! BF16-weight products with one K slice write FP32 rows directly into a caller's row stride.
const q = @import("qmmf.zig");
const launch = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
pub const symbols = [4][:0]const u8{ "tf_qmmf_shared16_half", "tf_qmmf_shared16", "tf_qmmf_shared32", "tf_qmmf_shared64" };

/// A forced strided launch supports one pinned BF16 K slice.
pub fn supported(l: q.Lane, m: usize, w: q.Weight) bool {
    return w.mode == .bf16 and q.planFor(m, w, l.clusters).sk == 1;
}

/// Prefer an available 128-row prompt projection over the smaller strided tile.
pub fn eligible(l: q.Lane, m: usize, w: q.Weight) bool {
    return supported(l, m, w) and !l.usesPrompt(m, w);
}

pub fn matmulF32Strided(l: q.Lane, s: Stream, x: u64, ldx: usize, m: usize, w: q.Weight, y: u64, ldy: usize) !void {
    if (!supported(l, m, w)) return error.StridedLaneNotSupported;
    if (m == 0 or w.n == 0 or w.k == 0 or w.k % q.group != 0 or w.npad < w.n or w.npad % 128 != 0 or
        ldy < w.n or ldx < w.k) return error.BadStridedLaneShape;
    const bm: usize = q.bucket(m);
    const rows = (m + bm - 1) / bm;
    const half = bm == 16 and rows * ((w.n + 63) / 64) < l.half_below;
    const index: usize = if (half) 0 else if (bm == 16) 1 else if (bm == 32) 2 else 3;
    const cols: usize = if (half) 32 else 64;
    const band = @max(1, @min(rows, (12 << 20) / (bm * w.k * 2)));
    var a: launch.Args = .{};
    a.add(x);
    a.add(w.codes);
    a.add(w.scale);
    a.add(y);
    for ([_]usize{ m, w.n, w.k, w.npad, if (m == 1) w.k else ldx, band, ldy }) |v| a.add(@as(i32, @intCast(v)));
    try launch.launch(l.shared_down[index], .{
        .grid = .{ .x = @intCast(rows * ((w.n + cols - 1) / cols)) },
        .block = .{ .x = 128 },
        .shared = if (half) q.halfShared(.bf16) else q.sharedBytes(.bf16, @intCast(bm)),
    }, s, &a);
}

test "shared down stride: geometry, output address and stride match the contiguous launcher" {
    const std = @import("std");
    const abi = @import("abi.zig");
    const Mock = struct {
        var id: usize = 0;
        var stride: i32 = 0;
        var dest: u64 = 0;
        var smem: u32 = 0;
        fn run(f: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            id = @intFromPtr(f.?);
            stride = @as(*const i32, @ptrCast(@alignCast(args.?[10].?))).*;
            dest = @as(*const u64, @ptrCast(@alignCast(args.?[3].?))).*;
            smem = sm;
            return 0;
        }
    };
    var d: @import("driver.zig").Driver = undefined;
    d.api.cuLaunchKernel = Mock.run;
    var l: q.Lane = undefined;
    l.clusters = true;
    l.prompt = null;
    l.half_below = 96;
    for (&l.shared_down, 0..) |*f, i| f.* = .{ .d = &d, .handle = @ptrFromInt(101 + i) };
    const s: Stream = .{ .d = &d, .handle = null };
    const w: q.Weight = .{ .mode = .bf16, .codes = 11, .scales = 0, .n = 2560, .k = 640, .npad = 2560 };
    for ([_]usize{ 1, 16, 17, 32, 64, 65, 4218 }, [_]usize{ 0, 0, 2, 2, 3, 3, 3 }) |rows, i| {
        try l.matmulF32Strided(s, 12, 640, rows, w, 1000, 28160);
        try std.testing.expectEqual(101 + i, Mock.id);
        try std.testing.expectEqual(@as(i32, 28160), Mock.stride);
        try std.testing.expectEqual(@as(u64, 1000), Mock.dest);
        try std.testing.expectEqual(if (i == 0) q.halfShared(.bf16) else q.sharedBytes(.bf16, q.bucket(rows)), Mock.smem);
    }
    var other = w;
    other.mode = .mxfp8;
    try std.testing.expectError(error.StridedLaneNotSupported, l.matmulF32Strided(s, 12, 640, 64, other, 1000, 28160));
    other = w;
    other.k = 6144;
    try std.testing.expect(!eligible(l, 64, other));
    try std.testing.expectError(error.BadStridedLaneShape, l.matmulF32Strided(s, 12, 640, 64, w, 1000, 2559));
    for ([_]bool{ false, true }) |loaded| {
        l.prompt = if (loaded) @splat(@splat(l.shared_down[0])) else null;
        for ([_]usize{ 511, 512, 513 }) |m| {
            try std.testing.expectEqual(loaded and m >= q.prompt_rows, l.usesPrompt(m, w));
            try std.testing.expectEqual(!loaded or m < q.prompt_rows, eligible(l, m, w));
            try std.testing.expect(supported(l, m, w));
            try l.matmulF32Strided(s, 12, 640, m, w, 1000, 28160);
            try std.testing.expectEqual(@as(usize, 104), Mock.id);
        }
    }
    other = w;
    other.k = 6144;
    other.sk = 1;
    try std.testing.expect(supported(l, 64, other));
    try std.testing.expect(eligible(l, 511, other));
    try std.testing.expect(!eligible(l, 512, other));
    other.sk = 4;
    try std.testing.expect(!supported(l, 64, other));
}
