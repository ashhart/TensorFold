//! The image tower's attention on CUDA: each frame group attends within itself.

const std = @import("std");
const cuda = @import("cuda");
const vision = @import("cuda_vision.zig");

pub const HEADS: i64 = 16;
pub const HD: i64 = 64; // a head's width
pub const CHUNK: i64 = 2048; // query rows a scores product covers

/// TF_GLM_VISION_FUSED=1: rank 0's tower runs vit's full attention; unset or 0, a head in chunks around cuBLASLt.
pub fn wanted() !bool {
    return @import("sys").environment.value(bool, "TF_GLM_VISION_FUSED", @import("profile").lookup("TF_GLM_VISION_FUSED"), false);
}

/// A pass's rows: qkv [N, 3 x 1024], att [N, 1024]; the chunked path's q and k [t, 16, seg, 64], v transposed, scratch.
pub const Buffers = struct { qkv: u64, qh: u64, kh: u64, vt: u64, att: u64, sc: u64, pr: u64, oh: u64, sums: u64 };

/// A block's q and k head-normed and rotated: in place in `b.qkv` for `t.full`, else into `b.qh`, `b.kh` and `b.vt`.
pub fn heads(t: *const vision.Tower, ru: *const vision.Run, b: Buffers, qn: u64, kn: u64, pos: u64, n: i64, seg: i64) !void {
    const rows = t.full != null;
    var a: cuda.Args = .{};
    for ([_]u64{ b.qkv, qn, kn, pos, t.inv_freq.ptr }) |p| a.add(p);
    if (!rows) for ([_]u64{ b.qh, b.kh, b.vt }) |p| a.add(p);
    a.add(@as(i32, @intCast(n)));
    if (!rows) a.add(@as(i32, @intCast(seg)));
    a.add(@as(i32, @intCast(HEADS)));
    a.add(vision.EPS);
    const grid: cuda.Dim3 = .{ .x = @intCast(std.math.divCeil(i64, n * HEADS, 8) catch unreachable) };
    try vision.launch(ru, if (rows) ru.f.qk else ru.f.qkv, grid, 256, &a);
}

/// Every frame group's heads into `b.att`: `t.full` over the group's rows, else two products around a softmax a head.
pub fn run(t: *vision.Tower, ru: *const vision.Run, b: Buffers, gt: i64, seg: i64) !void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
    var g: i64 = 0;
    while (g < gt) : (g += 1) {
        const at: u64 = @intCast(g * seg * HEADS * HD * 2);
        if (t.full) |*f| {
            try f.full(ru.stream, b.qkv + 3 * at, b.att + at, @intCast(seg), HEADS, HD, scale);
            continue;
        }
        var h: i64 = 0;
        while (h < HEADS) : (h += 1) try chunked(t, ru, b, @intCast((g * HEADS + h) * seg * HD * 2), b.att + at, seg, h, scale);
    }
}

/// The head in query chunks: the scores product, glm_vis_softmax, the values product, glm_vis_head_out.
fn chunked(t: *vision.Tower, ru: *const vision.Run, b: Buffers, blk: u64, att: u64, seg: i64, h: i64, scale: f32) !void {
    var c: i64 = 0;
    while (c < seg) : (c += CHUNK) {
        const rows = @min(CHUNK, seg - c);
        try t.linear(ru, b.qh + blk + @as(u64, @intCast(c * HD * 2)), b.kh + blk, b.sc, rows, seg, HD, true);
        var a: cuda.Args = .{};
        a.add(b.sc);
        a.add(b.pr);
        a.add(b.sums);
        a.add(@as(i32, @intCast(seg)));
        a.add(scale);
        try vision.launch(ru, ru.f.softmax, .{ .x = @intCast(rows) }, 256, &a);
        try t.linear(ru, b.pr, b.vt + blk, b.oh, rows, HD, seg, true);
        var o: cuda.Args = .{};
        o.add(b.oh);
        o.add(b.sums);
        o.add(att + @as(u64, @intCast(c * HEADS * HD * 2)));
        o.add(@as(i32, @intCast(rows)));
        o.add(@as(i32, @intCast(HEADS)));
        o.add(@as(i32, @intCast(h)));
        try vision.launch(ru, ru.f.head_out, vision.Tower.grid1(rows * HD), 256, &o);
    }
}

const Seen = struct {
    var n: usize = 0;
    var grid: [8][2]u32 = undefined;
    var block: [8]u32 = undefined;
    fn launch(_: cuda.abi.Function, gx: c_uint, gy: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        grid[n] = .{ gx, gy };
        block[n] = bx;
        n += 1;
        return cuda.abi.success;
    }
};

test "with vit's attention a block is one q and k launch and one attention launch a frame group, for any segment" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Seen.launch;
    d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(Seen.launch);
    const f: cuda.Function = .{ .d = &d, .handle = null };
    var ru: vision.Run = .{ .stream = .{ .d = &d, .handle = null }, .f = undefined };
    ru.f.qk = f;
    var tower: vision.Tower = undefined;
    tower.inv_freq.ptr = 0;
    tower.full = .{ .mod = undefined, .fns = @splat(f) };
    const b: Buffers = .{ .qkv = 0, .qh = 0, .kh = 0, .vt = 0, .att = 0, .sc = 0, .pr = 0, .oh = 0, .sums = 0 };
    inline for (.{ .{ 4, 16, 1 }, .{ 128, 512, 1 }, .{ 129, 516, 2 }, .{ 6528, 26112, 51 } }) |case| {
        Seen.n = 0;
        try heads(&tower, &ru, b, 0, 0, 0, 2 * case[0], case[0]);
        try run(&tower, &ru, b, 2, case[0]);
        try std.testing.expectEqual(@as(usize, 3), Seen.n);
        try std.testing.expectEqual([2]u32{ case[1], 1 }, Seen.grid[0]);
        for (1..3) |i| try std.testing.expectEqual([2]u32{ case[2], HEADS }, Seen.grid[i]);
        try std.testing.expectEqualSlices(u32, &.{ 256, 256, 256 }, Seen.block[0..3]);
    }
}
