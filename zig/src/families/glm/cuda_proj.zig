//! A GLM-5.3-Flash projection as its checkpoint stores it: bf16 through the glue's matmuls, block FP8 through qlinear.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");

/// A rank's projection: bf16 rows for the glue, or an FP8_PB_WO checkpoint's block FP8 packed for the lane.
pub const Proj = union(enum) { bf16: u64, fp8g: cuda.qmmf.Weight };

/// x [m][k] (rows `x_stride` apart) by `w` into out [m][n]: bf16, or fp32 with `wide`; `part` the bf16 path's slices.
pub fn matmul(set: glue.Set, stream: cuda.Stream, w: Proj, x: u64, x_stride: usize, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
    switch (w) {
        .bf16 => |p| return b16.matmul(set, stream, x, x_stride, p, out, part, m, n, k, wide),
        .fp8g => |q| {
            if (q.n != n or q.k != k) return error.ShapeMismatch;
            const lane = set.linear.lane orelse return error.FormatNotLoaded;
            // GB10's lane meets its K slices in a cluster; a GPU that sums them in memory would need partials held here
            if (lane.partBytes(m, q) != 0) return error.SlicePartialsNotHeld;
            const in: cuda.qlinear.Rows = .{ .x = x, .ldx = x_stride };
            const f: cuda.qlinear.Weight = .{ .fp8g = q };
            return if (wide) set.linear.decodeF32(stream, f, in, out, m) else set.linear.decode(stream, f, in, out, m);
        },
    }
}

test "a block-FP8 projection is refused before any launch when the call's shape or the loaded kernels do not fit" {
    const q = cuda.fp8.weight(0, 0, 64, 4096);
    const s: cuda.Stream = undefined; // never reached: the refusals come first
    var set: glue.Set = .{ .glue = undefined };
    try std.testing.expectError(error.ShapeMismatch, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 128, 4096, false));
    try std.testing.expectError(error.FormatNotLoaded, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 64, 4096, true));
    var lane: cuda.qmmf.Lane = undefined;
    lane.clusters = false; // a GPU before sm_90: 64 outputs from 4,096 inputs take K slices summed in memory
    set.linear = .{ .lane = &lane };
    try std.testing.expectError(error.SlicePartialsNotHeld, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 64, 4096, true));
}
