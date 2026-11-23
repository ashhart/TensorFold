//! Prompt projections on 16 x 16 MMA tiles (kernels/metal/inkling2/dense_mma.metal): X in fragment order first.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("../../core/exl3_projection.zig");

pub const View = exl3.View;

/// Outputs a side of a tile, its threads (eight simdgroups), and the K step of a chain round (K must be a multiple).
pub const tile = 16;
pub const threads = 256;
pub const chunk = 256;

const Args = extern struct { k: u32, n: u32, rows: u32 };

pub const Call = struct { x: View, w: View, y: View, rows: u32, n: u32, k: u32, f32_out: bool };

/// y = x w^T over slices of rows that fit `stage` (private, `stage_bytes`): each slice is permuted, then multiplied.
pub fn encode(e: mtl.ComputeEncoder, perm: mtl.Pipeline, mma: mtl.Pipeline, stage: mtl.Buffer, stage_bytes: usize, c: Call) !void {
    const slice: u32 = @intCast(@min(c.rows, stage_bytes / (2 * @as(usize, c.k))));
    if (slice == 0) return error.Inkling2ScratchTooSmall;
    const out_bytes: usize = if (c.f32_out) 4 else 2;
    var r0: u32 = 0;
    while (r0 < c.rows) : (r0 += slice) {
        const rows = @min(slice, c.rows - r0);
        const args = Args{ .k = c.k, .n = c.n, .rows = rows };
        e.setPipeline(perm);
        e.setBuffer(c.x.buffer, c.x.offset + @as(usize, r0) * c.k * 2, 0);
        e.setBuffer(stage, 0, 1);
        e.setValue(args, 2);
        e.dispatchThreads(mtl.Size.of(c.k / 8, rows, 1), mtl.Size.of(threads, 1, 1));
        e.setPipeline(mma);
        e.setBuffer(stage, 0, 0);
        e.setBuffer(c.w.buffer, c.w.offset, 1);
        e.setBuffer(c.y.buffer, c.y.offset + @as(usize, r0) * c.n * out_bytes, 2);
        e.setValue(args, 3);
        e.dispatchGroups(mtl.Size.of((c.n + tile - 1) / tile, (rows + tile - 1) / tile, 1), mtl.Size.of(threads, 1, 1));
    }
}
