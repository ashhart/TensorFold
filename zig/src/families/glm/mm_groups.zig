//! Prompt experts and dense products in column groups (glm_mm_groups.metal): core/affine_mm.zig's bits, faster.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const affine_mm = @import("../../core/affine_mm.zig");

pub const names = [5][:0]const u8{ "glm_gather_glu_32", "glm_gather_glu_64", "glm_gather_scatter_32", "glm_gather_scatter_64", "glm_mm_grouped" };

/// The column-group products in `names` order (bf16 out).
pub const Pipes = [5]mtl.Pipeline;

/// Column tiles a group (glm_mm_groups.metal's GT_GROUP): a launch's column tiles must be a multiple.
pub const group = 32;

/// Dense products' depth from which they take column groups (shallower ones were as fast or slower on an M5 Ultra).
pub const dense_from = 4096;

/// Experts' mean rows below which a launch takes 32-row tiles (gate/up, down); M5 Ultra, GLM's shapes.
pub const glu_32_below = 96;
pub const down_32_below = 64;

/// core/affine_mm.metal at 4/64, bf16 out, the column-group kernels after it: affine_mm.names, then ours.
pub fn source(device: mtl.Device, a: std.mem.Allocator) ![]u8 {
    const core = try affine_mm.source(device, a, .{ .exact_nonfinite = true });
    defer a.free(core);
    return std.mem.concat(a, u8, &.{ core, sources.glm_mm_groups });
}

fn bind(e: mtl.ComputeEncoder, i: usize, r: anytype) void {
    e.setBuffer(r.buf, r.off, i);
}

/// The one-launch gathers' tile count: each expert's last tile may be partial.
fn tiles(rows: u32, experts: u32, bm: u32) u32 {
    return @min(rows, (rows + bm - 1) / bm + experts - 1);
}

/// affine_mm.gatherGlu's activations (expert-sorted pairs, pair i: x row order[i] / topk) in column groups.
pub fn gatherGlu(e: mtl.ComputeEncoder, p: Pipes, x: anytype, sums: anytype, gate: anytype, up: anytype, offsets: anytype, order: anytype, act: anytype, rows: u32, experts: u32, topk: u32, limit: f32) void {
    std.debug.assert(gate.n % (32 * group) == 0 and gate.k % 64 == 0 and gate.n == up.n and gate.k == up.k);
    const bm: u32 = if (rows / experts < glu_32_below) 32 else 64;
    e.setPipeline(p[if (bm == 64) 1 else 0]);
    bind(e, 0, x);
    bind(e, 1, gate.w);
    bind(e, 2, gate.s);
    bind(e, 3, gate.b);
    bind(e, 4, sums);
    e.setValue(affine_mm.Args{ .rows = @intCast(rows), .n = @intCast(gate.n), .k = @intCast(gate.k), .experts = @intCast(experts) }, 5);
    bind(e, 6, act);
    bind(e, 7, offsets);
    bind(e, 8, order);
    bind(e, 9, up.w);
    bind(e, 10, up.s);
    bind(e, 11, up.b);
    e.setValue([2]f32{ @floatFromInt(topk), limit }, 12);
    e.dispatchGroups(mtl.Size.of(tiles(rows, experts, bm) * (gate.n / 32), 1, 1), mtl.Size.of(128, 1, 1));
}

/// affine_mm.gatherTo (sorted row i's outputs to row dst[i] of y) in column groups.
pub fn gatherTo(e: mtl.ComputeEncoder, p: Pipes, x: anytype, sums: anytype, q: anytype, offsets: anytype, y: anytype, dst: anytype, rows: u32, experts: u32) void {
    std.debug.assert(q.n % (64 * group) == 0 and q.k % 64 == 0);
    const bm: u32 = if (rows / experts < down_32_below) 32 else 64;
    e.setPipeline(p[if (bm == 64) 3 else 2]);
    bind(e, 0, x);
    bind(e, 1, q.w);
    bind(e, 2, q.s);
    bind(e, 3, q.b);
    bind(e, 4, sums);
    e.setValue(affine_mm.Args{ .rows = @intCast(rows), .n = @intCast(q.n), .k = @intCast(q.k), .experts = @intCast(experts) }, 5);
    bind(e, 6, y);
    bind(e, 7, offsets);
    bind(e, 8, dst);
    e.dispatchGroups(mtl.Size.of(tiles(rows, experts, bm) * (q.n / 64), 1, 1), mtl.Size.of(128, 1, 1));
}

/// affine_mm.denseBatch at one batch (y = x W^T, pitches `s`) in column groups.
pub fn dense(e: mtl.ComputeEncoder, p: Pipes, x: anytype, sums: anytype, q: anytype, y: anytype, rows: u32, s: affine_mm.Strides) void {
    const n = std.mem.alignForward(u32, q.n, 64);
    e.setPipeline(p[4]);
    bind(e, 0, x);
    bind(e, 1, q.w);
    bind(e, 2, q.s);
    bind(e, 3, q.b);
    bind(e, 4, sums);
    e.setValue(affine_mm.Args{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(q.k), .experts = 0 }, 5);
    bind(e, 6, y);
    e.setValue(s, 7);
    e.dispatchGroups(mtl.Size.of(((rows + 63) / 64) * (n / 64), 1, 1), mtl.Size.of(128, 1, 1));
}

/// Whether a routed layer's matrices fit the column groups (4-bit/64 gate/up of 1024s of rows, down of 2048s).
pub fn fits(gate: anytype, down: anytype) bool {
    return gate.n % (32 * group) == 0 and down.n % (64 * group) == 0;
}
