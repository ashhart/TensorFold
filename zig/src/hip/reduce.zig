//! The tensor-parallel reduce: ranks' fp32 shares summed before each residual add, logits slices joined in rank order.

const std = @import("std");
const ops = @import("ops/ops.zig");
const collective = @import("collective.zig");

const Ops = ops.Ops;
const Tensor = ops.Tensor;

pub const Error = ops.Error || collective.Error;

/// `n` fp32 values summed over the ranks in rank order, so a row's sum does not depend on the rows sharing the call.
pub fn sum(o: Ops, c: collective.Comm, y: u64, n: usize) Error!u64 {
    const out = try o.arena.of(f32, n);
    if (c.world == 2) {
        try c.allReduce(y, out, n, .f32, o.stream);
        return out;
    }
    const parts = try o.arena.of(f32, c.world * n);
    try c.allGather(y, parts, n, .f32, o.stream);
    const part = n * 4;
    try o.add(.{ .ptr = parts, .kind = .f32 }, .{ .ptr = parts + part, .kind = .f32 }, .{ .ptr = out, .kind = .f32 }, n);
    for (2..c.world) |r| try o.add(.{ .ptr = out, .kind = .f32 }, .{ .ptr = parts + r * part, .kind = .f32 }, .{ .ptr = out, .kind = .f32 }, n);
    return out;
}

/// x += y on the residual rows: an activation-dtype `y` as one rank adds it, an fp32 share summed over the ranks first.
pub fn residual(o: Ops, tp: ?collective.Comm, x: Tensor, y: Tensor, n: usize) Error!void {
    if (y.kind != .f32) return o.add(x, y, x, n);
    const total = try sum(o, tp.?, y.ptr, n);
    try o.addWide(x, total, x, n);
}

/// `residual`, then normed = rms(x) * weight, the add and the norm in one launch where `residualRms` takes them.
pub fn residualNorm(
    o: Ops,
    tp: ?collective.Comm,
    x: Tensor,
    y: Tensor,
    weight: u64,
    normed: Tensor,
    rows: usize,
    width: usize,
    eps: f32,
) Error!void {
    if (y.kind != .f32) return o.residualRms(x, y, weight, normed, rows, width, eps);
    const total = try sum(o, tp.?, y.ptr, rows * width);
    try o.residualRms(x, .{ .ptr = total, .kind = .f32 }, weight, normed, rows, width, eps);
}

/// The ranks' logits slices (world, rows, width) of the activation dtype, gathered into `out`.
pub fn gather(o: Ops, c: collective.Comm, slice: Tensor, out: u64, n: usize) Error!void {
    const dtype: collective.Dtype = if (slice.kind == .f16) .f16 else .bf16;
    try c.allGather(slice.ptr, out, n, dtype, o.stream);
}
