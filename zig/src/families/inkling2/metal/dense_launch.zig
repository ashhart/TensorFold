//! Inkling 2 Mini's dense projection dispatch and host-recorder entry points.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const dense_mma = @import("dense.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const Model = model_file.Model;
const Ref = state.Ref;

pub const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
pub const DenseStep = struct { variant: usize, rows: usize };

/// Ceiling-power-of-two matvec variant for `left` rows, reading weights once per <=32 rows.
pub fn denseStep(left: usize) DenseStep {
    std.debug.assert(left > 0);
    const take = @min(left, 32);
    const variant: usize = if (take <= 1) 0 else if (take <= 2) 1 else if (take <= 4) 2 else if (take <= 8) 3 else if (take <= 16) 4 else 5;
    return .{ .variant = variant, .rows = take };
}

/// Every projection uses the shared BF16 matvec's FP32 reduction and one BF16 store.
pub fn dense(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    encodeDense(mtl.ComputeEncoder, m, e, weight, input, out, rows);
}

/// Dispatch helper parameterized on encoder so host tests record exact launch counts and arguments.
pub fn encodeDense(comptime Encoder: type, m: anytype, e: Encoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    const k = weight.shape[weight.rank - 1];
    const n = weight.shape[weight.rank - 2];
    var base: usize = 0;
    while (base < rows) {
        const step = denseStep(rows - base);
        e.setPipeline(m.dense[step.variant]);
        e.setBuffer(input.buf, input.off + base * k * 2, 0);
        e.setBuffer(weight.buffer, weight.offset, 1);
        e.setValue(DenseArgs{ .rows = @intCast(step.rows), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
        e.setBuffer(out.buf, out.off + base * n * 2, 3);
        e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        base += step.rows;
    }
}

/// A visit's projection: `dense`, simdgroup matrices without tensor units, or register-tiled bf16 on tensor-unit GPUs.
pub fn project(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    const n: u32 = @intCast(weight.shape[weight.rank - 2]);
    const call = dense_mma.Call{ .rows = @intCast(rows), .n = n, .k = @intCast(weight.shape[weight.rank - 1]), .ldy = n };
    if (m.dense_mma) |mma| {
        if (dense_mma.Mma.serves(call)) return mma.launch(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = weight.buffer, .off = weight.offset }, .{ .buf = out.buf, .off = out.off }, call);
    } else if (m.dense_tiled) |tiled| {
        if (dense_mma.Tiled.serves(call)) return tiled.launch(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = weight.buffer, .off = weight.offset }, .{ .buf = out.buf, .off = out.off }, call);
    }
    dense(m, e, weight, input, out, rows);
}
