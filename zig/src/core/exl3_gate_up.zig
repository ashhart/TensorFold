//! An expert's gate (w1) and up (w3) EXL3 projections of the same rows, then DeepSeek V4.1's clamped SwiGLU.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("exl3_projection.zig");

/// The epilogue: both matrices' rotate_out, the clamped SwiGLU and its act_quant round trip in one launch.
pub const Pipelines = struct {
    swiglu: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_exl3_rows, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const swiglu = try mtl.Pipeline.init(device, lib, "tf_exl3_swiglu", false);
        errdefer swiglu.deinit();
        if (swiglu.simdWidth() != 32 or swiglu.maxThreads() < 32) return error.UnsupportedExl3Simd;
        return .{ .swiglu = swiglu };
    }

    pub fn deinit(p: *Pipelines) void {
        p.swiglu.deinit();
    }
};

/// Scratch for one call: each matrix's rotated rows and fp32 sums.
pub fn scratchBytes(k: usize, n: usize, rows: usize) !usize {
    return std.math.mul(usize, try ex.scratchBytes(k, n, rows), 2);
}

fn bind(e: mtl.ComputeEncoder, v: ex.View, index: usize) void {
    e.setBuffer(v.buffer, v.offset, index);
}

/// y = bf16 SwiGLU of 1-16 contiguous rows' gate and up clamped as `tf_dsv41_f_swiglu`, yq its act_quant; no alloc.
pub fn encode(p: *const ex.Pipelines, q: *const Pipelines, e: mtl.ComputeEncoder, gate: ex.Matrix, up: ex.Matrix, x: ex.View, y: ex.View, yq: ex.View, scratch: ex.View, r: ex.Rows, limit: f32) !void {
    if (gate.k != up.k or gate.n != up.n) return error.Exl3GateUpShape;
    if (r.count == 0 or r.count > 16 or r.routes != null or r.input_ids != null or r.output_ids != null or r.output != .bf16)
        return error.InvalidExl3GateUpRows;
    const half = try ex.scratchBytes(gate.k, gate.n, r.count);
    const sg = scratch;
    const su: ex.View = .{ .buffer = scratch.buffer, .offset = scratch.offset + half };
    const ag = try ex.matrixArgs(gate, x, y, sg, r);
    const au = try ex.matrixArgs(up, x, yq, su, r);
    // each matrix's reads against the other's writes as well
    _ = try ex.matrixArgs(gate, x, yq, su, r);
    _ = try ex.matrixArgs(up, x, y, sg, r);
    const ybytes = ((r.count - 1) * r.y_stride + gate.n) * 2;
    try ex.disjoint(y, ybytes, yq, ybytes);
    // the epilogue reads both sums while it writes y and yq, so neither output may lie on the other matrix's scratch
    try ex.disjoint(y, ybytes, su, half);
    try ex.disjoint(yq, ybytes, sg, half);
    ex.rotateIn(p, e, ag, x, gate.suh, sg, r, sg);
    ex.rotateIn(p, e, au, x, up.suh, su, r, su);
    e.barrier();
    ex.sums(p, e, ag, gate.trellis, sg, sg, sg);
    ex.sums(p, e, au, up.trellis, su, su, su);
    e.barrier();
    e.setPipeline(q.swiglu);
    const views = [_]ex.View{ ex.sumsView(sg, ag), ex.sumsView(su, au), gate.svh, up.svh, gate.bias orelse gate.svh, up.bias orelse up.svh, y, yq };
    for (views, 0..) |v, i| bind(e, v, i);
    e.setValue(ag, 8);
    e.setValue(au, 9);
    e.setValue(limit, 10);
    e.dispatchGroups(.{ .width = gate.n / 128, .height = r.count }, .{ .width = 32 });
    e.barrier();
}
