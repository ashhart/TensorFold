//! The EXL3 projection's encode entry points.
const std = @import("std");
const mtl = @import("metal");
const core = @import("exl3_projection_core.zig");
const pipe = @import("exl3_projection_pipelines.zig");
const swi = @import("exl3_projection_swiglu.zig");
const Pipelines = pipe.Pipelines;
const stages = swi.stages;
const moeSplitCount = core.moeSplitCount;
const Matrix = core.Matrix;
const View = core.View;
const Rows = core.Rows;
const Experts = core.Experts;
const Int8 = core.Int8;
const Args = core.Args;
const ExpertDescriptor = core.ExpertDescriptor;
const format = core.format;
const decoder = @import("exl3_projection_decode.zig");
const scratchBytes = core.scratchBytes;
const one_row_splits = core.one_row_splits;
const moe_split_rows = core.moe_split_rows;
const rowsArgs = core.rowsArgs;
const bind = core.bind;
const shape = core.shape;
const rotatedBytes = core.rotatedBytes;
const check = core.check;
const disjoint = core.disjoint;
const readonly = core.readonly;

pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, m: Matrix, x: View, y: View, scratch: View, r: Rows) !void {
    const storage_bytes = try m.checkStorage();
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    if (r.grouped) return error.InvalidExl3GroupedRows;
    try matrixWrites(m, storage_bytes, y, scratch, r);
    a.halves = m.width.halves;
    a.codebook = @intCast(@backingInt(m.codebook));
    a.su_signs = @intFromBool(m.suh_format == .packed_signs);
    a.sv_signs = @intFromBool(m.svh_format == .packed_signs);
    a.has_bias = @intFromBool(m.bias != null);
    a.tile_k_stride = try m.kStride();
    a.tile_n_stride = m.nStride();
    stages(p, e, a, m.trellis, m.suh, m.svh, m.bias orelse m.svh, scratch, x, y, scratch, r);
}

/// A kept-dense EXL3 matrix: a bf16 weight beside the input/output rotation scales, no trellis.
pub const DenseMatrix = struct {
    weight: View,
    suh: View,
    svh: View,
    bias: ?View = null,
    k: usize,
    n: usize,
};

/// Kept-dense: quantized rotate stages, trellis to dense, f32 in rotate_out sums; sequential k, as the fp64 reference.
pub fn encodeDense(p: *const Pipelines, e: mtl.ComputeEncoder, m: DenseMatrix, x: View, y: View, scratch: View, r: Rows) !void {
    try shape(m.k, m.n);
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    if (r.grouped) return error.InvalidExl3GroupedRows;
    a.su_signs = 0;
    a.sv_signs = 0;
    a.has_bias = @intFromBool(m.bias != null);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
    const sums_bytes = try std.math.mul(usize, try std.math.mul(usize, m.n, r.count), 4);
    try readonly(m.weight, try std.math.mul(usize, try std.math.mul(usize, m.n, m.k), 2), y, 0, scratch, sums_bytes);
    try readonly(m.suh, m.k * 2, y, 0, scratch, sums_bytes);
    try readonly(m.svh, m.n * 2, y, 0, scratch, sums_bytes);
    if (m.bias) |b| try readonly(b, m.n * 2, y, 0, scratch, sums_bytes);
    e.setPipeline(p.rotate_in);
    bind(e, x, 0);
    bind(e, m.suh, 1);
    bind(e, scratch, 2);
    e.setValue(a, 3);
    bind(e, scratch, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    e.dispatchGroups(.{ .width = a.k / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
    e.setPipeline(p.dense);
    bind(e, scratch, 0);
    bind(e, m.weight, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, scratch, 4);
    e.dispatchThreads(.{ .width = a.n, .height = a.rows }, .{ .width = 64 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    e.dispatchGroups(.{ .width = a.n / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
}

/// The int8 tile path for one dense call: rotate in, the int8 tiles into `sums`, rotate out.
pub fn encodeInt8Tile(p: *const Pipelines, e: mtl.ComputeEncoder, m: Matrix, x: View, y: View, scratch: View, r: Rows) !void {
    if (p.int8_tile == null or p.codebook_table == null) return error.UnsupportedExl3Int8;
    const storage_bytes = try m.checkStorage();
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    a.halves = m.width.halves;
    a.codebook = @intCast(@backingInt(m.codebook));
    a.su_signs = @intFromBool(m.suh_format == .packed_signs);
    a.sv_signs = @intFromBool(m.svh_format == .packed_signs);
    a.has_bias = @intFromBool(m.bias != null);
    a.tile_k_stride = try m.kStride();
    a.tile_n_stride = m.nStride();
    a.splits = 1;
    try matrixWrites(m, storage_bytes, y, scratch, r);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
    e.setPipeline(p.rotate_in);
    bind(e, x, 0);
    bind(e, m.suh, 1);
    bind(e, scratch, 2);
    e.setValue(a, 3);
    bind(e, r.input_ids orelse scratch, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    const rotate_width: usize = (a.k / 128 + 3) / 4;
    e.dispatchGroups(.{ .width = rotate_width, .height = a.rows }, .{ .width = 128 });
    e.barrier();
    e.setPipeline(p.int8_tile.?);
    bind(e, scratch, 0);
    bind(e, m.trellis, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, scratch, 5);
    e.setBuffer(p.codebook_table.?, 0, 7);
    e.dispatchGroups(.{ .width = (a.n + 15) / 16, .height = (a.rows + 15) / 16 }, .{ .width = 32 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, r.output_ids orelse scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    const rotate_out_width: usize = (a.n / 128 + 3) / 4;
    e.dispatchGroups(.{ .width = rotate_out_width, .height = a.rows }, .{ .width = 128 });
    e.barrier();
}

/// The prompt-row int8 path for a dense projection.
pub fn encodePrefillInt8(p: *const Pipelines, e: mtl.ComputeEncoder, m: Matrix, side: Int8, x: View, y: View, scratch: View, r: Rows, direct: bool) !void {
    if (p.prefill_fp8 == null) return error.UnsupportedExl3Int8;
    if (m.k != side.k or m.n != side.n) return error.InvalidExl3Stride;
    var a = try rowsArgs(m.k, m.n, r, x, y, scratch);
    if (r.grouped) return error.InvalidExl3GroupedRows;
    a.halves = 0;
    a.codebook = 2;
    a.su_signs = 0;
    a.sv_signs = 0;
    a.has_bias = 0;
    a.tile_k_stride = (m.n / 16) * 16;
    a.tile_n_stride = 16;
    const weight_bytes = try std.math.mul(usize, m.k, m.n);
    try check(side.weight, weight_bytes, 16);
    try check(side.scales, try std.math.mul(usize, m.n, 4), 4);
    if (direct) {
        if (r.output != .bf16 or p.prefill_cast == null) return error.UnsupportedExl3Output;
        if (core.knobs.i8_debug) std.log.info("i8c direct rows={d} k={d} n={d} xs={d} ys={d} yoff={d}", .{ a.rows, a.k, a.n, a.x_stride, a.y_stride, y.offset });
        try readonly(side.weight, weight_bytes, y, 0, scratch, try scratchBytes(m.k, m.n, r.count));
        try readonly(side.scales, try std.math.mul(usize, m.n, 4), y, 0, scratch, try scratchBytes(m.k, m.n, r.count));
        const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
        e.setPipeline(p.prefill_fp8.?);
        bind(e, x, 0);
        bind(e, side.weight, 1);
        bind(e, sums, 2);
        e.setValue(a, 3);
        bind(e, side.scales, 4);
        e.dispatchGroups(.{ .width = (a.n + 63) / 64, .height = (a.rows + 127) / 128 }, .{ .width = 256 });
        e.barrier();
        e.setPipeline(p.prefill_cast.?);
        bind(e, sums, 0);
        bind(e, y, 1);
        e.setValue(a, 2);
        e.dispatchThreads(.{ .width = a.n, .height = a.rows }, .{ .width = 64 });
        e.barrier();
        return;
    }
    try readonly(side.weight, weight_bytes, y, 0, scratch, try scratchBytes(m.k, m.n, r.count));
    try readonly(side.scales, try std.math.mul(usize, m.n, 4), y, 0, scratch, try scratchBytes(m.k, m.n, r.count));
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try rotatedBytes(m.k, r.count) };
    e.setPipeline(p.rotate_in);
    bind(e, x, 0);
    bind(e, m.suh, 1);
    bind(e, scratch, 2);
    e.setValue(a, 3);
    bind(e, scratch, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    e.dispatchGroups(.{ .width = a.k / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
    // The rotated activation is contiguous at k; the call x stride belongs to the raw input the rotation read.
    a.x_stride = m.k;
    e.setPipeline(p.prefill_fp8.?);
    bind(e, scratch, 0);
    bind(e, side.weight, 1);
    bind(e, sums, 2);
    e.setValue(a, 3);
    bind(e, side.scales, 4);
    e.dispatchGroups(.{ .width = (a.n + 63) / 64, .height = (a.rows + 127) / 128 }, .{ .width = 256 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    e.dispatchGroups(.{ .width = a.n / 128, .height = a.rows }, .{ .width = 32 });
    e.barrier();
}

/// GPU routes choose caller-validated resident descriptors; grouped padding uses maxInt(u32).
pub fn encodeIndexed(p: *const Pipelines, e: mtl.ComputeEncoder, experts: Experts, x: View, y: View, scratch: View, r: Rows) !void {
    if (r.routes == null or experts.count == 0 or experts.count > std.math.maxInt(u32)) return error.InvalidExl3Experts;
    if (r.grouped and (r.input_ids == null or r.output_ids == null or r.count % 8 != 0))
        return error.InvalidExl3GroupedRows;
    try shape(experts.k, experts.n);
    try check(experts.descriptors, try std.math.mul(usize, experts.count, @sizeOf(ExpertDescriptor)), 8);
    var a = try rowsArgs(experts.k, experts.n, r, x, y, scratch);
    const yrows = if (r.output_ids != null) r.output_rows else r.count;
    const ybytes = ((yrows - 1) * r.y_stride + experts.n) * (if (r.output == .f32) @as(usize, 4) else 2);
    try readonly(experts.descriptors, experts.count * @sizeOf(ExpertDescriptor), y, ybytes, scratch, try scratchBytes(experts.k, experts.n, r.count));
    a.indexed = if (r.grouped) 2 else 1;
    a.experts = @intCast(experts.count);
    stages(p, e, a, scratch, scratch, scratch, scratch, experts.descriptors, x, y, scratch, r);
}

pub fn matrixWrites(m: Matrix, storage_bytes: usize, y: View, scratch: View, r: Rows) !void {
    const yrows = if (r.output_ids != null) r.output_rows else r.count;
    const ybytes = ((yrows - 1) * r.y_stride + m.n) * (if (r.output == .f32) @as(usize, 4) else 2);
    const scratch_bytes = try scratchBytes(m.k, m.n, r.count);
    try readonly(m.trellis, storage_bytes, y, ybytes, scratch, scratch_bytes);
    try readonly(m.suh, m.k * 2 / (if (m.suh_format == .packed_signs) @as(usize, 16) else 1), y, ybytes, scratch, scratch_bytes);
    try readonly(m.svh, m.n * 2 / (if (m.svh_format == .packed_signs) @as(usize, 16) else 1), y, ybytes, scratch, scratch_bytes);
    if (m.bias) |b| try readonly(b, m.n * 2, y, ybytes, scratch, scratch_bytes);
}
