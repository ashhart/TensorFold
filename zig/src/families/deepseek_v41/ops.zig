//! Trusted plaintext encoders: callers validate nonzero shapes/storage; strides count elements, EXL3 is separate.
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const View = @import("view.zig").View;

pub const Pipelines = struct {
    rms: mtl.Pipeline,
    dense: mtl.Pipeline,
    embedding: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    combine: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_ops, options);
        defer lib.deinit();
        const rms = try mtl.Pipeline.init(device, lib, "dsv41_rms", false);
        errdefer rms.deinit();
        const dense = try mtl.Pipeline.init(device, lib, "dsv41_dense", false);
        errdefer dense.deinit();
        const embedding = try mtl.Pipeline.init(device, lib, "dsv41_embedding", false);
        errdefer embedding.deinit();
        const swiglu = try mtl.Pipeline.init(device, lib, "dsv41_swiglu", false);
        errdefer swiglu.deinit();
        const combine = try mtl.Pipeline.init(device, lib, "dsv41_combine", false);
        return .{ .rms = rms, .dense = dense, .embedding = embedding, .swiglu = swiglu, .combine = combine };
    }
    pub fn deinit(p: *Pipelines) void {
        p.rms.deinit();
        p.dense.deinit();
        p.embedding.deinit();
        p.swiglu.deinit();
        p.combine.deinit();
    }
};
fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
}
pub const Rms = struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: bool = false };
/// BF16 x/y [rows,width], BF16 or FP32 weight [width]; sums, mean and affine product are FP32.
pub fn encodeRms(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, weight: View, y: View, d: Rms) void {
    e.setPipeline(p.rms);
    bind(e, .{ x, weight, y });
    const D = extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: u32 };
    e.setValue(D{ .rows = d.rows, .width = d.width, .x_stride = d.x_stride, .y_stride = d.y_stride, .eps = d.eps, .weight_f32 = @intFromBool(d.weight_f32) }, 3);
    e.dispatchGroups(mtl.Size.of(d.rows, 1, 1), mtl.Size.of(256, 1, 1));
}
pub const Dtype = enum(u32) { bf16 = 0, f32 = 1, f16 = 2 };
pub const Dense = struct {
    rows: u32,
    k: u32,
    n: u32,
    groups: u32 = 1,
    x_stride: u32,
    w_stride: u32,
    y_stride: u32,
    x_dtype: Dtype = .bf16,
    w_dtype: Dtype = .bf16,
    y_dtype: Dtype = .bf16,
};
/// x[rows,groups,k], w[groups,n,k], y[rows,groups,n]; packed groups, padded token pitches, FP32 sums, final BF16.
pub fn encodeDense(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: Dense) void {
    e.setPipeline(p.dense);
    bind(e, .{ x, w, y });
    e.setValue([10]u32{ d.rows, d.k, d.n, d.groups, d.x_stride, d.w_stride, d.y_stride, @backingInt(d.x_dtype), @backingInt(d.w_dtype), @backingInt(d.y_dtype) }, 3);
    e.dispatchGroups(mtl.Size.of((d.n + 15) / 16, d.rows, d.groups), mtl.Size.of(128, 1, 1));
}
pub const Embedding = extern struct { rows: u32, width: u32, table_stride: u32, y_stride: u32, copies: u32 = 4 };
/// ids[rows] u32 must be in vocabulary; table[vocab,width] BF16 -> y[rows,copies,width] BF16.
pub fn encodeEmbedding(e: mtl.ComputeEncoder, p: *const Pipelines, ids: View, table: View, y: View, d: Embedding) void {
    e.setPipeline(p.embedding);
    bind(e, .{ ids, table, y });
    e.setValue(d, 3);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}
pub const Activation = struct {
    rows: u32,
    width: u32,
    input_stride: u32,
    output_stride: u32,
    weighted: bool = false,
    input_f32: bool = false,
};
/// Separate gate/up; apply FP32 weights before BF16 cast/down. Unweighted calls borrow but never read weights.
pub fn encodeSwiGLU(e: mtl.ComputeEncoder, p: *const Pipelines, gate: View, up: View, weights: View, y: View, d: Activation) void {
    e.setPipeline(p.swiglu);
    bind(e, .{ gate, up, weights, y });
    e.setValue([6]u32{ d.rows, d.width, d.input_stride, d.output_stride, @intFromBool(d.weighted), @intFromBool(d.input_f32) }, 4);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}
pub const Combine = struct {
    rows: u32,
    width: u32,
    routed_row_stride: u32,
    slot_stride: u32,
    shared_stride: u32,
    y_stride: u32,
    input_f32: bool = false,
    id_stride: u32 = 6,
};
/// Sum six weighted down slots by ascending expert ID in FP32, then shared; final BF16, invalid IDs yield NaN.
pub fn encodeCombine(e: mtl.ComputeEncoder, p: *const Pipelines, routed: View, shared: View, ids: View, y: View, d: Combine) void {
    e.setPipeline(p.combine);
    bind(e, .{ routed, shared, ids, y });
    e.setValue([8]u32{ d.rows, d.width, d.routed_row_stride, d.slot_stride, d.shared_stride, d.y_stride, @intFromBool(d.input_f32), d.id_stride }, 4);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}
