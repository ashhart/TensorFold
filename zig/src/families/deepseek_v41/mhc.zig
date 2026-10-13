//! Four-copy mHC; callers validate shapes/storage and own previous-pre coefficients.
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const View = @import("view.zig").View;
pub const Pipelines = struct {
    rms: mtl.Pipeline,
    split: mtl.Pipeline,
    collapse: mtl.Pipeline,
    post: mtl.Pipeline,
    expand: mtl.Pipeline,
    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_mhc, options);
        defer lib.deinit();
        const rms = try mtl.Pipeline.init(device, lib, "dsv41_hc_rms", false);
        errdefer rms.deinit();
        const split = try mtl.Pipeline.init(device, lib, "dsv41_hc_split", false);
        errdefer split.deinit();
        const collapse = try mtl.Pipeline.init(device, lib, "dsv41_hc_collapse", false);
        errdefer collapse.deinit();
        const post = try mtl.Pipeline.init(device, lib, "dsv41_hc_post", false);
        errdefer post.deinit();
        const expand = try mtl.Pipeline.init(device, lib, "dsv41_hc_expand", false);
        return .{ .rms = rms, .split = split, .collapse = collapse, .post = post, .expand = expand };
    }
    pub fn deinit(p: *Pipelines) void {
        p.rms.deinit();
        p.split.deinit();
        p.collapse.deinit();
        p.post.deinit();
        p.expand.deinit();
    }
};
fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
}
pub const Rms = extern struct { rows: u32, width: u32, stream_stride: u32, eps: f32 };
/// BF16 flattened [rows,4*width] -> FP32 inv[rows]; fixed CUDA single-row tree, not every batched CUDA grouping.
pub fn encodeRms(e: mtl.ComputeEncoder, p: *const Pipelines, stream: View, inv: View, d: Rms) void {
    e.setPipeline(p.rms);
    bind(e, .{ stream, inv });
    e.setValue(d, 2);
    e.dispatchGroups(mtl.Size.of(d.rows, 1, 1), mtl.Size.of(512, 1, 1));
}
pub const Split = extern struct {
    rows: u32,
    mix_stride: u32 = 24,
    coefficient_stride: u32 = 4,
    comb_stride: u32 = 16,
    eps: f32 = 1e-6,
};
/// FP32 mixes[rows,24], inv, scale[3], base[24] -> pre/post[rows,4], comb[rows,4,4]; 20 iterations.
pub fn encodeSplit(e: mtl.ComputeEncoder, p: *const Pipelines, mixes: View, inv: View, scale: View, base: View, pre: View, post: View, comb: View, d: Split) void {
    e.setPipeline(p.split);
    bind(e, .{ mixes, inv, scale, base, pre, post, comb });
    e.setValue(d, 7);
    e.dispatchThreads(mtl.Size.of(d.rows, 1, 1), mtl.Size.of(32, 1, 1));
}
pub const Collapse = extern struct { rows: u32, width: u32, stream_stride: u32, y_stride: u32, pre_stride: u32 = 4 };
/// Collapse BF16 [rows,4,width] with previous sublayer pre[rows,4]; head uses final FFN pre, initially one-hot.
pub fn encodeCollapse(e: mtl.ComputeEncoder, p: *const Pipelines, stream: View, previous_pre: View, y: View, d: Collapse) void {
    e.setPipeline(p.collapse);
    bind(e, .{ stream, previous_pre, y });
    e.setValue(d, 3);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}
pub const Post = extern struct {
    rows: u32,
    width: u32,
    branch_stride: u32,
    residual_stride: u32,
    y_stride: u32,
    coefficient_stride: u32 = 4,
    comb_stride: u32 = 16,
};
/// y[k,d]=post[k]*branch[d]+sum_j comb[j,k]*residual[j,d]; BF16 inputs/output, FP32 coefficients/math.

// Element pitches; exact residual alias is safe, but output must not alias branch/post/comb.
pub fn encodePost(e: mtl.ComputeEncoder, p: *const Pipelines, branch: View, residual: View, post: View, comb: View, y: View, d: Post) void {
    e.setPipeline(p.post);
    bind(e, .{ branch, residual, post, comb, y });
    e.setValue(d, 5);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}
pub const Expand = extern struct { rows: u32, width: u32, x_stride: u32, stream_stride: u32, pre_stride: u32 = 4 };
/// BF16 x[rows,width] copied into four streams and FP32 initial pre[rows,4] set to [1,0,0,0].
pub fn encodeExpand(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, stream: View, pre: View, d: Expand) void {
    e.setPipeline(p.expand);
    bind(e, .{ x, stream, pre });
    e.setValue(d, 3);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}
