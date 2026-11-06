//! Four-copy mHC; callers validate shapes/storage and own previous-pre coefficients.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const View = @import("view.zig").View;
pub const Pipelines = struct {
    rms: mtl.Pipeline,
    split: mtl.Pipeline,
    collapse: mtl.Pipeline,
    post: mtl.Pipeline,
    expand: mtl.Pipeline,
    mix_dense: mtl.Pipeline,
    dense_col: mtl.Pipeline,
    gate_tree: mtl.Pipeline,
    gate_tree4: mtl.Pipeline,
    /// ROUND-349: the wide gate -- eight rows and four columns a threadgroup, the tiles staged in threadgroup
    /// memory. The order is the shipped one's, so the logits are bit-identical.
    gate_tree8: mtl.Pipeline,
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
        const mix_dense = try mtl.Pipeline.init(device, lib, "dsv41_mix_dense", false);
        errdefer mix_dense.deinit();
        const dense_col = try mtl.Pipeline.init(device, lib, "dsv41_dense_col", false);
        errdefer dense_col.deinit();
        const expand = try mtl.Pipeline.init(device, lib, "dsv41_hc_expand", false);
        const gate_tree = try mtl.Pipeline.init(device, lib, "dsv41_gate_tree", false);
        errdefer gate_tree.deinit();
        const gate_tree4 = try mtl.Pipeline.init(device, lib, "dsv41_gate_tree4", false);
        errdefer gate_tree4.deinit();
        const gate_tree8 = try mtl.Pipeline.init(device, lib, "dsv41_gate_tree8", false);
        errdefer gate_tree8.deinit();
        return .{ .rms = rms, .split = split, .collapse = collapse, .post = post, .expand = expand, .mix_dense = mix_dense, .dense_col = dense_col, .gate_tree = gate_tree, .gate_tree4 = gate_tree4, .gate_tree8 = gate_tree8 };
    }
    pub fn deinit(p: *Pipelines) void {
        p.rms.deinit();
        p.split.deinit();
        p.collapse.deinit();
        p.post.deinit();
        p.expand.deinit();
        p.mix_dense.deinit();
        p.dense_col.deinit();
        p.gate_tree.deinit();
        p.gate_tree4.deinit();
        p.gate_tree8.deinit();
    }
};
/// The fused boundary (kernels/metal/deepseek_v41/hc_boundary.metal): expand+mix+squares in one pass over
/// the streams, then the split with the lagged collapse and the norm in one threadgroup a row.
pub const BoundaryPipelines = struct {
    expand_mix: mtl.Pipeline,
    mix_part: mtl.Pipeline,
    boundary: mtl.Pipeline,
    plain: mtl.Pipeline,
    /// ROUND-407: the launch-overhead probe (a dependent, work-free dispatch).
    probe: mtl.Pipeline,
    /// ROUND-407: the INDEPENDENT form of the same probe (each dispatch its own sink buffer, so the
    /// chain carries no dependency between the probe dispatches) and the sinks it writes into.
    probe_indep: mtl.Pipeline,
    probe_sinks: [64]mtl.Buffer,
    pub fn load(device: mtl.Device) !BoundaryPipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_hc_boundary, options);
        defer lib.deinit();
        const expand_mix = try mtl.Pipeline.init(device, lib, "tf_hc_expand_mix", false);
        errdefer expand_mix.deinit();
        const mix_part = try mtl.Pipeline.init(device, lib, "tf_hc_mix_part", false);
        errdefer mix_part.deinit();
        const boundary = try mtl.Pipeline.init(device, lib, "tf_hc_boundary_mix", false);
        errdefer boundary.deinit();
        const plain = try mtl.Pipeline.init(device, lib, "tf_hc_boundary_plain", false);
        const probe = try mtl.Pipeline.init(device, lib, "tf_hc_probe", false);
        const probe_indep = try mtl.Pipeline.init(device, lib, "tf_hc_probe_indep", false);
        var sinks: [64]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (sinks[0..made]) |b| b.deinit();
        while (made < sinks.len) : (made += 1)
            sinks[made] = try device.buffer(256, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return .{
            .expand_mix = expand_mix,
            .mix_part = mix_part,
            .boundary = boundary,
            .plain = plain,
            .probe = probe,
            .probe_indep = probe_indep,
            .probe_sinks = sinks,
        };
    }
    pub fn deinit(p: *BoundaryPipelines) void {
        p.expand_mix.deinit();
        p.mix_part.deinit();
        p.boundary.deinit();
        p.plain.deinit();
        p.probe.deinit();
        p.probe_indep.deinit();
        for (p.probe_sinks) |b| b.deinit();
    }
};
/// The boundary's fixed shape: the streams' width, the split's eps and iteration count, the partial sums a row.
pub const boundary_width: u32 = 5120;
pub const boundary_slices: u32 = 20;
pub fn boundaryPartBytes(rows: usize) usize {
    return rows * boundary_slices * 25 * 4;
}
pub const Boundary = struct { rows: u32, expand: bool, mixes: bool = true, eps: f32, hc_eps: f32, iters: u32 };
/// One boundary: expand the pending branch into `x_new` (or the mixes of `x_old` alone), then the split with
/// `pre_in`'s lagged collapse, `norm`'s weight and this boundary's coefficients out.
pub fn encodeBoundary(e: mtl.ComputeEncoder, p: *const BoundaryPipelines, d: Boundary, v: anytype) void {
    const expand_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_HC_EXPAND") != null) 2 else 1;
    for (0..expand_passes) |_| {
        if (d.expand) {
            e.setPipeline(p.expand_mix);
            inline for (.{ v.x_old, v.branch, v.post, v.comb, v.fn_, v.part, v.x_new }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
            e.dispatchGroups(.{ .width = 20, .height = d.rows }, .{ .width = 256 });
        } else {
            e.setPipeline(p.mix_part);
            inline for (.{ v.x_old, v.fn_, v.part }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
            e.dispatchGroups(.{ .width = 20, .height = d.rows }, .{ .width = 256 });
        }
    }
    // ROUND-407: THE LAUNCH-OVERHEAD PROBE. N work-free dispatches on the boundary's own critical path:
    // the slope against N is the GPU's dispatch latency, the piece every glue fusion would save.
    // TF_DSV41_HC_PROBE_THREADS sweeps the probe's threadgroup size, so the setup cost's shape (the mHC's
    // split runs 1024 threads on ONE threadgroup) is priced too. TF_DSV41_HC_PROBE_INDEP=1 gives each
    // probe dispatch its OWN sink buffer, so the probe dispatches carry no dependency between them: the
    // difference between the two forms is the DEPENDENT-dispatch latency (the part only a fold removes).
    if (probeDispatches()) |n| {
        if (probeIndependent()) {
            e.setPipeline(p.probe_indep);
            e.setBuffer(v.part.buffer, v.part.offset, 0);
            for (0..n) |i| {
                const sink = p.probe_sinks[i % p.probe_sinks.len];
                e.setBuffer(sink, 0, 1);
                e.dispatchGroups(.{ .width = d.rows }, .{ .width = probeThreads() });
            }
        } else {
            e.setPipeline(p.probe);
            e.setBuffer(v.part.buffer, v.part.offset, 0);
            e.setBuffer(v.part.buffer, v.part.offset, 1);
            for (0..n) |_| e.dispatchGroups(.{ .width = d.rows }, .{ .width = probeThreads() });
        }
    }
    e.barrier();
    if (std.c.getenv("TF_DSV41_SKIP_HC_EXPAND") != null) return;
    if (d.mixes) {
        e.setPipeline(p.boundary);
    } else e.setPipeline(p.plain);
    inline for (.{ if (d.expand) v.x_new else v.x_old, v.part, v.base, v.scale, v.pre_in, v.norm }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
    e.setValue(BoundaryArgs{ .eps = d.eps, .hc_eps = d.hc_eps, .iters = @intCast(d.iters) }, 6);
    inline for (.{ v.pre_out, v.post_out, v.comb_out, v.normed }, 7..) |r, i| e.setBuffer(r.buffer, r.offset, i);
    // Measurement: the split is idempotent (it reads X/PART and writes the same outputs), so a second launch
    // prices it without changing a value.
    const split_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_HC_SPLIT") != null) 2 else 1;
    for (0..split_passes) |_| if (std.c.getenv("TF_DSV41_SKIP_HC_SPLIT") == null) e.dispatchGroups(.{ .width = d.rows }, .{ .width = 1024 });
    e.barrier();
}
pub const BoundaryArgs = extern struct { eps: f32, hc_eps: f32, iters: i32 };

/// ROUND-407: the probe's dispatch count (zero = off) and its threadgroup size.
var probe_count: ?u32 = null;
var probe_threads: ?u32 = null;
/// ROUND-407: the independent form (each probe dispatch its own sink buffer).
var probe_indep_flag: ?bool = null;
fn probeIndependent() bool {
    if (probe_indep_flag == null) probe_indep_flag = std.c.getenv("TF_DSV41_HC_PROBE_INDEP") != null;
    return probe_indep_flag.?;
}
fn probeDispatches() ?u32 {
    if (probe_count == null) {
        const raw = std.c.getenv("TF_DSV41_HC_PROBE");
        probe_count = if (raw) |r| std.fmt.parseInt(u32, std.mem.trim(u8, std.mem.span(r), " \t"), 10) catch 0 else 0;
    }
    return if (probe_count.? == 0) null else probe_count.?;
}
fn probeThreads() u32 {
    if (probe_threads == null) {
        const raw = std.c.getenv("TF_DSV41_HC_PROBE_THREADS");
        probe_threads = if (raw) |r| std.fmt.parseInt(u32, std.mem.trim(u8, std.mem.span(r), " \t"), 10) catch 256 else 256;
        if (probe_threads.? < 32) probe_threads = 32;
        if (probe_threads.? > 1024) probe_threads = 1024;
    }
    return probe_threads.?;
}

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
pub const MixDense = extern struct { rows: u32, k: u32, n: u32, x_stride: u32 };
/// FP32 mixes[rows,24] = stream[rows,20480] @ projection[24,20480]: one threadgroup a column, so the weight
/// matrix is read coalesced instead of by a handful of lanes.
pub fn encodeMixDense(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: MixDense) void {
    e.setPipeline(p.mix_dense);
    bind(e, .{ x, w, y });
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of(d.n, d.rows, 1), mtl.Size.of(512, 1, 1));
}

pub const ColDense = extern struct { rows: u32, k: u32, n: u32, x_stride: u32, bf16_w: u32 };
/// A narrow projection one threadgroup a column; the generic dense gives `n/16` threadgroups, which starves
/// a row-1 call of the bandwidth the weight matrix needs.
pub fn encodeColDense(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: ColDense) void {
    e.setPipeline(p.dense_col);
    bind(e, .{ x, w, y });
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of(d.n, d.rows, 1), mtl.Size.of(512, 1, 1));
}

pub const GateTree = extern struct { rows: u32, k: u32, n: u32, x_stride: u32, bf16_w: u32 };
/// The bit-exact tiled twin of encodeColDense: a 4x4 cell tile per 128-thread threadgroup, the shipped reduction
/// order reproduced in registers. Same arguments and destination as encodeColDense.
pub fn encodeGateTree(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: GateTree) void {
    e.setPipeline(p.gate_tree);
    bind(e, .{ x, w, y });
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of((d.n + 3) / 4, (d.rows + 3) / 4, 1), mtl.Size.of(128, 1, 1));
}

pub const GateTree4 = extern struct { rows: u32, k: u32, n: u32, x_stride: u32, bf16_w: u32 };
/// The vectorized bit-exact gate: one warp a 2x2 cell tile, the shipped reduction order in registers, 1.93x the
/// shipped kernel at 3,619 x 5,120 x 384 and bit-identical to it (tf-gate-bench).
pub fn encodeGateTree4(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: GateTree4) void {
    e.setPipeline(p.gate_tree4);
    bind(e, .{ x, w, y });
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of((d.n + 1) / 2, (d.rows + 1) / 2, 1), mtl.Size.of(32, 1, 1));
}

/// ROUND-349: the wide gate's dispatch: eight rows and four columns a threadgroup, eight warps.
pub fn encodeGateTree8(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: GateTree4) void {
    e.setPipeline(p.gate_tree8);
    bind(e, .{ x, w, y });
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of((d.n + 3) / 4, (d.rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
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
    comptime {
        @import("metal/index_mirror.zig").check(Post, sources.deepseek_v41_mhc, "Post");
    }
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
