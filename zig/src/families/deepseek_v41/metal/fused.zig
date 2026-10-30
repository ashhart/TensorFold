//! Decode launches that split or fuse the phase-1 glue (fused.metal): the same bits in fewer, wider dispatches.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// The hc mix's sums: 24 matvecs and the squares, eight simdgroup partials each a row.
pub const mix_sums = 25;
pub const mix_parts = 8;

const names = [_][:0]const u8{ "tf_dsv41_f_hcmix_part", "tf_dsv41_f_hcmix_fin", "tf_dsv41_f_kv", "tf_dsv41_f_swiglu", "tf_dsv41_f_joinpost", "tf_dsv41_f_actq" };

/// The norm kernels' widths: values a thread holds; a row of `dim` takes the first NV with 256 NV >= dim.
const nvs = [_]u32{ 1, 2, 5, 8, 20, 32 };

fn nvOf(dim: u32) !usize {
    for (nvs, 0..) |v, i| if (256 * v >= dim) return i;
    return error.BadNormShape;
}

pub const Pipelines = struct {
    hc_part: mtl.Pipeline,
    hc_fin: mtl.Pipeline,
    kv: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    joinpost: mtl.Pipeline,
    actq: mtl.Pipeline,
    prenorm: [nvs.len]mtl.Pipeline,
    normq: [nvs.len]mtl.Pipeline,
    /// the wide rows' prenorm in two launches: a simdgroup a group for the sums (20 or 32 steps), then a thread a value
    pre_a: [2]mtl.Pipeline,
    pre_b: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_fused, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        inline for (@typeInfo(Pipelines).@"struct".field_names[0..names.len], names) |f, n| @field(p, f) = try mtl.Pipeline.init(device, lib, n, false);
        inline for (nvs, 0..) |v, i| {
            p.prenorm[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_dsv41_f_prenorm_v{d}", .{v}), false);
            p.normq[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_dsv41_f_normq_v{d}", .{v}), false);
        }
        p.pre_a[0] = try mtl.Pipeline.init(device, lib, "tf_dsv41_f_pre_a_v20", false);
        p.pre_a[1] = try mtl.Pipeline.init(device, lib, "tf_dsv41_f_pre_a_v32", false);
        p.pre_b = try mtl.Pipeline.init(device, lib, "tf_dsv41_f_pre_b", false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.pre_a) |x| x.deinit();
        p.pre_b.deinit();
        inline for (@typeInfo(Pipelines).@"struct".field_names[0..names.len]) |f| @field(p, f).deinit();
        for (p.prenorm) |x| x.deinit();
        for (p.normq) |x| x.deinit();
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

pub const Hc = struct { dim: u32, iters: u32, eps: f32, hc_eps: f32, alpha: f32 };

/// Bytes of the partial sums a launch of `rows` needs.
pub fn mixScratch(rows: usize) usize {
    return rows * mix_sums * mix_parts * 4;
}

pub const Mix = struct { x: Ref, w: [3]Ref, out: [3]Ref, part: Ref, h: Hc };

fn mixArgs(m: Mix, rows: u32) extern struct { dim: u32, rows: u32, iters: u32, eps: f32, hc_eps: f32, alpha: f32 } {
    return .{ .dim = m.h.dim, .rows = rows, .iters = m.h.iters, .eps = m.h.eps, .hc_eps = m.h.hc_eps, .alpha = m.h.alpha };
}

fn mixPart(p: *const Pipelines, e: mtl.ComputeEncoder, m: Mix, rows: u32) void {
    e.setPipeline(p.hc_part);
    bind(e, &.{ m.x, m.w[0], m.part });
    e.setValue(mixArgs(m, rows), 3);
    e.dispatchGroups(mtl.Size.of(mix_sums, mix_parts, rows), mtl.Size.of(32, 1, 1));
}

fn mixFin(p: *const Pipelines, e: mtl.ComputeEncoder, m: Mix, rows: u32) void {
    e.setPipeline(p.hc_fin);
    bind(e, &.{ m.part, m.w[1], m.w[2], m.out[0], m.out[1], m.out[2] });
    e.setValue(mixArgs(m, rows), 6);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
}

/// glue_tmp's hcmix in two launches: (24 + 1) x 8 simdgroups a row for the sums, then a simdgroup a row for the rest.
pub fn hcmix(p: *const Pipelines, e: mtl.ComputeEncoder, m: Mix, rows: u32) void {
    mixPart(p, e, m, rows);
    e.barrier();
    mixFin(p, e, m, rows);
}

/// A sublayer's input: the hc mixes' sums (`m`) beside `prenorm` (`n`); `mixFinish` runs after the next barrier.
pub fn mixNorm(p: *const Pipelines, e: mtl.ComputeEncoder, m: Mix, n: Norm, rows: u32) !void {
    mixPart(p, e, m, rows);
    if (n.dim > 2048) {
        try preA(p, e, n, rows);
        e.barrier();
        return preB(p, e, n, rows);
    }
    try prenorm(p, e, n, rows);
}

/// The hc mixes' sums alone (`mixFinish` reads them after the next barrier), for a caller with its own norm.
pub fn mixSums(p: *const Pipelines, e: mtl.ComputeEncoder, m: Mix, rows: u32) void {
    mixPart(p, e, m, rows);
}

/// The hc mixes from `mixNorm`'s sums: pre, post and comb.
pub fn mixFinish(p: *const Pipelines, e: mtl.ComputeEncoder, m: Mix, rows: u32) void {
    mixFin(p, e, m, rows);
}

const NormArgs = extern struct { dim: u32, rows: u32, eps: f32 };

/// Bytes of the wide prenorm's partial sums for `rows`.
pub fn normScratch(rows: usize) usize {
    return rows * 8 * 4;
}

/// glue_tmp's hcpre (x by pre into h), rmsnorm (hn) and actq (hq); rows over 2048 wide in two launches via `part`.
pub const Norm = struct { x: Ref, pre: Ref, w: Ref, h: Ref, hn: Ref, hq: Ref, part: Ref, dim: u32, eps: f32 };

pub fn prenorm(p: *const Pipelines, e: mtl.ComputeEncoder, n: Norm, rows: u32) !void {
    if (n.dim % 32 != 0) return error.BadNormShape;
    if (n.dim > 2048) {
        try preA(p, e, n, rows);
        e.barrier();
        return preB(p, e, n, rows);
    }
    e.setPipeline(p.prenorm[try nvOf(n.dim)]);
    bind(e, &.{ n.x, n.pre, n.w, n.h, n.hn, n.hq });
    e.setValue(NormArgs{ .dim = n.dim, .rows = rows, .eps = n.eps }, 6);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

fn preA(p: *const Pipelines, e: mtl.ComputeEncoder, n: Norm, rows: u32) !void {
    if (n.dim % 32 != 0 or n.dim > 256 * 32) return error.BadNormShape;
    e.setPipeline(p.pre_a[if (n.dim <= 256 * 20) 0 else 1]);
    bind(e, &.{ n.x, n.pre, n.h, n.part });
    e.setValue(NormArgs{ .dim = n.dim, .rows = rows, .eps = n.eps }, 4);
    e.dispatchGroups(mtl.Size.of(8, rows, 1), mtl.Size.of(32, 1, 1));
}

fn preB(p: *const Pipelines, e: mtl.ComputeEncoder, n: Norm, rows: u32) void {
    e.setPipeline(p.pre_b);
    bind(e, &.{ n.h, n.part, n.w, n.hn, n.hq });
    e.setValue(NormArgs{ .dim = n.dim, .rows = rows, .eps = n.eps }, 5);
    e.dispatchThreads(mtl.Size.of(n.dim, rows, 1), mtl.Size.of(256, 1, 1));
}

/// RMSNorm of rows x [rows][dim] (weight w) into y and its act_quant round trip into yq, a group of 256 a row.
pub fn normq(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, yq: Ref, rows: u32, dim: u32, eps: f32) !void {
    if (dim % 32 != 0) return error.BadNormShape;
    e.setPipeline(p.normq[try nvOf(dim)]);
    bind(e, &.{ x, x, w, x, y, yq });
    e.setValue(NormArgs{ .dim = dim, .rows = rows, .eps = eps }, 6);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// Window keys kv [rows][512]: kv_norm into kvn, RoPE into kvr, act_quant into kvq and ring rows dest [rows][reps].
pub const Kv = struct { kv: Ref, w: Ref, kvn: Ref, kvr: Ref, kvq: Ref, ring: Ref, pos: Ref, table: Ref, dest: Ref, reps: u32 };

pub fn kv(p: *const Pipelines, e: mtl.ComputeEncoder, b: Kv, rows: u32, dim: u32, eps: f32) !void {
    if (dim != 512) return error.BadKvShape;
    e.setPipeline(p.kv);
    bind(e, &.{ b.kv, b.w, b.kvn, b.kvr, b.kvq, b.ring, b.pos, b.table, b.dest });
    e.setValue(extern struct { dim: u32, rows: u32, reps: u32, eps: f32 }{ .dim = dim, .rows = rows, .reps = b.reps, .eps = eps }, 9);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// The clamped SwiGLU of n gate and up values into y and its act_quant round trip into yq; n a multiple of 32.
pub fn swiglu(p: *const Pipelines, e: mtl.ComputeEncoder, g: Ref, u: Ref, y: Ref, yq: Ref, n: u32, limit: f32) void {
    e.setPipeline(p.swiglu);
    bind(e, &.{ g, u, y, yq });
    e.setValue(extern struct { n: u32, limit: f32 }{ .n = n, .limit = limit }, 4);
    e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

/// b = bf16(routed fp32 + shared bf16), then hc_post of b and the residual streams res by post and comb into y.
pub fn joinpost(p: *const Pipelines, e: mtl.ComputeEncoder, routed: Ref, shared: Ref, b: Ref, res: Ref, post: Ref, comb: Ref, y: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.joinpost);
    bind(e, &.{ routed, shared, b, res, post, comb, y });
    e.setValue([2]u32{ dim, rows }, 7);
    e.dispatchThreads(mtl.Size.of(dim, rows, 1), mtl.Size.of(64, 1, 1));
}

/// glue.zig's act_quant round trip (its Strides) a thread a value, a simdgroup a block; k a multiple of 32.
pub fn actq(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, s: anytype) !void {
    if (s.k % 32 != 0) return error.BadQuantShape;
    e.setPipeline(p.actq);
    bind(e, &.{ x, y });
    e.setValue([7]u32{ s.k, s.nb, s.na, s.sa, s.sb, s.ta, s.tb }, 2);
    e.dispatchThreads(mtl.Size.of(s.k, s.nb, s.na), mtl.Size.of(@min(s.k, 256), 1, 1));
}
