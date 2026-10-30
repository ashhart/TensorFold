//! Decode launches that fold a dependent glue launch into its neighbour (fold.metal): the same bits, one wave fewer.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// The widest row `prenorm` takes (its threadgroup holds the row in fp32).
pub const max_dim = 5120;
/// Threads of the prenorm's group: the collapse a thread a value, the sum of squares on the first 256.
pub const prenorm_threads = 1024;
pub const post_threads = 64;

pub const Pipelines = struct {
    combpost: mtl.Pipeline,
    prenorm: mtl.Pipeline,
    postnorm: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.deepseek_v41_fold);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.combpost = try mtl.Pipeline.init(device, lib, "tf_dsv41_o_combpost", false);
        errdefer p.combpost.deinit();
        p.prenorm = try mtl.Pipeline.init(device, lib, "tf_dsv41_o_prenorm", false);
        errdefer p.prenorm.deinit();
        p.postnorm = try mtl.Pipeline.init(device, lib, "tf_dsv41_o_postnorm", false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.combpost.deinit();
        p.prenorm.deinit();
        p.postnorm.deinit();
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// The routed experts' outputs [pair position][dim] (bf16), where, picks and weights; their fp32 sum goes to `sum`.
pub const Routed = struct { out: Ref, where: Ref, weights: Ref, ids: Ref, sum: Ref, topk: u32, per_group: u32 };

/// Kimi's combine of `r` plus the shared expert sh into b (bf16), then hc_post of b and res by post and comb into y.
pub fn combpost(p: *const Pipelines, e: mtl.ComputeEncoder, r: Routed, sh: Ref, b: Ref, res: Ref, post: Ref, comb: Ref, y: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.combpost);
    bind(e, &.{ r.out, r.where, r.weights, r.ids, sh, b, res, post, comb, y });
    e.setValue([5]u32{ dim, rows, r.topk, r.per_group, 8 }, 10);
    e.setBuffer(r.sum.buf, r.sum.off, 11);
    e.dispatchThreads(mtl.Size.of(dim, rows, 1), mtl.Size.of(post_threads, 1, 1));
}

/// hc_pre of x [rows][4][dim] by pre [rows][4] into h, its RMS norm (fp32 weight w) into hn and act_quant into hq.
pub fn prenorm(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, pre: Ref, w: Ref, h: Ref, hn: Ref, hq: Ref, rows: u32, dim: u32, eps: f32) !void {
    if (dim % 32 != 0 or dim > max_dim or dim < 256) return error.BadNormShape;
    e.setPipeline(p.prenorm);
    bind(e, &.{ x, pre, w, h, hn, hq, x, x, x, x, x });
    norm(e, rows, dim, eps);
}

/// hc_post's inputs: the branch [rows][dim] bf16, post [rows][4], comb [rows][4][4]; its streams go to x1.
pub const Post = struct { branch: Ref, post: Ref, comb: Ref, x1: Ref };

/// `prenorm` of hc_post(branch, x) (written to q.x1) instead of x: x is then the residual streams [rows][4][dim].
pub fn prenormPost(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, pre: Ref, w: Ref, h: Ref, hn: Ref, hq: Ref, q: Post, rows: u32, dim: u32, eps: f32) !void {
    if (dim % 32 != 0 or dim > max_dim or dim < 256) return error.BadNormShape;
    e.setPipeline(p.postnorm);
    bind(e, &.{ x, pre, w, h, hn, hq, x, q.branch, q.post, q.comb, q.x1 });
    norm(e, rows, dim, eps);
}

fn norm(e: mtl.ComputeEncoder, rows: u32, dim: u32, eps: f32) void {
    e.setValue(extern struct { dim: u32, rows: u32, eps: f32 }{ .dim = dim, .rows = rows, .eps = eps }, 6);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(prenorm_threads, 1, 1));
}
