//! DeepSeek V4.1's hyper-connection boundary: the mixes, hc_pre with the lagged pre mix, and hc_post, in fixed orders.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const tg = 256;
pub const mult = 4;

const MixArgs = extern struct { dim: u32, rows: u32, iters: u32, eps: f32, hc_eps: f32, alpha: f32 };
const RowArgs = extern struct { dim: u32, rows: u32 };

/// The coefficients one sublayer makes, and the ones it consumes (made by the sublayer before it).
pub const Hc = struct {
    dim: u32, // hc_mult * hidden: the flattened stream the mixes are read from
    iters: u32,
    eps: f32,
    hc_eps: f32,
    alpha: f32,
};

pub const Pipelines = struct {
    mix: mtl.Pipeline,
    pre: mtl.Pipeline,
    post: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.hc);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.mix = try mtl.Pipeline.init(device, lib, "tf_dsv41_hc_mix", false);
        p.pre = try mtl.Pipeline.init(device, lib, "tf_dsv41_hc_pre", false);
        p.post = try mtl.Pipeline.init(device, lib, "tf_dsv41_hc_post", false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.mix.deinit();
        p.pre.deinit();
        p.post.deinit();
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// hc_mixes and hc_split_sinkhorn of each row's stream: pre [rows][4], post [rows][4], comb [rows][4][4], fp32.
pub fn mix(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: [3]Ref, out: [3]Ref, rows: u32, h: Hc) void {
    e.setPipeline(p.mix);
    bind(e, &.{ x, w[0], w[1], w[2], out[0], out[1], out[2] });
    e.setValue(MixArgs{ .dim = h.dim, .rows = rows, .iters = h.iters, .eps = h.eps, .hc_eps = h.hc_eps, .alpha = h.alpha }, 7);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(tg, 1, 1));
}

/// hc_pre: x [rows][4][dim] collapsed by the previous sublayer's `pre` [rows][4] into y [rows][dim] bf16 (O4).
pub fn pre(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, pre_mix: Ref, y: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.pre);
    bind(e, &.{ x, pre_mix, y });
    e.setValue(RowArgs{ .dim = dim, .rows = rows }, 3);
    e.dispatchThreads(mtl.Size.of(dim, rows, 1), mtl.Size.of(64, 1, 1));
}

/// hc_post (model.py:964-969): the branch [rows][dim] and the residual [rows][4][dim] into y [rows][4][dim] (O5).
pub fn post(p: *const Pipelines, e: mtl.ComputeEncoder, branch: Ref, res: Ref, post_mix: Ref, comb: Ref, y: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.post);
    bind(e, &.{ branch, res, post_mix, comb, y });
    e.setValue(RowArgs{ .dim = dim, .rows = rows }, 5);
    e.dispatchThreads(mtl.Size.of(dim, rows, 1), mtl.Size.of(64, 1, 1));
}
