//! Launch encoders of the phase-1 glue (glue_tmp.metal), the MoE route and the index top-512.
const std = @import("std");
const mtl = @import("metal");
const mvk = @import("mv.zig");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
const tg = 256;

pub const names = [_][:0]const u8{
    "tf_dsv41_g_rmsnorm", "tf_dsv41_g_rope",    "tf_dsv41_g_fp4pack", "tf_dsv41_g_fp4rt",
    "tf_dsv41_g_pool2",   "tf_dsv41_g_cand",    "tf_dsv41_g_hcmix",   "tf_dsv41_g_hcpre",
    "tf_dsv41_g_hcpost",  "tf_dsv41_g_scatter", "tf_dsv41_g_embed",   "tf_dsv41_g_hcmean",
};
const Kind = enum { rmsnorm, rope, fp4pack, fp4rt, pool2, cand, hcmix, hcpre, hcpost, scatter, embed, hcmean };

/// The glue kernels by name.
pub const Pipelines = struct {
    p: [names.len]mtl.Pipeline,
    mv: mvk.Pipelines,

    pub fn load(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = undefined;
        for (names, 0..) |n, i| out.p[i] = try mtl.Pipeline.init(device, lib, n, false);
        out.mv = try mvk.Pipelines.load(device);
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.p) |x| x.deinit();
        p.mv.deinit();
    }

    fn of(p: *const Pipelines, k: Kind) mtl.Pipeline {
        return p.p[@backingInt(k)];
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

fn grid(e: mtl.ComputeEncoder, x: usize, y: usize, z: usize) void {
    e.dispatchThreads(mtl.Size.of(x, y, z), mtl.Size.of(@min(x, 64), 1, 1));
}

/// RMSNorm of `rows` bf16 rows of `dim` with an fp32 weight, bf16 out.
pub fn rmsnorm(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: u32, dim: u32, eps: f32) void {
    e.setPipeline(p.of(.rmsnorm));
    bind(e, &.{ x, w, y });
    e.setValue(extern struct { dim: u32, ins: u32, outs: u32, eps: f32 }{ .dim = dim, .ins = dim, .outs = dim, .eps = eps }, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(tg, 1, 1));
}

/// Element strides of the act_quant round trip: value (a, b, k) reads a*sa + b*sb + k and writes a*ta + b*tb + k.
pub const Strides = struct { k: u32, nb: u32, na: u32 = 1, sa: u32 = 0, sb: u32, ta: u32 = 0, tb: u32 };

/// Rows of `heads` heads of `head` dims, copied with the last 64 dims of each head rotated at the row's position.
pub fn rope(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, pos: Ref, table: Ref, rows: u32, heads: u32, head: u32) void {
    e.setPipeline(p.of(.rope));
    bind(e, &.{ x, y, pos, table });
    e.setValue([5]u32{ head / 2, heads, head, heads * head, head - 64 }, 4);
    grid(e, head / 2, heads, rows);
}

/// A bf16-weight matvec (mv.zig): x [rows][k] bf16, w [n][k] bf16, y [rows][n] in fp32 or bf16.
pub fn mv(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: u32, k: u32, n: u32, f32_out: bool) void {
    if (rows == 0) return;
    const call = mvk.Call{ .k = k, .n = n, .rows = rows, .out = if (f32_out) .f32 else .bf16 };
    mvk.encode(&p.mv, e, .{ .buf = x.buf, .off = x.off }, .{ .buf = w.buf, .off = w.off }, .{ .buf = y.buf, .off = y.off }, call) catch unreachable;
}

/// Candidate blocks of each row's scores [rows][keys]: a byte a block, `blocks` a row; `bs` is fp32 scratch alike.
pub fn cand(p: *const Pipelines, e: mtl.ComputeEncoder, s: Ref, vis: Ref, bs: Ref, mask: Ref, rows: u32, keys: u32, blocks: u32, topk: u32) void {
    e.setPipeline(p.of(.cand));
    bind(e, &.{ s, vis, bs, mask });
    e.setValue([3]u32{ keys, blocks, topk }, 4);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(tg, 1, 1));
}

pub const Hc = struct { dim: u32, iters: u32, eps: f32, hc_eps: f32, alpha: f32 };

/// The mixes of each row's four streams: pre [rows][4], post [rows][4], comb [rows][4][4].
pub fn hcmix(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: [3]Ref, out: [3]Ref, rows: u32, h: Hc) void {
    e.setPipeline(p.of(.hcmix));
    bind(e, &.{ x, w[0], w[1], w[2], out[0], out[1], out[2] });
    e.setValue(extern struct { dim: u32, iters: u32, eps: f32, hc_eps: f32, alpha: f32 }{ .dim = h.dim, .iters = h.iters, .eps = h.eps, .hc_eps = h.hc_eps, .alpha = h.alpha }, 7);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(tg, 1, 1));
}

/// hc_pre: x [rows][4][dim] collapsed by pre [rows][4] into y [rows][dim].
pub fn hcpre(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, pre: Ref, y: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.of(.hcpre));
    bind(e, &.{ x, pre, y });
    e.setValue([2]u32{ dim, rows }, 3);
    grid(e, dim, rows, 1);
}

/// hc_post: the branch [rows][dim] and the residual [rows][4][dim] into y [rows][4][dim].
pub fn hcpost(p: *const Pipelines, e: mtl.ComputeEncoder, branch: Ref, res: Ref, post: Ref, comb: Ref, y: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.of(.hcpost));
    bind(e, &.{ branch, res, post, comb, y });
    e.setValue([2]u32{ dim, rows }, 5);
    grid(e, dim, rows, 1);
}

/// Rows of `bytes` (a multiple of 4) from x row i / rep to y row dest[i], `items` of them.
pub fn scatter(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, dest: Ref, items: u32, bytes: u32, rep: u32) void {
    e.setPipeline(p.of(.scatter));
    bind(e, &.{ x, y, dest });
    e.setValue([3]u32{ bytes / 4, items, rep }, 3);
    grid(e, bytes / 4, items, 1);
}

/// The rows of a bf16 table [vocab][dim] at `ids` [rows] (u32) into all four streams of y; pre gets (1, 0, 0, 0).
pub fn embed(p: *const Pipelines, e: mtl.ComputeEncoder, table: Ref, ids: Ref, y: Ref, pre: Ref, rows: u32, dim: u32) void {
    e.setPipeline(p.of(.embed));
    bind(e, &.{ table, ids, y, pre });
    e.setValue([2]u32{ dim, rows }, 4);
    grid(e, dim, rows, 1);
}

/// The mean of each row's four streams x [rows][4][dim], bf16, row r at y + r * stride values.
pub fn hcmean(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32, dim: u32, stride: u32) void {
    e.setPipeline(p.of(.hcmean));
    bind(e, &.{ x, y });
    e.setValue([3]u32{ dim, rows, stride }, 2);
    grid(e, dim, rows, 1);
}

/// The MoE route: logits [rows][E] fp32, bias [E] (bias_vl beside it), image [rows] bytes; picks and weights [rows][k].
pub const Route = struct { logits: Ref, bias: Ref, bias_vl: Ref, image: Ref, picks: Ref, weights: Ref };

pub fn route(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, r: Route, rows: u32, topk: u32, scale: f32) void {
    e.setPipeline(pipe);
    bind(e, &.{ r.logits, r.bias, r.bias_vl, r.image });
    e.setValue(extern struct { rows: i32, topk: i32, scale: f32, eps: f32 }{ .rows = @intCast(rows), .topk = @intCast(topk), .scale = scale, .eps = 1e-20 }, 4);
    e.setBuffer(r.picks.buf, r.picks.off, 5);
    e.setBuffer(r.weights.buf, r.weights.off, 6);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
}

/// The index top-512 of scores [rows][width] (row stride `width`), counting visible[row] keys; selected [rows][512].
pub fn topk512(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, scores: Ref, visible: Ref, selected: Ref, rows: u32, width: u32) void {
    e.setPipeline(pipe);
    e.setValue([1]u32{width}, 0);
    e.setBuffer(scores.buf, scores.off, 1);
    e.setBuffer(visible.buf, visible.off, 2);
    e.setBuffer(selected.buf, selected.off, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
}
