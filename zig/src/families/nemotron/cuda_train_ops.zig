//! The training kernels (train.cu, train_mixers.cu) as calls on a stream: the change, the loss, products backward.
const std = @import("std");
const cuda = @import("cuda");
const dims = @import("slide_dims.zig");
const weights = @import("cuda_weights.zig");

/// A Mamba layer's shape as train_mixers.cu reads it, with the dt clamp.
pub const Shape = extern struct { rows: c_int, heads: c_int, dh: c_int, groups: c_int, n: c_int, inner: c_int, conv: c_int, proj: c_int, lo: f32, hi: f32 };

/// Attention's shape as train_mixers.cu reads it.
pub const Heads = extern struct { rows: c_int, heads: c_int, kv_heads: c_int, dim: c_int, nqkv: c_int, scale: f32 };

/// tf_slide_out stages 16 rows of xa at every rank the change may use.
pub const out_smem: u32 = 16 * dims.max_rank * 4;

/// Rows a decode window holds at most: the change takes its window kernels up to here.
const window_rows = 16;

const train_names = .{ "slide_xa", "slide_out", "slide_xa_rows", "slide_out_rows", "gate", "lora_db", "lora_dxa", "lora_dx", "project", "sketch", "softmax", "rms_back", "dequant_t", "narrow", "widen", "gemm", "gemm_sum", "pairs_in", "experts_back", "relu2_back", "pairs_out", "route_back", "rest_back", "adam" };
const mixer_names = .{ "dt", "ssm_fwd", "ssm_back", "ssm_bc", "gate_back", "conv_back", "dt_back", "attn_q", "attn_kv" };

/// Every training kernel by its extern "C" name (tf_slide_* for the forward's two, tf_train_* for the rest).
pub const Fns = struct {
    slide_xa: cuda.Function,
    slide_out: cuda.Function,
    slide_xa_rows: cuda.Function,
    slide_out_rows: cuda.Function,
    gate: cuda.Function,
    lora_db: cuda.Function,
    lora_dxa: cuda.Function,
    lora_dx: cuda.Function,
    project: cuda.Function,
    sketch: cuda.Function,
    softmax: cuda.Function,
    rms_back: cuda.Function,
    dequant_t: cuda.Function,
    narrow: cuda.Function,
    widen: cuda.Function,
    gemm: cuda.Function,
    gemm_sum: cuda.Function,
    pairs_in: cuda.Function,
    experts_back: cuda.Function,
    relu2_back: cuda.Function,
    pairs_out: cuda.Function,
    route_back: cuda.Function,
    rest_back: cuda.Function,
    adam: cuda.Function,
    dt: cuda.Function,
    ssm_fwd: cuda.Function,
    ssm_back: cuda.Function,
    ssm_bc: cuda.Function,
    gate_back: cuda.Function,
    conv_back: cuda.Function,
    dt_back: cuda.Function,
    attn_q: cuda.Function,
    attn_kv: cuda.Function,

    /// The functions of train.cu's module and train_mixers.cu's.
    pub fn resolve(train: cuda.Module, mixers: cuda.Module) !Fns {
        var f: Fns = undefined;
        inline for (train_names) |name| {
            const prefix = if (comptime std.mem.startsWith(u8, name, "slide_")) "tf_" else "tf_train_";
            @field(f, name) = try train.function(prefix ++ name);
        }
        inline for (mixer_names) |name| @field(f, name) = try mixers.function("tf_train_" ++ name);
        try f.slide_out.allowDynamicShared(out_smem);
        return f;
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// Launches on one stream.
pub const Train = struct {
    f: *const Fns,
    s: cuda.Stream,
    d: *const cuda.Driver,

    fn go(t: Train, f: cuda.Function, grid: [3]usize, block: u32, shared: u32, a: *cuda.Args) !void {
        const g: cuda.Dim3 = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) };
        try cuda.launch.launch(f, .{ .grid = g, .block = .{ .x = block }, .shared = shared }, t.s, a);
    }

    fn args(xs: anytype) cuda.Args {
        var a: cuda.Args = .{};
        inline for (xs) |x| a.add(x);
        return a;
    }

    /// y += scale (x a^T) b over each row's open blocks: x rows x_stride apart, y's at (r row_mul + row_add) y_stride.
    pub fn adapt(t: Train, ad: weights.Adapter, x: u64, x_stride: usize, y: u64, y_f32: bool, y_stride: usize, row_mul: usize, row_add: usize, rows: usize) !void {
        var a = args(.{ ad.xa, ad.xn, ad.tau, ad.b, ad.rank, y, @as(c_int, @intFromBool(y_f32)), int(y_stride), int(row_mul), int(row_add), int(rows), int(ad.out), dims.scale, dims.unit, int(dims.max_rank) });
        if (rows <= window_rows) {
            // a window: a warp a dot product, so a few rows still fill the GPU
            var b = args(.{ x, int(x_stride), ad.a, ad.rank, ad.xa, ad.xn, int(ad.in), int(dims.max_rank) });
            try t.go(t.f.slide_xa_rows, .{ dims.max_rank / 8, rows, 1 }, 256, 0, &b);
            return t.go(t.f.slide_out_rows, .{ cdiv(ad.out, 32), 1, 1 }, 256, 0, &a);
        }
        try t.xa(ad, x, x_stride, ad.xa, ad.xn, rows);
        try t.go(t.f.slide_out, .{ cdiv(ad.out, 256), cdiv(rows, 16), 1 }, 256, out_smem, &a);
    }

    /// xa = x a^T over every rank in use, and xn = each row's |x|^2, into the given scratch.
    pub fn xa(t: Train, ad: weights.Adapter, x: u64, x_stride: usize, xa_out: u64, xn: u64, rows: usize) !void {
        var a = args(.{ x, int(x_stride), ad.a, ad.rank, xa_out, xn, int(rows), int(ad.in), int(dims.max_rank) });
        try t.go(t.f.slide_xa, .{ dims.max_blocks, cdiv(rows, 16), 1 }, 256, 0, &a);
    }

    /// Each row's closed blocks zeroed in xa, and which are open in gates [rows, max_blocks].
    pub fn gate(t: Train, ad: weights.Adapter, xa_in: u64, xn: u64, gates: u64, rows: usize, rank: usize) !void {
        var a = args(.{ xa_in, xn, ad.tau, gates, int(rows), int(rank), int(dims.max_rank), dims.unit });
        try t.go(t.f.gate, .{ 1, rows, 1 }, dims.max_blocks, 0, &a);
    }

    /// db [block, out] += scale xa[:, first..]^T g, the open block's gradient.
    pub fn loraDb(t: Train, xa_in: u64, g: u64, db: u64, rows: usize, out: usize, first: usize) !void {
        var a = args(.{ xa_in, g, db, int(rows), int(out), int(first), int(dims.max_rank), dims.scale });
        try t.go(t.f.lora_db, .{ cdiv(out, 256), dims.block, 1 }, 256, 0, &a);
    }

    /// dxa [rows, rank] = scale g b^T where each block is open on the row.
    pub fn loraDxa(t: Train, g: u64, b: u64, gates: u64, dxa: u64, rows: usize, out: usize, rank: usize) !void {
        var a = args(.{ g, b, gates, dxa, int(rows), int(out), int(rank), int(dims.max_rank), dims.scale });
        try t.go(t.f.lora_dxa, .{ cdiv(32 * rank, 256), rows, 1 }, 256, 0, &a);
    }

    /// dx [rows, in] (row stride dx_stride) += dxa a.
    pub fn loraDx(t: Train, dxa: u64, a_in: u64, dx: u64, dx_stride: usize, rows: usize, in: usize, rank: usize) !void {
        var a = args(.{ dxa, a_in, dx, int(dx_stride), int(rows), int(in), int(rank), int(dims.max_rank) });
        try t.go(t.f.lora_dx, .{ cdiv(in, 256), rows, 1 }, 256, 0, &a);
    }

    /// p [rows, k] = each unit input row along the k directions f [k, in].
    pub fn project(t: Train, x: u64, x_stride: usize, f: u64, p: u64, rows: usize, in: usize, k: usize) !void {
        var a = args(.{ x, int(x_stride), f, p, int(rows), int(in), int(k) });
        try t.go(t.f.project, .{ cdiv(32 * k, 256), rows, 1 }, 256, 0, &a);
    }

    /// y [k, in] += w sum_r coin(first + r, j) x[r]: input rows into a random sketch.
    pub fn sketch(t: Train, x: u64, x_stride: usize, y: u64, rows: usize, in: usize, k: usize, first: usize, seed: u32, w: f32) !void {
        var a = args(.{ x, int(x_stride), y, int(rows), int(in), int(k), @as(u32, @truncate(first)), seed, w });
        try t.go(t.f.sketch, .{ cdiv(in, 256), k, 1 }, 256, 0, &a);
    }

    /// Each row's loss and target probability into stats, its logits replaced by their gradient.
    pub fn softmax(t: Train, logits: u64, targets: u64, wts: u64, stats: u64, rows: usize, vocab: usize) !void {
        var a = args(.{ logits, targets, wts, stats, int(vocab) });
        try t.go(t.f.softmax, .{ rows, 1, 1 }, 1024, 0, &a);
    }

    /// g += the gradient through an input RMS norm of h (bf16) with weight w, given dx at its output.
    pub fn rmsBack(t: Train, h: u64, w: u64, dx: u64, g: u64, rows: usize, dim: usize, eps: f32) !void {
        var a = args(.{ h, w, dx, g, int(dim), eps });
        try t.go(t.f.rms_back, .{ rows, 1, 1 }, 256, 0, &a);
    }

    /// A tiled projection [n, k] dequantized as bf16 [k, n].
    pub fn dequantT(t: Train, q: weights.QLinear, out: u64) !void {
        var a = args(.{ q.w, q.s, q.b, out, int(q.n), int(q.k), int(q.npad) });
        try t.go(t.f.dequant_t, .{ cdiv(q.n * q.k, 256), 1, 1 }, 256, 0, &a);
    }

    pub fn narrow(t: Train, x: u64, out: u64, n: usize) !void {
        var a = args(.{ x, out, @as(i64, @intCast(n)) });
        try t.go(t.f.narrow, .{ cdiv(n, 256), 1, 1 }, 256, 0, &a);
    }

    pub fn widen(t: Train, x: u64, out: u64, n: usize) !void {
        var a = args(.{ x, out, @as(i64, @intCast(n)) });
        try t.go(t.f.widen, .{ cdiv(n, 256), 1, 1 }, 256, 0, &a);
    }

    /// c [m, n] += a [m, k] b [n, k]^T (bf16 in, fp32 out) in `split` K slices, added in order through `parts`.
    pub fn gemm(t: Train, a_in: u64, lda: usize, b: u64, ldb: usize, c: u64, ldc: usize, m: usize, n: usize, k: usize, split: usize, parts: u64) !void {
        if (k % 32 != 0) return error.GemmShape;
        var a = args(.{ a_in, int(lda), b, int(ldb), c, int(ldc), parts, int(m), int(n), int(k), int(split) });
        try t.go(t.f.gemm, .{ cdiv(n, 64), cdiv(m, 64), split }, 128, 0, &a);
        if (split == 1) return;
        var s = args(.{ parts, c, int(ldc), int(m), int(n), int(split) });
        try t.go(t.f.gemm_sum, .{ cdiv(n, 256), m, 1 }, 256, 0, &s);
    }

    /// dy [pairs, dim] = each pair's routing weight times its row's g.
    pub fn pairsIn(t: Train, wt: u64, g: u64, dy: u64, pairs: usize, slots: usize, dim: usize) !void {
        var a = args(.{ wt, g, dy, int(slots), int(dim) });
        try t.go(t.f.pairs_in, .{ cdiv(dim, 256), pairs, 1 }, 256, 0, &a);
    }

    /// out [pairs, k] = x [pairs, n] times each pair's expert of a packed table [E, n rows, k inputs], by plan item.
    pub fn expertsBack(t: Train, x: u64, x_stride: usize, w: u64, items: u64, counts: u64, members: u64, out: u64, out_stride: usize, n: usize, k: usize, max_items: usize) !void {
        var a = args(.{ x, int(x_stride), w, items, counts, members, out, int(out_stride), int(k / 64), int(n / 32) });
        try t.go(t.f.experts_back, .{ max_items, k / 64, 1 }, 256, 0, &a);
    }

    /// du = da 2 sqrt(act) over n values: relu^2 undone from its output.
    pub fn relu2Back(t: Train, act: u64, da: u64, du: u64, n: usize) !void {
        var a = args(.{ act, da, du, @as(i64, @intCast(n)) });
        try t.go(t.f.relu2_back, .{ cdiv(n, 256), 1, 1 }, 256, 0, &a);
    }

    /// dx [rows, dim] += each row's pairs' input gradients.
    pub fn pairsOut(t: Train, dxp: u64, dx: u64, rows: usize, slots: usize, dim: usize) !void {
        var a = args(.{ dxp, dx, int(slots), int(dim) });
        try t.go(t.f.pairs_out, .{ cdiv(dim, 256), rows, 1 }, 256, 0, &a);
    }

    /// dx [rows, dim] += the routing weights' gradient carried into the router's input.
    pub fn routeBack(t: Train, part: u64, sk: usize, ids: u64, ys: u64, g: u64, gate_w: u64, dx: u64, rows: usize, experts: usize, top_k: usize, slots: usize, dim: usize, scale: f32) !void {
        var a = args(.{ part, int(sk), ids, ys, g, gate_w, dx, int(rows), int(experts), int(top_k), int(slots), int(dim), scale });
        try t.go(t.f.route_back, .{ rows, 1, 1 }, 256, 0, &a);
    }

    /// dx [rows, width] (row stride dx_stride) += g [rows, n] rest [n, width].
    pub fn restBack(t: Train, g: u64, rest: u64, dx: u64, dx_stride: usize, rows: usize, n: usize, width: usize) !void {
        var a = args(.{ g, rest, dx, int(dx_stride), int(rows), int(n), int(width) });
        try t.go(t.f.rest_back, .{ cdiv(width, 256), rows, 1 }, 256, 0, &a);
    }

    /// One Adam step on n parameters (hp: rate, decays, epsilon; corr: bias corrections); their gradient cleared.
    pub fn adam(t: Train, p: u64, g: u64, m: u64, v: u64, n: usize, hp: [4]f32, corr: [2]f32) !void {
        var a = args(.{ p, g, m, v, int(n), hp[0], hp[1], hp[2], hp[3], corr[0], corr[1] });
        try t.go(t.f.adam, .{ cdiv(n, 256), 1, 1 }, 256, 0, &a);
    }

    pub fn dt(t: Train, proj: u64, bias: u64, out: u64, s: Shape) !void {
        var a = args(.{ proj, bias, out, s });
        try t.go(t.f.dt, .{ cdiv(@intCast(s.heads), 64), @intCast(s.rows), 1 }, 64, 0, &a);
    }

    pub fn ssmFwd(t: Train, act: u64, dt_in: u64, a_neg: u64, d_skip: u64, y: u64, ckpt: u64, s: Shape) !void {
        var a = args(.{ act, dt_in, a_neg, d_skip, y, ckpt, s });
        try t.go(t.f.ssm_fwd, .{ @intCast(s.heads), 1, 1 }, 1024, 0, &a);
    }

    pub fn ssmBack(t: Train, act: u64, dt_in: u64, a_neg: u64, d_skip: u64, dy: u64, ckpt: u64, states: u64, dact: u64, dbp: u64, dcp: u64, ddt: u64, s: Shape) !void {
        var a = args(.{ act, dt_in, a_neg, d_skip, dy, ckpt, states, dact, dbp, dcp, ddt, s });
        try t.go(t.f.ssm_back, .{ @intCast(s.heads), 1, 1 }, 1024, 0, &a);
    }

    pub fn ssmBc(t: Train, dbp: u64, dcp: u64, dact: u64, s: Shape) !void {
        var a = args(.{ dbp, dcp, dact, s });
        try t.go(t.f.ssm_bc, .{ cdiv(@intCast(s.n), 128), @intCast(s.groups), @intCast(s.rows) }, 128, 0, &a);
    }

    pub fn gateBack(t: Train, y: u64, proj: u64, w: u64, dn: u64, dy: u64, dproj: u64, s: Shape, eps: f32) !void {
        var a = args(.{ y, proj, w, dn, dy, dproj, s, eps });
        try t.go(t.f.gate_back, .{ @intCast(s.groups), @intCast(s.rows), 1 }, 256, 0, &a);
    }

    pub fn convBack(t: Train, dact: u64, proj: u64, cw: u64, cb: u64, dproj: u64, s: Shape) !void {
        var a = args(.{ dact, proj, cw, cb, dproj, s });
        try t.go(t.f.conv_back, .{ cdiv(@intCast(s.conv), 256), @intCast(s.rows), 1 }, 256, 0, &a);
    }

    pub fn dtBack(t: Train, proj: u64, bias: u64, ddt: u64, dproj: u64, s: Shape) !void {
        var a = args(.{ proj, bias, ddt, dproj, s });
        try t.go(t.f.dt_back, .{ cdiv(@intCast(s.heads), 64), @intCast(s.rows), 1 }, 64, 0, &a);
    }

    pub fn attnQ(t: Train, qkv: u64, dout: u64, probs: u64, dscores: u64, dqkv: u64, h: Heads) !void {
        var a = args(.{ qkv, dout, probs, dscores, dqkv, h });
        try t.go(t.f.attn_q, .{ @intCast(h.heads), @intCast(h.rows), 1 }, 256, 0, &a);
    }

    pub fn attnKv(t: Train, qkv: u64, dout: u64, probs: u64, dscores: u64, dqkv: u64, h: Heads) !void {
        var a = args(.{ qkv, dout, probs, dscores, dqkv, h });
        try t.go(t.f.attn_kv, .{ @intCast(h.kv_heads), @intCast(h.rows), 1 }, @intCast(h.dim), 0, &a);
    }

    /// `bytes` (a multiple of four) of zeros at `ptr`.
    pub fn zero(t: Train, ptr: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try t.d.check(t.d.api.cuMemsetD32Async(ptr, 0, bytes / 4, t.s.handle), "cuMemsetD32Async");
    }

    pub fn copy(t: Train, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try t.d.check(t.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, t.s.handle), "cuMemcpyDtoDAsync");
    }
};
