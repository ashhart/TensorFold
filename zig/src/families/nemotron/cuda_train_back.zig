//! Each kind of layer undone on CUDA for a sequence: the gradient at its output, through its mixer, to its input.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const kern = @import("cuda_kernels.zig");
const weights = @import("cuda_weights.zig");
const sites = @import("cuda_sites.zig");
const ops = @import("cuda_train_ops.zig");
const dims = @import("slide_dims.zig");

/// The f32 gradients a backward works in at `rows` rows (attention reads at most 256), and the products' bf16 scratch.
pub const Bufs = struct {
    mem: cuda.DeviceBuffer,
    g: u64, // [rows, D]: the residual's gradient
    dx: u64, // [rows, D]: the gradient at a mixer's normed input
    xa: u64, // [rows, max_rank]: a site's input along the change's directions, closed blocks zeroed
    xn: u64, // [rows]
    gates: u64, // [rows, max_blocks]
    dxa: u64, // [rows, max_rank]
    dyn: u64, // [rows, inner]
    ytot: u64, // [rows, inner]
    dytot: u64, // [rows, inner]
    dt: u64, // [rows, heads]
    ddt: u64, // [rows, heads]
    dact: u64, // [rows, conv]
    dproj: u64, // [rows, proj]
    ckpt: u64, // [heads, rows / 16 + 1, dh * n]
    states: u64, // [heads, 16, dh * n]
    dbp: u64, // [rows, heads, n]
    dcp: u64, // [rows, heads, n]
    d_o: u64, // [rows, heads * hd]
    dqkv: u64, // [rows, qkv]
    probs: u64, // [heads, rows, rows]
    dscores: u64, // [heads, rows, rows]
    dy: u64, // [pairs, D]
    d_act: u64, // [pairs, W]
    du: u64, // [pairs, W]
    dxp: u64, // [pairs, D]
    xb: u64, // bf16 [rows, widest]: a gradient narrowed for its product
    wt: u64, // bf16 [widest k, widest n]: a projection dequantized transposed
    parts: u64, // f32 [split, m, n]: a split product's slices before they add in order (split m n within split_cap)

    const names = .{ "g", "dx", "xa", "xn", "gates", "dxa", "dyn", "ytot", "dytot", "dt", "ddt", "dact", "dproj", "ckpt", "states", "dbp", "dcp", "d_o", "dqkv", "probs", "dscores", "dy", "d_act", "du", "dxp", "xb", "wt", "parts" };

    pub fn init(d: *const cuda.Driver, c: cfg.Config, rows: usize) !Bufs {
        const bytes = sizes(c, rows);
        var b: Bufs = undefined;
        b.mem = try cuda.DeviceBuffer.alloc(d, total(c, rows));
        var at: usize = 0;
        inline for (names, 0..) |n, i| {
            @field(b, n) = b.mem.ptr + at;
            at += std.mem.alignForward(usize, bytes[i], 256);
        }
        return b;
    }

    /// The device bytes `init` takes at `rows` rows.
    pub fn total(c: cfg.Config, rows: usize) usize {
        var n: usize = 0;
        for (sizes(c, rows)) |x| n += std.mem.alignForward(usize, x, 256);
        return n;
    }

    fn sizes(c: cfg.Config, rows: usize) [names.len]usize {
        const D = c.hidden;
        const pairs = rows * c.slots();
        const qw = c.heads * c.head_dim;
        const state = c.mamba_head_dim * c.state;
        const h = c.mamba_heads;
        const widest = @max(c.projDim(), c.qkvDim(), D, c.inner(), qw);
        const plane = @max(D * c.projDim(), D * c.qkvDim(), D * c.inner(), D * qw);
        return .{
            rows * D * 4,              rows * D * 4,           rows * dims.max_rank * 4,   rows * 4,                        rows * dims.max_blocks * 4,
            rows * dims.max_rank * 4,  rows * c.inner() * 4,   rows * c.inner() * 4,       rows * c.inner() * 4,            rows * h * 4,
            rows * h * 4,              rows * c.convDim() * 4, rows * c.projDim() * 4,     h * (rows / 16 + 1) * state * 4, h * 16 * state * 4,
            rows * h * c.state * 4,    rows * h * c.state * 4, rows * qw * 4,              rows * c.qkvDim() * 4,           c.heads * rows * rows * 4,
            c.heads * rows * rows * 4, pairs * D * 4,          pairs * c.expert_width * 4, pairs * c.expert_width * 4,      pairs * D * 4,
            rows * widest * 2,         plane * 2,              split_cap * 4,
        };
    }

    pub fn deinit(b: *Bufs) void {
        b.mem.free();
    }
};

/// What a layer's backward reads: the launches, the model, its block and change, its forward's saved parts.
pub const Layer = struct {
    t: ops.Train,
    o: kern.Ops,
    c: cfg.Config,
    blk: *const weights.Block,
    site: *const sites.Site,
    sites: *const sites.Sites,
    acts: [5]u64, // the parts train.zig keeps for this kind of layer, in its order
    b: *const Bufs,
    rows: usize,
    sk: usize, // the router's K slices
    plan: kern.Plan,
};

/// y [rows, k] (+)= x [rows, n] times the tiled 4-bit [n, k] projection q: q dequantized transposed, then the GEMM.
fn product(x: Layer, in: u64, q: weights.QLinear, y: u64, add: bool) !void {
    const t = x.t;
    const rows = x.rows;
    try t.narrow(in, x.b.xb, rows * q.n);
    try t.dequantT(q, x.b.wt);
    if (!add) try t.zero(y, rows * q.k * 4);
    try t.gemm(x.b.xb, q.n, x.b.wt, q.n, y, q.k, rows, q.k, q.n, split(rows, q.k, q.n), x.b.parts);
}

/// Blocks a split GEMM fills the GPU with.
const split_blocks = 192;
/// The most floats a split product's slices take: split keeps slices times 64 x 64 tiles within split_blocks.
pub const split_cap = split_blocks * 64 * 64;

/// K slices for a GEMM of m x n outputs: enough blocks to fill the GPU, each slice at least 256 deep.
pub fn split(m: usize, n: usize, k: usize) usize {
    const tiles = ((n + 63) / 64) * ((m + 63) / 64);
    return std.math.clamp(split_blocks / @max(tiles, 1), 1, @max(k / 256, 1));
}

/// The open block's gradient from its site's input (bf16, row stride in_stride) and g; dx gains every block's part.
fn adapterBack(x: Layer, in: u64, in_stride: usize, dx: u64, dx_stride: usize) !void {
    const s = x.sites;
    if (s.rank == 0) return;
    const t = x.t;
    const b = x.b;
    const ad = x.site.adapter(s);
    try t.xa(ad, in, in_stride, b.xa, b.xn, x.rows);
    try t.gate(ad, b.xa, b.xn, b.gates, x.rows, s.rank);
    try t.loraDb(b.xa, b.g, x.site.gb.dev, x.rows, ad.out, s.first());
    try t.loraDxa(b.g, ad.b, b.gates, b.dxa, x.rows, ad.out, s.rank);
    try t.loraDx(b.dxa, ad.a, dx, dx_stride, x.rows, ad.in, s.rank);
}

/// The MoE layer: every pair's products undone on the packed experts, the shared halves' rest and change, then routing.
pub fn moe(x: Layer) !void {
    const c = x.c;
    const m = x.blk.moe;
    const b = x.b;
    const t = x.t;
    const rows = x.rows;
    const ex = m.experts;
    const ns = c.slots();
    const pairs = rows * ns;
    const W = ex.width;
    const D = c.hidden;
    const pick = x.acts[0];
    const act = x.acts[3];
    try x.o.plan(pick, pairs, ex.count, 64, x.plan);
    const items = kern.maxItems(pairs, ex.count, 64);
    try t.pairsIn(x.acts[1], b.g, b.dy, pairs, ns, D);
    try t.expertsBack(b.dy, D, ex.down, x.plan.items, x.plan.counts, x.plan.members, b.d_act, W, D, W, items);
    for (m.rest, 0..) |r, h| if (r != 0) try t.restBack(b.g, r, b.d_act + (c.top_k + h) * W * 4, ns * W, rows, D, W);
    try adapterBack(x, act + c.top_k * W * 2, ns * W, b.d_act + c.top_k * W * 4, ns * W);
    try t.relu2Back(act, b.d_act, b.du, pairs * W);
    try t.expertsBack(b.du, W, ex.up, x.plan.items, x.plan.counts, x.plan.members, b.dxp, D, W, D, items);
    try t.zero(b.dx, rows * D * 4);
    try t.pairsOut(b.dxp, b.dx, rows, ns, D);
    try t.routeBack(x.acts[2], x.sk, pick, x.acts[4], b.g, m.router, b.dx, rows, c.experts, c.top_k, ns, D, c.routed_scaling);
}

/// The Mamba-2 layer: out_proj, its rest and change, the gated norm, the scan, dt, the conv, in_proj.
pub fn mamba(x: Layer) !void {
    const c = x.c;
    const m = x.blk.mamba;
    const b = x.b;
    const t = x.t;
    const rows = x.rows;
    const proj = x.acts[0];
    const xc = x.acts[1];
    const shape: ops.Shape = .{ .rows = @intCast(rows), .heads = @intCast(c.mamba_heads), .dh = @intCast(c.mamba_head_dim), .groups = @intCast(c.groups), .n = @intCast(c.state), .inner = @intCast(c.inner()), .conv = @intCast(c.convDim()), .proj = @intCast(c.projDim()), .lo = c.dt_min, .hi = c.dt_max };
    try product(x, b.g, m.out_proj, b.dyn, false);
    if (m.out_rest != 0) try t.restBack(b.g, m.out_rest, b.dyn, c.inner(), rows, c.hidden, c.inner());
    try adapterBack(x, x.acts[2], c.inner(), b.dyn, c.inner());
    try t.dt(proj, m.dt_bias, b.dt, shape);
    try t.ssmFwd(xc, b.dt, m.a, m.d, b.ytot, b.ckpt, shape);
    try t.gateBack(b.ytot, proj, m.gnorm, b.dyn, b.dytot, b.dproj, shape, c.eps);
    try t.ssmBack(xc, b.dt, m.a, m.d, b.dytot, b.ckpt, b.states, b.dact, b.dbp, b.dcp, b.ddt, shape);
    try t.ssmBc(b.dbp, b.dcp, b.dact, shape);
    try t.convBack(b.dact, proj, m.conv_w, m.conv_b, b.dproj, shape);
    try t.dtBack(proj, m.dt_bias, b.ddt, b.dproj, shape);
    try product(x, b.dproj, m.in_proj, b.dx, false);
}

/// The attention layer: o_proj, its rest and change, causal softmax attention with shared kv heads, then q, k and v.
pub fn attention(x: Layer) !void {
    const c = x.c;
    const a = x.blk.attn;
    const b = x.b;
    const t = x.t;
    const rows = x.rows;
    const qkv = x.acts[0];
    const qw = c.heads * c.head_dim;
    try product(x, b.g, a.o, b.d_o, false);
    if (a.o_rest != 0) try t.restBack(b.g, a.o_rest, b.d_o, qw, rows, c.hidden, qw);
    try adapterBack(x, x.acts[1], qw, b.d_o, qw);
    const h: ops.Heads = .{ .rows = @intCast(rows), .heads = @intCast(c.heads), .kv_heads = @intCast(c.kv_heads), .dim = @intCast(c.head_dim), .nqkv = @intCast(c.qkvDim()), .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.head_dim)))) };
    try t.attnQ(qkv, b.d_o, b.probs, b.dscores, b.dqkv, h);
    try t.attnKv(qkv, b.d_o, b.probs, b.dscores, b.dqkv, h);
    try product(x, b.dqkv, a.qkv, b.dx, false);
}

test "a GEMM's K slices fill the GPU without slicing K below 256" {
    try std.testing.expectEqual(@as(usize, 4), split(48, 2688, 10304));
    try std.testing.expectEqual(@as(usize, 1), split(256, 4096, 2688));
    try std.testing.expectEqual(@as(usize, 4), split(30, 2688, 131072));
    try std.testing.expectEqual(@as(usize, 1), split(16, 64, 128));
}
