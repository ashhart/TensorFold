//! Where the CUDA trainer's device buffers sit: one allocation, rows for a whole pass, a sequence's for its own parts.
const std = @import("std");
const cuda = @import("cuda");
const train = @import("cuda_train.zig");
const ops = @import("cuda_train_ops.zig");
const sites_mod = @import("cuda_sites.zig");

const Trainer = train.Trainer;
const pack_rows = train.pack_rows;
const max_rows = train.max_rows;

/// Sub-allocations of one device allocation, 256-byte aligned.
pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    pub fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

/// The trainer's device arena: its bytes (`count`), else its pointers handed out in the same order.
pub fn layout(t: *Trainer, count: bool) usize {
    const c = t.c;
    const R: usize = pack_rows;
    const Q: usize = max_rows; // a sequence's own rows: its scan's checkpoints, its attention's probabilities
    const D: usize = c.hidden;
    const S: usize = c.slots();
    const W: usize = c.expert_width;
    const F: usize = c.shared_width;
    const pairs = R * c.top_k;
    const qw = c.heads * c.head_dim;
    const kw = c.kv_heads * c.head_dim;
    const xd = c.inner();
    const cd = c.convDim();
    const pd = c.projDim();
    const H = c.mamba_heads;
    const size = c.mamba_head_dim * c.state;
    const widest_k = @max(pd, qw, F, D, xd);
    var widest_n: usize = 0;
    for (t.raw) |r| switch (r) {
        .mamba => |m| widest_n = @max(widest_n, m.in_proj.n * m.in_proj.k, m.out_proj.n * m.out_proj.k),
        .attention => |a| widest_n = @max(widest_n, a.q.n * a.q.k, a.o.n * a.o.k),
        .moe => |m| widest_n = @max(widest_n, m.up.n * m.up.k, m.down.n * m.down.k),
    };
    var a: Arena = if (count) .{ .buf = .{ .d = undefined, .ptr = 0, .len = std.math.maxInt(usize) } } else t.mem;
    var total: usize = 0;
    const Take = struct {
        fn f(ar: *Arena, n: usize, tot: *usize) u64 {
            tot.* += std.mem.alignForward(usize, n, 256) + 256;
            return ar.take(n);
        }
    };
    const T = Take.f;
    t.saved = T(&a, (c.layers + 1) * R * D * 2, &total);
    t.logits = T(&a, R * c.vocab * 2, &total);
    for (t.kept, 0..) |*k, l| {
        k.* = .{};
        switch (c.kinds[l]) {
            .mamba => {
                k.proj = T(&a, R * pd * 2, &total);
                k.ya = T(&a, R * xd * 2, &total);
            },
            .attention => {
                k.x = T(&a, R * D * 2, &total);
                k.ya = T(&a, R * qw * 2, &total);
            },
            .moe => {
                k.x = T(&a, R * D * 2, &total);
                k.upr = T(&a, R * F * 2, &total);
                k.act = T(&a, R * S * W * 2, &total);
                k.ys = T(&a, R * S * D * 2, &total);
                k.ids = T(&a, R * S * 4, &total);
                k.wts = T(&a, R * S * 4, &total);
            },
        }
    }
    for ([_]*u64{ &t.g, &t.dx, &t.gs, &t.dhn }) |p| p.* = T(&a, R * D * 4, &total);
    t.dxa = T(&a, R * sites_mod.max_rank * 4, &total);
    t.xb = T(&a, R * widest_k * 2, &total);
    t.wt = T(&a, widest_n * 2, &total);
    t.d_upr = T(&a, R * F * 4, &total);
    t.du = T(&a, R * F * 4, &total);
    t.dyp = T(&a, pairs * D * 4, &total);
    t.dxp = T(&a, pairs * D * 4, &total);
    t.d_y1r = T(&a, pairs * W * 4, &total);
    t.du1 = T(&a, pairs * W * 4, &total);
    t.y1s = T(&a, pairs * W * 2, &total);
    t.tiles = T(&a, ops.tileCap(pairs, c.experts) * 8, &total);
    t.order = T(&a, pairs * 4, &total);
    t.inv = T(&a, pairs * 4, &total);
    t.offsets = T(&a, c.experts * 4, &total);
    t.lg = T(&a, R * c.experts * 4, &total);
    for ([_]*u64{ &t.dyn, &t.ytot, &t.dytot }) |p| p.* = T(&a, R * xd * 4, &total);
    t.dt = T(&a, R * H * 4, &total);
    t.ddt = T(&a, R * H * 4, &total);
    t.dact = T(&a, R * cd * 4, &total);
    t.dproj = T(&a, R * pd * 4, &total);
    t.ckpt = T(&a, H * (Q / 8 + 1) * size * 4, &total);
    t.dbp = T(&a, R * H * c.state * 4, &total);
    t.dcp = T(&a, R * H * c.state * 4, &total);
    t.conv_out = T(&a, R * cd * 2, &total);
    t.act_m = T(&a, R * cd * 2, &total);
    t.qkv = T(&a, R * c.qkvDim() * 2, &total);
    t.d_o = T(&a, R * qw * 4, &total);
    t.dq = T(&a, R * qw * 4, &total);
    t.dk = T(&a, R * kw * 4, &total);
    t.dv = T(&a, R * kw * 4, &total);
    t.probs = T(&a, c.heads * Q * Q * 4, &total);
    t.dscores = T(&a, c.heads * Q * Q * 4, &total);
    if (!count) t.mem = a;
    return total;
}
