//! A CUDA training pass's mixers over several sequences: each its own conv, scan and attention, the rest shared.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const state = @import("cuda_state.zig");
const weights = @import("cuda_weights.zig");
const fwd = @import("cuda_forward.zig");
const train_ops = @import("cuda_train_ops.zig");
const Span = @import("living").train_mode.Span;
const Store = @import("cuda_train_prefix.zig").Store;

/// The open fact's pocket neurons scaled row by row: `rows` f32 [rows] on the device, at layer l from `from[l]` on.
pub const Scale = struct { rows: u64, from: []const u32 };

/// Where the spans' prompt prefixes are kept, and whether this pass keeps each span's end states there (prime).
pub const Prefix = struct { store: *const Store, prime: bool = false };

/// Block i's mixer (cuda_forward.zig's chunkMixer) on every span from its prefix's states, else from zero.
pub fn mixer(f: *const fwd.Forward, w: *const fwd.Walk, i: usize, spans: []const Span, pf: Prefix) !void {
    const c = f.c;
    const b = f.b;
    const o = f.ops;
    switch (f.w.blocks[i].kind) {
        .mamba => {
            const m = f.w.blocks[i].mamba;
            const ssm = b.ssm + @as(u64, w.mj) * c.mamba_heads * c.mamba_head_dim * c.state * 4;
            const base = b.conv_base + @as(u64, w.mj) * 3 * c.convDim() * 2;
            const ssm_bytes = c.mamba_heads * c.mamba_head_dim * c.state * 4;
            const conv_bytes = 3 * c.convDim() * 2;
            for (spans) |sp| {
                if (sp.skip > 0) {
                    try o.copy(ssm, pf.store.ssmAt(sp.slot, w.mj), ssm_bytes);
                    try o.copy(base, pf.store.convAt(sp.slot, w.mj), conv_bytes);
                } else {
                    try o.fill32(ssm, 0, ssm_bytes / 4);
                    try o.fill32(base, 0, conv_bytes / 4);
                }
                const proj = b.proj + @as(u64, sp.at) * c.projDim() * 2;
                const xc = b.p_xc + @as(u64, sp.at) * c.convDim() * 2;
                try f.glue.convRows(proj, base, xc, m.conv_w, m.conv_b, sp.rows, mshape(f));
                try o.scanRows(proj, xc, ssm, m.a, m.d, m.dt_bias, b.sy + @as(u64, sp.at) * c.inner() * 2, sp.rows, c.projDim(), c.mamba_heads, c.mamba_head_dim, c.convDim(), c.groups, c.dt_min, c.dt_max, .{});
                if (pf.prime) {
                    try o.copy(pf.store.ssmAt(sp.slot, w.mj), ssm, ssm_bytes);
                    try o.copy(pf.store.convAt(sp.slot, w.mj), base, conv_bytes);
                }
            }
        },
        .attention => {
            const per = f.max_len * c.kv_heads * c.head_dim * 2;
            const kc = b.k_cache + @as(u64, w.aj) * per;
            const vc = b.v_cache + @as(u64, w.aj) * per;
            const qd = c.heads * c.head_dim;
            const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.head_dim), -0.5));
            const kvd = c.kv_heads * c.head_dim * 2;
            for (spans) |sp| {
                const qkv = b.qkv + @as(u64, sp.at) * c.qkvDim() * 2;
                const q = b.q + @as(u64, sp.at) * qd * 2;
                // the prefix's keys and values at the cache's first positions, the span's own after them
                if (sp.skip > 0) {
                    try o.copy(kc, pf.store.keysAt(sp.slot, w.aj, false), sp.skip * kvd);
                    try o.copy(vc, pf.store.keysAt(sp.slot, w.aj, true), sp.skip * kvd);
                }
                try o.fill32(b.p_meta, @intCast(sp.skip), 4);
                try f.glue.kvWrite(qkv, kc, vc, b.p_meta, sp.rows, f.ashape());
                try o.torch().copyRows(qkv, c.qkvDim() * 2, q, qd * 2, qd * 2, sp.rows);
                try o.prefillAttention(q, kc, vc, b.att + @as(u64, sp.at) * qd * 2, sp.skip, sp.rows, c.heads, c.kv_heads, scale, 0, 0); // the trainer's own rows
                if (pf.prime) {
                    try o.copy(pf.store.keysAt(sp.slot, w.aj, false), kc, sp.rows * kvd);
                    try o.copy(pf.store.keysAt(sp.slot, w.aj, true), vc, sp.rows * kvd);
                }
            }
        },
        .moe => {},
    }
}

fn mshape(f: *const fwd.Forward) @import("cuda_glue.zig").Shape {
    const c = f.c;
    return .{ .proj = c.projDim(), .xd = c.inner(), .cd = c.convDim(), .heads = c.mamba_heads, .dh = c.mamba_head_dim, .groups = c.groups, .state = c.state, .rmax = state.windowRows(c) };
}

/// The MoE block (cuda_forward.zig's experts and sharedRest), the pocket's open neurons scaled row by row.
pub fn experts(f: *const fwd.Forward, m: weights.MoE, rows: usize, scale: Scale, layer: usize) !void {
    const c = f.c;
    const b = f.b;
    const o = f.ops;
    const ex = m.experts;
    const pairs = rows * c.slots();
    try f.glue.route(b.y, m.router, m.bias, b.part, b.pick, b.wts, rows, c.hidden, c.experts, c.top_k, c.slots(), c.routed_scaling, c.norm_topk);
    const tile = cuda.experts.promptTile(ex.format);
    try o.plan(b.pick, pairs, ex.experts, tile, b.plan);
    const items = kern.maxItems(pairs, ex.experts, tile);
    const g = o.grouped();
    const in = f.rowsIn(ex);
    try g.prompt(o.s, .up, in[0], ex, b.plan, b.act, items, -1);
    try g.prompt(o.s, .down_bf16, in[1], ex, b.plan, b.ymoe, items, -1);
    for (m.rest, 0..) |r, h| if (r != 0) try o.restRows(b.act, ex.width, r, b.ymoe, ex.dims, false, rows, c.slots(), c.top_k + h, ex.dims, ex.width);
    const t: train_ops.Ops = .{ .k = &o.k.train, .s = o.s };
    if (m.pocket) |p| try t.pocketRows(b.y, ex.dims, p.up, p.down, p.n, b.ymoe, false, rows, c.slots(), c.top_k, scale.rows, scale.from[layer]);
    const ad = m.adapter orelse return;
    const x = b.act + @as(u64, c.top_k) * ex.width * 2;
    const y = b.ymoe + @as(u64, c.top_k) * ex.dims * 2;
    try t.loraIn(x, c.slots() * ex.width, ad.a, ad.xa, rows, ad.in, ad.rank, ad.xn);
    var r = f.route;
    if (ad.whole) r.whole = 1;
    try t.gate(ad.xa, ad.xn, ad.tau, ad.gates, rows, ad.rank, ad.unit, &r, ad.latch, ad.score);
    try t.loraOut(ad.xa, ad.b, y, c.slots() * ex.dims, false, rows, ad.out, ad.rank, ad.scale);
}
