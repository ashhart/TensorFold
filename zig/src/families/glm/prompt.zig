//! Prompt chunks of up to `max_rows` rows: projections and routed experts on the M5 tensor units (MLX's NAX qmm and
//! sorted-expert gather), every other op the decode path's row kernels over all the chunk's rows.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const pk = @import("../nemotron/prefill_kernels.zig");
const Ref = wts.Ref;
const bind = fwd.bind;
const size = fwd.size;

pub const max_rows = 4096;
/// Sparse rows whose block scores one selection pass holds.
const select_rows = 256;

const qmm_name = "custom_kernel_tf_qmm_t_nax_bf16_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";
const gather_names = [2][]const u8{
    "custom_kernel_tf_gather_qmm_rhs_nax_bf16_32_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_t_int32_t_bfloat16_t",
    "custom_kernel_tf_gather_qmm_rhs_nax_bf16_64_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_t_int32_t_bfloat16_t",
};

/// The chunk's buffers beyond the decode scratch's (whose stream fields `streams` points at prompt-sized ones).
pub const Prompt = struct {
    k: pk.Kernels,
    streams: st.Scratch, // the decode scratch with x, h, normed, branch, post, comb, inv, mixes, raw, hidden resized
    proj: Ref,
    y: Ref,
    xp: Ref,
    qr: Ref,
    qp: Ref,
    ql: Ref,
    att: Ref,
    vals: Ref,
    iw: Ref,
    indices: Ref,
    sscore: Ref,
    xf: Ref,
    logits_r: Ref,
    pick: Ref,
    wts: Ref,
    counts: Ref,
    starts: Ref,
    offsets: Ref,
    order: Ref,
    sorted: Ref,
    inverse: Ref,
    xs: Ref,
    g: Ref,
    u: Ref,
    act: Ref,
    ye: Ref,
    yp: Ref,
    sgu: Ref,
    sact: Ref,
    ys: Ref,
    gu: Ref,
    actd: Ref,
    m_emb: Ref,
    m_eh: Ref,
    m_x: Ref,
    m_xn: Ref,
    m_out: Ref,

    pub fn deinit(p: *Prompt) void {
        p.k.deinit();
    }
};

/// The chunk buffers for `cap`-token caches (selection passes read cap / 4 block scores a row).
pub fn init(gpa: std.mem.Allocator, arena: *st.Arena, device: mtl.Device, c: *const cfg.Config, decode: *const st.Scratch, cap: u32) !Prompt {
    const R: usize = max_rows;
    const D: usize = c.hidden;
    const n: usize = R * c.topk;
    const blocks = (n + 255) / 256;
    var p: Prompt = undefined;
    p.k = try pk.load(gpa, device);
    errdefer p.k.deinit();
    p.streams = decode.*;
    const big = struct {
        fn of(a: *st.Arena, bytes: usize) !Ref {
            return a.buffer(bytes);
        }
    };
    p.streams.x = .{ try big.of(arena, R * 4 * D * 2), try big.of(arena, R * 4 * D * 2) };
    p.streams.h = try big.of(arena, R * D * 2);
    p.streams.normed = try big.of(arena, R * D * 2);
    p.streams.branch = try big.of(arena, R * D * 2);
    p.streams.post = try big.of(arena, R * 4 * 4);
    p.streams.comb = try big.of(arena, R * 16 * 4);
    p.streams.inv = try big.of(arena, R * 4);
    p.streams.mixes = try big.of(arena, R * 24 * 4);
    p.streams.raw = try big.of(arena, R * D * 2);
    p.streams.hidden = try big.of(arena, R * D * 2);
    const xp_w = std.mem.alignForward(usize, c.xProj(), 64);
    p.proj = try big.of(arena, R * c.kdaProj() * 2);
    p.y = try big.of(arena, R * c.kdaWidth() * 2);
    p.xp = try big.of(arena, R * xp_w * 2);
    p.qr = try big.of(arena, R * c.q_lora * 2);
    p.qp = try big.of(arena, R * c.qrProj() * 2);
    p.ql = try big.of(arena, R * c.mla_heads * c.kv_lora * 2);
    p.att = try big.of(arena, R * c.mla_heads * c.kv_lora * 2);
    p.vals = try big.of(arena, R * c.mla_heads * c.v_dim * 2);
    p.iw = try big.of(arena, R * c.i_heads * 2);
    p.indices = try big.of(arena, R * c.keyWidth() * 4);
    p.sscore = try big.of(arena, select_rows * (@as(usize, cap) / c.kpool + 1) * 4);
    p.xf = try big.of(arena, R * D * 4);
    p.logits_r = try big.of(arena, R * c.experts * 4);
    p.pick = try big.of(arena, n * 4);
    p.wts = try big.of(arena, n * 4);
    p.counts = try big.of(arena, blocks * c.experts * 4);
    p.starts = try big.of(arena, blocks * c.experts * 4);
    p.offsets = try big.of(arena, (c.experts + 1) * 4);
    p.order = try big.of(arena, n * 4);
    p.sorted = try big.of(arena, n * 4);
    p.inverse = try big.of(arena, n * 4);
    p.xs = try big.of(arena, n * D * 2);
    p.g = try big.of(arena, n * c.moe_inter * 2);
    p.u = try big.of(arena, n * c.moe_inter * 2);
    p.act = try big.of(arena, n * c.moe_inter * 2);
    p.ye = try big.of(arena, n * D * 2);
    p.yp = try big.of(arena, n * D * 2);
    p.sgu = try big.of(arena, R * 2 * c.moe_inter * 2);
    p.sact = try big.of(arena, R * c.moe_inter * 2);
    p.ys = try big.of(arena, R * D * 2);
    p.gu = try big.of(arena, R * 2 * c.dense_inter * 2);
    p.actd = try big.of(arena, R * c.dense_inter * 2);
    p.m_emb = try big.of(arena, R * D * 2);
    p.m_eh = try big.of(arena, R * 2 * D * 2);
    p.m_x = try big.of(arena, R * D * 2);
    p.m_xn = try big.of(arena, R * D * 2);
    p.m_out = try big.of(arena, R * D * 2);
    return p;
}

/// The int32 parameter array the NAX and sort kernels read (16 entries, zero-padded).
fn params(e: mtl.ComputeEncoder, index: usize, values: anytype) void {
    var v: [16]i32 = @splat(0);
    inline for (values, 0..) |x, i| v[i] = @intCast(x);
    e.setBytes(std.mem.asBytes(&v), index);
}

/// MLX's threadgroup for a grid of threads: the group, but no larger than the grid.
fn run(e: mtl.ComputeEncoder, grid: [3]usize, group: [3]usize) void {
    e.dispatchThreads(size(grid[0], grid[1], grid[2]), size(@min(group[0], grid[0]), @min(group[1], grid[1]), @min(group[2], grid[2])));
}

/// y [M, N] = x [M, K] W^T for a 4-bit matrix (affine_qmm_t_nax, 64x64 tiles; N rounded up to its padded rows).
fn qmm(p: *const Prompt, e: mtl.ComputeEncoder, x: Ref, q: wts.Q4, y: Ref, M: u32) void {
    const N = std.mem.alignForward(u32, q.n, 64);
    e.setPipeline(p.k.get(qmm_name));
    bind(e, 0, .{ q.w, q.s, q.b, x });
    params(e, 4, .{ q.k, N, M });
    bind(e, 5, .{y});
    run(e, .{ N / 64 * 32, (M + 63) / 64 * 2, 2 }, .{ 32, 2, 2 });
}

/// Rows sorted by expert times their expert's 4-bit W^T (affine_gather_qmm_rhs_nax) from `offsets`.
fn gather(p: *const Prompt, e: mtl.ComputeEncoder, xs: Ref, q: wts.Q4, offsets: Ref, y: Ref, n: u32, experts: u32) void {
    const bm: u32 = if (n / experts < 64) 32 else 64;
    e.setPipeline(p.k.get(gather_names[@intFromBool(bm == 64)]));
    bind(e, 0, .{ xs, q.w, q.s, q.b, offsets });
    params(e, 5, .{ n, q.n, q.k, experts });
    bind(e, 6, .{y});
    run(e, .{ (q.n + 63) / 64 * 32, @min(n, (n + bm - 1) / bm + experts - 1) * 2, 2 }, .{ 32, 2, 2 });
}

fn swiglu(x: *const fwd.Ctx, e: mtl.ComputeEncoder, gu: Ref, act: Ref, rows: u32, width: u32) void {
    e.setPipeline(x.k.swiglu);
    bind(e, 0, .{ gu, act });
    e.setValue(fwd.Rows{ .rows = @intCast(rows), .width = @intCast(width), .x_stride = @intCast(2 * width), .y_stride = @intCast(width), .eps = x.c.swiglu_limit }, 2);
    e.dispatchThreads(size(width, rows, 1), size(256, 1, 1));
}

fn add(x: *const fwd.Ctx, e: mtl.ComputeEncoder, a: Ref, b: Ref, out: Ref, n: u32) void {
    e.setPipeline(x.k.add);
    bind(e, 0, .{ a, b, out });
    e.setValue(n, 3);
    e.dispatchThreads(size(n, 1, 1), size(256, 1, 1));
}

/// The MoE block on M rows: the shared expert, routing (the decode router 16 rows at a time, the decode route's
/// arithmetic), a stable sort of (row, slot) pairs by expert, gathered gate/up/down, then the decode combine.
fn moe(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32) void {
    const c = x.c;
    const k = x.k;
    const D = c.hidden;
    const E = c.experts;
    const n = M * c.topk;
    const out = p.streams.branch;
    qmm(p, e, x_in, w.sh_gate_up, p.sgu, M);
    swiglu(x, e, p.sgu, p.sact, M, c.moe_inter);
    qmm(p, e, p.sact, w.sh_down, p.ys, M);
    e.setPipeline(k.cast_f32);
    bind(e, 0, .{ x_in, p.xf });
    e.setValue(M * D, 2);
    e.dispatchThreads(size(M * D, 1, 1), size(256, 1, 1));
    var r0: u32 = 0;
    while (r0 < M) : (r0 += st.max_rows) {
        const rr = @min(st.max_rows, M - r0);
        e.setPipeline(k.router[rr - 1]);
        bind(e, 0, .{ p.xf.at(@as(usize, r0) * D * 4), w.router, p.logits_r.at(@as(usize, r0) * E * 4) });
        e.dispatchThreads(size(1024 * E / 16, 1, 1), size(1024, 1, 1));
    }
    e.setPipeline(k.route_rows);
    bind(e, 0, .{ p.logits_r, w.bias });
    e.setValue(c.routed_scale, 2);
    bind(e, 3, .{ p.pick, p.wts });
    e.setValue(M, 5);
    e.dispatchThreads(size(32 * M, 1, 1), size(256, 1, 1));
    const blocks = (n + 255) / 256;
    e.setPipeline(p.k.get("custom_kernel_tf_sort_count_uint32_t_int32_t_int32_t"));
    bind(e, 0, .{p.pick});
    params(e, 1, .{ n, E });
    bind(e, 2, .{p.counts});
    run(e, .{ blocks * 256, 1, 1 }, .{ 256, 1, 1 });
    e.setPipeline(p.k.get("custom_kernel_tf_sort_starts_int32_t_int32_t_int32_t_int32_t"));
    bind(e, 0, .{p.counts});
    params(e, 1, .{ n, E });
    bind(e, 2, .{ p.starts, p.offsets });
    run(e, .{ E, 1, 1 }, .{ E, 1, 1 });
    e.setPipeline(p.k.get("custom_kernel_tf_sort_place_uint32_t_int32_t_int32_t_uint32_t_uint32_t"));
    bind(e, 0, .{ p.pick, p.starts });
    params(e, 2, .{ n, E });
    bind(e, 3, .{ p.order, p.sorted });
    run(e, .{ blocks * 256, 1, 1 }, .{ 256, 1, 1 });
    e.setPipeline(p.k.get("custom_kernel_tf_rows_take_bfloat16_t_uint32_t_int32_t_bfloat16_t"));
    bind(e, 0, .{ x_in, p.order });
    params(e, 2, .{ n, D, c.topk });
    bind(e, 3, .{p.xs});
    run(e, .{ D, n, 1 }, .{ 256, 1, 1 });
    gather(p, e, p.xs, w.gate, p.offsets, p.g, n, E);
    gather(p, e, p.xs, w.up, p.offsets, p.u, n, E);
    e.setPipeline(k.act2);
    bind(e, 0, .{ p.g, p.u, p.act });
    e.setValue(c.swiglu_limit, 3);
    e.setValue(n * c.moe_inter, 4);
    e.dispatchThreads(size(n * c.moe_inter, 1, 1), size(256, 1, 1));
    gather(p, e, p.act, w.down, p.offsets, p.ye, n, E);
    e.setPipeline(p.k.get("custom_kernel_tf_sort_inverse_uint32_t_int32_t_uint32_t"));
    bind(e, 0, .{p.order});
    params(e, 1, .{n});
    bind(e, 2, .{p.inverse});
    run(e, .{ n, 1, 1 }, .{ 256, 1, 1 });
    e.setPipeline(p.k.get("custom_kernel_tf_rows_take_bfloat16_t_uint32_t_int32_t_bfloat16_t"));
    bind(e, 0, .{ p.ye, p.inverse });
    params(e, 2, .{ n, D, 1 });
    bind(e, 3, .{p.yp});
    run(e, .{ D, n, 1 }, .{ 256, 1, 1 });
    e.setPipeline(k.moe_combine);
    bind(e, 0, .{ p.ys, p.yp, p.wts });
    fwd.shape(e, 3, .{ M, c.topk });
    bind(e, 4, .{out});
    e.dispatchThreads(size(M * D, 1, 1), size(256, 1, 1));
}

/// MLA layer `mi` on M rows at positions pos..: tensor-unit projections, the decode path's cache writes, absorb and
/// unabsorb, and every row through the sparse kernel over its key list (all keys up to index_topk, else selected).
fn mla(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, w: *const wts.Mla, x_in: Ref, M: u32, pos: u32) void {
    const c = x.c;
    const XP: u32 = @intCast(std.mem.alignForward(usize, c.xProj(), 64));
    const width = c.keyWidth();
    qmm(p, e, x_in, w.x_proj, p.xp, M);
    fwd.rms(x, e, p.xp, w.q_norm, p.qr, M, c.q_lora, XP, c.q_lora, c.eps);
    qmm(p, e, p.qr, w.qr_proj, p.qp, M);
    fwd.mlaCache(x, e, mi, w, x_in, p.xp, XP, p.iw, M, pos);
    fwd.absorb(x, e, w, p.qp, p.ql, M);
    var dense: u32 = 0;
    while (dense < M and pos + dense + 1 <= c.i_topk) dense += 1;
    if (dense > 0) {
        e.setPipeline(x.k.dense_indices);
        bind(e, 0, .{p.indices});
        e.setValue([4]u32{ width, pos, dense, 0 }, 1);
        e.dispatchThreads(size(width, dense, 1), size(256, 1, 1));
    }
    var r0 = dense;
    while (r0 < M) : (r0 += select_rows) {
        const sb = @min(select_rows, M - r0);
        fwd.selectKeys(x, e, mi, p.qp.at((@as(usize, r0) * c.qrProj() + c.mla_heads * c.nope) * 2), c.qrProj(), p.iw.at(@as(usize, r0) * c.i_heads * 2), p.sscore, p.indices.at(@as(usize, r0) * width * 4), sb, pos + r0);
    }
    fwd.attendIndexed(x, e, mi, p.ql, p.indices, p.att, M, pos + M);
    fwd.unabsorb(x, e, w, p.att, p.vals, M);
    qmm(p, e, p.vals, w.o_proj, p.streams.branch, M);
}

/// The backbone over a chunk (tokens in `ids`) at positions pos..: final-normed rows into `streams.hidden`.
pub fn backbone(p: *const Prompt, x: *fwd.Ctx, e: mtl.ComputeEncoder, ids: Ref, M: u32, pos: u32) void {
    const c = x.c;
    const ss = &p.streams;
    std.debug.assert(x.sc == ss and M <= max_rows);
    fwd.embed(x, e, ids, M);
    var pending = false;
    var ki: usize = 0;
    var mi: usize = 0;
    for (0..c.run) |li| {
        const L = &x.w.layers[li];
        const hcs = L.hc.?;
        fwd.boundary(x, e, M, pending, hcs[0], L.in_norm);
        switch (L.attn) {
            .kda => |*a| {
                qmm(p, e, ss.normed, a.in_proj, p.proj, M);
                fwd.kdaStep(x, e, ki, a, p.proj, M, p.y);
                qmm(p, e, p.y, a.o_proj, ss.branch, M);
                ki += 1;
            },
            .mla => |*a| {
                mla(p, x, e, mi, a, ss.normed, M, pos);
                mi += 1;
            },
        }
        fwd.boundary(x, e, M, true, hcs[1], L.post_norm);
        switch (L.mlp) {
            .dense => |*d| {
                qmm(p, e, ss.normed, d.gate_up, p.gu, M);
                swiglu(x, e, p.gu, p.actd, M, c.dense_inter);
                qmm(p, e, p.actd, d.down, ss.branch, M);
            },
            .moe => |*m| moe(p, x, e, m, ss.normed, M),
        }
        pending = true;
    }
    fwd.boundary(x, e, M, true, null, null);
    e.setPipeline(x.k.stream_mean);
    bind(e, 0, .{ ss.x[x.xi], ss.raw });
    e.setValue([2]u32{ c.hidden, M }, 2);
    e.dispatchThreads(size(c.hidden, M, 1), size(256, 1, 1));
    fwd.rms(x, e, ss.raw, x.w.norm, ss.hidden, M, c.hidden, c.hidden, c.hidden, c.eps);
}

/// The MTP head over M prompt rows (final-normed `h`, the tokens after them in `next`) at head positions pos..: the
/// rows enter its cache; the drafts come later, from the decode path.
pub fn mtp(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, h: Ref, next: Ref, M: u32, pos: u32) void {
    const c = x.c;
    const D = c.hidden;
    const L = &x.w.layers[c.layers];
    const m = x.w.mtp.?;
    fwd.embedRows(x, e, next, p.m_emb, M);
    fwd.rms(x, e, p.m_emb, m.enorm, p.m_eh, M, D, D, 2 * D, c.eps);
    fwd.rms(x, e, h, m.hnorm, p.m_eh.at(@as(usize, D) * 2), M, D, D, 2 * D, c.eps);
    qmm(p, e, p.m_eh, m.eh_proj, p.m_x, M);
    fwd.rms(x, e, p.m_x, L.in_norm, p.m_xn, M, D, D, D, c.eps);
    mla(p, x, e, c.countKind(.mla), &L.attn.mla, p.m_xn, M, pos);
    add(x, e, p.m_x, p.streams.branch, p.m_out, M * D);
    fwd.rms(x, e, p.m_out, L.post_norm, p.m_xn, M, D, D, D, c.eps);
    moe(p, x, e, &L.mlp.moe, p.m_xn, M);
    add(x, e, p.m_out, p.streams.branch, p.m_x, M * D);
}
