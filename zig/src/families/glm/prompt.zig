//! Prompt chunks: matmuls on the M5 tensor units at the row kernels' arithmetic (core/affine_mm.zig), the rest on the row kernels.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const pk = @import("../nemotron/prefill_kernels.zig");
const ep_mod = @import("ep.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const hc = @import("../../core/hc.zig");
const kernels = @import("kernels.zig");
const Kernels = kernels.Kernels;
const moe_route = @import("../../core/moe_route.zig");
const Ref = wts.Ref;
const bind = fwd.bind;
const size = fwd.size;

/// Chunk heights a load picks from (the tallest whose buffers fit this Mac's limit); expert parallel keeps one exchange's.
pub const heights = [_]u32{ 16384, 8192, 6144, 4096, 2048 };
pub const max_rows = heights[0];
/// Sparse rows whose block scores one selection pass holds.
const select_rows = 256;

/// The chunk's buffers beyond the decode scratch's (whose stream fields `streams` points at prompt-sized ones).
pub const Prompt = struct {
    k: pk.Kernels,
    mm: *const Kernels, // the engine's: core/affine_mm.zig's pipelines among them
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
    logits_r: Ref,
    pick: Ref,
    wts: Ref,
    counts: Ref,
    starts: Ref,
    offsets: Ref,
    order: Ref,
    sorted: Ref,
    act: Ref,
    yp: Ref,
    yf: Ref, // by rows: fp32 [rows * topk, hidden], the down partials by (row, slot); one Mac: the planes `buffers` lists
    part: Ref, // fp32 [rows, hidden]: TP2's partial of a row-split projection
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
    sums: Ref, // fp32 group sums of a matmul's rows (affine_mm's bias operand)
    sparse_nax: bool, // the sparse MLA attention on the tensor units (GLM_SPARSE_NAX=0: the row kernel)
    rows: u32, // a chunk's rows at most (the buffers' height)
    kda_rows: u32, // the rows a KDA block's planes hold (half a chunk)

    pub fn deinit(p: *Prompt) void {
        p.k.deinit();
    }
};

/// The chunk buffers of `rows`-row chunks for `cap`-token caches (selection passes read cap / 4 block scores a row).
pub fn init(gpa: std.mem.Allocator, arena: *st.Arena, device: mtl.Device, c: *const cfg.Config, decode: *const st.Scratch, engine_kernels: *const Kernels, cap: u32, rows: u32) !Prompt {
    var p: Prompt = undefined;
    p.k = try pk.load(gpa, device);
    errdefer p.k.deinit();
    p.mm = engine_kernels;
    try buffers(&p, arena, c, decode, cap, rows);
    return p;
}

/// The prompt's sparse MLA on the tensor units (GLM_SPARSE_NAX=0: the row kernel, its output beside its queries).
fn sparseNax() bool {
    return if (std.c.getenv("GLM_SPARSE_NAX")) |v| v[0] != '0' else true;
}

/// The bytes `init` takes from the arena for `rows`-row chunks.
pub fn chunkBytes(gpa: std.mem.Allocator, c: *const cfg.Config, decode: *const st.Scratch, cap: u32, rows: u32) usize {
    var dry: st.Arena = .{ .device = undefined, .gpa = gpa, .dry = true };
    var p: Prompt = undefined;
    buffers(&p, &dry, c, decode, cap, rows) catch unreachable;
    return dry.bytes;
}

fn buffers(p: *Prompt, arena: *st.Arena, c: *const cfg.Config, decode: *const st.Scratch, cap: u32, rows: u32) !void {
    const R: usize = rows;
    const D: usize = c.hidden;
    const n: usize = R * c.topk;
    const blocks = (n + 255) / 256;
    const up = struct {
        fn to(v: usize) usize {
            return std.mem.alignForward(usize, v, 256);
        }
    }.to;
    p.rows = rows;
    p.kda_rows = @max(rows / 2, 1);
    p.sparse_nax = sparseNax();
    p.streams = decode.*;
    const big = struct {
        fn of(a: *st.Arena, bytes: usize) !Ref {
            return a.buffer(bytes);
        }
    };
    const xs = try big.of(arena, R * 4 * D * 2); // one buffer: the expands write the streams over themselves (core/hc.zig)
    p.streams.x = .{ xs, xs };
    p.streams.normed = try big.of(arena, R * D * 2);
    p.streams.branch = try big.of(arena, R * D * 2);
    p.streams.post = try big.of(arena, R * 4 * 4);
    p.streams.comb = try big.of(arena, R * 16 * 4);
    p.streams.inv = try big.of(arena, R * 4);
    p.streams.mixes = try big.of(arena, R * hc.partBytes(.{ .width = @intCast(D), .sinkhorn = 0, .eps_e9 = 0 }));
    p.sscore = try big.of(arena, select_rows * (@as(usize, cap) / c.kpool + 1) * 4);
    p.pick = try big.of(arena, n * 4);
    p.wts = try big.of(arena, n * 4);
    p.counts = try big.of(arena, blocks * c.experts * 4);
    p.starts = try big.of(arena, blocks * c.experts * 4);
    p.offsets = try big.of(arena, (c.experts + 1) * 4);
    p.order = try big.of(arena, n * 4);
    p.sorted = try big.of(arena, n * 4);
    p.sums = try big.of(arena, @max(R * 16384, n * D) / 64 * 4); // the widest dense K (MLA's out-projection), or a gather's
    // the rest share three regions by layer; proj: KDA's projection, MLA's queries and key lists then values, the experts' planes, the dense gate and up
    const pitch = std.mem.alignForward(usize, c.kdaProj(), 64); // the matmul's padded pitch
    const qp = up(R * c.qrProj() * 2);
    const act = up(n * c.moe_inter * 2);
    const ys = up(R * D * 2);
    p.proj = try big.of(arena, @max(@max(R * pitch * 2, qp + R * c.keyWidth() * 4), @max(act + ys + R * c.experts * 4, R * 2 * c.dense_inter * 2)));
    p.qp = p.proj;
    p.indices = p.proj.at(qp);
    p.vals = p.proj; // written once the attention has read the queries and key lists
    p.sgu = p.proj; // done before the routed experts write their activations (one Mac) or after their down (by rows)
    p.sact = p.proj.at(up(R * 2 * c.moe_inter * 2));
    p.act = p.proj;
    p.ys = p.proj.at(act);
    p.logits_r = p.ys.at(ys);
    p.gu = p.proj;
    // y: KDA's output; MLA's input projection, normed queries and index weights
    const xp = up(R * std.mem.alignForward(usize, c.xProj(), 64) * 2);
    const qr = up(R * c.q_lora * 2);
    p.y = try big.of(arena, @max(R * c.kdaWidth() * 2, xp + qr + R * c.i_heads * 2));
    p.xp = p.y;
    p.qr = p.y.at(xp);
    p.iw = p.y.at(xp + qr);
    // yf: embeddings, KDA planes, MLA latent queries and attention, expert outputs, dense activations, the final rows (read before the MTP head writes here)
    const plane = @as(usize, p.kda_rows) * c.kdaWidth() * 2;
    const lat = R * c.mla_heads * c.kv_lora * 2;
    const part = up(R * c.dense_inter * 2);
    const row = up(R * D * 2);
    const one = @max(@max(7 * plane + @as(usize, p.kda_rows) * c.kda_heads * 4, if (p.sparse_nax) lat else 2 * lat), @max(@max(n * D * 2, part + R * D * 4), 4 * row));
    p.yf = try big.of(arena, if (c.byRows()) @max(one, n * D * 4) else one);
    p.streams.h = p.yf;
    p.streams.raw = p.yf; // the final mean, long after the embedding rows were read
    p.streams.hidden = p.yf.at(row);
    p.ql = p.yf;
    p.att = if (p.sparse_nax) p.yf else p.yf.at(lat); // the tensor-unit kernel reads a row's queries before it writes over them
    p.yp = p.yf;
    p.actd = p.yf;
    p.part = p.yf.at(part);
    p.m_eh = p.yf.at(2 * row);
    p.m_emb = xs; // the MTP head's rows in the streams (the backbone is done)
    p.m_x = xs.at(row);
    p.m_xn = xs.at(2 * row);
    p.m_out = xs.at(3 * row);
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

/// y [M, N] = x [M, K] W^T for a 4-bit matrix (64x64 tiles; N rounded up to the weights' padded rows, y's row pitch).
fn qmm(p: *const Prompt, e: mtl.ComputeEncoder, x: Ref, q: wts.Q4, y: Ref, M: u32) void {
    affine_mm.rowSums(e, p.mm.mm_bf16, 64, x, p.sums, M, q.k);
    affine_mm.dense(e, p.mm.mm_bf16, x, p.sums, q, y, M);
}

/// `qmm` over x's rows `ld` elements apart (a slice of a wider row).
fn qmmAt(p: *const Prompt, e: mtl.ComputeEncoder, x: Ref, ld: u32, q: wts.Q4, y: Ref, M: u32) void {
    affine_mm.rowSumsAt(e, p.mm.mm_bf16, 64, x, p.sums, M, q.k, ld);
    affine_mm.denseBatch(e, p.mm.mm_bf16, x, p.sums, q, y, M, 1, .{ .x_row = @intCast(ld), .y_row = @intCast(std.mem.alignForward(u32, q.n, 64)) });
}

/// `qmm` with fp32 out (TP2: one Mac's partial of a row-split projection).
fn qmmF32(p: *const Prompt, e: mtl.ComputeEncoder, x: Ref, q: wts.Q4, y: Ref, M: u32) void {
    affine_mm.rowSums(e, p.mm.mm_bf16, 64, x, p.sums, M, q.k);
    affine_mm.dense(e, p.mm.mm_f32, x, p.sums, q, y, M);
}

/// The routed experts' gate, up and SwiGLU on the M rows of `x_in` its pairs read (in expert order) into `act` ([n, gate.n]).
fn gateUp(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32, n: u32) void {
    affine_mm.rowSums(e, p.mm.mm_bf16, 64, x_in, p.sums, M, w.gate.k);
    affine_mm.gatherGlu(e, p.mm.mm_bf16, x_in, p.sums, w.gate, w.up, p.offsets, p.order, p.act, n, x.c.experts, x.c.topk, x.c.swiglu_limit);
    affine_mm.rowSums(e, p.mm.mm_bf16, 64, p.act, p.sums, n, w.down.k);
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

/// The shared expert on M rows into `ys`.
fn shared(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32) void {
    qmm(p, e, x_in, w.sh_gate_up, p.sgu, M);
    swiglu(x, e, p.sgu, p.sact, M, x.c.moe_inter);
    qmm(p, e, p.sact, w.sh_down, p.ys, M);
}

/// The MoE block on M rows: shared expert, route, experts gathered by expert; by rows, this Mac's halves summed in slot order and swapped.
fn moe(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32, ep: ?*ep_mod.Ep) void {
    const c = x.c;
    const k = x.k;
    const D = c.hidden;
    const E = c.experts;
    const n = M * c.topk;
    const out = p.streams.branch;
    const s = x.skip; // classes a profile leaves out
    const Class = fwd.Class;
    if (ep) |t| {
        if (s & Class.exchange == 0) t.begin();
    } else if (s & Class.shared == 0) shared(p, x, e, w, x_in, M);
    if (s & Class.route == 0) route(p, x, e, w, x_in, M);
    if (ep) |t| {
        if (s & Class.routed == 0) { // down's fp32 partials put back in (row, slot) order for the decode's combine
            gateUp(p, x, e, w, x_in, M, n);
            affine_mm.gatherTo(e, p.mm.mm_f32, p.act, p.sums, w.down, p.offsets, p.yf, p.order, n, E);
        }
        if (s & Class.exchange == 0) t.sendRows(e, p.yf, p.wts, M);
        if (s & Class.shared == 0) shared(p, x, e, w, x_in, M);
        if (s & Class.combine == 0) t.receiveRows(e, p.ys, out, M);
        return;
    }
    if (s & Class.routed == 0) { // down's rows put back in (row, slot) order for the combine
        gateUp(p, x, e, w, x_in, M, n);
        affine_mm.gatherTo(e, p.mm.mm_bf16, p.act, p.sums, w.down, p.offsets, p.yp, p.order, n, E);
    }
    if (s & Class.combine != 0) return;
    e.setPipeline(k.moe_combine);
    bind(e, 0, .{ p.ys, p.yp, p.wts });
    fwd.shape(e, 3, .{ M, c.topk });
    bind(e, 4, .{out});
    e.dispatchThreads(size(M * D, 1, 1), size(256, 1, 1));
}

/// The route on M rows: the decode's router and top-k, pairs sorted by expert (`order`: each sorted pair's index).
fn route(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, M: u32) void {
    const c = x.c;
    const k = x.k;
    const E = c.experts;
    const n = M * c.topk;
    moe_route.logits(e, k.route_logits, kernels.route_shape, x_in, w.router, p.logits_r, M); // the decode's bits, any rows
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
}

/// KDA layer `ki` over a chunk (glm_kda_prompt.metal) by blocks of `kda_rows`: f_b and g_b on the tensor units, prep, the recurrence (from the second block on, the state carried in place), the norm.
fn kdaChunk(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, ki: usize, w: *const wts.Kda, M: u32) void {
    const c = x.c;
    const k = x.k;
    const L = &x.s.kda[ki];
    const cur = L.cur;
    const H = c.kda_heads;
    const W = c.kdaWidth();
    const tp = c.tp > 1; // TP2: this Mac's heads
    const pitch: u32 = std.mem.alignForward(u32, c.kdaProj(), 64);
    const fa = 3 * W; // the projection row: q, k, v, then f_a, g_a and beta
    var r0: u32 = 0;
    while (r0 < M) : (r0 += p.kda_rows) {
        const m = @min(p.kda_rows, M - r0);
        const plane = @as(usize, m) * W * 2; // [m, heads * dim] bf16
        const q = p.yf;
        const kk = q.at(plane);
        const v = kk.at(plane);
        const sy = v.at(plane);
        const g = sy.at(plane); // fp32
        const gate = g.at(2 * plane);
        const beta = gate.at(plane);
        if (fwd.on(x, "kda_pre")) {
            const at = @as(usize, r0) * pitch * 2; // the block's projection rows
            qmmAt(p, e, p.proj.at(at + @as(usize, fa) * 2), pitch, w.f_b, sy, m); // f_b's rows into sy's plane, read before the scan writes it
            qmmAt(p, e, p.proj.at(at + @as(usize, fa + c.kda_dim) * 2), pitch, w.g_b, gate, m);
            e.setPipeline(if (tp) k.kda_prep_tp else k.kda_prep);
            bind(e, 0, .{p.proj});
            fwd.shape(e, 1, .{ M, pitch, r0 });
            bind(e, 2, .{ L.cs[cur], w.conv_w, sy, w.a, w.dt_bias });
            e.setValue(c.lower_bound, 7);
            bind(e, 8, .{ q, kk, v, g, beta });
            e.dispatchGroups(size(m, H / 4, 1), size(128, 1, 1));
        }
        if (fwd.on(x, "kda_scan")) {
            e.setPipeline(if (tp) k.kda_scan_tp else k.kda_scan);
            bind(e, 0, .{ q, kk, v, g, beta, L.st[if (r0 == 0) cur else 1 - cur], L.st[1 - cur], sy });
            e.setValue(@as(i32, @intCast(m)), 8);
            e.dispatchGroups(size(c.kda_dim / 64, H, 1), size(512, 1, 1));
        }
        if (!fwd.on(x, "kda_post")) continue;
        e.setPipeline(if (tp) k.kda_post_tp else k.kda_post);
        bind(e, 0, .{ sy, gate, w.o_norm });
        e.setValue(c.eps, 3);
        bind(e, 4, .{ p.y.at(@as(usize, r0) * W * 2), p.proj });
        fwd.shape(e, 6, .{ M, pitch }); // every block writes the next chunk's conv window, the same rows
        bind(e, 7, .{ L.cs[cur], L.cs[1 - cur] });
        e.dispatchGroups(size(m, H, 1), size(32, 1, 1));
    }
}

/// Every row's 64 heads' latent outputs to values on the tensor units: each head's value half of kv_b, one dense product of a batch.
fn unabsorb(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, w: *const wts.Mla, M: u32) void {
    const c = x.c;
    const H = c.mla_heads;
    const K = c.kv_lora;
    const per = c.nope + c.v_dim; // kv_b's rows a head: its key half, then its value half
    var q = w.kv_b;
    q.w = q.w.at(@as(usize, c.nope) * K / 2);
    q.s = q.s.at(@as(usize, c.nope) * (K / 64) * 2);
    q.b = q.b.at(@as(usize, c.nope) * (K / 64) * 2);
    q.n = c.v_dim;
    affine_mm.rowSums(e, p.mm.mm_bf16, 64, p.att, p.sums, M * H, K);
    affine_mm.denseBatch(e, p.mm.mm_bf16, p.att, p.sums, q, p.vals, M, H, .{ .x_row = @intCast(H * K), .y_row = @intCast(H * c.v_dim), .sums_row = @intCast(H), .x_batch = @intCast(K), .y_batch = @intCast(c.v_dim), .sums_batch = 1, .w_batch = @intCast(per) });
}

/// `n` rows' key lists from their index scores over every pooled block they read (positions p0..), the scores on the tensor units.
fn selectKeys(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, iq: Ref, iw: Ref, indices: Ref, n: u32, p0: u32) void {
    const c = x.c;
    const s_stride = x.s.cap / c.kpool + 1;
    const most = (p0 + n) / c.kpool; // the last row's blocks
    e.setPipeline(x.k.index_scores_nax);
    bind(e, 0, .{ iq, iw, x.s.mla[mi].pool, p.sscore });
    e.setValue([5]u32{ p0, c.qrProj(), c.i_heads, s_stride, n }, 4);
    e.dispatchGroups(size((most + 63) / 64, (n + 63) / 64, 1), size(128, 1, 1));
    e.setPipeline(x.k.index_select);
    bind(e, 0, .{ p.sscore, indices });
    e.setValue([5]u32{ p0, c.i_topk / c.kpool, c.keyWidth(), s_stride, c.keyWidth() }, 2);
    e.dispatchGroups(size(n, 1, 1), size(1024, 1, 1));
}

/// MLA layer `mi` on M rows at pos..: tensor-unit projections, the decode cache writes, each row's key list in the sparse kernel.
fn mla(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, mi: usize, w: *const wts.Mla, x_in: Ref, M: u32, pos: u32) void {
    const c = x.c;
    const XP: u32 = @intCast(std.mem.alignForward(usize, c.xProj(), 64));
    const width = c.keyWidth();
    if (fwd.on(x, "mla_proj")) {
        qmm(p, e, x_in, w.x_proj, p.xp, M);
        fwd.rms(x, e, p.xp, w.q_norm, p.qr, M, c.q_lora, XP, c.q_lora, c.eps);
        qmm(p, e, p.qr, w.qr_proj, p.qp, M);
    }
    if (fwd.on(x, "mla_cache")) fwd.mlaCache(x, e, mi, w, x_in, p.xp, XP, p.iw, M, pos);
    if (fwd.on(x, "mla_absorb")) {
        e.setPipeline(if (c.tp > 1) x.k.absorb_nax_tp else x.k.absorb_nax); // the row kernel's qvm, weights in three bf16 parts
        bind(e, 0, .{ w.kv_b.w, w.kv_b.s, w.kv_b.b, p.qp, p.ql });
        e.setValue([2]i32{ @intCast(M), @intCast(c.qrProj()) }, 5);
        e.dispatchGroups(size(c.kv_lora / 64, (M + 63) / 64, c.mla_heads), size(128, 1, 1));
    }
    var dense: u32 = 0;
    while (dense < M and pos + dense + 1 <= c.i_topk) dense += 1;
    if (fwd.on(x, "mla_select")) {
        if (dense > 0) {
            e.setPipeline(x.k.dense_indices);
            bind(e, 0, .{p.indices});
            e.setValue([4]u32{ width, pos, dense, 0 }, 1);
            e.dispatchThreads(size(width, dense, 1), size(256, 1, 1));
        }
        var r0 = dense;
        while (r0 < M) : (r0 += select_rows) {
            const sb = @min(select_rows, M - r0);
            selectKeys(p, x, e, mi, p.qp.at((@as(usize, r0) * c.qrProj() + c.mla_heads * c.nope) * 2), p.iw.at(@as(usize, r0) * c.i_heads * 2), p.indices.at(@as(usize, r0) * width * 4), sb, pos + r0);
        }
    }
    if (fwd.on(x, "mla_attn")) {
        if (p.sparse_nax) {
            e.setPipeline(if (c.tp > 1) x.k.sparse_nax_tp else x.k.sparse_nax);
            bind(e, 0, .{ p.ql, x.s.mla[mi].keys, p.indices });
            e.setValue(@as(f32, 1.0 / 16.0), 3);
            e.setValue([4]i32{ @intCast(width), @intCast(pos + M), 0, 0 }, 4);
            bind(e, 5, .{p.att});
            e.dispatchGroups(size(c.mla_heads / 16, M, 1), size(128, 1, 1));
        } else fwd.attendIndexed(x, e, mi, p.ql, p.indices, p.att, M, pos + M);
    }
    if (fwd.on(x, "mla_unabs")) unabsorb(p, x, e, w, M);
    if (!fwd.on(x, "mla_out")) return;
    if (c.tp == 1) return qmm(p, e, p.vals, w.o_proj, p.streams.branch, M);
    qmmF32(p, e, p.vals, w.o_proj, p.part, M); // TP2: this Mac's heads' partial, summed with the peer's
    x.ep.?.reduce(e, p.part, p.streams.branch, M);
}

/// The backbone over a chunk (tokens in `ids`) at positions pos..: final-normed rows into `streams.hidden`.
pub fn backbone(p: *const Prompt, x: *fwd.Ctx, e: mtl.ComputeEncoder, ids: Ref, M: u32, pos: u32) void {
    const c = x.c;
    const ss = &p.streams;
    std.debug.assert(x.sc == ss and M <= p.rows);
    const s = x.skip; // classes a profile leaves out
    const Class = fwd.Class;
    if (s & Class.ends == 0) fwd.embed(x, e, ids, M, pos);
    const plane = @as(usize, M) * c.hidden * 2; // a trace's capture points: the decode backbone's
    fwd.snap(x, e, ss.h, plane);
    var pending = false;
    var ki: usize = 0;
    var mi: usize = 0;
    for (0..c.run) |li| {
        const L = &x.w.layers[li];
        const hcs = L.hc.?;
        if (s & Class.hc == 0) fwd.boundary(x, e, M, pending, hcs[0], L.in_norm);
        fwd.snap(x, e, ss.normed, plane);
        switch (L.attn) {
            .kda => |*a| {
                if (s & Class.kda == 0) {
                    if (fwd.on(x, "kda_in")) qmm(p, e, ss.normed, a.in_proj, p.proj, M);
                    kdaChunk(p, x, e, ki, a, M);
                    if (c.tp > 1) { // TP2: this Mac's heads' out-projection partial, summed with the peer's
                        qmmF32(p, e, p.y, a.o_proj, p.part, M);
                        x.ep.?.reduce(e, p.part, ss.branch, M);
                    } else if (fwd.on(x, "kda_out")) qmm(p, e, p.y, a.o_proj, ss.branch, M);
                }
                ki += 1;
            },
            .mla => |*a| {
                if (s & Class.mla == 0) mla(p, x, e, mi, a, ss.normed, M, pos);
                mi += 1;
            },
        }
        fwd.snap(x, e, ss.branch, plane);
        if (s & Class.hc == 0) fwd.boundary(x, e, M, true, hcs[1], L.post_norm);
        fwd.snap(x, e, ss.normed, plane);
        switch (L.mlp) {
            .dense => |*d| if (s & Class.dense == 0) {
                qmm(p, e, ss.normed, d.gate_up, p.gu, M);
                swiglu(x, e, p.gu, p.actd, M, c.dense_inter);
                if (c.tp > 1) { // TP2: this Mac's half, summed with the peer's
                    qmmF32(p, e, p.actd, d.down, p.part, M);
                    x.ep.?.reduce(e, p.part, ss.branch, M);
                } else qmm(p, e, p.actd, d.down, ss.branch, M);
            },
            .moe => |*m| moe(p, x, e, m, ss.normed, M, if (c.byRows()) x.ep else null),
        }
        fwd.snap(x, e, ss.branch, plane);
        pending = true;
    }
    if (s & Class.hc == 0) fwd.boundary(x, e, M, true, null, null);
    if (s & Class.ends != 0) return;
    e.setPipeline(x.k.stream_mean);
    bind(e, 0, .{ ss.x[x.xi], ss.raw });
    e.setValue([2]u32{ c.hidden, M }, 2);
    e.dispatchThreads(size(c.hidden, M, 1), size(256, 1, 1));
    fwd.rms(x, e, ss.raw, x.w.norm, ss.hidden, M, c.hidden, c.hidden, c.hidden, c.eps);
    fwd.snap(x, e, ss.hidden, plane);
}

/// The MTP head over M prompt rows (`h`, next tokens) at head positions pos..: their cache entries only (no one reads the block's output).
pub fn mtp(p: *const Prompt, x: *const fwd.Ctx, e: mtl.ComputeEncoder, h: Ref, next: Ref, M: u32, pos: u32) void {
    const c = x.c;
    const D = c.hidden;
    const L = &x.w.layers[c.layers];
    const m = x.w.mtp.?;
    fwd.embedRows(x, e, next, p.m_emb, M);
    fwd.inject(x, e, p.m_emb, M, pos + 1); // the head reads each row's next token: an image's rows one place on
    fwd.rms(x, e, p.m_emb, m.enorm, p.m_eh, M, D, D, 2 * D, c.eps);
    fwd.rms(x, e, h, m.hnorm, p.m_eh.at(@as(usize, D) * 2), M, D, D, 2 * D, c.eps);
    qmm(p, e, p.m_eh, m.eh_proj, p.m_x, M);
    fwd.rms(x, e, p.m_x, L.in_norm, p.m_xn, M, D, D, D, c.eps);
    qmm(p, e, p.m_xn, L.attn.mla.x_proj, p.xp, M);
    fwd.mlaCache(x, e, c.countKind(.mla), &L.attn.mla, p.m_xn, p.xp, @intCast(std.mem.alignForward(usize, c.xProj(), 64)), p.iw, M, pos);
}
