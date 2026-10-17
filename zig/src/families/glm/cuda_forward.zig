//! GLM-5.3-Flash's TP forward on CUDA (the Python family's forward.compute and commit), partials over NCCL or copied.

const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const weights = @import("cuda_weights.zig");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const experts = @import("cuda_experts.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const state = @import("cuda_state.zig");
const mm_mod = @import("cuda_mm.zig");
const prof_mod = @import("cuda_prof.zig");
const images_mod = @import("cuda_images.zig");
const exl3 = @import("cuda_exl3.zig");
const shared = @import("cuda_shared.zig");

pub const prefill_rows = state.prefill_rows;
pub const window_rows = state.window_rows;
pub const State = state.State;
pub const Buffers = state.Buffers;

/// The modules a forward launches from: our fatbins.
pub const Modules = struct {
    glue: glue.Kernels,
    mods: [9]cuda.Module,
    kda: kda.Kernels,
    lane: cuda.qmmf.Lane, // the NVFP4 linears' decode rows (qlinear)
    prompt: cuda.nvfp4.Prompt, // and their prompt rows
    experts: experts.Kernels,
    exl3: exl3.Kernels, // an EXL3 checkpoint's routed experts
    mm: mm_mod.Kernels, // glm_mm: the bf16 linears on tiled weights
    tiled: bool = false, // the bf16 linears' weights are tiled (Engine.init, TF_GLM_MM unset or 1): glm_mm runs them
    packs: ?*const mm_mod.Packed = null, // the linears packed to 4, 6 or 8 bits (TF_GLM_DENSE_BITS, _HEAD_BITS)
    copy_rows: cuda.Function,
    put_slot: cuda.Function,
    pack_dense: cuda.Function,
    quant: quant.Kernels,
    q4: q4.Kernels,

    pub fn load(d: *const cuda.Driver, ctx: *const cuda.Context) !Modules {
        const k = cuda.kernels;
        var m: Modules = undefined;
        var n: usize = 0;
        errdefer for (m.mods[0..n]) |*x| x.unload();
        for (&m.mods, [_][]const u8{ k.glm_kda, k.glm_ops, k.experts, k.affine4_pack, k.glm_quant, k.qmm_group, k.glm_glue, k.glm_exl3, k.glm_mm }) |*x, image| {
            x.* = try cuda.Module.load(d, image);
            n += 1;
        }
        m.lane = try cuda.qmmf.Lane.load(d, @intCast(try ctx.attribute(.compute_capability_major)));
        errdefer m.lane.unload();
        m.prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer m.prompt.unload();
        m.kda = try kda.Kernels.load(m.mods[0]);
        m.experts = try experts.Kernels.load(d, m.mods[2], @intCast(try ctx.attribute(.multiprocessor_count)));
        errdefer m.experts.unload();
        m.copy_rows = try m.mods[1].function("tf_glm_copy_rows");
        m.put_slot = try m.mods[1].function("tf_glm_put_slot");
        m.pack_dense = try m.mods[3].function("tf_pack_dense");
        m.quant = try quant.Kernels.load(m.mods[4]);
        m.q4 = try q4.Kernels.load(m.mods[5], m.mods[3], ctx);
        m.glue = try glue.Kernels.load(m.mods[6]);
        m.exl3 = try exl3.Kernels.load(m.mods[7]);
        m.mm = try mm_mod.Kernels.load(m.mods[8]);
        m.tiled = false;
        m.packs = null; // `m` starts undefined: the field default does not apply
        return m;
    }

    /// What weights.load runs: the dense packing, and for the MTP head the quantizers and the 4-bit tiling.
    pub fn tools(m: *const Modules) weights.Tools {
        return .{ .pack_dense = m.pack_dense, .quant = &m.quant, .q4 = &m.q4 };
    }

    pub fn ops(m: *const Modules) cuda.Module {
        return m.mods[1];
    }

    pub fn deinit(m: *Modules) void {
        for (&m.mods) |*x| x.unload();
        m.lane.unload();
        m.prompt.unload();
        m.experts.unload();
        m.* = undefined;
    }
};

/// overlap.ROW_STEP and PIECES_MAX: piece boundaries on the BF16 prompt matmul's 128-row blocks, at most 16 pieces.
pub const row_step = 128;
pub const max_pieces = 16;

/// overlap.ranges: [lo, hi) row pieces of a chunk of `rows`, about equal on row_step boundaries (fewer when short).
pub fn ranges(rows: usize, pieces: usize, out: *[max_pieces][2]usize) usize {
    const per = (rows + pieces - 1) / pieces;
    const size = (per + row_step - 1) / row_step * row_step;
    var n: usize = 0;
    var lo: usize = 0;
    while (lo < rows) : (lo += size) {
        out[n] = .{ lo, @min(rows, lo + size) };
        n += 1;
    }
    return n;
}

/// A prompt chunk's pieced exchanges (overlap.Overlap): second stream and events; `count`: TF_GLM_OVERLAP_PIECES (3).
pub const Pieces = struct {
    count: usize = 4,
    side: cuda.Stream,
    filled: [max_pieces]cuda.Event,
    gathered: [max_pieces]cuda.Event,
    glues: [max_pieces]cuda.Event, // split: a piece's own rows glued
    shared: [max_pieces]cuda.Event, // split: every rank's glued rows of a piece in

    pub fn init(d: *const cuda.Driver, count: usize) !Pieces {
        if (count == 0 or count > max_pieces) return error.BadPieces;
        var p: Pieces = .{ .count = count, .side = try cuda.Stream.init(d, false), .filled = undefined, .gathered = undefined, .glues = undefined, .shared = undefined };
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.* = try cuda.Event.init(d, false);
            b.* = try cuda.Event.init(d, false);
            g.* = try cuda.Event.init(d, false);
            s.* = try cuda.Event.init(d, false);
        }
        return p;
    }

    pub fn deinit(p: *Pieces) void {
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.deinit();
            b.deinit();
            g.deinit();
            s.deinit();
        }
        p.side.deinit();
        p.* = undefined;
    }
};

/// How the ranks' partials meet: every rank in-process copying into the others' slots, or one rank over NCCL.
pub const Exchange = union(enum) {
    local,
    // group: the main stream's all-gathers through it (RoCE writes for small ones on two ranks); side streams stay raw
    nccl: struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, group: ?*const cuda.tp_group.Group = null },
};

/// One rank of the forward: its weights, its sequence's caches, the window's buffers.
pub const Rank = struct {
    w: *const weights.Weights,
    st: *State,
    b: *Buffers,
    /// A shared window (Buffers.initShared): each stream's rows in row order; empty: every row on `st`.
    segs: []const Seg = &.{},
};

/// The engine's image rows and the sequence the forward runs on (they splice into that sequence's rows only).
pub const ImageRows = struct { v: *images_mod.Images, slot: u32 };

/// One stream's rows of a shared window: rows row0..row0 + rows, on sequence `st` from its position.
pub const Seg = struct { st: *State, row0: usize, rows: usize };

pub const Forward = struct {
    d: *const cuda.Driver,
    m: *const Modules,
    stream: cuda.Stream,
    ranks: []const Rank, // every rank of a local check, or this process's one
    exchange: Exchange,
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE (the Python family's default): decode windows on KDA's three-kernel chain
    pieces: ?*const Pieces = null, // a prompt chunk's exchanges in row pieces on a second stream (null: whole)
    split: bool = true, // TF_GLM_PREFILL_REDUCE=split (the default with send/recv): each rank glues its own rows
    images: ?ImageRows = null, // a prompt pass's image rows over their placeholders' embeddings (its sequence only)
    prof: ?*prof_mod.Prof = null, // TF_GLM_PROF=1: events between the sections

    pub fn set(f: *const Forward) glue.Set {
        return .{ .glue = &f.m.glue, .mm = if (f.m.tiled) &f.m.mm else null, .packs = f.m.packs };
    }

    /// reduce.sums' gather: every rank's partial of `rows` rows into every rank's gathered buffer, rank order.
    fn gather(f: *const Forward, rows: usize) !void {
        const c = &f.ranks[0].w.config;
        const bytes = rows * c.hidden * 4;
        switch (f.exchange) {
            .local => for (f.ranks) |to| for (f.ranks) |from| {
                if (to.w.share.rank == from.w.share.rank) continue;
                const off = from.w.share.rank * bytes;
                try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(to.b.gathered + off, from.b.gathered + off, bytes, f.stream.handle), "cuMemcpyDtoDAsync");
            },
            .nccl => |n| {
                const r = f.ranks[0];
                if (n.group) |g| return g.allGather(f.stream, r.b.gathered + r.w.share.rank * bytes, r.b.gathered, bytes);
                const count = rows * c.hidden;
                try n.lib.check(n.lib.api.ncclAllGather(r.b.gathered + r.w.share.rank * bytes, r.b.gathered, count, .f32, n.comm, f.stream.handle), "ncclAllGather");
            },
        }
    }

    fn slot(r: Rank, rows: usize) u64 {
        return r.b.gathered + r.w.share.rank * rows * r.w.config.hidden * 4;
    }

    /// forward._pre: hc_pre of every row at a site into b.normed, xs, post and comb.
    fn pre(f: *const Forward, r: Rank, hc: weights.Hc, norm: u64, rows: usize) !void {
        const c = &r.w.config;
        const b = r.b;
        try glue.hcPre(f.set(), f.stream, b.x, hc.fn_w, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, rows, c.hidden, c.hc);
    }

    fn post(f: *const Forward, r: Rank, rows: usize) !void {
        const c = &r.w.config;
        try glue.hcPost(f.set(), f.stream, r.b.x, r.b.x, r.b.gathered, r.b.post, r.b.comb, rows, c.hidden, c.hc, r.w.share.world);
    }

    /// forward._mixer: the layer's KDA or DSA block on b.normed; `li`/`di`: its index among loaded KDA / DSA layers.
    fn mixer(f: *const Forward, r: Rank, l: weights.Layer, li: usize, di: usize, rows: usize, nch: usize, out: ?u64) !void {
        const c = &r.w.config;
        const b = r.b;
        const st = r.st;
        switch (l.mixer) {
            .kda => |k| {
                const cur = st.cur[li];
                var s = b.kda;
                s.p = if (b.prefill) b.kproj else st.proj + li * st.proj_layer * 2;
                s.saves.state_out = st.recAt(1 - cur, li);
                if (!b.prefill) { // a window's saved rows stay with the layer until its commit
                    const kv = li * window_rows * st.heads * kda.dk;
                    s.saves.k = st.save_k + kv * 4;
                    s.saves.v = st.save_v + kv * 2;
                    s.saves.g = st.save_g + kv * 4;
                    s.saves.b = st.save_b + li * window_rows * st.heads * 4;
                }
                const conv = st.conv + li * 3 * st.conv_c * 2;
                try kda_block.body(f.set(), &f.m.kda, f.stream, k, s, b.normed, c.hidden, rows, conv, st.recAt(cur, li), !b.prefill and f.kda_wide, b.prefill);
                if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
                if (b.prefill) st.cur[li] = 1 - cur; // a prompt chunk keeps every row: the layer commits now
            },
            .mla => |a| {
                try dsa_block.body(.{ .set = f.set(), .ops = f.m.ops() }, f.stream, a, st.caches[di], b.dsa, b.normed, c.hidden, rows, st.pos_dev, st.pos, nch, b.part_rows);
                if (out) |o| try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, o);
            },
        }
    }

    /// forward._ffn: the dense MLP or MoE block on b.normed into this rank's slot.
    fn ffn(f: *const Forward, r: Rank, l: weights.Layer, rows: usize, out: ?u64) !void {
        const c = &r.w.config;
        const b = r.b;
        switch (l.ffn) {
            .dense => |m| {
                try mlp.body(f.mlpKernels(), f.stream, m, b.dense, b.normed, c.hidden, rows, b.prefill);
                if (out) |o| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, 0, rows, b.prefill, o);
            },
            .moe => |m| {
                try moe.body(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, rows, b.prefill);
                if (out) |o| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, 0, rows, b.prefill, o);
            },
        }
    }

    fn mlpKernels(f: *const Forward) mlp.Kernels {
        return .{ .set = f.set(), .linear = .{ .lane = &f.m.lane, .nvfp4_prompt = &f.m.prompt }, .copy_rows = f.m.copy_rows };
    }

    fn moeKernels(f: *const Forward) moe.Kernels {
        return .{ .set = f.set(), .experts = &f.m.experts, .exl3 = &f.m.exl3, .put_slot = f.m.put_slot, .prof = f.prof };
    }

    const Site = enum { mixer, ffn };

    /// forward.out_proj's fill of rows lo..hi at a site of layer `l` into `out` (this rank's rows of a piece).
    fn fillRows(f: *const Forward, r: Rank, l: weights.Layer, site: Site, lo: usize, hi: usize, out: u64) !void {
        const c = &r.w.config;
        const b = r.b;
        switch (site) {
            .mixer => switch (l.mixer) {
                .kda => |k| try kda_block.fill(f.set(), f.stream, k, b.kda, lo, hi, out),
                .mla => |a| try dsa_block.fill(f.set(), f.stream, a, b.dsa, lo, hi, out),
            },
            .ffn => switch (l.ffn) {
                .dense => |m| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, lo, hi, b.prefill, out),
                .moe => |m| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, lo, hi, b.prefill, out),
            },
        }
    }

    /// Piece [lo, hi)'s room in a rank's gathered buffer (pieces never share bytes): [world][hi - lo][dims] fp32.
    fn room(r: Rank, lo: usize) u64 {
        return r.b.gathered + r.w.share.world * lo * r.w.config.hidden * 4;
    }

    /// reduce.shares: [lo, hi) of the rows of a piece of `n` each rank owns: about equal, the first n % world one more.
    fn share(n: usize, world: usize, rank: usize) [2]usize {
        const base = n / world;
        const extra = n % world;
        const lo = rank * base + @min(rank, extra);
        return .{ lo, lo + base + @intFromBool(rank < extra) };
    }

    /// Whether piece [lo, lo + n) splits (TF_GLM_PREFILL_REDUCE=split): at least one row for every rank.
    fn splits(f: *const Forward, n: usize) bool {
        return f.split and n >= f.ranks[0].w.share.world;
    }

    fn copy(f: *const Forward, stream: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), "cuMemcpyDtoDAsync");
    }

    /// overlap.Overlap.partials: each piece filled on the main stream, exchanged on the second (gathered or split).
    fn exchangePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
        const D = f.ranks[0].w.config.hidden;
        const world = f.ranks[0].w.share.world;
        for (cut, 0..) |piece, q| {
            const lo = piece[0];
            const n = piece[1] - lo;
            const split = f.splits(n);
            for (f.ranks) |r| try f.fillRows(r, r.w.layers[j], site, lo, piece[1], if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4);
            try p.filled[q].record(f.stream);
            try p.side.wait(p.filled[q]);
            const bytes = n * D * 4;
            if (!split) switch (f.exchange) {
                .local => for (f.ranks) |to| for (f.ranks) |from| {
                    if (to.w.share.rank == from.w.share.rank) continue;
                    const off = from.w.share.rank * bytes;
                    try f.copy(p.side, room(to, lo) + off, room(from, lo) + off, bytes);
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    try x.lib.check(x.lib.api.ncclAllGather(room(r, lo) + r.w.share.rank * bytes, room(r, lo), n * D, .f32, x.comm, p.side.handle), "ncclAllGather");
                },
            } else switch (f.exchange) {
                .local => for (f.ranks) |to| {
                    const mine = share(n, world, to.w.share.rank);
                    const own = mine[1] - mine[0];
                    for (f.ranks) |from| try f.copy(p.side, room(to, lo) + from.w.share.rank * own * D * 4, from.b.part + (lo + mine[0]) * D * 4, own * D * 4);
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    const me = r.w.share.rank;
                    const mine = share(n, world, me);
                    const own = mine[1] - mine[0];
                    try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                    for (0..world) |peer| if (peer != me) {
                        const theirs = share(n, world, peer);
                        try x.lib.check(x.lib.api.ncclSend(r.b.part + (lo + theirs[0]) * D * 4, (theirs[1] - theirs[0]) * D, .f32, @intCast(peer), x.comm, p.side.handle), "ncclSend");
                        try x.lib.check(x.lib.api.ncclRecv(room(r, lo) + peer * own * D * 4, own * D, .f32, @intCast(peer), x.comm, p.side.handle), "ncclRecv");
                    };
                    try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
                    try f.copy(p.side, room(r, lo) + me * own * D * 4, r.b.part + (lo + mine[0]) * D * 4, own * D * 4);
                },
            }
            try p.gathered[q].record(p.side);
        }
    }

    /// hc_post of rows lo..hi from partials `g` [world][hi - lo][D] (a gathered room or a split one's own rows).
    fn postRows(f: *const Forward, r: Rank, lo: usize, hi: usize, g: u64) !void {
        const c = &r.w.config;
        const b = r.b;
        const S: usize = c.hc;
        const D: usize = c.hidden;
        const x = b.x + lo * S * D * 2;
        try glue.hcPost(f.set(), f.stream, x, x, g, b.post + lo * S * 4, b.comb + lo * S * S * 4, hi - lo, D, S, r.w.share.world);
    }

    /// forward._pre of rows lo..hi.
    fn preRows(f: *const Forward, r: Rank, hc: weights.Hc, norm: u64, lo: usize, hi: usize) !void {
        const c = &r.w.config;
        const b = r.b;
        const S: usize = c.hc;
        const D: usize = c.hidden;
        try glue.hcPre(f.set(), f.stream, b.x + lo * S * D * 2, hc.fn_w, hc.base, hc.scale, norm, b.normed + lo * D * 2, b.xs + lo * (D / 64) * 4, b.post + lo * S * 4, b.comb + lo * S * S * 4, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, hi - lo, D, S);
    }

    /// One piece's glue after a site: hc_post, then the next layer's hc_pre or (after the last layer) the stream mean.
    fn glueRows(f: *const Forward, r: Rank, j: usize, site: Site, lo: usize, hi: usize, g: u64) !void {
        const c = &r.w.config;
        const layers = r.w.layers;
        try f.postRows(r, lo, hi, g);
        if (site == .mixer) return f.preRows(r, layers[j].ffn_hc, layers[j].post_norm, lo, hi);
        if (j + 1 < layers.len) return f.preRows(r, layers[j + 1].attn_hc, layers[j + 1].in_norm, lo, hi);
        try glue.streamMean(f.set(), f.stream, r.b.x + lo * c.hc * c.hidden * 2, r.b.hidden + lo * c.hidden * 2, hi - lo, c.hidden, c.hc);
    }

    /// overlap.Overlap.glue: each piece's glue once its exchange is in, its rows then shared on the second stream.
    fn gluePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
        const c = &f.ranks[0].w.config;
        const D: usize = c.hidden;
        const world = f.ranks[0].w.share.world;
        const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
        var sent: usize = 0;
        for (cut, 0..) |piece, q| {
            const lo = piece[0];
            const n = piece[1] - lo;
            try f.stream.wait(p.gathered[q]);
            if (!f.splits(n)) {
                for (f.ranks) |r| try f.glueRows(r, j, site, lo, piece[1], room(r, lo));
                continue;
            }
            for (f.ranks) |r| {
                const mine = share(n, world, r.w.share.rank);
                try f.glueRows(r, j, site, lo + mine[0], lo + mine[1], room(r, lo));
            }
            try p.glues[q].record(f.stream);
            try p.side.wait(p.glues[q]);
            switch (f.exchange) {
                .local => for (f.ranks) |from| {
                    const mine = share(n, world, from.w.share.rank);
                    const bytes = (mine[1] - mine[0]) * D * 2;
                    for (f.ranks) |to| if (to.w.share.rank != from.w.share.rank) {
                        const at = (lo + mine[0]) * D * 2;
                        try f.copy(p.side, (if (last) to.b.hidden else to.b.normed) + at, (if (last) from.b.hidden else from.b.normed) + at, bytes);
                    };
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    const me = r.w.share.rank;
                    const rows_of = if (last) r.b.hidden else r.b.normed;
                    const mine = share(n, world, me);
                    try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                    for (0..world) |peer| if (peer != me) {
                        const theirs = share(n, world, peer);
                        try x.lib.check(x.lib.api.ncclSend(rows_of + (lo + mine[0]) * D * 2, (mine[1] - mine[0]) * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclSend");
                        try x.lib.check(x.lib.api.ncclRecv(rows_of + (lo + theirs[0]) * D * 2, (theirs[1] - theirs[0]) * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclRecv");
                    };
                    try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
                },
            }
            try p.shared[q].record(p.side);
            sent |= @as(usize, 1) << @intCast(q);
        }
        for (0..cut.len) |q| if (sent >> @intCast(q) & 1 == 1) try f.stream.wait(p.shared[q]);
    }

    /// forward.pieced_layers: a prompt chunk's layers with exchanges in row pieces; same kernels and bits as unpieced.
    fn pieced(f: *const Forward, p: *const Pieces, R: usize, cut: []const [2]usize) !void {
        const layers = f.ranks[0].w.layers;
        for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
        var li: usize = 0;
        var di: usize = 0;
        for (layers, 0..) |_, j| {
            for (f.ranks) |r| try f.mixer(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null);
            try f.exchangePieces(p, j, .mixer, cut);
            try f.gluePieces(p, j, .mixer, cut);
            for (f.ranks) |r| try f.ffn(r, r.w.layers[j], R, null);
            try f.exchangePieces(p, j, .ffn, cut);
            try f.gluePieces(p, j, .ffn, cut);
            switch (layers[j].mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
        }
    }

    /// forward.forward of `ids`: logits in b.logits; a prompt chunk's KDA layers committed, a window's at `commit`.
    pub fn run(f: *const Forward, ids: []const i32) !void {
        const R = ids.len;
        for (f.ranks) |r| {
            if (R == 0 or R > r.b.rows) return error.BadWindow;
            if (r.segs.len == 0) {
                if (r.st.pos + R > r.st.capacity) return error.PastCapacity;
                r.st.pending = R;
                r.st.prompt = r.b.prefill;
            } else try shared.begin(r, R);
            try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(r.b.ids, ids.ptr, R * 4, f.stream.handle), "cuMemcpyHtoDAsync");
        }
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
        if (f.images) |im| for (f.ranks) |r| try im.v.splice(f.stream, im.slot, r.st.pos, r.st.pos + R, r.b.x, c.hc);
        if (f.prof) |p| p.mark(f.stream, .embed);
        var cut: [max_pieces][2]usize = undefined;
        const pieces = if (f.pieces) |p| (if (f.ranks[0].b.prefill) ranges(R, p.count, &cut) else 0) else 0;
        // overlap.Overlap.begin: pieced with two pieces or more, and split takes a chunk of one piece too
        if (pieces > 1 or (pieces == 1 and f.splits(R))) try f.pieced(f.pieces.?, R, cut[0..pieces]) else try f.whole(R);
        try f.finish(R);
        if (f.prof) |p| p.mark(f.stream, .head);
    }

    /// The unpieced loop over the layers (and the stream mean): each site's partials whole, then the glue.
    fn whole(f: *const Forward, R: usize) !void {
        const c = &f.ranks[0].w.config;
        var li: usize = 0;
        var di: usize = 0;
        for (f.ranks[0].w.layers, 0..) |_, j| {
            for (f.ranks) |r| {
                const l = r.w.layers[j];
                try f.pre(r, l.attn_hc, l.in_norm, R);
                if (r.segs.len > 0) try shared.mixer(f, r, l, li, di, R, slot(r, R)) else try f.mixer(r, l, li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), slot(r, R));
            }
            if (f.prof) |p| p.mark(f.stream, if (f.ranks[0].w.layers[j].mixer == .kda) .kda else .mla);
            try f.gather(R);
            if (f.prof) |p| p.mark(f.stream, .gather_a);
            for (f.ranks) |r| {
                const l = r.w.layers[j];
                try f.post(r, R);
                try f.pre(r, l.ffn_hc, l.post_norm, R);
                try f.ffn(r, l, R, slot(r, R));
            }
            if (f.prof) |p| p.mark(f.stream, if (f.ranks[0].w.layers[j].ffn == .dense) .dense else .shared); // MoE: its fill
            try f.gather(R);
            for (f.ranks) |r| try f.post(r, R);
            if (f.prof) |p| p.mark(f.stream, .gather_b);
            switch (f.ranks[0].w.layers[j].mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
        }
        for (f.ranks) |r| try glue.streamMean(f.set(), f.stream, r.b.x, r.b.hidden, R, c.hidden, c.hc);
    }

    /// The final norm and this rank's rows of the head (a prompt buffer's last row only).
    fn finish(f: *const Forward, R: usize) !void {
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| {
            const b = r.b;
            try glue.rmsnorm(f.set(), f.stream, b.hidden, c.hidden, r.w.norm, b.fnormed, c.hidden, b.fxs, c.eps, R, c.hidden);
            const head_rows: usize = if (b.prefill) 1 else R;
            const first = if (b.prefill) R - 1 else 0;
            try b16.matmul(f.set(), f.stream, b.fnormed + first * c.hidden * 2, c.hidden, r.w.head, b.logits, b.kda.mm_part, head_rows, r.w.share.vocab.len(), c.hidden, false);
        }
    }

    /// mtp.mtp_forward: caches from mtp_len, stream in b.mx, a window's draft logits in b.logits (prompt: no head).
    pub fn mtp(f: *const Forward, ids: []const i32, hidden: []const u64) !void {
        const n = ids.len;
        const c = &f.ranks[0].w.config;
        const D: usize = c.hidden;
        for (f.ranks, hidden) |r, h| {
            if (r.w.mtp == null) return error.NoMtpHead;
            if (n == 0 or n > r.b.rows) return error.BadWindow;
            if (r.st.mtp_len + n > r.st.capacity) return error.PastCapacity;
            try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(r.b.ids, ids.ptr, n * 4, f.stream.handle), "cuMemcpyHtoDAsync");
            if (h != r.b.hin) try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(r.b.hin, h, n * D * 2, f.stream.handle), "cuMemcpyDtoDAsync");
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.embedB16(f.set(), f.stream, b.ids, r.w.embed, b.me, n, D, 1);
            // the head reads each row's next token: an image's rows one place on
            if (f.images) |im| try im.v.splice(f.stream, im.slot, r.st.mtp_len + 1, r.st.mtp_len + 1 + n, b.me, 1);
            if (r.st.mtp_len == 0) try f.d.check(f.d.api.cuMemsetD8Async(b.me, 0, D * 2, f.stream.handle), "cuMemsetD8Async"); // position 0's embedding is zeroed
            try glue.rmsnorm(f.set(), f.stream, b.me, D, m.enorm, b.mcat, 2 * D, null, c.eps, n, D);
            try glue.rmsnorm(f.set(), f.stream, b.hin, D, m.hnorm, b.mcat + D * 2, 2 * D, null, c.eps, n, D);
            try b16.matmul(f.set(), f.stream, b.mcat, 2 * D, m.eh, b.mx, b.kda.mm_part, n, D, 2 * D, false);
            try glue.rmsnorm(f.set(), f.stream, b.mx, D, m.layer.in_norm, b.normed, D, b.xs, c.eps, n, D);
            const nch = @min(latent.chunksFor(r.st.mtp_len + n), b.nch);
            try dsa_block.block(.{ .set = f.set(), .ops = f.m.ops() }, f.stream, m.layer.mixer.mla, r.st.mtp_caches, b.dsa, b.normed, D, n, r.st.mtp_pos_dev, r.st.mtp_len, nch, b.part_rows, slot(r, n));
        }
        try f.gather(n);
        for (f.ranks) |r| {
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.residualAdd(f.set(), f.stream, b.mx, b.mx, b.gathered, n, D, r.w.share.world);
            try glue.rmsnorm(f.set(), f.stream, b.mx, D, m.layer.post_norm, b.normed, D, b.xs, c.eps, n, D);
            try f.ffn(r, m.layer, n, slot(r, n));
        }
        try f.gather(n);
        for (f.ranks) |r| {
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.residualAdd(f.set(), f.stream, b.mx, b.mx, b.gathered, n, D, r.w.share.world);
            try glue.rmsnorm(f.set(), f.stream, b.mx + (n - 1) * D * 2, D, m.norm, b.fnormed, D, b.fxs, c.eps, 1, D);
            if (!b.prefill) try q4.matmul(&f.m.q4, f.stream, b.fnormed, D, b.fxs, m.draft, b.logits, 1);
        }
    }

    /// forward.commit: keep the first `keep` rows; window KDA layers take new states (replayed if fewer), conv shifted.
    pub fn commit(f: *const Forward, keep: usize) !void {
        for (f.ranks) |r| {
            const st = r.st;
            if (keep == 0 or keep > st.pending or (st.prompt and keep != st.pending)) return error.BadCommit;
            if (!st.prompt and st.kda_layers > 0) {
                const cur = st.cur[0];
                for (st.cur) |x| if (x != cur) return error.KdaLayersOutOfStep;
                if (keep < st.pending) {
                    const h = st.heads;
                    var a: cuda.Args = .{};
                    a.add(@as(c_int, @intCast(h)));
                    a.add(st.recAt(cur, 0));
                    a.add(@as(usize, st.rec_layer / 4));
                    for ([_]u64{ st.save_k, st.save_v, st.save_g, st.save_b }) |x| a.add(x);
                    a.add(@as(usize, window_rows * h * kda.dk));
                    a.add(@as(usize, window_rows * h));
                    a.add(@as(c_int, @intCast(keep)));
                    a.add(st.recAt(1 - cur, 0));
                    try cuda.launch.launch(f.m.kda.replay_layers, .{ .grid = .{ .x = @intCast(st.kda_layers * h) }, .block = .{ .x = 1024 } }, f.stream, &a);
                }
                for (st.cur) |*x| x.* = 1 - x.*;
                if (st.kda_layers > 0) try glue.convShift(f.set(), f.stream, st.conv, st.proj, keep, st.proj_layer, st.proj_row, st.kda_layers, st.conv_c);
            }
            st.pending = 0;
            try st.setPos(f.stream, st.pos + keep);
        }
    }
};
