//! GLM-5.3-Flash's per-rank CUDA memory: forward Buffers (shared by sequences) and a sequence's State, an arena each.

const std = @import("std");
const cuda = @import("cuda");
const exl3 = @import("cuda_exl3.zig");
const weights = @import("cuda_weights.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const experts = @import("cuda_experts.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const moe = @import("cuda_moe.zig");
const glue = @import("cuda_glue.zig");
const mlp = @import("cuda_mlp.zig");
const kda_block = @import("cuda_kda_block.zig");

/// A prompt chunk's rows (decode.PREFILL_ROWS).
pub const prefill_rows = 2048;
/// A decode window's rows (Engine's max_rows).
pub const window_rows = 8;
/// The context the buffers' dense attention partials reach (Buffers' min(capacity, 2560)).
pub const dense_reach = 2560;

/// Sub-allocations of one device allocation, 256-byte aligned like the driver's own.
const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

/// Allocates `sizes` (bytes) in one zeroed device buffer and hands out each field's address in order.
fn arena(d: *const cuda.Driver, sizes: []const usize, out: []u64) !Arena {
    var total: usize = 0;
    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, @max(total, 256)) };
    errdefer a.buf.free();
    try a.buf.fill8(0, null);
    for (sizes, out) |n, *p| p.* = a.take(n);
    return a;
}

/// A rank's B16 matmul shapes ([n, k]) whose split-K partials the buffers hold.
fn b16Shapes(w: *const weights.Weights) [12][2]usize {
    const c = &w.config;
    const s = &w.share;
    const n = s.kda.len() * c.kda_dim;
    const kw = 3 * n + 2 * c.kda_dim + s.kda.len();
    return .{
        .{ kw, c.hidden },                   .{ n, c.kda_dim },                    .{ c.hidden, n },
        .{ c.q_lora + c.kv_lora, c.hidden }, .{ s.mla.len() * c.nope, c.q_lora },  .{ c.i_dim + c.i_heads, c.hidden },
        .{ c.i_heads * c.i_dim, c.q_lora },  .{ c.hidden, s.mla.len() * c.v_dim }, .{ 2 * s.shared.len(), c.hidden },
        .{ c.hidden, 2 * c.hidden }, // the MTP head's eh_proj
        if (c.exl3) .{ 2 * s.dense.len(), c.hidden } else .{ c.hidden, 2 * c.hidden }, // bf16 dense layers: [gate | up]
        if (c.exl3) .{ c.hidden, s.dense.len() } else .{ c.hidden, 2 * c.hidden }, // and down
    };
}

/// Scratch for windows of up to `rows` rows (the Python family's Buffers): a prompt buffer (`prefill`) or a decode's.
pub const Buffers = struct {
    arena: Arena,
    rows: usize,
    prefill: bool,
    part_rows: usize, // rows one dense attention call takes
    nch: usize, // the dense pass's most 512-key chunks
    ids: u64,
    x: u64,
    normed: u64,
    xs: u64,
    post: u64,
    comb: u64,
    hcpart: u64,
    gathered: u64, // fp32 [world][rows][dims]: every rank's partial in rank order
    kproj: u64, // a prompt chunk's or a shared window's KDA projection rows (a window's stay in the state, for its commit)
    kda: kda_block.Scratch,
    dsa: dsa_block.Scratch,
    dense: mlp.Scratch,
    moe: moe.Scratch,
    hidden: u64,
    fnormed: u64,
    fxs: u64,
    logits: u64, // bf16 [head rows][this rank's vocabulary]
    // the MTP head (every Buffers holds them, as the Python family's)
    hin: u64, // bf16 [rows][dims]: the hidden rows it reads
    me: u64, // bf16 [rows][dims]: the next tokens' embedding
    mcat: u64, // bf16 [rows][2 dims]: [enorm(me) | hnorm(hin)]
    mx: u64, // bf16 [rows][dims]: its stream
    part: u64, // fp32 [rows][dims]: this rank's partial rows a split exchange sends from

    pub fn init(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, prefill: bool, capacity: usize) !Buffers {
        return make(d, w, rows, prefill, capacity, rows);
    }

    /// A shared window's scratch: `rows` rows of several streams, each one's attention and KDA chain on `window_rows`.
    pub fn initShared(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, capacity: usize) !Buffers {
        return make(d, w, rows, false, capacity, window_rows);
    }

    /// `seg`: the most rows one stream's attention takes (a decode window's rows, or a shared window's segment).
    fn make(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, prefill: bool, capacity: usize, seg: usize) !Buffers {
        const c = &w.config;
        const s = &w.share;
        const D: usize = c.hidden;
        const S: usize = c.hc;
        const LL = s.kda.len();
        const HL = s.mla.len();
        const n = LL * c.kda_dim;
        const kw = 3 * n + 2 * c.kda_dim + LL;
        const mw = s.dense.len();
        const ew = s.moe.len();
        const sh = s.shared.len();
        const slots: usize = c.topk + 1;
        const pairs = rows * slots;
        const count: usize = c.experts;
        const lw: usize = c.kv_lora;
        const part_rows = if (prefill) @min(rows, dsa_block.prompt_att_rows) else seg;
        const nch = latent.chunksFor(@min(capacity, dense_reach) + seg);
        const partial_rows = @max(nch * part_rows, latent.chunksFor(sparse.width) * seg); // dense calls, or one sparse
        const np_max = capacity / sparse.pool;
        var mm: usize = 4;
        // windows under rows_from rows keep their K slices apart (a prompt's tail chunk on the prefill buffers too)
        for (b16Shapes(w)) |sh_| mm = @max(mm, b16.partBytes(@min(rows, b16.rows_from - 1), sh_[0], sh_[1]));
        const x3: [4]usize = if (c.exl3) exl3.scratchBytes(rows, D, ew, prefill, w.exl3_shx) else @splat(4);
        const hist_len = if (pairs > cuda.grouped.small) (pairs + cuda.grouped.small - 1) / cuda.grouped.small * (count + 1) else 1;
        const sizes = [_]usize{
            rows * 4,     rows * S * D * 2, rows * D * 2,           rows * (D / 64) * 4,
            rows * S * 4, rows * S * S * 4, glue.hcPartBytes(rows), s.world * rows * D * 4,
            if (prefill or seg < rows) rows * kw * 2 else 4, rows * n * 2, rows * n * 2, // kproj, ka, kg
            rows * n * 2, rows * n * 4, rows * n * 2, rows * n * 4, // saves out, k, v, g
            rows * LL * 4,                    rows * n * 4,               rows * n * 2, // saves b, wide q, y
            rows * (c.q_lora + lw) * 2,       rows * c.q_lora * 2,        rows * (c.q_lora / 64) * 4,
            rows * lw * 2,                    rows * (lw / 64) * 4,       rows * HL * c.nope * 2,
            rows * (c.i_dim + c.i_heads) * 2, rows * c.i_dim * 4,         glue.router_ks * rows * @max(count, c.i_dim) * 4,
            rows * HL * lw * 2,               rows * HL * lw * 2,         rows * HL * c.v_dim * 2,
            rows * c.i_heads * c.i_dim * 2,   partial_rows * HL * lw * 4, partial_rows * HL * 4,
            partial_rows * HL * 4,
            4, // dummy
            @min(seg, sparse.select_rows) * np_max * 4,
            rows * sparse.select_list * 4,
            rows * sparse.select_list * 4,
            rows * sparse.topk_pools * 8,
            4,
            rows * sparse.width * 4,
            rows * 4,
            mm,
            4, // a prompt block's position
            rows * mw * 2,
            rows * 2 * mw * 2,
            rows * mw * 2,
            rows * (mw / 64) * 4,
            rows * count * 4, pairs * 4, pairs * 4, pairs * 4, // mlog, pick, wts, members
            experts.maxItems(pairs, count + 1, experts.tile) * 3 * 4, // items (the decode tile's count bounds the prompt's)
            8,              @max(1, pairs) * 4, hist_len * 4, // counts, rank, hist
            pairs * ew * 2, pairs * D * 4,      rows * 2 * sh * 2,
            rows * sh * 2,
            rows * (sh / 64) * 4, rows * D * 4, // sxs, sy
            rows * D * 2,         rows * D * 2,
            rows * (D / 64) * 4,  (if (prefill) 1 else rows) * s.vocab.len() * 2,
            rows * D * 2, rows * D * 2, rows * 2 * D * 2, rows * D * 2, // hin, me, mcat, mx
            rows * D * 4, // part: this rank's fp32 partial rows a split exchange sends from
            x3[0], x3[1], x3[2], x3[3], // EXL3 experts: 8-slot picks, rotated inputs, a decode window's partials
        };
        var p: [sizes.len]u64 = undefined;
        const a = try arena(d, &sizes, &p);
        var out: Buffers = .{
            .arena = a,
            .rows = rows,
            .prefill = prefill,
            .part_rows = part_rows,
            .nch = nch,
            .ids = p[0],
            .x = p[1],
            .normed = p[2],
            .xs = p[3],
            .post = p[4],
            .comb = p[5],
            .hcpart = p[6],
            .gathered = p[7],
            .kproj = p[8],
            .kda = .{ .p = 0, .ka = p[9], .kg = p[10], .saves = .{ .out = p[11], .state_out = 0, .k = p[12], .v = p[13], .g = p[14], .b = p[15] }, .wide = .{ .q = p[16], .y = p[17] }, .mm_part = p[42] },
            .dsa = .{
                .dp = p[18],
                .qr = p[19],
                .xs_qr = p[20],
                .lat = p[21],
                .xs_lat = p[22],
                .q = p[23],
                .ikr = p[24],
                .igr = p[25],
                .router_part = p[26],
                .qa = p[27],
                .ol = p[28],
                .vn = p[29],
                .qi = p[30],
                .partials = .{ .po = p[31], .pm = p[32], .pl = p[33] },
                .dummy = p[34],
                .select = .{ .scores = p[35], .ls = p[36], .li = p[37], .pools = p[38], .at = p[39] },
                .tokens = p[40],
                .counts = p[41],
                .mm_part = p[42],
                .at = p[43],
            },
            .dense = .{ .tmp = p[44], .gu = p[45], .act = p[46], .xs_act = p[47], .mm_part = p[42] },
            .moe = .{ .mlog = p[48], .router_part = p[26], .pick = p[49], .wts = p[50], .plan = .{ .members = p[51], .items = p[52], .counts = p[53], .rank = p[54], .hist = p[55] }, .eact = p[56], .ey = p[57], .sgu = p[58], .sact = p[59], .sxs = p[60], .sy = p[61], .mm_part = p[42] },
            .hidden = p[62],
            .fnormed = p[63],
            .fxs = p[64],
            .logits = p[65],
            .hin = p[66],
            .me = p[67],
            .mcat = p[68],
            .mx = p[69],
            .part = p[70],
        };
        out.moe.x3 = .{ .picks = p[71], .xg = p[72], .xu = p[73], .z = p[74] };
        return out;
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.buf.free();
        b.* = undefined;
    }
};

/// A sequence's committed caches on a rank (the Python family's State): KDA conv and states, saved rows, DSA caches.
pub const State = struct {
    arena: Arena,
    gpa: std.mem.Allocator,
    capacity: usize,
    pos: usize = 0,
    pending: usize = 0, // rows of the last forward, until its commit
    prompt: bool = false, // whether that forward was a prompt chunk
    pos_dev: u64, // i32
    conv: u64, // bf16 [kda layers][3][3 heads 128]
    rec: u64, // fp32 [2][kda layers][heads][128][128]
    proj: u64, // bf16 [kda layers][window rows][kda width]
    save_k: u64, // fp32 [kda layers][window rows][heads][128]
    save_v: u64, // bf16, as save_k
    save_g: u64, // fp32, as save_k
    save_b: u64, // fp32 [kda layers][window rows][heads]
    cur: []u1,
    kda_layers: usize,
    heads: usize, // this rank's KDA heads
    caches: []dsa_block.Caches,
    mtp_len: usize = 0, // rows in the MTP layer's caches
    mtp_pos_dev: u64, // i32
    mtp_caches: dsa_block.Caches, // the MTP layer's latent rows, index keys, gates and pooled keys (the Python family's index[-1])
    conv_c: usize,
    rec_layer: usize,
    proj_layer: usize,
    proj_row: usize,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, capacity: usize) !State {
        const c = &w.config;
        const s = &w.share;
        var nk: usize = 0;
        var nd: usize = 0;
        for (w.layers) |l| switch (l.mixer) {
            .kda => nk += 1,
            .mla => nd += 1,
        };
        const n = s.kda.len() * c.kda_dim;
        const kw = 3 * n + 2 * c.kda_dim + s.kda.len();
        const pk_rows = capacity / sparse.pool + 2;
        var sizes: std.ArrayList(usize) = .empty;
        defer sizes.deinit(gpa);
        try sizes.appendSlice(gpa, &.{ 4, nk * 3 * 3 * n * 2, 2 * nk * s.kda.len() * kda.dk * kda.dv * 4, nk * window_rows * kw * 2 });
        try sizes.appendSlice(gpa, &.{ nk * window_rows * n * 4, nk * window_rows * n * 2, nk * window_rows * n * 4, nk * window_rows * s.kda.len() * 4 });
        for (0..nd + 1) |_| try sizes.appendSlice(gpa, &.{ capacity * c.kv_lora * 2, capacity * c.i_dim * 2, capacity * c.i_dim * 2, pk_rows * c.i_dim * 2 });
        try sizes.append(gpa, 4); // the MTP head's position
        const p = try gpa.alloc(u64, sizes.items.len);
        defer gpa.free(p);
        var a = try arena(d, sizes.items, p);
        errdefer a.buf.free();
        const cur = try gpa.alloc(u1, nk);
        errdefer gpa.free(cur);
        @memset(cur, 0);
        const caches = try gpa.alloc(dsa_block.Caches, nd);
        for (caches, 0..) |*x, j| x.* = .{ .lc = p[8 + 4 * j], .ik = p[9 + 4 * j], .ig = p[10 + 4 * j], .pk = p[11 + 4 * j], .pk_rows = pk_rows };
        return .{ .arena = a, .gpa = gpa, .capacity = capacity, .pos_dev = p[0], .conv = p[1], .rec = p[2], .proj = p[3], .save_k = p[4], .save_v = p[5], .save_g = p[6], .save_b = p[7], .cur = cur, .heads = s.kda.len(), .mtp_caches = .{ .lc = p[8 + 4 * nd], .ik = p[9 + 4 * nd], .ig = p[10 + 4 * nd], .pk = p[11 + 4 * nd], .pk_rows = pk_rows }, .mtp_pos_dev = p[12 + 4 * nd], .kda_layers = nk, .caches = caches, .conv_c = 3 * n, .rec_layer = s.kda.len() * kda.dk * kda.dv * 4, .proj_layer = window_rows * kw, .proj_row = kw };
    }

    pub fn deinit(st: *State) void {
        st.arena.buf.free();
        st.gpa.free(st.cur);
        st.gpa.free(st.caches);
        st.* = undefined;
    }

    /// State.reset: a fresh sequence (zero conv windows and states, position 0); the caches past it are never read.
    pub fn reset(st: *State, stream: cuda.Stream) !void {
        const d = st.arena.buf.d;
        try d.check(d.api.cuMemsetD8Async(st.conv, 0, st.kda_layers * 3 * st.conv_c * 2, stream.handle), "cuMemsetD8Async");
        try d.check(d.api.cuMemsetD8Async(st.rec, 0, 2 * st.kda_layers * st.rec_layer, stream.handle), "cuMemsetD8Async");
        @memset(st.cur, 0);
        st.pending = 0;
        try st.setPos(stream, 0);
        try st.setMtpLen(stream, 0);
    }

    /// Host side of a state just copied back (a kept prompt state): position, MTP rows, KDA parity, nothing pending.
    pub fn restoreHost(st: *State, stream: cuda.Stream, pos: usize, mtp_len: usize, cur: []const u1) !void {
        @memcpy(st.cur, cur);
        st.pending = 0;
        st.prompt = false;
        try st.setPos(stream, pos);
        try st.setMtpLen(stream, mtp_len);
    }

    /// State.set_mtp_len: rows in the MTP layer's caches (the next MTP step writes from here).
    pub fn setMtpLen(st: *State, stream: cuda.Stream, n: usize) !void {
        st.mtp_len = n;
        const v: i32 = @intCast(n);
        try st.arena.buf.d.check(st.arena.buf.d.api.cuMemsetD32Async(st.mtp_pos_dev, @bitCast(v), 1, stream.handle), "cuMemsetD32Async");
    }

    pub fn setPos(st: *State, stream: cuda.Stream, pos: usize) !void {
        st.pos = pos;
        const v: i32 = @intCast(pos);
        try st.arena.buf.d.check(st.arena.buf.d.api.cuMemsetD32Async(st.pos_dev, @bitCast(v), 1, stream.handle), "cuMemsetD32Async");
    }

    pub fn recAt(st: *const State, copy: u1, li: usize) u64 {
        return st.rec + (@as(usize, copy) * st.kda_layers + li) * st.rec_layer;
    }
};
