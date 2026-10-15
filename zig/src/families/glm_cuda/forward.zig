//! GLM-5.3-Flash on CUDA, one rank of two: each row's bits its own at any mix of streams; both ranks call alike.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("api.zig");
const kern = @import("kern.zig");
const sq = @import("seq.zig");
const blocks = @import("blocks.zig");

pub const Seq = sq.Seq;
pub const Media = sq.Media;
const K = kern.K;
const D = kern.D;
const WIDE = kern.WIDE;
const HL = kern.HL;
const LL = kern.LL;

pub const MAX_STREAMS: usize = 16;
pub const PREFILL_ROWS_MAX: i64 = 4096; // a prompt chunk at most (buffers sized once take the served value)
/// Rows a prompt chunk takes: TF_GLM_PREFILL_ROWS (2048; 4096 halves per-row expert decode, ~2 GiB more). Read once.
pub fn prefillRows() i64 {
    const v = std.c.getenv("TF_GLM_PREFILL_ROWS") orelse return 2048;
    const n = std.fmt.parseInt(i64, std.mem.span(v), 10) catch return 2048;
    return @max(256, @min(PREFILL_ROWS_MAX, @divFloor(n, 256) * 256));
}
pub const MAX_ROWS: i64 = sq.MAX_ROWS;
pub const STEP_ROWS: i64 = @as(i64, MAX_STREAMS) * 8; // a batched step's rows at most (the lane core keeps a step within them)
pub const HEAD_N: i64 = 77440; // a rank's vocabulary shard
pub const IDX_KW: i64 = kern.IDX_DIM + kern.IDX_HEADS; // the indexer's key rows and head weights
pub const MOE_W: i64 = 1024; // a rank's expert width
pub const DENSE_W: i64 = 6144; // a rank's dense MLP width

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

const Carve = struct {
    at: u64 = 0,
    fn take(self: *Carve, n: i64) u64 {
        const p = self.at;
        self.at += std.mem.alignForward(u64, @intCast(@max(n, 1)), 512);
        return p;
    }
};

/// One stream's window in a batched step, laid out as glm_kda_chain_seg reads it.
pub const KdaSeg = extern struct { first: i32, rows: i32, rec_in: u64, rec_out: u64, conv: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, proj: u64 };

/// Scratch for steps of up to ``rows`` rows: a batched decode step's, the MTP head's, a prompt chunk's.
pub const Buffers = struct {
    rows: i64,
    prefill: bool,
    np_max: i64, // the indexer's pool count at the capacity (the scores' columns at most)
    mem: cuda.DeviceBuffer = undefined,
    ids: u64 = 0,
    x: u64 = 0,
    normed: u64 = 0,
    xs: u64 = 0,
    post: u64 = 0,
    comb: u64 = 0,
    hcpart: u64 = 0,
    ka: u64 = 0,
    kg: u64 = 0,
    kproj: u64 = 0,
    kout: u64 = 0,
    ks: kern.KdaScratch = undefined,
    q_tmp: u64 = 0,
    y_tmp: u64 = 0,
    dp: u64 = 0,
    qr: u64 = 0,
    xs_qr: u64 = 0,
    lat: u64 = 0,
    xs_lat: u64 = 0,
    q: u64 = 0,
    vn: u64 = 0,
    ikr: u64 = 0,
    igr: u64 = 0,
    qi: u64 = 0,
    qa: u64 = 0,
    ol: u64 = 0,
    po: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    scores: u64 = 0,
    pools: u64 = 0,
    tokens: u64 = 0,
    counts: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    xs_act: u64 = 0,
    mlog: u64 = 0,
    pick: u64 = 0,
    wts: u64 = 0,
    rpick: u64 = 0,
    eact: u64 = 0,
    ey: u64 = 0,
    rey: u64 = 0,
    plan: kern.Plan = undefined,
    x3: kern.Exl3Scratch = .{ .xg = 0, .xu = 0, .xd = 0, .z = 0 }, // the EXL3 experts' scratch (exl3 checkpoints)
    sgu: u64 = 0,
    sact: u64 = 0,
    sxs: u64 = 0,
    sy: u64 = 0,
    part: u64 = 0,
    gath: u64 = 0,
    sk: u64 = 0,
    hidden: u64 = 0,
    fnormed: u64 = 0,
    fxs: u64 = 0,
    hrows: u64 = 0,
    logits: u64 = 0,
    me: u64 = 0,
    mcat: u64 = 0,
    mx: u64 = 0,
    hin: u64 = 0,
    cand: u64 = 0,
    cgath: u64 = 0,
    invt: u64 = 0,
    tk: u64 = 0,
    gtk: u64 = 0,
    /// The step's tables: positions [rows] i32, then lc, ik, ig, pk [12][rows] u64, the KDA segments, the head rows.
    tab: u64 = 0,
    tab_bytes: u64 = 0,
    segs: u64 = 0,
    last: u64 = 0,

    pub fn rowsTab(b: *const Buffers) kern.Rows {
        const r: u64 = @intCast(b.rows);
        const t = b.tab;
        return .{ .pos = t, .lc = t + r * 4, .ik = t + r * 4 + 12 * r * 8, .ig = t + r * 4 + 24 * r * 8, .pk = t + r * 4 + 36 * r * 8, .cap = b.rows };
    }

    pub fn ey_f32(b: *const Buffers) bool {
        return !b.prefill;
    }

    /// Rows one attention call takes (its chunk partials' scratch).
    pub fn attRows(b: *const Buffers) i64 {
        return @min(b.rows, kern.ATT_ROWS);
    }

    fn init(d: *const cuda.Driver, rows: i64, is_prefill: bool, np_max: i64, exl3: bool, shx: bool) !Buffers {
        var b: Buffers = .{ .rows = rows, .prefill = is_prefill, .np_max = np_max };
        var c: Carve = .{};
        const pairs = rows * kern.TOPK;
        const sel_rows = @min(rows, kern.SELECT_ROWS);
        const att = b.attRows();
        const nch = @max(kern.DENSE_CHUNKS, kern.SPARSE_CHUNKS);
        const head_rows: i64 = if (is_prefill) 1 else rows;
        const ey_elt: i64 = if (is_prefill and !exl3) 2 else 4;
        const sk_cols: i64 = 16384; // splitK keeps sk * n at or below this for every linear of the model
        b.tab_bytes = @intCast(rows * 4 + 48 * rows * 8 + @as(i64, MAX_STREAMS) * @sizeOf(KdaSeg) + @as(i64, MAX_STREAMS) * 4);
        inline for (.{ .{ "ids", rows * 4 }, .{ "x", rows * WIDE * 2 }, .{ "normed", rows * D * 2 }, .{ "xs", rows * @divExact(D, 64) * 4 }, .{ "post", rows * kern.S * 4 }, .{ "comb", rows * kern.S * kern.S * 4 }, .{ "hcpart", rows * 16 * 25 * 4 }, .{ "ka", rows * LL * 128 * 2 }, .{ "kg", rows * LL * 128 * 2 }, .{ "kproj", rows * sq.PROJ_W * 2 }, .{ "kout", rows * LL * 128 * 2 }, .{ "dp", rows * (kern.Q_LORA + kern.KV_LORA) * 2 }, .{ "qr", rows * kern.Q_LORA * 2 }, .{ "xs_qr", rows * @divExact(kern.Q_LORA, 64) * 4 }, .{ "lat", rows * kern.KV_LORA * 2 }, .{ "xs_lat", rows * @divExact(kern.KV_LORA, 64) * 4 }, .{ "q", rows * HL * kern.QK * 2 }, .{ "vn", rows * HL * kern.DV * 2 }, .{ "ikr", rows * IDX_KW * 2 }, .{ "igr", rows * kern.IDX_DIM * 4 }, .{ "qi", rows * kern.IDX_HEADS * kern.IDX_DIM * 2 }, .{ "qa", rows * HL * kern.KV_LORA * 2 }, .{ "ol", rows * HL * kern.KV_LORA * 2 }, .{ "po", nch * att * HL * kern.KV_LORA * 4 }, .{ "pm", nch * att * HL * 4 }, .{ "pl", nch * att * HL * 4 }, .{ "scores", sel_rows * @max(np_max, 1024) * 4 }, .{ "pools", rows * kern.TOPK_POOLS * 8 }, .{ "tokens", rows * kern.SEL_W * 4 }, .{ "counts", rows * 4 }, .{ "gu", rows * 2 * DENSE_W * 2 }, .{ "act", rows * DENSE_W * 2 }, .{ "xs_act", rows * @divExact(DENSE_W, 64) * 4 }, .{ "mlog", rows * kern.EXPERTS * 4 }, .{ "pick", rows * kern.SLOTS * 4 }, .{ "wts", rows * kern.SLOTS * 4 }, .{ "rpick", rows * kern.TOPK * 4 }, .{ "eact", rows * kern.SLOTS * MOE_W * 2 }, .{ "ey", rows * kern.SLOTS * D * ey_elt }, .{ "rey", rows * kern.TOPK * D * ey_elt }, .{ "sgu", rows * 2 * MOE_W * 2 }, .{ "sact", rows * MOE_W * 2 }, .{ "sxs", rows * @divExact(MOE_W, 64) * 4 }, .{ "sy", rows * D * 4 }, .{ "part", rows * D * 4 }, .{ "gath", 2 * rows * D * 4 }, .{ "sk", rows * sk_cols * 4 }, .{ "hidden", rows * D * 2 }, .{ "fnormed", rows * D * 2 }, .{ "fxs", rows * @divExact(D, 64) * 4 }, .{ "hrows", @as(i64, MAX_STREAMS) * D * 2 }, .{ "logits", head_rows * HEAD_N * 2 }, .{ "me", rows * D * 2 }, .{ "mcat", rows * 2 * D * 2 }, .{ "mx", rows * D * 2 }, .{ "hin", rows * D * 2 }, .{ "cand", rows * 4 * 4 }, .{ "cgath", 2 * rows * 4 * 4 }, .{ "invt", head_rows * 4 }, .{ "tk", head_rows * @as(i64, @intCast(@import("draw.zig").W)) * 4 }, .{ "gtk", 2 * head_rows * @as(i64, @intCast(@import("draw.zig").W)) * 4 }, .{ "tab", @as(i64, @intCast(b.tab_bytes)) } }) |f|
            @field(b, f[0]) = c.take(f[1]);
        const items = kern.maxItems(pairs, kern.EXPERTS, 16);
        const wide = pairs > 1024;
        b.plan = .{ .members = c.take(pairs * 4), .items = c.take(items * 3 * 4), .counts = c.take(2 * 4), .rank = c.take(if (wide) pairs * 4 else 4), .hist = c.take(if (wide) cdiv(pairs, 1024) * kern.EXPERTS * 4 else 4), .tile = if (is_prefill) kern.promptTile() else 16 };
        if (exl3) {
            const P = rows * kern.TOPK;
            const sk: i64 = if (is_prefill) (if (kern.promptOldKernel()) 1 else kern.promptSplit()) else kern.EXL3_SK; // the staged kernel's splits (its partials: 2 * sk * P * MOE_W fp32)
            const xrows: i64 = if (is_prefill and shx and !kern.promptOldKernel()) rows else P; // one rotated row a token (the staged prompt kernel)
            b.x3 = .{ .xg = c.take(xrows * D * 2), .xu = c.take((if (xrows == rows) 128 else P * D) * 2), .xd = c.take(P * MOE_W * 2), .z = c.take(@max(2 * sk * P * MOE_W, P * D) * 4) };
        }
        if (is_prefill) {
            b.ks = .{ .out = c.take(rows * LL * 128 * 2), .k = c.take(rows * LL * 128 * 4), .v = c.take(rows * LL * 128 * 2), .g = c.take(rows * LL * 128 * 4), .b = c.take(rows * LL * 4) };
            b.q_tmp = c.take(rows * LL * 128 * 4);
            b.y_tmp = c.take(rows * LL * 128 * 2);
        }
        b.mem = try cuda.DeviceBuffer.alloc(d, c.at);
        const base = b.mem.ptr;
        const info = @typeInfo(Buffers).@"struct";
        inline for (info.field_names, info.field_types) |name, ty| {
            if (ty == u64 and !std.mem.eql(u8, name, "tab_bytes")) @field(b, name) += base;
        }
        inline for (.{ "members", "items", "counts", "rank", "hist" }) |f| @field(b.plan, f) += base;
        if (exl3) inline for (.{ "xg", "xu", "xd", "z" }) |f| {
            @field(b.x3, f) += base;
        };
        if (is_prefill) inline for (.{ "out", "k", "v", "g", "b" }) |f| {
            @field(b.ks, f) += base;
        };
        const r: u64 = @intCast(rows);
        b.segs = b.tab + r * 4 + 48 * r * 8;
        b.last = b.segs + MAX_STREAMS * @sizeOf(KdaSeg);
        try d.check(d.api.cuMemsetD8_v2(b.mem.ptr, 0, c.at), "zero buffers");
        return b;
    }
};

pub const Options = struct { context: i64 };

pub const Skip = struct {
    gather: bool = false,
    experts: bool = false,
    dense: bool = false,
    attn: bool = false,
    hc: bool = false,

    fn fromEnv() Skip {
        var s: Skip = .{};
        const v = std.c.getenv("TF_GLM_SKIP") orelse return s;
        var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
        while (it.next()) |w| inline for (@typeInfo(Skip).@"struct".field_names) |name| {
            if (std.mem.eql(u8, w, name)) @field(s, name) = true;
        };
        return s;
    }
};

/// A step's stream and its rows (consecutive in the step, in segment order).
pub const Seg = struct { s: *Seq, rows: i64 };

pub const Kind = enum(u8) { verify = 0, mtp = 1 };

/// A decode step's shape beyond its tables: rows, streams, dense attention chunks, any sparse row, the pool bucket.
const GKey = struct { kind: Kind, rows: i64, nseg: usize, nch: i64, np: i64, sparse: bool };
const Cached = struct { ex: cuda.graph.Exec, used: u64 };

fn graphCap() usize {
    const v = std.c.getenv("TF_GLM_GRAPHS") orelse return 96;
    return @max(8, std.fmt.parseInt(usize, std.mem.span(v), 10) catch 96);
}

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: *api.Ctx,
    k: K,
    store: *const api.Store,
    w: blocks.Weights = undefined,
    rank: u8 = 0,
    buf: Buffers = undefined, // batched verify steps
    mbuf: Buffers = undefined, // the MTP head's steps
    pbuf: Buffers = undefined, // prompt chunks
    capacity: i64 = 0, // the largest stream's cache slots
    host: []align(16) u8 = &.{},
    tab_host: []align(16) u8 = &.{}, // a step's tables before their upload
    cand_host: []i32 = &.{}, // both ranks' candidate blocks of a step (sampled draws)
    cut: cuda.DeviceBuffer = undefined, // a prompt chunk's cut: every KDA layer's state and window at the kept point
    cut_rec: u64 = 0,
    cut_conv: u64 = 0,
    cut_at: i64 = 0, // the cut's row in the chunk being computed (0: none)
    full: cuda.DeviceBuffer = undefined, // both ranks' logits of one row (a draw past the candidates)
    full_host: []u16 = &.{},
    /// Development: every launch synchronizes (and names the one that faulted).
    sync_each: bool = false,
    /// Decode steps as CUDA graphs, one a ``GKey`` shape, at most ``graph_cap`` (TF_GLM_GRAPHS), LRU dropped first.
    use_graphs: bool = true,
    graphs: std.AutoHashMap(GKey, Cached) = undefined,
    graph_cap: usize = 96,
    graph_clock: u64 = 0,
    captures: usize = 0,
    evictions: usize = 0,
    updates: usize = 0, // captures that took an older graph's instantiation (oldestGraph)
    capture_ms: f64 = 0,
    /// DFlash2 (opt-in): its taps, each stream's ring bytes, and the verify steps run (a draft needs the latest).
    drafter: ?*@import("dflash.zig").Drafter = null,
    ring_bytes: u64 = 0,
    verifies: u64 = 0,
    /// The step being run (its segments and shape), for the blocks.
    step: Step = .{},
    /// Development (TF_GLM_SKIP=gather,experts,dense,attn,hc): work left out to time it; replies wrong while set.
    skip: Skip = .{},
};

/// The step being computed: its segments, their kind, and the host's view of its rows (for the launches' shape).
pub const Step = struct {
    segs: []const Seg = &.{},
    kind: Kind = .verify,
    rows: i64 = 0,
    first_pos: i64 = 0, // a one-segment step's first position (a prompt chunk's)
    nch: i64 = 0, // dense attention chunks (0: no dense row)
    sparse: bool = false, // some row attends to its selected tokens
    np: i64 = 0, // pools the sparse rows score
};

pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *api.Ctx, kernels: *api.Kernels, store: *const api.Store, opts: Options) !*Engine {
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{ .gpa = gpa, .io = io, .ctx = ctx, .k = try K.init(gpa, ctx.d, ctx.ctx, ctx.stream, &kernels.ext), .store = store };
    e.rank = ctx.rank;
    e.skip = Skip.fromEnv();
    kern.readPdl();
    e.w = try blocks.Weights.load(&e.k, store);
    e.k.experts_sym = e.w.experts_sym;
    e.k.experts3 = e.w.experts3;
    e.capacity = opts.context + MAX_ROWS;
    const np = @divFloor(e.capacity, 4);
    e.buf = try Buffers.init(ctx.d, STEP_ROWS, false, np, e.w.exl3, false);
    e.mbuf = try Buffers.init(ctx.d, STEP_ROWS, false, np, e.w.exl3, false);
    e.pbuf = try Buffers.init(ctx.d, prefillRows(), true, np, e.w.exl3, e.w.exl3_shx);
    e.host = try gpa.alignedAlloc(u8, .@"16", 1 << 20);
    e.tab_host = try gpa.alignedAlloc(u8, .@"16", @intCast(e.pbuf.tab_bytes));
    e.graphs = .init(gpa);
    e.graph_cap = graphCap();
    e.cand_host = try gpa.alloc(i32, 2 * @as(usize, @intCast(STEP_ROWS)) * @import("draw.zig").W);
    e.full = try cuda.DeviceBuffer.alloc(ctx.d, 2 * HEAD_N * 2);
    e.full_host = try gpa.alloc(u16, 2 * HEAD_N);
    {
        const n: u64 = @intCast(e.w.nkda);
        const rec_bytes = n * @as(u64, @intCast(sq.REC * 4));
        e.cut = try cuda.DeviceBuffer.alloc(ctx.d, rec_bytes + n * @as(u64, @intCast(3 * sq.CONV_C * 2)));
        e.cut_rec = e.cut.ptr;
        e.cut_conv = e.cut.ptr + rec_bytes;
    }
    return e;
}

pub fn deinit(e: *Engine) void {
    e.buf.mem.free();
    e.mbuf.mem.free();
    e.pbuf.mem.free();
    e.gpa.free(e.host);
    e.gpa.free(e.tab_host);
    e.gpa.free(e.cand_host);
    e.gpa.free(e.full_host);
    e.full.free();
    e.cut.free();
    var it = e.graphs.valueIterator();
    while (it.next()) |c| c.ex.deinit();
    e.graphs.deinit();
    e.k.cache.deinit();
    e.w.deinit(e.gpa);
    e.gpa.destroy(e);
}

/// Device bytes a stream of ``capacity`` cache slots takes.
pub fn seqBytes(e: *const Engine, capacity: i64) u64 {
    return sq.bytes(e.w.nkda, e.w.caches(), capacity, e.ring_bytes);
}

pub fn newSeq(e: *Engine, capacity: i64) !*Seq {
    return sq.alloc(e.gpa, e.ctx.d, e.w.nkda, e.w.ndsa, e.w.caches(), capacity, e.ring_bytes);
}

pub fn freeSeq(e: *Engine, s: *Seq) void {
    sq.free(e.gpa, s);
}

/// The next prefill's images (null: none): their features replace the embeddings (MTP a row later), then dropped.
pub fn attach(s: *Seq, m: ?Media) void {
    s.media = m;
}

/// The step's tables for ``segs`` into ``b`` (uploaded, waited for) and shape; ``kind`` .mtp: MTP cache and mtp_len.
pub fn prepare(e: *Engine, b: *Buffers, segs: []const Seg, kind: Kind) !void {
    var R: i64 = 0;
    for (segs) |g| R += g.rows;
    if (R == 0 or R > b.rows or segs.len > MAX_STREAMS) return error.WindowTooWide;
    const cap: usize = @intCast(b.rows);
    const t = e.tab_host[0..@intCast(b.tab_bytes)];
    const pos: []align(1) i32 = std.mem.bytesAsSlice(i32, t[0 .. cap * 4]);
    const tabs: []align(1) u64 = std.mem.bytesAsSlice(u64, t[cap * 4 .. cap * 4 + 48 * cap * 8]);
    const kseg: []align(1) KdaSeg = std.mem.bytesAsSlice(KdaSeg, t[cap * 4 + 48 * cap * 8 ..][0 .. MAX_STREAMS * @sizeOf(KdaSeg)]);
    const last: []align(1) i32 = std.mem.bytesAsSlice(i32, t[cap * 4 + 48 * cap * 8 + MAX_STREAMS * @sizeOf(KdaSeg) ..][0 .. MAX_STREAMS * 4]);
    if (kind == .verify and !b.prefill) e.verifies += 1;
    var st: Step = .{ .segs = segs, .kind = kind, .rows = R };
    var row: usize = 0;
    var last_pos: i64 = 0;
    for (segs, 0..) |g, si| {
        const s = g.s;
        const p0 = if (kind == .mtp) s.mtp_len else s.pos;
        if (p0 + g.rows > s.capacity) return error.ContextFull;
        try sq.ensureRing(&e.k, s, if (kind == .mtp) 1 else 0, p0, g.rows);
        if (si == 0) st.first_pos = p0;
        if (kind == .verify and !b.prefill) {
            s.v_step = e.verifies;
            s.v_row0 = @intCast(row);
            s.v_pos0 = p0;
            s.v_rows = @intCast(g.rows);
        }
        kseg[si] = .{ .first = @intCast(row), .rows = @intCast(g.rows), .rec_in = s.recAt(s.cur, 0), .rec_out = s.recAt(1 - s.cur, 0), .conv = s.conv, .k_save = s.ks.k, .v_save = s.ks.v, .g_save = s.ks.g, .b_save = s.ks.b, .proj = s.proj };
        last[si] = @intCast(row + @as(usize, @intCast(g.rows)) - 1);
        for (0..@intCast(g.rows)) |r| {
            const q = p0 + @as(i64, @intCast(r));
            pos[row] = @intCast(q);
            if (q < kern.DENSE_LIMIT) st.nch = @max(st.nch, cdiv(q + 1, kern.CHUNK)) else st.sparse = true;
            last_pos = @max(last_pos, q);
            const lo: usize = if (kind == .mtp) e.w.ndsa else 0;
            const hi: usize = if (kind == .mtp) e.w.ndsa + 1 else e.w.ndsa;
            for (lo..hi) |ci| {
                tabs[ci * cap + row] = s.kc[ci];
                tabs[12 * cap + ci * cap + row] = s.ringPtr(s.ik[ci], ci);
                tabs[24 * cap + ci * cap + row] = s.ringPtr(s.ig[ci], ci);
                tabs[36 * cap + ci * cap + row] = s.pk[ci];
            }
            row += 1;
        }
    }
    if (st.sparse) st.np = kern.poolBucket(last_pos, b.np_max);
    e.step = st;
    try e.k.upload(b.tab, t);
    try e.k.stream.synchronize();
}

fn keyOf(e: *const Engine) GKey {
    const st = e.step;
    return .{ .kind = st.kind, .rows = st.rows, .nseg = st.segs.len, .nch = st.nch, .np = st.np, .sparse = st.sparse };
}

/// The step's GPU work (capturable): a verify forward and its candidates, or the MTP head and last rows' candidates.
fn body(e: *Engine) !void {
    switch (e.step.kind) {
        .verify => {
            try blocks.compute(e, &e.buf);
            try candidates(e, &e.buf, e.step.rows);
        },
        .mtp => {
            try blocks.mtpCompute(e, &e.mbuf);
            try candidates(e, &e.mbuf, @intCast(e.step.segs.len));
        },
    }
}

/// ``body`` from its shape's graph (captured the first time the shape runs), or eager in development modes.
pub fn run(e: *Engine) !void {
    if (!e.use_graphs or e.sync_each) return body(e);
    const key = keyOf(e);
    e.graph_clock += 1;
    if (e.graphs.getPtr(key)) |c| {
        c.used = e.graph_clock;
        return c.ex.launchOn(e.k.stream);
    }
    const t0 = std.Io.Timestamp.now(e.io, .awake);
    defer e.capture_ms += @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(e.io, .awake)).toNanoseconds())) / 1e6;
    try cuda.graph.beginCapture(e.k.stream, .thread_local);
    body(e) catch |err| {
        if (cuda.graph.endCapture(e.k.stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return err;
    };
    var g = try cuda.graph.endCapture(e.k.stream);
    defer g.deinit();
    e.captures += 1;
    if (e.graphs.count() >= e.graph_cap) {
        // full: the LRU graph of the same layout takes the new parameters in place; one that won't update is dropped
        if (oldestGraph(e, key, true)) |k| {
            var c = e.graphs.fetchRemove(k).?.value;
            const took = if (c.ex.update(g)) |r| r == .success else |_| false;
            if (took) {
                e.updates += 1;
                try e.graphs.put(key, .{ .ex = c.ex, .used = e.graph_clock });
                return c.ex.launchOn(e.k.stream);
            }
            c.ex.deinit();
            e.evictions += 1;
        } else if (oldestGraph(e, key, false)) |k| {
            var c = e.graphs.fetchRemove(k).?.value;
            c.ex.deinit();
            e.evictions += 1;
        }
    }
    var ex = try g.instantiate();
    errdefer ex.deinit();
    try e.graphs.put(key, .{ .ex = ex, .used = e.graph_clock });
    try ex.launchOn(e.k.stream);
}

/// The least recently run graph (``alike``: of ``key``'s kind and attention layout, the ones an update can take).
fn oldestGraph(e: *const Engine, key: GKey, alike: bool) ?GKey {
    var oldest: ?GKey = null;
    var at: u64 = std.math.maxInt(u64);
    var it = e.graphs.iterator();
    while (it.next()) |x| {
        const k = x.key_ptr.*;
        if (alike and (k.kind != key.kind or k.sparse != key.sparse or (k.nch > 0) != (key.nch > 0) or k.nseg != key.nseg)) continue;
        if (x.value_ptr.used < at) {
            at = x.value_ptr.used;
            oldest = k;
        }
    }
    return oldest;
}

/// One rank's greedy candidate of each logits row -> both ranks' candidates in b.cgath (rank order).
pub fn candidates(e: *Engine, b: *Buffers, rows: i64) !void {
    try kern.rowsTop(&e.k, b.logits, HEAD_N, 0, e.w.vocab_offset, b.cand, rows);
    try e.ctx.allGather(b.cand, b.cgath, @intCast(4 * rows), .i32, "candidates");
}

/// Every rank's fp32 partial b.part[:R] in rank order -> b.gath [2, R, D] (summed rank 0 first by the consumer).
pub fn gather(e: *Engine, b: *Buffers, R: i64) !u64 {
    if (e.skip.gather) return b.part;
    if (e.k.prof) |m| {
        try e.k.stream.synchronize();
        const t0 = std.Io.Timestamp.now(e.io, .awake);
        try e.ctx.allGather(b.part, b.gath, @intCast(R * D), .f32, "all-gather");
        try e.k.stream.synchronize();
        const gop = try m.getOrPut(if (e.ctx.roce != null) "(all-gather)" else "(nccl all-gather)");
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(e.io, .awake)).toNanoseconds())) / 1e6;
        return b.gath;
    }
    try e.ctx.allGather(b.part, b.gath, @intCast(R * D), .f32, "all-gather");
    return b.gath;
}

/// The ids of a step's rows (in segment order) into b.ids.
pub fn stageIds(e: *Engine, b: *Buffers, ids: []const u32) !void {
    if (@as(i64, @intCast(ids.len)) > b.rows) return error.WindowTooWide;
    const w: []i32 = @alignCast(std.mem.bytesAsSlice(i32, e.host[0 .. ids.len * 4]));
    for (ids, w) |x, *y| y.* = @intCast(x);
    try e.k.upload(b.ids, std.mem.sliceAsBytes(w));
    // the upload reads the host buffer when the stream gets to it: later stages reuse it, so wait here
    try e.k.stream.synchronize();
}

pub fn dropGraphs(e: *Engine) void {
    var it = e.graphs.valueIterator();
    while (it.next()) |c| c.ex.deinit();
    e.graphs.clearRetainingCapacity();
}
