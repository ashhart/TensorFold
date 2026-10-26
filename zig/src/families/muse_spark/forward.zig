//! Muse Spark's decode forward: windows of decode rows (any slots, any positions), four all-gathers a layer at TP=2.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const load_ = @import("load.zig");
const dflash2 = @import("dflash2.zig");
const ops = @import("ops.zig");
const exl3 = @import("exl3.zig");
const Weights = load_.Weights;
const Stream = cuda.Stream;

pub const CK = 256; // keys an attention chunk
pub const DECODE_ROWS = 128; // the EXL3 GEMV's row limit
pub const NORM_THREADS = 1024;
pub const BUCKET = 4096; // graphs: the smallest key bucket (full layers read bucket / CK chunks)
pub const MAX_GRAPH_ROWS = 32; // wider windows (prompt chunks) run eagerly
pub const MAX_CHUNKS = 64; // full layers' key chunks at most: chunks grow past CK keys with the context
/// Graphs key windows under this bucket at it (at most the context): one graph a width, the same bits (graphed()).
pub const GRAPH_BUCKET = 1 << 20;

/// The key bucket of a window ending at end - 1: a power of two from BUCKET, at most the context (CK-rounded).
pub fn bucketOf(context: usize, end: usize) usize {
    const full = (context + CK - 1) / CK * CK;
    return @min(full, @max(BUCKET, std.math.ceilPowerOfTwo(usize, end) catch unreachable));
}

/// Keys a full layer's chunk holds for a bucket: CK, or bucket / MAX_CHUNKS rounded up to CK past 16K keys.
pub fn chunkKeys(bucket: usize) usize {
    return @max(CK, std.mem.alignForward(usize, (bucket + MAX_CHUNKS - 1) / MAX_CHUNKS, CK));
}

/// linear_multi.cu's instantiations (EXL3 bits a weight; mul1 codebook, 4 warps).
pub const multi_k2s = [_]u32{ 2, 3, 4, 5, 6, 8 };

/// Key chunks a full layer's row at position pos reads eagerly (its own chunk width).
pub fn rowChunks(context: usize, pos: usize) usize {
    return pos / chunkKeys(bucketOf(context, pos + 1)) + 1;
}

/// One decode row: a stream's cache slot, its position, its input token.
pub const Row = struct { slot: u32, pos: u32, token: i64 };

/// Sliding cache rows a slot keeps: the window, the row itself and a window's other rows of the same slot.
pub fn ringSize(window: usize, rows: usize) usize {
    return std.mem.alignForward(usize, window + 1 + rows, 64);
}

fn u(x: usize) u32 {
    return @intCast(x);
}
fn ci(x: usize) c_int {
    return @intCast(x);
}

const Bufs = struct {
    ids: u64, // i64 [R]
    pos: u64, // i32 [R]
    slot: u64, // i32 [R]
    res: u64, // f32 [R, D]
    x: u64, // f16 [R, D]: the normed sublayer input
    xf: u64, // f32 [R, D]: the normed MoE input unrounded (the router's)
    q: u64, // f16 [R, H * 64]
    k: u64, // f16 [R, HK * 64]
    v: u64,
    g: u64, // f16 [R, H * 64]: the output gate's logits
    qo: u64, // f16 [R, H * 64]
    att: u64, // f16 [R, H * 64]: gated attention output
    po: u64, // f32 [R, H, NCH, 64]
    pm: u64,
    pl: u64,
    xh: u64, // f16 rot_in scratch [R, max K]
    xh4: u64, // f16 [4, R, D]: q / k / v / gate's rotated inputs (one launch)
    z4: u64, // f32: their split partials, one region each
    z: u64, // f32 split partials
    part: u64, // f32 [R, D]
    logits: u64, // f32 [R, E]
    pick: u64, // i32 [R, K]
    wts: u64, // f32 [R, K]
    lat: u64, // f16 [R, latent]
    latn: u64, // f16 [R, latent]
    acc: u64, // f16 [R, latent]
    head: u64, // f32 [R, head stride]: this rank's head rows
    recv: u64, // f32 [world, R, max(D, K * latent, head stride)]: all-gathers
    latg: u64, // f16 [world, R, latent / world]: the gathered latent halves
    am: u64, // i32 [R]
    erow: u64, // bf16 [R, D]: the window's embedding rows, read through iota
    iota: u64, // i64 [R]: 0 .. R - 1
    sc: exl3.Scratch,
};

pub const Model = struct {
    w: *const Weights,
    d: *const cuda.Driver,
    k3: *const exl3.Kernels,
    comm: ?tp.Collective,
    glue: cuda.Module,
    attn: cuda.Module,
    lin: cuda.Module, // linear_multi.cu: q / k / v / gate in one launch
    f_multi: [multi_k2s.len]cuda.Function,
    f_rotm: cuda.Function,
    f_lin2: [multi_k2s.len][2]cuda.Function, // linear_body2 (32-row passes, the same bits) at 4 and 8 warps
    f_embed: cuda.Function,
    f_rms: cuda.Function,
    f_gate: cuda.Function,
    f_gate_c: cuda.Function, // the cluster versions (8 blocks a row, the same bits)
    f_esum_c: cuda.Function,
    f_router: cuda.Function,
    f_topk: cuda.Function,
    f_esum: cuda.Function,
    f_softcap: cuda.Function,
    f_argmax: cuda.Function,
    f_prep: cuda.Function,
    f_rows: cuda.Function,
    f_merge: cuda.Function,
    context: usize,
    graphs: std.AutoHashMapUnmanaged([2]u32, cuda.graph.Exec) = .empty,
    replays: usize = 0,
    captures: usize = 0,
    kv_bytes: usize = 0, // the KV caches' device bytes
    ring: usize,
    rows: usize,
    slots: usize,
    nch_max: usize,
    kc: []u64,
    vc: []u64,
    cos: u64, // fp32 [context, 32] (one theta: every RoPE layer's)
    sin: u64,
    theta: f64,
    b: Bufs,
    arena: load_.Arena,
    gpa: std.mem.Allocator,
    h_erow: []u16 = &.{}, // host staging of the window's embedding rows
    dr: ?*dflash2.Drafter = null, // fed inside run: taps after its layers, the context update after the last layer

    /// fp16 K/V caches: full layers [slots, context, HK, 64], sliding layers rings of ringSize rows.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const Weights, k3: *const exl3.Kernels, comm: ?tp.Collective, context: usize, rows: usize, slots: usize, s: Stream) !Model {
        const cfg = w.cfg;
        if (w.world > 1 and comm == null) return error.NeedComm;
        if (rows > DECODE_ROWS or rows == 0) return error.TooManyRows;
        var m: Model = undefined;
        m.w = w;
        m.d = d;
        m.k3 = k3;
        m.comm = if (w.world > 1) comm else null;
        m.context = context;
        m.graphs = .empty;
        m.replays = 0;
        m.captures = 0;
        m.kv_bytes = 0;
        m.ring = ringSize(cfg.window, rows);
        m.rows = rows;
        m.slots = slots;
        m.nch_max = @max(@min(MAX_CHUNKS, (context + CK - 1) / CK), (cfg.window + 1 + CK - 1) / CK);
        m.gpa = gpa;
        m.dr = null;
        m.arena = .{ .d = d, .gpa = gpa };
        errdefer m.arena.deinit();
        m.glue = try cuda.Module.load(d, cuda.kernels.muse_glue);
        errdefer m.glue.unload();
        m.attn = try cuda.Module.load(d, cuda.kernels.muse_attention);
        errdefer m.attn.unload();
        m.f_embed = try m.glue.function("muse_embed_norm");
        m.f_rms = try m.glue.function("muse_rms");
        m.f_gate = try m.glue.function("muse_gate_norm");
        m.f_gate_c = try m.glue.function("muse_gate_norm_c");
        m.f_esum_c = try m.glue.function("muse_expert_sum_c");
        m.f_router = try m.glue.function("muse_router");
        m.f_topk = try m.glue.function("muse_topk");
        m.f_esum = try m.glue.function("muse_expert_sum");
        m.f_softcap = try m.glue.function("muse_softcap");
        m.f_argmax = try m.glue.function("muse_argmax");
        m.f_prep = try m.attn.function("muse_qk_prep");
        m.f_rows = try m.attn.function("muse_attn_rows");
        m.f_merge = try m.attn.function("muse_attn_merge");
        m.lin = try cuda.Module.load(d, cuda.kernels.muse_linear);
        errdefer m.lin.unload();
        inline for (multi_k2s, 0..) |k2, j| m.f_multi[j] = try m.lin.function(std.fmt.comptimePrint("muse_linear_multi_k{d}", .{k2}));
        m.f_rotm = try m.lin.function("muse_rot_multi");
        inline for (multi_k2s, 0..) |k2, j| inline for (.{ 4, 8 }, 0..) |wk, q| {
            m.f_lin2[j][q] = try m.lin.function(std.fmt.comptimePrint("muse_linear2_k{d}_w{d}", .{ k2, wk }));
        };
        const ar = &m.arena;
        const L = w.layers.len;
        const HK = cfg.kv_heads;
        const hd = cfg.head_dim;
        m.kc = try gpa.alloc(u64, L);
        m.vc = try gpa.alloc(u64, L);
        for (0..L) |i| {
            const n = if (cfg.full[i]) context else m.ring;
            m.kc[i] = try ar.zeros(slots * n * HK * hd * 2);
            m.vc[i] = try ar.zeros(slots * n * HK * hd * 2);
            m.kv_bytes += 2 * slots * n * HK * hd * 2;
        }
        // RoPE tables: every RoPE layer of Muse Spark has the same theta (5e5); refuse a second one
        m.theta = 0;
        for (0..L) |i| if (cfg.rope(i)) {
            if (m.theta != 0 and m.theta != cfg.theta[i]) return error.SeveralThetas;
            m.theta = cfg.theta[i];
        };
        m.cos = try ar.alloc(context * (hd / 2) * 4);
        m.sin = try ar.alloc(context * (hd / 2) * 4);
        if (m.theta != 0) {
            const f = try m.glue.function("muse_rope_tables");
            var a: cuda.Args = .{};
            a.add(m.cos);
            a.add(m.sin);
            a.add(ci(context));
            a.add(m.theta);
            a.add(ci(hd));
            try cuda.launch.launch(f, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        }
        const R = rows;
        const D = cfg.hidden;
        const HQ = cfg.heads * hd;
        const HKD = HK * hd;
        const lat = cfg.latent;
        const vr = w.head_stride;
        var max_z: usize = zBytes(w.head, R);
        for (w.layers) |ly| inline for (.{ "q", "k", "v", "g", "o", "pre", "post" }) |f| {
            max_z = @max(max_z, zBytes(@field(ly, f), R));
        };
        var b: Bufs = undefined;
        b.ids = try ar.zeros(R * 8);
        b.pos = try ar.zeros(R * 4);
        b.slot = try ar.zeros(R * 4);
        b.res = try ar.zeros(R * D * 4);
        b.x = try ar.zeros(R * D * 2);
        b.xf = try ar.zeros(R * D * 4);
        b.q = try ar.zeros(R * HQ * 2);
        b.k = try ar.zeros(R * HKD * 2);
        b.v = try ar.zeros(R * HKD * 2);
        b.g = try ar.zeros(R * HQ * 2);
        b.qo = try ar.zeros(R * HQ * 2);
        b.att = try ar.zeros(R * HQ * 2);
        b.po = try ar.zeros(R * cfg.heads * m.nch_max * hd * 4);
        b.pm = try ar.zeros(R * cfg.heads * m.nch_max * 4);
        b.pl = try ar.zeros(R * cfg.heads * m.nch_max * 4);
        b.xh = try ar.zeros(R * @max(D, HQ) * 2);
        b.z = try ar.zeros(@max(max_z, 4));
        var z4: usize = 0;
        for (w.layers) |ly| z4 = @max(z4, zBytes(ly.q, R) + zBytes(ly.k, R) + zBytes(ly.v, R) + zBytes(ly.g, R));
        b.xh4 = try ar.zeros(4 * R * D * 2);
        b.z4 = try ar.zeros(@max(z4, 4));
        b.part = try ar.zeros(R * D * 4);
        b.logits = try ar.zeros(R * cfg.experts * 4);
        b.pick = try ar.zeros(R * cfg.top_k * 4);
        b.wts = try ar.zeros(R * cfg.top_k * 4);
        b.lat = try ar.zeros(R * lat * 2);
        b.latn = try ar.zeros(R * lat * 2);
        b.acc = try ar.zeros(R * lat * 2);
        b.head = try ar.zeros(R * vr * 4);
        b.recv = if (w.world > 1) try ar.zeros(w.world * R * @max(@max(D, cfg.top_k * lat), vr) * 4) else 0;
        b.latg = if (w.world > 1) try ar.zeros(R * lat * 2) else 0;
        b.am = try ar.zeros(R * 4);
        b.erow = try ar.zeros(R * D * 2);
        {
            const io_ = try gpa.alloc(i64, R);
            defer gpa.free(io_);
            for (io_, 0..) |*x, j| x.* = @intCast(j);
            b.iota = try ar.upload(std.mem.sliceAsBytes(io_));
        }
        m.h_erow = try gpa.alloc(u16, R * D);
        const first = w.layers[0].experts;
        const sz = try exl3.sizes(first, R, cfg.top_k);
        b.sc = .{ .xg = try ar.zeros(sz.xg), .xu = try ar.zeros(sz.xg), .xd = try ar.zeros(sz.xd), .z = try ar.zeros(sz.z), .y = try ar.zeros(sz.y), .ids = try ar.zeros(sz.ids), .count = try ar.zeros(4), .members = try ar.alloc(sz.members), .rows = R, .slots = cfg.top_k };
        try d.check(d.api.cuMemsetD32_v2(b.sc.members, 0xFFFF_FFFF, sz.members / 4), "members -1");
        m.b = b;
        try s.synchronize();
        return m;
    }

    fn zBytes(l: exl3.Linear, R: usize) usize {
        return if (l.split[0] > 1) l.split[0] * R * l.n * 4 else 0;
    }

    pub fn deinit(m: *Model) void {
        var it = m.graphs.valueIterator();
        while (it.next()) |e| e.deinit();
        m.graphs.deinit(m.gpa);
        m.gpa.free(m.kc);
        m.gpa.free(m.vc);
        m.gpa.free(m.h_erow);
        m.glue.unload();
        m.attn.unload();
        m.lin.unload();
        m.arena.deinit();
    }

    pub const Tap = enum { input, attn, mlp };
    /// A hook per layer: `.input` before it (teacher forcing), `.attn` after the gate, `.mlp` the block output.
    pub const Capture = struct {
        hook: *const fn (ctx: *anyopaque, what: Tap, layer: usize, dev: u64, n: usize) anyerror!void,
        ctx: *anyopaque,
        teacher: bool = false,
    };

    /// Rows' tokens, positions and slots into the fixed device inputs.
    pub fn stage(m: *Model, rows: []const Row) !void {
        if (rows.len == 0 or rows.len > m.rows) return error.WindowTooWide;
        var ids: [DECODE_ROWS]i64 = undefined;
        var pos: [DECODE_ROWS]i32 = undefined;
        var slot: [DECODE_ROWS]i32 = undefined;
        var per: [DECODE_ROWS]usize = @splat(0);
        for (rows, 0..) |r, i| {
            if (r.slot >= m.slots) return error.BadSlot;
            if (r.pos >= m.context) return error.PastContext;
            if (r.token < 0 or r.token >= m.w.full.vocab) return error.BadToken;
            ids[i] = r.token;
            pos[i] = @intCast(r.pos);
            slot[i] = @intCast(r.slot);
            per[r.slot] += 1;
        }
        for (per[0..m.slots]) |c| if (c + m.w.cfg.window + 1 > m.ring) return error.RingOverrun;
        const n = rows.len;
        try m.embedRows(ids[0..n], m.h_erow, m.b.erow);
        try upload(m.d, m.b.ids, std.mem.sliceAsBytes(ids[0..n]));
        try upload(m.d, m.b.pos, std.mem.sliceAsBytes(pos[0..n]));
        try upload(m.d, m.b.slot, std.mem.sliceAsBytes(slot[0..n]));
    }

    /// The tokens' table rows (bf16) through host staging `h` into `dev` [n, D].
    pub fn embedRows(m: *const Model, tokens: []const i64, h: []u16, dev: u64) !void {
        const D = m.w.full.hidden;
        const tab = m.w.embed_host;
        if (h.len < tokens.len * D) return error.WindowTooWide;
        for (tokens, 0..) |t, j| @memcpy(std.mem.sliceAsBytes(h[j * D ..][0..D]), tab[@as(usize, @intCast(t)) * D * 2 ..][0 .. D * 2]);
        try upload(m.d, dev, std.mem.sliceAsBytes(h[0 .. tokens.len * D]));
    }

    /// The embedding launch's (table, ids): the staged rows and iota (the table stays on the host).
    pub fn embedSrc(_: *const Model, erow: u64, iota: u64) [2]u64 {
        return .{ erow, iota };
    }

    /// Eager decode window: softcapped fp32 logits of every row into `out` [n, vocab] (or none).
    pub fn forward(m: *Model, s: Stream, rows: []const Row, out: ?[]f32, cap: ?Capture) !void {
        try m.stage(rows);
        var chunks: usize = 1; // the most any row's own chunking takes
        for (rows) |r| chunks = @max(chunks, rowChunks(m.context, r.pos));
        try m.run(s, rows.len, chunks, cap);
        if (out) |o| try m.logitsTo(s, rows.len, o) else try s.synchronize();
    }

    /// Graphed decode window: a graph per (rows, key bucket) after an eager warm-up; replay bits equal eager bits.
    pub fn graphed(m: *Model, s: Stream, rows: []const Row, out: ?[]f32) !void {
        if (rows.len > MAX_GRAPH_ROWS) return m.forward(s, rows, out, null); // prompt windows: eager, the same bits
        var end: usize = 0;
        for (rows) |r| end = @max(end, r.pos + 1);
        var bucket = bucketOf(m.context, end);
        if (GRAPH_BUCKET > bucket) bucket = @min(GRAPH_BUCKET, (m.context + CK - 1) / CK * CK);
        const chunks = (bucket + chunkKeys(bucket) - 1) / chunkKeys(bucket); // covers every row at or under bucket
        const key: [2]u32 = .{ @intCast(rows.len), @intCast(bucket) };
        try m.stage(rows);
        if (m.graphs.get(key)) |exec| {
            try exec.launchOn(s);
            m.replays += 1;
        } else {
            try m.run(s, rows.len, chunks, null); // warm-up: writes the same cache rows the replay will
            try s.synchronize();
            try cuda.graph.beginCapture(s, .thread_local);
            m.run(s, rows.len, chunks, null) catch |e| {
                if (cuda.graph.endCapture(s)) |g| {
                    var gg = g;
                    gg.deinit();
                } else |_| {}
                return e;
            };
            var g = try cuda.graph.endCapture(s);
            defer g.deinit();
            const exec = try g.instantiate();
            try m.graphs.put(m.gpa, key, exec);
            try exec.launchOn(s);
            m.captures += 1;
        }
        if (out) |o| try m.logitsTo(s, rows.len, o) else try s.synchronize();
    }

    pub const norm = ops.norm;
    pub const normParts = ops.normParts;
    pub const gateNormAt = ops.gateNormAt;
    pub const qkvg = ops.qkvg;
    pub const rowLaunch = ops.rowLaunch;
    pub const proj = ops.proj;
    pub const AttnIo = ops.AttnIo;
    pub const attentionIo = ops.attentionIo;

    /// The window's kernels on staged inputs, no host work; full layers read `chunks` key chunks (each row's width).
    pub fn run(m: *Model, s: Stream, n: usize, chunks: usize, cap: ?Capture) !void {
        const cfg = m.w.cfg;
        const D = cfg.hidden;
        if (chunks > m.nch_max) return error.TooManyChunks;
        {
            var a: cuda.Args = .{};
            const es = m.embedSrc(m.b.erow, m.b.iota);
            a.add(es[0]);
            a.add(es[1]);
            a.add(m.b.res);
            a.add(ci(D));
            a.add(@as(f32, @floatCast(cfg.eps)));
            try cuda.launch.launch(m.f_embed, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = NORM_THREADS, .y = 1, .z = 1 } }, s, &a);
        }
        try m.norm(s, m.b.res, false, m.w.layers[0].in_norm, m.b.x, n, D, cfg.eps);
        for (m.w.layers, 0..) |L, i| {
            if (cap) |c| if (c.teacher) {
                try c.hook(c.ctx, .input, i, m.b.res, n);
                try m.norm(s, m.b.res, false, L.in_norm, m.b.x, n, D, cfg.eps);
            };
            try m.qkvg(s, L, m.b.x, m.b.q, m.b.k, m.b.v, m.b.g, n);
            try ops.attention(m, s, i, n, chunks);
            try m.proj(s, L.o, m.b.att, m.b.part, true, n);
            try ops.gateNorm(m, s, m.b.part, 0, L.attn_ab, L.ffn_norm, m.b.xf, n);
            if (cap) |c| try c.hook(c.ctx, .attn, i, m.b.res, n);
            try ops.moe(m, s, L, n);
            const after = if (i + 1 < m.w.layers.len) m.w.layers[i + 1].in_norm else m.w.norm;
            try ops.gateNorm(m, s, m.b.part, L.post_ffn, L.mlp_ab, after, 0, n);
            if (cap) |c| try c.hook(c.ctx, .mlp, i, m.b.res, n);
            if (m.dr) |dr| if (cap == null) try dr.tap(s, i, m.b.res, n);
        }
        if (m.dr) |dr| if (cap == null) try dr.update(s, n);
        try m.proj(s, m.w.head, m.b.x, m.b.head, true, n);
        const cfgf = m.w.full;
        var a: cuda.Args = .{};
        a.add(m.b.head);
        a.add(@as(i64, @intCast(n * m.w.head.n)));
        a.add(@as(f32, @floatCast(cfgf.out_mult)));
        a.add(@as(f32, @floatCast(cfgf.softcap)));
        try cuda.launch.launch(m.f_softcap, .{ .grid = .{ .x = u(@min((n * m.w.head.n + 255) / 256, 4096)), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        // each rank's [n, its rows] block, gathered as n * stride elements a rank (rank 1's slice is a block shorter)
        if (m.comm) |c| try c.allGather(m.b.head, m.b.recv, n * m.w.head_stride, .f32, s.handle);
    }

    /// Where the rows' head slices are: (rank 0's [n, w0], rank 1's [n, w1] offset in elements, w0, w1).
    fn heads(m: *const Model, n: usize) struct { base: u64, part: usize, w0: usize, w1: usize } {
        if (m.comm == null) return .{ .base = m.b.head, .part = 0, .w0 = m.w.head.n, .w1 = 0 };
        const st = m.w.head_stride;
        return .{ .base = m.b.recv, .part = n * st, .w0 = st, .w1 = m.w.head_rows - st }; // rank 1: the padded head's rest
    }

    /// Each row's argmax over the real vocabulary on the device; only the ids reach the host.
    pub fn argmaxTo(m: *Model, s: Stream, n: usize, out: []i32) !void {
        const h = m.heads(n);
        var a: cuda.Args = .{};
        a.add(h.base);
        a.add(m.b.am);
        a.add(ci(h.w0));
        a.add(ci(h.w1));
        a.add(@as(i64, @intCast(h.part)));
        a.add(ci(m.w.full.vocab));
        try cuda.launch.launch(m.f_argmax, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        try s.synchronize();
        try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), m.b.am, n * 4), "argmax");
    }

    /// The window's logits on the host: [n, vocab] (the head's padding rows dropped).
    pub fn logitsTo(m: *Model, s: Stream, n: usize, out: []f32) !void {
        const V = m.w.full.vocab;
        const h = m.heads(n);
        try s.synchronize();
        for (0..n) |r| {
            const c0 = @min(V, h.w0);
            try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out[r * V ..].ptr), h.base + r * h.w0 * 4, c0 * 4), "logits");
            if (V > c0) try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out[r * V + c0 ..].ptr), h.base + (h.part + r * h.w1) * 4, (V - c0) * 4), "logits");
        }
    }
};

fn upload(d: *const cuda.Driver, at: u64, bytes: []const u8) !void {
    try d.check(d.api.cuMemcpyHtoD_v2(at, bytes.ptr, bytes.len), "upload");
}

test "chunks stay at 64 or fewer, and at CK keys up to 16K" {
    try std.testing.expectEqual(@as(usize, 4096), bucketOf(4096, 2615));
    try std.testing.expectEqual(@as(usize, 256), chunkKeys(4096));
    try std.testing.expectEqual(@as(usize, 256), chunkKeys(16384));
    try std.testing.expectEqual(@as(usize, 512), chunkKeys(32768));
    try std.testing.expectEqual(@as(usize, 2048), chunkKeys(131072));
    try std.testing.expectEqual(@as(usize, 40960), bucketOf(40960, 40000));
    try std.testing.expect((40960 + chunkKeys(40960) - 1) / chunkKeys(40960) <= MAX_CHUNKS);
}

test "ring" {
    try std.testing.expectEqual(@as(usize, 2176), ringSize(2047, 128));
    try std.testing.expectEqual(@as(usize, 2112), ringSize(2047, 1));
}
