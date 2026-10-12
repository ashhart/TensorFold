//! MiMo-V2's decode forward, launch for launch as the Python engine's: captured Triton, EXL3 and glue kernels.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const load_ = @import("load.zig");
const dflash = @import("dflash.zig");
const kv8_ = @import("kv8.zig");
const exl3 = cuda.exl3;
const aot = cuda.aot;
const Weights = load_.Weights;
const Stream = cuda.Stream;

pub const CK = 256; // attention.CK: keys a decode chunk
pub const BN = 64;
pub const GP = 16;
pub const PROMPT_CHUNK = 16384; // forward.PROMPT_CHUNK: sizes the sliding ring
pub const DECODE_ROWS = 128; // ops.DECODE_ROWS: the EXL3 GEMV's row limit
pub const BUCKET = 8192; // graphs.BUCKET
pub const MAX_GRAPH_ROWS = 32; // graphs.MAX_ROWS: the widest chain graphed

/// One decode row: a stream's cache slot, its position, its input token.
pub const Row = struct { slot: u32, pos: u32, token: i64 };

/// attention.ring_size
pub fn ringSize(rows: usize, window: usize) usize {
    return (rows + window + BN - 1) / BN * BN;
}

fn pow2(x: usize) usize {
    return std.math.ceilPowerOfTwo(usize, x) catch unreachable;
}

const p = aot.ptr;
fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}
fn ii(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}
fn u(x: usize) u32 {
    return @intCast(x);
}

/// Device scratch for up to `rows` decode rows.
const Bufs = struct {
    ids: u64, // i64 [R]
    pos: u64, // i32 [R]
    slot: u64, // i32 [R]
    res: u64, // f32 [R, D]
    x: u64, // f16 [R, D]
    q: u64, // f16 [R, H * 192]
    k: u64, // f16 [R, HK * 192]
    v: u64,
    qo: u64, // f16 [R, H, 192]
    att: u64, // f16 [R, H, 128]
    po: u64, // f32 [R * NCH * H * 128]
    pm: u64,
    pl: u64,
    xh: u64, // f16 rot_in scratch [R, max K]
    z: u64, // f32 split partials
    part: u64, // f32 [R, D]
    recv: u64, // f32 [2, R, max(D, Vr)]
    sum: u64, // f32 [R, D]
    logits: u64, // f32 [R, E]
    pick: u64, // i32 [R, 8]
    wts: u64, // f32 [R, 8]
    g: u64, // f16 [R, ND]
    up: u64,
    h: u64,
    head: u64, // f32 [R, Vr]
    am: u64, // i32 [R]: device argmax of each row
    sc: exl3.Scratch,
};

pub const Model = struct {
    w: *const Weights,
    d: *const cuda.Driver,
    set: *const aot.Set,
    k3: *const exl3.Kernels,
    glue: cuda.Module,
    f_embed: cuda.Function,
    f_add2: cuda.Function,
    f_argmax: cuda.Function,
    comm: ?tp.Collective,
    context: usize,
    ring: usize,
    rows: usize,
    slots: usize,
    nch_max: usize,
    graphs: std.AutoHashMapUnmanaged([2]u32, cuda.graph.Exec) = .empty,
    dr: ?*dflash.Drafter = null,
    replays: usize = 0,
    captures: usize = 0,
    kc: []u64,
    vc: []u64,
    kv8: ?kv8_.Kv8 = null, // --policy kv=fp8: the full layers' 8-bit cache
    cos_full: u64,
    sin_full: u64,
    cos_swa: u64,
    sin_swa: u64,
    scale: f32,
    b: Bufs,
    arena: load_.Arena,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const Weights, set: *const aot.Set, k3: *const exl3.Kernels, comm: ?tp.Collective, context: usize, rows: usize, slots: usize, kv8: bool, s: Stream) !Model {
        const cfg = w.cfg;
        var m: Model = undefined;
        m.w = w;
        m.d = d;
        m.set = set;
        m.k3 = k3;
        m.comm = if (w.world > 1) comm orelse return error.NeedComm else null;
        m.context = context;
        m.ring = ringSize(PROMPT_CHUNK, cfg.window);
        if (rows > DECODE_ROWS) return error.TooManyRows;
        m.rows = rows;
        m.slots = slots;
        m.graphs = .empty;
        m.dr = null;
        m.replays = 0;
        m.captures = 0;
        m.nch_max = (context + CK - 1) / CK; // a graph's bucket / CK at most, and max(pos) / CK + 1 eagerly
        m.gpa = gpa;
        m.arena = .{ .d = d, .gpa = gpa };
        errdefer m.arena.deinit();
        m.glue = try cuda.Module.load(d, cuda.kernels.mimo_glue);
        m.f_embed = try m.glue.function("mimo_embed_f32");
        m.f_add2 = try m.glue.function("mimo_add2_f32");
        m.f_argmax = try m.glue.function("mimo_argmax");
        const ar = &m.arena;
        // caches: keys [1, LEN, HK, 192], values [1, LEN, HK, 128], zeroed (forward.Model.__init__)
        m.kc = try gpa.alloc(u64, cfg.layers);
        m.vc = try gpa.alloc(u64, cfg.layers);
        m.kv8 = if (kv8) try kv8_.Kv8.init(gpa, d, &m.arena, cfg, slots, context) else null;
        errdefer if (m.kv8) |*k| k.deinit(gpa);
        for (0..cfg.layers) |i| {
            const n = if (cfg.full[i]) context else m.ring;
            const hk = cfg.kvHeads(i);
            const bytes: usize = if (m.kv8 != null and cfg.full[i]) 1 else 2; // e4m3 or fp16
            m.kc[i] = try ar.zeros(slots * n * hk * 192 * bytes);
            m.vc[i] = try ar.zeros(slots * n * hk * 128 * bytes);
        }
        // rope tables per theta: cos, sin [context, rope_dim / 2] fp32
        const half = cfg.rope_dim / 2;
        const rope = try m.glue.function("mimo_rope_tables");
        inline for (.{ .{ "cos_full", "sin_full", cfg.theta_full }, .{ "cos_swa", "sin_swa", cfg.theta_swa } }) |t| {
            @field(m, t[0]) = try ar.alloc(context * half * 4);
            @field(m, t[1]) = try ar.alloc(context * half * 4);
            var a: cuda.Args = .{};
            a.add(@field(m, t[0]));
            a.add(@field(m, t[1]));
            a.add(@as(c_int, @intCast(context)));
            a.add(@as(f64, t[2]));
            a.add(@as(c_int, @intCast(cfg.rope_dim)));
            try cuda.launch.launch(rope, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        }
        // qk_dim ** -0.5 as Python's float(), then fp32
        m.scale = @floatCast(std.math.pow(f64, @floatFromInt(cfg.qk_dim), -0.5));
        // scratch
        const R = rows;
        const D = cfg.hidden;
        const H = cfg.heads;
        const hk_max = @max(cfg.kv_full, cfg.kv_swa);
        const vr = w.head.n;
        const nd = cfg.dense_width / w.world;
        var max_k: usize = @max(D, H * 128);
        max_k = @max(max_k, nd);
        var max_z: usize = 0;
        for (w.layers) |L| {
            inline for (.{ "q", "k", "v", "o" }) |f| max_z = @max(max_z, zBytes(@field(L, f), R));
            if (L.gate) |g| max_z = @max(max_z, zBytes(g, R));
            if (L.up) |g| max_z = @max(max_z, zBytes(g, R));
            if (L.down) |g| max_z = @max(max_z, zBytes(g, R));
        }
        max_z = @max(max_z, zBytes(w.head, R));
        var b: Bufs = undefined;
        b.ids = try ar.zeros(R * 8);
        b.pos = try ar.zeros(R * 4);
        b.slot = try ar.zeros(R * 4);
        b.res = try ar.zeros(R * D * 4);
        b.x = try ar.zeros(R * D * 2);
        b.q = try ar.zeros(R * H * 192 * 2);
        b.k = try ar.zeros(R * hk_max * 192 * 2);
        b.v = try ar.zeros(R * hk_max * 192 * 2);
        b.qo = try ar.zeros(R * H * 192 * 2);
        b.att = try ar.zeros(R * H * 128 * 2);
        b.po = try ar.zeros(R * m.nch_max * H * 128 * 4);
        b.pm = try ar.zeros(R * m.nch_max * H * 4);
        b.pl = try ar.zeros(R * m.nch_max * H * 4);
        b.xh = try ar.zeros(R * max_k * 2);
        b.z = try ar.zeros(@max(max_z, 4));
        b.part = try ar.zeros(R * D * 4);
        b.recv = try ar.zeros(2 * R * @max(D, vr) * 4);
        b.sum = try ar.zeros(R * D * 4);
        b.logits = try ar.zeros(R * cfg.experts * 4);
        b.pick = try ar.zeros(R * cfg.top_k * 4);
        b.wts = try ar.zeros(R * cfg.top_k * 4);
        b.g = try ar.zeros(R * nd * 2);
        b.up = try ar.zeros(R * nd * 2);
        b.h = try ar.zeros(R * nd * 2);
        b.head = try ar.zeros(R * vr * 4);
        b.am = try ar.zeros(R * 4);
        // experts.Scratch(ex, WINDOW, top_k): the decode path's (rows only size it)
        const first = for (w.layers) |L| {
            if (L.experts) |e| break e;
        } else return error.NoMoE;
        const cfg_gu = try exl3.defaultConfig(first.dims, first.width, true);
        const cfg_d = try exl3.defaultConfig(first.width, first.dims, false);
        const sz = exl3.Scratch.bytes(first, R, cfg.top_k, cfg_gu, cfg_d);
        b.sc = .{ .xg = try ar.zeros(sz.xg), .xu = try ar.zeros(sz.xg), .xd = try ar.zeros(sz.xd), .z = try ar.zeros(sz.z), .y = try ar.zeros(sz.y), .ids = try ar.zeros(sz.ids), .count = try ar.zeros(4), .members = try ar.alloc(sz.members), .rows = R, .slots = cfg.top_k, .cfg_gu = cfg_gu, .cfg_d = cfg_d };
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
        if (m.kv8) |*k| k.deinit(m.gpa);
        m.glue.unload();
        m.arena.deinit();
    }

    /// The fp32 residual (or partial) of `n` rows summed over the ranks in rank order (forward._sum).
    fn sum(m: *Model, s: Stream, part: u64, n: usize) !u64 {
        const c = m.comm orelse return part;
        const D = m.w.cfg.hidden;
        try c.allGather(part, m.b.recv, n * D, .f32, s.handle);
        try m.add2(s, m.b.recv, m.b.recv + n * D * 4, m.b.sum, n * D);
        return m.b.sum;
    }

    pub fn add2(m: *Model, s: Stream, a_: u64, b_: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(a_);
        a.add(b_);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(m.f_add2, .{ .grid = .{ .x = u(@min((n + 255) / 256, 1024)), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    fn rms(m: *Model, s: Stream, x: u64, w: u64, out: u64, n: usize) !void {
        const D = m.w.cfg.hidden;
        try m.set.run(s, "_rms", .{ u(n), 1, 1 }, &.{ p("X", "*fp32", x), p("W", "*fp32", w), p("OUT", "*fp16", out) }, &.{ ci("N", D), aot.cf("EPS", @floatCast(m.w.cfg.eps)), ci("BLOCK", pow2(D)) });
    }

    /// The ranks' partials summed into the residual and RMS-normed into `out` (forward._sum, ops._add_rms).
    fn sumNorm(m: *Model, s: Stream, part: u64, w: u64, out: u64, n: usize) !void {
        try m.addRms(s, try m.sum(s, part, n), w, out, n);
    }

    fn addRms(m: *Model, s: Stream, x: u64, w: u64, out: u64, n: usize) !void {
        const D = m.w.cfg.hidden;
        try m.set.run(s, "_add_rms", .{ u(n), 1, 1 }, &.{ p("X", "*fp32", x), p("RES", "*fp32", m.b.res), p("W", "*fp32", w), p("OUT", "*fp16", out) }, &.{ ci("N", D), aot.cf("EPS", @floatCast(m.w.cfg.eps)), ci("BLOCK", pow2(D)) });
    }

    /// ops.Proj on the decode path: Exl3Linear(x [n, K] fp16) -> out [n, N] (fp16, or fp32 for o / down / head).
    fn proj(m: *Model, s: Stream, l: exl3.Linear, x: u64, out: u64, out_f32: bool, n: usize) !void {
        if (n > 128) return error.DecodeRows;
        try m.k3.linear(s, l, x, .f16, m.b.xh, out, if (out_f32) .f32 else .f16, if (l.split[0] > 1) m.b.z else 0, n);
    }

    pub const Tap = enum { attn, mlp, att };
    pub const Capture = struct {
        hook: ?*const fn (ctx: *anyopaque, what: Tap, layer: usize, dev: u64, bytes: usize) anyerror!void = null,
        ctx: *anyopaque = undefined,
    };

    /// Rows' tokens, positions and slots into the fixed device inputs (graphs.Staged.refresh).
    pub fn stage(m: *Model, rows: []const Row) !void {
        const cfg = m.w.cfg;
        if (rows.len == 0 or rows.len > m.rows) return error.WindowTooWide;
        var ids: [DECODE_ROWS]i64 = undefined;
        var pos: [DECODE_ROWS]i32 = undefined;
        var slot: [DECODE_ROWS]i32 = undefined;
        for (rows, 0..) |r, i| {
            if (r.slot >= m.slots) return error.BadSlot;
            if (r.pos >= m.context) return error.PastContext;
            ids[i] = r.token;
            pos[i] = @intCast(r.pos);
            slot[i] = @intCast(r.slot);
        }
        // forward.check: a chain's rows must fit the sliding ring before its first row reads the window back
        var per: [DECODE_ROWS]usize = @splat(0);
        for (rows) |r| per[r.slot] += 1;
        for (per[0..m.slots]) |c| if (c > m.ring - cfg.window) return error.RingOverrun;
        const n = rows.len;
        try upload(m.d, m.b.ids, std.mem.sliceAsBytes(ids[0..n]));
        try upload(m.d, m.b.pos, std.mem.sliceAsBytes(pos[0..n]));
        try upload(m.d, m.b.slot, std.mem.sliceAsBytes(slot[0..n]));
    }

    /// Eager decode window: fp32 logits of every row into `out` [n, V] (Model.forward(prompt=False, rows=all)).
    pub fn forward(m: *Model, s: Stream, rows: []const Row, out: ?[]f32, cap: ?Capture) !void {
        try m.stage(rows);
        var end: usize = 0;
        for (rows) |r| end = @max(end, r.pos);
        try m.run(s, rows.len, end / CK + 1, cap);
        if (out) |o| try m.logitsTo(s, rows.len, o) else try s.synchronize();
    }

    /// graphs.Graphs.forward: a graph per (rows, key bucket), captured after a warm-up; wider chains run eagerly.
    pub fn graphed(m: *Model, s: Stream, rows: []const Row, out: ?[]f32) !void {
        var per: [DECODE_ROWS]usize = @splat(0);
        var end: usize = 0;
        for (rows) |r| {
            per[r.slot] += 1;
            end = @max(end, r.pos + 1);
        }
        if (std.mem.max(usize, per[0..m.slots]) > MAX_GRAPH_ROWS) return m.forward(s, rows, out, null);
        const full = (m.context + CK - 1) / CK * CK;
        const bucket = @min(full, @max(BUCKET, std.math.ceilPowerOfTwo(usize, end) catch unreachable));
        const key: [2]u32 = .{ @intCast(rows.len), @intCast(bucket) };
        try m.stage(rows);
        if (m.graphs.get(key)) |exec| {
            try exec.launchOn(s);
            m.replays += 1;
        } else {
            try m.run(s, rows.len, bucket / CK, null); // warm-up: writes the same cache rows the replay will
            try s.synchronize();
            try cuda.graph.beginCapture(s, .thread_local);
            m.run(s, rows.len, bucket / CK, null) catch |e| {
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

    /// attention._qkv, then attention.rows and _merge: sliding layers over the window with sinks, full from key 0.
    fn attention(m: *Model, s: Stream, i: usize, layer_sinks: u64, n: usize, chunks: usize) !void {
        const cfg = m.w.cfg;
        const H = cfg.heads;
        const hk = cfg.kvHeads(i);
        const full = cfg.full[i];
        const len = if (full) m.context else m.ring;
        try m.set.run(s, "_qkv", .{ u(n), u(H + 2 * hk), 1 }, &.{
            p("Q", "*fp16", m.b.q),                                 p("K", "*fp16", m.b.k),                                 p("V", "*fp16", m.b.v),
            p("COS", "*fp32", if (full) m.cos_full else m.cos_swa), p("SIN", "*fp32", if (full) m.sin_full else m.sin_swa), p("POS", "*i32", m.b.pos),
            p("SLOT", "*i32", m.b.slot),                            p("QO", "*fp16", m.b.qo),                               p("KC", "*fp16", m.kc[i]),
            p("VC", "*fp16", m.vc[i]),                              ii("LEN", len),
        }, &.{ ci("H", H), ci("HK", hk), ci("RD", cfg.rope_dim) });
        const win: usize = if (full) 0 else cfg.window;
        const nch: usize = if (full) chunks else (cfg.window + CK - 1) / CK + 1;
        try m.set.run(s, "_rows", .{ u(n), u(hk), u(nch) }, &.{
            p("Q", "*fp16", m.b.qo),   p("KC", "*fp16", m.kc[i]),   p("VC", "*fp16", m.vc[i]),
            p("POS", "*i32", m.b.pos), p("SLOT", "*i32", m.b.slot), p("PO", "*fp32", m.b.po),
            p("PM", "*fp32", m.b.pm),  p("PL", "*fp32", m.b.pl),    ii("LEN", len),
            ii("NCH", nch),
        }, &.{ ci("H", H), ci("HK", hk), ci("G", H / hk), ci("WIN", win), aot.cf("SCALE", m.scale), ci("CK", CK), ci("BN", BN), ci("GP", GP) });
        const sinks = if (full) 0 else layer_sinks;
        try m.set.run(s, "_merge", .{ u(n), u(H), 1 }, &.{
            p("PO", "*fp32", m.b.po),                               p("PM", "*fp32", m.b.pm),   p("PL", "*fp32", m.b.pl),
            p("SINKS", "*fp32", if (sinks != 0) sinks else m.b.pm), p("OUT", "*fp16", m.b.att), ii("NCH", nch),
        }, &.{ ci("H", H), ci("HAS_SINK", @intFromBool(sinks != 0)) });
    }

    /// The window's kernels on staged inputs, no host work; full layers read `chunks` key chunks of CK.
    pub fn run(m: *Model, s: Stream, n: usize, chunks: usize, cap: ?Capture) !void {
        const cfg = m.w.cfg;
        const D = cfg.hidden;
        const H = cfg.heads;
        if (chunks > m.nch_max) return error.TooManyChunks;
        // res = embed[ids].float()
        {
            var a: cuda.Args = .{};
            a.add(m.w.embed);
            a.add(m.b.ids);
            a.add(m.b.res);
            a.add(@as(c_int, @intCast(D)));
            try cuda.launch.launch(m.f_embed, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        }
        try m.rms(s, m.b.res, m.w.layers[0].input_norm, m.b.x, n);
        for (m.w.layers, 0..) |L, i| {
            try m.proj(s, L.q, m.b.x, m.b.q, false, n);
            try m.proj(s, L.k, m.b.x, m.b.k, false, n);
            try m.proj(s, L.v, m.b.x, m.b.v, false, n);
            if (cfg.full[i] and m.kv8 != null) try m.kv8.?.attend(m, s, i, n, chunks) else try m.attention(s, i, L.sinks, n, chunks);
            if (cap) |c| if (c.hook) |h| try h(c.ctx, .att, i, m.b.att, n * H * 128 * 2);
            try m.proj(s, L.o, m.b.att, m.b.part, true, n);
            try m.sumNorm(s, m.b.part, L.post_norm, m.b.x, n);
            if (cap) |c| if (c.hook) |h| try h(c.ctx, .attn, i, m.b.res, n * D * 4);
            if (L.experts) |e| {
                // moe.MoE: router logits (shared.router's tile by row count), top 8 (sigmoid + bias), routed experts
                const bm = routerTile(n);
                const be: usize, const bk: usize = if (bm == 16) .{ 32, 256 } else .{ 64, 64 };
                try m.set.run(s, "_router", .{ u((n + bm - 1) / bm), u((cfg.experts + be - 1) / be), 1 }, &.{
                    p("X", "*fp16", m.b.x), p("W", "*fp16", L.router), p("OUT", "*fp32", m.b.logits), ii("M", n), ii("x_stride", D),
                }, &.{ ci("D", D), ci("NE", cfg.experts), ci("BM", bm), ci("BLOCK_E", be), ci("BK", bk) });
                try m.set.run(s, "_topk", .{ u(n), 1, 1 }, &.{
                    p("L", "*fp32", m.b.logits), p("BIAS", "*fp32", L.bias), p("PICK", "*i32", m.b.pick), p("WTS", "*fp32", m.b.wts), aot.float("SCALE", L.route_scale),
                }, &.{ ci("NE", cfg.experts), ci("TOPK", cfg.top_k), ci("BLOCK", pow2(cfg.experts)), ci("KP", pow2(cfg.top_k)) });
                try m.k3.routed(s, m.b.x, false, D, m.b.pick, m.b.wts, e, m.b.sc, m.b.part, n, std.math.inf(f32), 1, true);
            } else {
                try m.proj(s, L.gate.?, m.b.x, m.b.g, false, n);
                try m.proj(s, L.up.?, m.b.x, m.b.up, false, n);
                const nd = L.gate.?.n;
                try m.set.run(s, "_silu_mul", .{ u(n), u(nd / 1024), 1 }, &.{ p("G", "*fp16", m.b.g), p("U", "*fp16", m.b.up), p("OUT", "*fp16", m.b.h) }, &.{ ci("N", nd), ci("BLOCK", 1024) });
                try m.proj(s, L.down.?, m.b.h, m.b.part, true, n);
            }
            const after = if (i + 1 < m.w.layers.len) m.w.layers[i + 1].input_norm else m.w.norm;
            try m.sumNorm(s, m.b.part, after, m.b.x, n);
            if (cap) |c| if (c.hook) |h| try h(c.ctx, .mlp, i, m.b.res, n * D * 4);
            if (m.dr) |dr| if (cap == null) try dr.tap(s, i, m.b.res, n);
        }
        if (m.dr) |dr| if (cap == null) try dr.update(s, n);
        // _head / _logits: this rank's vocab slice; gathered over the ranks into recv [world, n, Vr]
        try m.proj(s, m.w.head, m.b.x, m.b.head, true, n);
        if (m.comm) |c| try c.allGather(m.b.head, m.b.recv, n * m.w.head.n, .f32, s.handle);
    }

    /// Each row's argmax (torch.argmax over the gathered vocab) on the device; only the ids reach the host.
    pub fn argmaxTo(m: *Model, s: Stream, n: usize, out: []i32) !void {
        var a: cuda.Args = .{};
        a.add(if (m.comm != null) m.b.recv else m.b.head);
        a.add(m.b.am);
        a.add(@as(c_int, @intCast(m.w.world)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(m.w.head.n)));
        try cuda.launch.launch(m.f_argmax, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        try s.synchronize();
        try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), m.b.am, n * 4), "argmax");
    }

    /// The window's logits on the host: row r = [rank 0's slice, rank 1's slice, ...] (permute(1, 0, 2)).
    pub fn logitsTo(m: *Model, s: Stream, n: usize, out: []f32) !void {
        const vr = m.w.head.n;
        try s.synchronize();
        if (m.comm == null) return m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), m.b.head, n * vr * 4), "logits");
        for (0..n) |r| for (0..m.w.world) |k| {
            try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out[(r * m.w.world + k) * vr ..].ptr), m.b.recv + ((k * n + r) * vr) * 4, vr * 4), "logits");
        };
    }
};

/// shared._tile: rows a router program takes (16 to 128).
pub fn routerTile(n: usize) usize {
    for ([_]usize{ 16, 32, 64, 128 }) |b| if (n <= b) return b;
    return 128;
}

fn upload(d: *const cuda.Driver, at: u64, bytes: []const u8) !void {
    try d.check(d.api.cuMemcpyHtoD_v2(at, bytes.ptr, bytes.len), "upload");
}

/// torch.argmax of fp32 logits: the first index of the largest value (NaN counts as largest).
pub fn argmax(xs: []const f32) usize {
    var best: usize = 0;
    for (xs, 0..) |x, i| {
        const b = xs[best];
        if (std.math.isNan(b)) return best;
        if (std.math.isNan(x) or x > b) best = i;
    }
    return best;
}

test "ring size and argmax" {
    try std.testing.expectEqual(@as(usize, 16512), ringSize(16384, 128));
    try std.testing.expectEqual(@as(usize, 1), argmax(&.{ 1.0, 3.0, 3.0, 2.0 }));
}
