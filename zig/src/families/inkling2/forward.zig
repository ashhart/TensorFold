//! Inkling 2's decoder forward on one rank: streams' rows (prompt chunks or decode rows) to softcapped logits.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("shape.zig");
const kernels = @import("kernels.zig");
const ops = @import("ops.zig");
const experts = @import("experts.zig");
const weights = @import("pack_weights.zig");
const vision = @import("vision.zig");
const prompt_dense = @import("prompt_dense.zig");

const Buf = cuda.DeviceBuffer;

/// The kernels a forward launches: main's dense and grouped EXL3, the family's attention and glue.
pub const Modules = struct {
    dense: cuda.exl3.Dense,
    experts: experts.Kernels,
    attn: kernels.Kernels,
    ops: ops.Kernels,
    side: Side,
    prompt: ?prompt_dense.Prompt = null, // prompt rows' dense projections (Engine.init sets it once the weights load)

    pub fn load(d: *const cuda.Driver) !Modules {
        var dense = try cuda.exl3.Dense.load(d);
        errdefer dense.unload();
        var ex = try experts.Kernels.load(d);
        errdefer ex.unload();
        var attn = try kernels.Kernels.load(d);
        errdefer attn.unload();
        var o = try ops.Kernels.load(d);
        errdefer o.unload();
        return .{ .dense = dense, .experts = ex, .attn = attn, .ops = o, .side = try Side.init(d) };
    }

    pub fn deinit(m: *Modules) void {
        if (m.prompt) |*p| p.deinit();
        m.side.deinit();
        m.ops.unload();
        m.attn.unload();
        m.experts.unload();
        m.dense.unload();
    }
};

/// A layer's narrow projections (r, k, v, row dots) run here beside q and the gate, on their own EXL3 scratch.
pub const Side = struct {
    stream: cuda.Stream,
    fork: cuda.Event,
    join: cuda.Event,
    dense: cuda.exl3.Dense,

    fn init(d: *const cuda.Driver) !Side {
        var st = try cuda.Stream.init(d, true);
        errdefer st.deinit();
        var fork = try cuda.Event.init(d, false);
        errdefer fork.deinit();
        var join = try cuda.Event.init(d, false);
        errdefer join.deinit();
        return .{ .stream = st, .fork = fork, .join = join, .dense = try cuda.exl3.Dense.load(d) };
    }

    fn deinit(x: *Side) void {
        x.dense.unload();
        x.join.deinit();
        x.fork.deinit();
        x.stream.deinit();
    }
};

/// The geometry a stream's caches and a call's scratch are sized for.
pub const Shape = struct {
    rows: usize, // most rows a call takes
    context: usize, // most positions a stream reaches
    heads: usize, // the rank's query heads
    kv: usize, // the rank's key heads
    ring_rows: usize = 0, // rows a call may write into a sliding ring (0: rows)

    pub fn sliding(s: Shape, c: *const cfg.Config) kernels.Layer {
        return kernels.Layer.sliding(c.window, @intCast(if (s.ring_rows > 0) s.ring_rows else s.rows));
    }

    pub fn global(s: Shape) kernels.Layer {
        return kernels.Layer.global(@intCast(s.context));
    }

    fn layer(s: Shape, c: *const cfg.Config, i: usize) kernels.Layer {
        return if (c.layer[i] == .sliding) s.sliding(c) else s.global();
    }
};

/// Partials a (row, head) needs for a call whose rows reach `last_pos`: a sliding window's, or the global chunks'.
pub fn stride(c: *const cfg.Config, last_pos: usize) u32 {
    const sl = kernels.Layer.sliding(c.window, 1);
    return @max(kernels.slotsFor(sl, 1), kernels.chunks(last_pos, kernels.global_chunk));
}

/// Bytes of one cache slot of a stream: K and V rings a layer plus four conv states (the per-stream cost).
pub fn slotBytes(c: *const cfg.Config, s: Shape) u64 {
    var total: u64 = 0;
    for (0..c.layers) |i| {
        total += 2 * @as(u64, s.layer(c, i).cacheBytes(s.kv));
        total += 2 * 3 * @as(u64, s.kv) * c.head_dim * 4 + 2 * 3 * @as(u64, c.hidden) * 4;
    }
    return total;
}

fn alloc(d: *const cuda.Driver, list: *std.ArrayList(Buf), gpa: std.mem.Allocator, bytes: usize) !u64 {
    var b = try Buf.alloc(d, @max(bytes, 256));
    errdefer b.free();
    try b.fill8(0, null);
    try list.append(gpa, b);
    return b.ptr;
}

/// `slots` streams' caches and conv states: K and V rings a layer, the last three inputs of each short convolution.
pub const State = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(Buf) = .empty,
    kc: []u64 = &.{},
    vc: []u64 = &.{},
    conv: [][4]u64 = &.{}, // k, v (fp32 [slots][3][kv * 128]) and attention, MLP (fp32 [slots][3][hidden]) a layer
    pos: []usize = &.{}, // positions each slot's caches hold
    ring: [2]usize = .{ 0, 0 }, // sliding and global ring slots a key head
    kv: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: *const cfg.Config, s: Shape, slots: usize) !State {
        var st: State = .{ .gpa = gpa, .kv = s.kv };
        errdefer st.deinit();
        st.kc = try gpa.alloc(u64, c.layers);
        st.vc = try gpa.alloc(u64, c.layers);
        st.conv = try gpa.alloc([4]u64, c.layers);
        st.pos = try gpa.alloc(usize, slots);
        @memset(st.pos, 0);
        st.ring = .{ s.sliding(c).ring, s.global().ring };
        for (0..c.layers) |i| {
            const l = s.layer(c, i);
            st.kc[i] = try alloc(d, &st.bufs, gpa, slots * l.cacheBytes(s.kv));
            st.vc[i] = try alloc(d, &st.bufs, gpa, slots * l.cacheBytes(s.kv));
            for (0..2) |j| st.conv[i][j] = try alloc(d, &st.bufs, gpa, slots * 3 * s.kv * c.head_dim * 4);
            for (2..4) |j| st.conv[i][j] = try alloc(d, &st.bufs, gpa, slots * 3 * @as(usize, c.hidden) * 4);
        }
        return st;
    }

    pub fn cache(st: *const State, i: usize) kernels.Cache {
        return .{ .k = st.kc[i], .v = st.vc[i] };
    }

    /// Back to empty streams: caches and conv states zeroed.
    pub fn reset(st: *State) !void {
        for (st.bufs.items) |b| try b.fill8(0, null);
        @memset(st.pos, 0);
    }

    /// One slot's conv states zeroed on `s` (a fresh pass; its rings are only read at positions it writes).
    pub fn clearSlot(st: *State, c: *const cfg.Config, d: *const cuda.Driver, slot: usize, s: cuda.Stream) !void {
        for (st.conv) |cv| for (cv, 0..) |base, j| {
            const n = 3 * (if (j < 2) st.kv * c.head_dim else @as(usize, c.hidden)) * 4;
            const view: Buf = .{ .d = d, .ptr = base + slot * n, .len = n, .borrowed = true };
            try view.fill8(0, s.handle);
        };
        st.pos[slot] = 0;
    }

    pub fn deinit(st: *State) void {
        for (st.bufs.items) |*b| b.free();
        st.bufs.deinit(st.gpa);
        st.gpa.free(st.kc);
        st.gpa.free(st.vc);
        st.gpa.free(st.conv);
        st.gpa.free(st.pos);
    }
};

/// Rows a grouped-experts call takes: its grouping kernel holds rows x 8 picks in 64 KiB of shared memory.
const expert_rows = 2048;

/// Most streams one call takes.
pub const max_items = 64;

/// One stream's rows in a call: its slot and the ids at its next positions.
pub const Item = struct { slot: usize, ids: []const i32 };

/// Attention partials a call's rows write: `slots` (row, head, chunk) entries, fp32 sums, each chunk's max and total.
pub const Partials = struct { o: u64, ml: u64, slots: usize };

/// A call's scratch, sized for Shape.rows rows; the partials may be another Buffers' (one stream runs them in order).
pub const Buffers = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(Buf) = .empty,
    ids: Buf = undefined, // int32 rows, then pos, seg_of, the segments and the tiles
    x: u64 = 0, // bf16 [rows][hidden]: the residual stream between layers
    resid: u64 = 0, // fp32 [rows][hidden]: the residual inside a layer
    blocks: u64 = 0, // bf16 [8][rows][hidden]: the completed blocks' inputs of these rows
    normed: u64 = 0, // bf16 [3][rows][hidden]
    q: u64 = 0,
    r: u64 = 0,
    gate: u64 = 0,
    kf: u64 = 0,
    vf: u64 = 0,
    kraw: u64 = 0,
    vraw: u64 = 0,
    dk: u64 = 0,
    dv: u64 = 0,
    sink: u64 = 0,
    qo: u64 = 0,
    relo: u64 = 0,
    tab: u64 = 0,
    parts: Partials = .{ .o = 0, .ml = 0, .slots = 0 },
    att: u64 = 0,
    ao: u64 = 0,
    scores: u64 = 0,
    pick: u64 = 0,
    wts: u64 = 0,
    moe: u64 = 0,
    gathered: u64 = 0, // fp32 [world][rows][hidden]: every rank's partial of a site (world > 1)
    exs: cuda.exl3_experts.Scratch = undefined,
    logits: Buf = undefined, // fp32 [rows][padded vocab of the rank]

    /// Scratch for `s.rows` rows; `shared` partials, else its own for every row at the shape's context.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, s: Shape, shared: ?Partials) !Buffers {
        const c = &w.config;
        var b: Buffers = .{ .gpa = gpa };
        errdefer b.deinit();
        const n = s.rows;
        const h: usize = c.hidden;
        const hd: usize = c.head_dim;
        const A = struct {
            fn f(bb: *Buffers, dd: *const cuda.Driver, bytes: usize) !u64 {
                return alloc(dd, &bb.bufs, bb.gpa, bytes);
            }
        }.f;
        b.ids = try Buf.alloc(d, stageBytes(n, max_items, kernels.tileCount(n, c.group()) + max_items));
        b.x = try A(&b, d, n * h * 2);
        b.resid = try A(&b, d, n * h * 4);
        b.blocks = try A(&b, d, ops.max_blocks * n * h * 2);
        b.normed = try A(&b, d, 3 * n * h * 2);
        b.q = try A(&b, d, n * s.heads * hd * 2);
        b.r = try A(&b, d, n * s.heads * c.d_rel * 2);
        b.gate = try A(&b, d, n * s.heads * hd * 2);
        b.kf = try A(&b, d, n * s.kv * hd * 4);
        b.vf = try A(&b, d, n * s.kv * hd * 4);
        b.kraw = try A(&b, d, n * s.kv * hd * 2);
        b.vraw = try A(&b, d, n * s.kv * hd * 2);
        b.dk = try A(&b, d, n * s.kv * c.conv * 4);
        b.dv = try A(&b, d, n * s.kv * c.conv * 4);
        b.sink = try A(&b, d, n * s.heads * 2);
        b.qo = try A(&b, d, n * s.heads * hd * 2);
        b.relo = try A(&b, d, n * s.heads * c.d_rel * 2);
        b.tab = try A(&b, d, kernels.tableBytes(n, @intCast(s.heads)));
        b.parts = shared orelse blk: {
            const slots = n * s.heads * stride(c, s.context - 1);
            break :blk .{ .o = try A(&b, d, slots * hd * 4), .ml = try A(&b, d, slots * 8), .slots = slots };
        };
        b.att = try A(&b, d, n * s.heads * hd * 2);
        b.ao = try A(&b, d, n * h * 4);
        b.scores = try A(&b, d, n * @as(usize, c.experts) * 4);
        b.pick = try A(&b, d, n * c.topk * 4);
        b.wts = try A(&b, d, n * c.topk * 4);
        b.moe = try A(&b, d, n * h * 4);
        if (w.share.world > 1) b.gathered = try A(&b, d, w.share.world * n * h * 4);
        const el = w.layers[0].experts;
        b.exs = try cuda.exl3_experts.scratch(try A(&b, d, try experts.scratchBytes(el, @min(n, expert_rows), c.topk)), el, @min(n, expert_rows), c.topk);
        b.logits = try Buf.alloc(d, n * w.share.vocab.len() * 4);
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        for (b.bufs.items) |*x| x.free();
        b.bufs.deinit(b.gpa);
        if (b.ids.len != 0) b.ids.free();
        if (b.logits.len != 0) b.logits.free();
    }
};

/// Partials for `rows` rows of `heads` heads at `context`, to share between a rank's Buffers.
pub fn partialsFor(d: *const cuda.Driver, list: *std.ArrayList(Buf), gpa: std.mem.Allocator, slots: usize) !Partials {
    return .{ .o = try alloc(d, list, gpa, slots * kernels.head_dim * 4), .ml = try alloc(d, list, gpa, slots * 8), .slots = slots };
}

/// How the ranks' partials (after o_proj and the experts) meet: one rank alone, or gathered in rank order.
pub const Exchange = union(enum) {
    local,
    nccl: struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, group: ?*const cuda.tp_group.Group = null },
};

/// A conv input made of `parts` fp32 rank partials, `stride` elements apart.
const Parts = struct { x: u64, parts: i32, stride: i64 };

/// Where a call's staged words sit in Buffers.ids: ids, pos, seg_of (int32 a row), then segments and tiles.
const Layout = struct { segs: [2]usize, conv: usize, tiles: usize, end: usize };

fn layout(n: usize, ni: usize, nt: usize) Layout {
    const at0 = std.mem.alignForward(usize, n * 12, 16);
    const seg = ni * @sizeOf(kernels.Seg);
    return .{ .segs = .{ at0, at0 + seg }, .conv = at0 + 2 * seg, .tiles = at0 + 2 * seg + ni * 16, .end = at0 + 2 * seg + ni * 16 + nt * @sizeOf(kernels.Tile) };
}

fn stageBytes(n: usize, ni: usize, nt: usize) usize {
    return layout(n, ni, nt).end + 256;
}

/// A prompt call's image rows (bf16 [tokens][hidden]) and spans; its rows stand at prompt positions lo, lo + 1, ...
pub const Images = struct { rows: u64, spans: []const vision.Span, lo: usize };

/// A staged call: its rows, streams, tiles, the latest row's position and the partials' stride.
pub const Plan = struct { n: usize, ni: usize, nt: usize, last: usize, stride: u32 };

pub const Forward = struct {
    d: *const cuda.Driver,
    m: *Modules,
    w: *const weights.Weights,
    b: *Buffers,
    shape: Shape,
    stream: cuda.Stream,
    exchange: Exchange = .local,
    images: ?Images = null, // a prompt call's pictures: its one stream's rows from prompt position `lo`
    last_logits: bool = false, // the head on the call's last row only (a prompt chunk draws from it alone)

    /// Where a site's fp32 partial of `n` rows goes: this rank's slot of the gathered buffer, or `own` at one rank.
    fn slot(f: Forward, own: u64, n: usize) u64 {
        if (f.w.share.world == 1) return own;
        return f.b.gathered + f.w.share.rank * n * f.w.config.hidden * 4;
    }

    /// Every rank's partial into every rank's gathered buffer (RoCE writes when they fit); convs add them in order.
    fn gather(f: Forward, n: usize) !Parts {
        const h: usize = f.w.config.hidden;
        switch (f.exchange) {
            .local => return error.Inkling2NoExchange,
            .nccl => |x| if (x.group) |g| try g.allGather(f.stream, f.slot(0, n), f.b.gathered, n * h * 4) else try x.lib.check(x.lib.api.ncclAllGather(f.slot(0, n), f.b.gathered, n * h, .f32, x.comm, f.stream.handle), "ncclAllGather"),
        }
        return .{ .x = f.b.gathered, .parts = @intCast(f.w.share.world), .stride = @intCast(n * h) };
    }

    /// The conv input of a partial site: the local buffer at one rank, else the gathered parts.
    fn summed(f: Forward, own: u64, n: usize) !Parts {
        if (f.w.share.world == 1) return .{ .x = own, .parts = 1, .stride = 0 };
        return f.gather(n);
    }

    fn dense(f: Forward, wt: cuda.exl3.Weight, x: u64, rows: usize, y: u64, dt: cuda.exl3.DType) !void {
        try f.proj(f.stream, &f.m.dense, wt, x, rows, y, dt);
    }

    /// A projection of bf16 rows: prompt rows by the GEMM where it takes the weight, else the decode kernels.
    fn proj(f: Forward, s: cuda.Stream, de: *cuda.exl3.Dense, wt: cuda.exl3.Weight, x: u64, rows: usize, y: u64, dt: cuda.exl3.DType) !void {
        if (f.m.prompt) |*p| if (dt != .f16 and p.takes(wt, rows)) return p.run(s, wt, x, rows, y, dt == .f32);
        try de.decode(s, wt, x, .bf16, rows, y, dt);
    }

    /// Each item's ids at its stream's next positions through every layer; the rows' logits land in b.logits.
    pub fn run(f: Forward, st: *State, items: []const Item) !void {
        const p = try f.stage(st, items);
        try f.launch(st, p);
        advance(st, items);
    }

    /// The items' streams stand past their rows.
    pub fn advance(st: *State, items: []const Item) void {
        for (items) |it| st.pos[it.slot] += it.ids.len;
    }

    /// The call's ids, positions, segments and tiles uploaded (after the stream drains); launch runs on them.
    pub fn stage(f: Forward, st: *const State, items: []const Item) !Plan {
        const c = &f.w.config;
        var n: usize = 0;
        for (items) |it| {
            if (it.ids.len == 0 or st.pos[it.slot] + it.ids.len > f.shape.context) return error.Inkling2Rows;
            n += it.ids.len;
        }
        if (n == 0 or n > f.shape.rows or items.len > max_items) return error.Inkling2Rows;
        const gpa = f.w.gpa;
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(gpa);
        for (items) |it| try host.appendSlice(gpa, std.mem.sliceAsBytes(it.ids));
        for (items) |it| for (0..it.ids.len) |i| try host.appendSlice(gpa, std.mem.asBytes(&@as(i32, @intCast(st.pos[it.slot] + i))));
        for (items, 0..) |it, j| for (0..it.ids.len) |_| try host.appendSlice(gpa, std.mem.asBytes(&@as(i32, @intCast(j))));
        var segs: [2][max_items]kernels.Seg = undefined;
        var cs: [max_items][4]i32 = undefined;
        var last: usize = 0;
        var row0: usize = 0;
        const per = [2]usize{ f.shape.kv * st.ring[0] * c.head_dim, f.shape.kv * st.ring[1] * c.head_dim };
        for (items, 0..) |it, j| {
            const p0 = st.pos[it.slot];
            for (0..2) |r| segs[r][j] = .{ .row0 = @intCast(row0), .n = @intCast(it.ids.len), .p0 = @intCast(p0), .off = @intCast(it.slot * per[r]) };
            cs[j] = .{ @intCast(row0), @intCast(it.ids.len), @intCast(it.slot), 0 };
            last = @max(last, p0 + it.ids.len - 1);
            row0 += it.ids.len;
        }
        const ni = items.len;
        var tl: [8192]kernels.Tile = undefined;
        const nt = try kernels.tiles(segs[0][0..ni], c.group(), &tl);
        const lay = layout(n, ni, nt);
        try host.appendNTimes(gpa, 0, lay.segs[0] - host.items.len);
        try host.appendSlice(gpa, std.mem.sliceAsBytes(segs[0][0..ni]));
        try host.appendSlice(gpa, std.mem.sliceAsBytes(segs[1][0..ni]));
        try host.appendSlice(gpa, std.mem.sliceAsBytes(cs[0..ni]));
        try host.appendSlice(gpa, std.mem.sliceAsBytes(tl[0..nt]));
        std.debug.assert(host.items.len == lay.end);
        const p: Plan = .{ .n = n, .ni = ni, .nt = nt, .last = last, .stride = stride(c, last) };
        if (n * f.shape.heads * p.stride > f.b.parts.slots) return error.Inkling2Partials;
        try f.stream.synchronize(); // the previous call's kernels may still read the staged rows
        try f.b.ids.upload(0, host.items);
        return p;
    }

    /// A staged call's kernels only (no host work): a graph captures exactly this.
    pub fn launch(f: Forward, st: *const State, p: Plan) !void {
        const c = &f.w.config;
        const n = p.n;
        const h: usize = c.hidden;
        const s = f.stream;
        const k = &f.m.ops;
        const b = f.b;
        const base = b.ids.ptr;
        const lay = layout(n, p.ni, p.nt);
        var rows_by: [2]kernels.Rows = undefined;
        for (&rows_by, lay.segs) |*r, at| r.* = .{ .rows = @intCast(n), .heads = @intCast(f.shape.heads), .kv_heads = @intCast(f.shape.kv), .pos = base + n * 4, .seg_of = base + n * 8, .segs = base + at, .tiles = base + lay.tiles, .n_tiles = @intCast(p.nt), .last_pos = @intCast(p.last), .streams = @intCast(p.ni) };
        const conv_segs = base + lay.conv;
        const streams = p.ni;
        try ops.embed(k, s, base, n, f.w.embed, f.w.embed_norm, b.x, h, c.eps);
        if (f.images) |im| try vision.splice(f.d, s, c.hidden, im.rows, im.spans, im.lo, im.lo + n, b.x);
        var nb: usize = 0;
        const hd: usize = c.head_dim;
        const kvw = f.shape.kv * hd;
        for (f.w.layers, 0..) |*l, i| {
            const global = l.kind == .global;
            var mix: ops.MixArgs = .{ .partial = b.x, .query = l.attn_query, .out = b.normed, .n_blocks = @intCast(nb), .partial_bf16 = 1, .nq = @intCast(l.nq), .rows = @intCast(n), .c = @intCast(h), .eps = c.eps };
            for (0..nb) |j| mix.blocks[j] = b.blocks + j * n * h * 2;
            mix.norm = l.norms;
            try ops.mix(k, s, mix);
            const xq = b.normed;
            const xk = if (global) b.normed + n * h * 2 else xq;
            const xv = if (global) b.normed + 2 * n * h * 2 else xq;
            const a = &l.attn;
            const sd = &f.m.side;
            // prompt rows run the narrow projections on the main stream: one GEMM scratch, and they fill the GPU alone
            const ss = if (f.m.prompt != null and n >= prompt_dense.min_rows) s else sd.stream;
            try sd.fork.record(s);
            try ss.wait(sd.fork);
            try f.proj(ss, &sd.dense, a.r, xq, n, b.r, .bf16);
            try f.proj(ss, &sd.dense, a.k, xk, n, b.kf, .f32);
            try f.proj(ss, &sd.dense, a.v, xv, n, b.vf, .f32);
            try ops.rowdot(k, ss, xq, a.sink, b.sink, n, f.shape.heads, h, true, false);
            if (global) {
                try ops.rowdot(k, ss, xk, a.k_proj, b.dk, n, f.shape.kv * c.conv, h, false, false);
                try ops.rowdot(k, ss, xv, a.v_proj, b.dv, n, f.shape.kv * c.conv, h, false, false);
            }
            try sd.join.record(ss);
            try f.dense(a.q, xq, n, b.q, .bf16);
            try f.dense(a.gate, xq, n, b.gate, .bf16);
            try s.wait(sd.join);
            const cv = st.conv[i];
            try ops.conv(k, s, .{ .x = b.kf, .state = cv[0], .w = a.k_conv, .dyn = if (global) b.dk else 0, .heads = @intCast(f.shape.kv), .c = @intCast(kvw), .rows = @intCast(n), .mode = 0, .out = b.kraw, .segs = conv_segs, .state_stride = @intCast(3 * kvw) }, streams);
            try ops.conv(k, s, .{ .x = b.vf, .state = cv[1], .w = a.v_conv, .dyn = if (global) b.dv else 0, .heads = @intCast(f.shape.kv), .c = @intCast(kvw), .rows = @intCast(n), .mode = 0, .out = b.vraw, .segs = conv_segs, .state_stride = @intCast(3 * kvw) }, streams);
            var lay_i = f.shape.layer(c, i);
            lay_i.ring = @intCast(st.ring[@intFromBool(global)]); // the caches' rings, whatever rows this call's scratch holds
            const rows = rows_by[@intFromBool(global)];
            const prep: kernels.Prep = .{ .qk_clip = c.qk_clip, .rel_clip = c.rel_clip, .floor = c.log_floor, .alpha = c.log_alpha, .eps = c.eps };
            try kernels.prep(&f.m.attn, s, rows, lay_i, prep, .{ b.q, b.kraw, b.vraw, b.r }, .{ a.q_scale, a.rel_scale }, b.qo, b.relo, st.cache(i));
            try kernels.relTable(&f.m.attn, s, rows, b.relo, a.rel_proj, b.tab);
            try kernels.attend(&f.m.attn, s, rows, lay_i, p.stride, b.parts.slots, b.qo, st.cache(i), b.tab, b.parts.o, b.parts.ml, b.sink, b.gate, b.att);
            try f.dense(a.o, b.att, n, f.slot(b.ao, n), .f32);
            const ao = try f.summed(b.ao, n);
            const boundary = i % c.block_layers == 0;
            if (boundary) {
                if (nb == ops.max_blocks) return error.Inkling2Blocks;
                try copy(f.d, b.blocks + nb * n * h * 2, b.x, n * h * 2, s);
                nb += 1;
            }
            try ops.conv(k, s, .{ .x = ao.x, .parts = ao.parts, .part_stride = ao.stride, .state = cv[2], .w = l.attn_conv, .base = if (boundary) 0 else b.x, .resid = b.resid, .round_in = 1, .c = @intCast(h), .rows = @intCast(n), .mode = 1, .segs = conv_segs, .state_stride = @intCast(3 * h) }, streams);
            var mm: ops.MixArgs = .{ .partial = b.resid, .query = l.mlp_query, .out = b.normed, .n_blocks = @intCast(nb), .partial_bf16 = 0, .nq = 1, .rows = @intCast(n), .c = @intCast(h), .eps = c.eps };
            for (0..nb) |j| mm.blocks[j] = b.blocks + j * n * h * 2;
            mm.norm[0] = l.mlp_norm;
            try ops.mix(k, s, mm);
            try ops.rowdot(k, s, b.normed, l.router, b.scores, n, c.experts, h, false, true);
            try ops.topk(k, s, b.scores, l.bias, b.pick, b.wts, n, c.experts, c.topk, c.route_scale);
            const out = f.slot(b.moe, n);
            var r0: usize = 0;
            while (r0 < n) : (r0 += expert_rows) {
                const w = @min(expert_rows, n - r0);
                const caps: experts.Caps = .{ .gate = c.gate_softcap, .up = c.up_softcap };
                const sel = r0 * c.topk * 4;
                try experts.run(&f.m.experts, s, l.experts, caps, l.bits, b.normed + r0 * h * 2, h, .bf16, b.pick + sel, b.wts + sel, w, out + r0 * h * 4, b.exs);
            }
            const mo = try f.summed(b.moe, n);
            try ops.conv(k, s, .{ .x = mo.x, .parts = mo.parts, .part_stride = mo.stride, .state = cv[3], .w = l.mlp_conv, .resid = b.resid, .out = b.x, .round_in = 1, .c = @intCast(h), .rows = @intCast(n), .mode = 2, .segs = conv_segs, .state_stride = @intCast(3 * h) }, streams);
        }
        var fm: ops.MixArgs = .{ .partial = b.x, .query = f.w.out_query, .out = b.normed, .n_blocks = @intCast(nb), .partial_bf16 = 1, .nq = 1, .rows = @intCast(n), .c = @intCast(h), .eps = c.eps };
        for (0..nb) |j| fm.blocks[j] = b.blocks + j * n * h * 2;
        fm.norm[0] = f.w.norm;
        try ops.mix(k, s, fm);
        const nv = f.w.share.vocab.len();
        const hr: usize = if (f.last_logits) n - 1 else 0; // rows before hr keep no logits
        try f.dense(f.w.head, b.normed + hr * h * 2, n - hr, b.logits.ptr + hr * nv * 4, .f32);
        try ops.softcap(k, s, b.logits.ptr + hr * nv * 4, (n - hr) * nv, c.logit_softcap);
    }
};

fn copy(d: *const cuda.Driver, dst: u64, src: u64, bytes: usize, s: cuda.Stream) !void {
    const view: Buf = .{ .d = d, .ptr = dst, .len = bytes, .borrowed = true };
    try view.copyFrom(0, src, bytes, s.handle);
}

test "forward: a staged call's words sit where launch reads them" {
    const l = layout(3, 2, 2);
    try std.testing.expectEqual(@as(usize, 48), l.segs[0]);
    try std.testing.expectEqual(@as(usize, 48 + 2 * 2 * 24), l.conv); // two segment arrays (sliding, global)
    try std.testing.expectEqual(@as(usize, 48 + 2 * 2 * 24 + 2 * 16), l.tiles);
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(kernels.Seg));
    const c: cfg.Config = .{};
    try std.testing.expectEqual(@as(u32, 5), stride(&c, 100));
    try std.testing.expectEqual(@as(u32, 64), stride(&c, 1048575));
}
