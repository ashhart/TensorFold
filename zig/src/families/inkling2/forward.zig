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
const state = @import("state.zig");
const loop = @import("loop_ops.zig");
const visits = @import("visits.zig");
const pair = @import("pair.zig");

const Buf = cuda.DeviceBuffer;

/// The kernels a forward launches: main's dense and grouped EXL3, the family's attention and glue.
pub const Modules = struct {
    dense: cuda.exl3.Dense,
    experts: experts.Kernels,
    attn: kernels.Kernels,
    ops: ops.Kernels,
    side: Side,
    loop: loop.Kernels,
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
        var lk = try loop.Kernels.load(d);
        errdefer lk.unload();
        return .{ .dense = dense, .experts = ex, .attn = attn, .ops = o, .loop = lk, .side = try Side.init(d) };
    }

    pub fn deinit(m: *Modules) void {
        if (m.prompt) |*p| p.deinit();
        m.side.deinit();
        m.loop.unload();
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

    /// Cache slot `i`'s attention geometry (its layer's kind).
    pub fn layer(s: Shape, c: *const cfg.Config, i: usize) kernels.Layer {
        return if (c.slotKind(i) == .sliding) s.sliding(c) else s.global();
    }
};

/// Partials a (row, head) needs for a call whose rows reach `last_pos`: a sliding window's, or the global chunks'.
pub fn stride(c: *const cfg.Config, last_pos: usize) u32 {
    const sl = kernels.Layer.sliding(c.window, 1);
    return @max(kernels.slotsFor(sl, 1), kernels.chunks(last_pos, kernels.global_chunk));
}

pub const State = state.State;
pub const slotBytes = state.slotBytes;
const alloc = state.alloc;

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
    ids: Buf, // int32 rows, then pos, seg_of, the segments and the tiles
    x: u64 = 0, // bf16 [rows][hidden]: the residual stream between layers
    resid: u64 = 0, // fp32 [rows][hidden]: the residual inside a layer
    blocks: u64 = 0, // bf16 [8][rows][hidden]: the completed blocks' inputs of these rows
    normed: u64 = 0, // bf16 [3][rows][hidden]
    q: u64 = 0,
    r: u64 = 0,
    rw: u64 = 0, // bf16 [rows][all heads x d_rel]: r held whole on a rank whose rel columns are not 128-aligned
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
    logits: Buf, // fp32 [rows][padded vocab of the rank]
    lp: loop.Scratch = .{}, // Mini's loop: the banks, mixes and gates of these rows (rows 0 without a loop)

    /// Scratch for `s.rows` rows; `shared` partials, else its own for every row at the shape's context.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, s: Shape, shared: ?Partials) !Buffers {
        const c = &w.config;
        var b: Buffers = .{ .gpa = gpa, .ids = .{ .d = d, .ptr = 0, .len = 0 }, .logits = .{ .d = d, .ptr = 0, .len = 0 } };
        errdefer b.deinit(); // ids and logits start empty, so a failure before either is made frees neither
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
        if (w.layers[0].attn.r_lo != null) b.rw = try A(&b, d, n * @as(usize, c.heads) * c.d_rel * 2);
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
        if (c.loop != null) {
            const Ctx = struct {
                b: *Buffers,
                d: *const cuda.Driver,
                fn f(x: @This(), bytes: usize) !u64 {
                    return alloc(x.d, &x.b.bufs, x.b.gpa, bytes);
                }
            };
            b.lp = try loop.Scratch.init(n, w.plan.bankSources(), Ctx{ .b = &b, .d = d }, Ctx.f);
        }
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
    pair: struct { p: *pair.Pair, rank: usize }, // two virtual ranks on one GPU (test tools only)
    serial, // the serial reference: both ranks' partials written on this GPU, rank 0's then rank 1's (test tools only)
};

/// The serial reference's rank 1 (test tools only): its cut, its caches and its logits rows, run after rank 0's.
pub const Peer = struct { w: *const weights.Weights, st: *State, logits: u64 };

/// The rank whose partials a site computes: its weights and caches.
const Cut = struct { w: *const weights.Weights, st: *const State };

/// Where rank `rank`'s fp32 partial of `n` rows of `h` sits in the gathered buffer: the convs read part q at q.
pub fn partAt(gathered: u64, rank: usize, n: usize, h: usize) u64 {
    return gathered + rank * n * h * 4;
}

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
    zero_messages: bool = false, // a planted control (test tools only): the banks stay zero, so every message is 0
    peer: ?Peer = null, // the serial reference: rank 1's cut, run after this rank's at every partial site

    /// Where a site's fp32 partial of `n` rows goes: the rank's slot of the gathered buffer, or `own` at one rank.
    fn slot(f: Forward, w: *const weights.Weights, own: u64, n: usize) u64 {
        if (w.share.world == 1) return own;
        return partAt(f.b.gathered, w.share.rank, n, w.config.hidden);
    }

    /// Every rank's partial into every rank's gathered buffer (RoCE writes when they fit); convs add them in order.
    fn gather(f: Forward, n: usize) !Parts {
        const h: usize = f.w.config.hidden;
        switch (f.exchange) {
            .local => return error.Inkling2NoExchange,
            .nccl => |x| if (x.group) |g| try g.allGather(f.stream, f.slot(f.w, 0, n), f.b.gathered, n * h * 4) else try x.lib.check(x.lib.api.ncclAllGather(f.slot(f.w, 0, n), f.b.gathered, n * h, .f32, x.comm, f.stream.handle), "ncclAllGather"),
            .pair => |x| try x.p.allGather(x.rank, f.stream, f.b.gathered, n * h * 4),
            .serial => {}, // both parts already written, rank 0's then rank 1's
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
        if (f.peer) |q| advance(q.st, items);
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
        const plan = &f.w.plan;
        const passes: u32 = if (c.loop) |lp| lp.passes else 1;
        const sources = plan.bankSources();
        var blocks: [ops.max_blocks]u64 = undefined;
        for (&blocks, 0..) |*x, j| x.* = b.blocks + j * n * h * 2;
        for (plan.visits[0..plan.n]) |*v| {
            const i = v.layer;
            const sl = v.slot; // the visit's caches and conv states
            const l = &f.w.layers[i];
            if (v.restart) { // a later pass starts again from the prelude's hidden state, its first core block
                nb = plan.prelude_blocks;
                try copy(f.d, b.x, blocks[nb], n * h * 2, s);
            }
            const global = l.kind == .global;
            const bank_in = b.lp.bank[(v.pass + 1) % 2]; // the previous pass's bank (pass p writes bank p % 2)
            if (v.messages) {
                const m: loop.Mix = .{ .blocks = blocks[0..nb], .partial = b.x, .partial_bf16 = true, .query = l.attn_query, .nq = l.nq };
                try loop.attention(&f.m.loop, s, &b.lp, l.loop.?, m, bank_in, sources, l.norms, n, c.eps, b.normed);
            } else {
                var mix: ops.MixArgs = .{ .partial = b.x, .query = l.attn_query, .out = b.normed, .n_blocks = @intCast(nb), .partial_bf16 = 1, .nq = @intCast(l.nq), .rows = @intCast(n), .c = @intCast(h), .eps = c.eps };
                for (0..nb) |j| mix.blocks[j] = blocks[j];
                mix.norm = l.norms;
                try ops.mix(k, s, mix);
            }
            const apart = global or v.messages; // q, k and v inputs differ: three normed rows
            const xs = [3]u64{ b.normed, if (apart) b.normed + n * h * 2 else b.normed, if (apart) b.normed + 2 * n * h * 2 else b.normed };
            try f.attention(.{ .w = f.w, .st = st }, v, xs, rows_by, conv_segs, p);
            if (f.peer) |q| try f.attention(.{ .w = q.w, .st = q.st }, v, xs, rows_by, conv_segs, p);
            const cv = st.conv[sl];
            const ao = try f.summed(b.ao, n);
            const boundary = i % c.block_layers == 0;
            if (boundary) {
                if (nb == ops.max_blocks) return error.Inkling2Blocks;
                try copy(f.d, blocks[nb], b.x, n * h * 2, s);
                nb += 1;
                if (visits.savesBlock(plan, v, passes) and !f.zero_messages) try loop.save(&f.m.loop, s, b.x, b.lp.bank[v.pass % 2], sources, visits.bankIndex(plan, i), n);
            }
            try ops.conv(k, s, .{ .x = ao.x, .parts = ao.parts, .part_stride = ao.stride, .state = cv[2], .w = l.attn_conv, .base = if (boundary) 0 else b.x, .resid = b.resid, .round_in = 1, .c = @intCast(h), .rows = @intCast(n), .mode = 1, .segs = conv_segs, .state_stride = @intCast(3 * h) }, streams);
            if (v.messages) {
                const m: loop.Mix = .{ .blocks = blocks[0..nb], .partial = b.resid, .partial_bf16 = false, .query = l.mlp_query, .nq = 1 };
                try loop.mlp(&f.m.loop, s, &b.lp, l.loop.?, m, bank_in, sources, l.mlp_norm, n, c.eps, b.normed);
            } else {
                var mm: ops.MixArgs = .{ .partial = b.resid, .query = l.mlp_query, .out = b.normed, .n_blocks = @intCast(nb), .partial_bf16 = 0, .nq = 1, .rows = @intCast(n), .c = @intCast(h), .eps = c.eps };
                for (0..nb) |j| mm.blocks[j] = blocks[j];
                mm.norm[0] = l.mlp_norm;
                try ops.mix(k, s, mm);
            }
            try ops.rowdot(k, s, b.normed, l.router, b.scores, n, c.experts, h, false, true);
            try ops.topk(k, s, b.scores, l.bias, b.pick, b.wts, n, c.experts, c.topk, c.route_scale);
            try f.moe(f.w, i, n);
            if (f.peer) |q| try f.moe(q.w, i, n);
            const mo = try f.summed(b.moe, n);
            try ops.conv(k, s, .{ .x = mo.x, .parts = mo.parts, .part_stride = mo.stride, .state = cv[3], .w = l.mlp_conv, .resid = b.resid, .out = b.x, .round_in = 1, .c = @intCast(h), .rows = @intCast(n), .mode = 2, .segs = conv_segs, .state_stride = @intCast(3 * h) }, streams);
            if (v.saves_bank and !f.zero_messages) try loop.save(&f.m.loop, s, b.x, b.lp.bank[v.pass % 2], sources, plan.bank_n, n);
        }
        var fm: ops.MixArgs = .{ .partial = b.x, .query = f.w.out_query, .out = b.normed, .n_blocks = @intCast(nb), .partial_bf16 = 1, .nq = 1, .rows = @intCast(n), .c = @intCast(h), .eps = c.eps };
        for (0..nb) |j| fm.blocks[j] = blocks[j];
        fm.norm[0] = f.w.norm;
        try ops.mix(k, s, fm);
        const hr: usize = if (f.last_logits) n - 1 else 0; // rows before hr keep no logits
        try f.head(f.w, hr, n, b.logits.ptr);
        if (f.peer) |q| try f.head(q.w, hr, n, q.logits); // rank 1's padded vocabulary span, in its own rows
    }

    /// One rank's attention partial of visit `v`: its heads' projections, caches and attention, o_proj into its slot.
    fn attention(f: Forward, r: Cut, v: *const visits.Visit, xs: [3]u64, rows_by: [2]kernels.Rows, conv_segs: u64, p: Plan) !void {
        const c = &f.w.config;
        const n = p.n;
        const h: usize = c.hidden;
        const s = f.stream;
        const k = &f.m.ops;
        const b = f.b;
        const hd: usize = c.head_dim;
        const kvw = f.shape.kv * hd;
        const streams = p.ni;
        const sl = v.slot;
        const l = &r.w.layers[v.layer];
        const st = r.st;
        const global = l.kind == .global;
        const xq = xs[0];
        const xk = xs[1];
        const xv = xs[2];
        const a = &l.attn;
        const sd = &f.m.side;
        // prompt rows run the narrow projections on the main stream: one GEMM scratch, and they fill the GPU alone
        const ss = if (f.m.prompt != null and n >= prompt_dense.min_rows) s else sd.stream;
        try sd.fork.record(s);
        try ss.wait(sd.fork);
        if (a.r_lo) |lo| { // every head's r, then this rank's columns
            try f.proj(ss, &sd.dense, a.r, xq, n, b.rw, .bf16);
            const all: usize = c.heads * c.d_rel * 2;
            const m: cuda.abi.Memcpy2D = .{ .src = b.rw + lo * 2, .src_pitch = all, .dst = b.r, .dst_pitch = f.shape.heads * c.d_rel * 2, .width = f.shape.heads * c.d_rel * 2, .height = n };
            try f.d.check(f.d.api.cuMemcpy2DAsync_v2(&m, ss.handle), "cuMemcpy2DAsync");
        } else try f.proj(ss, &sd.dense, a.r, xq, n, b.r, .bf16);
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
        const cv = st.conv[sl];
        try ops.conv(k, s, .{ .x = b.kf, .state = cv[0], .w = a.k_conv, .dyn = if (global) b.dk else 0, .heads = @intCast(f.shape.kv), .c = @intCast(kvw), .rows = @intCast(n), .mode = 0, .out = b.kraw, .segs = conv_segs, .state_stride = @intCast(3 * kvw) }, streams);
        try ops.conv(k, s, .{ .x = b.vf, .state = cv[1], .w = a.v_conv, .dyn = if (global) b.dv else 0, .heads = @intCast(f.shape.kv), .c = @intCast(kvw), .rows = @intCast(n), .mode = 0, .out = b.vraw, .segs = conv_segs, .state_stride = @intCast(3 * kvw) }, streams);
        var lay_i = f.shape.layer(c, sl);
        // the caches' rings, whatever rows this call's scratch holds
        lay_i.ring = @intCast(st.ring[@intFromBool(global)]);
        const rows = rows_by[@intFromBool(global)];
        const prep: kernels.Prep = .{ .qk_clip = c.qk_clip, .rel_clip = c.rel_clip, .floor = c.log_floor, .alpha = c.log_alpha, .eps = c.eps };
        try kernels.prep(&f.m.attn, s, rows, lay_i, prep, .{ b.q, b.kraw, b.vraw, b.r }, .{ a.q_scale, a.rel_scale }, b.qo, b.relo, st.cache(sl));
        try kernels.relTable(&f.m.attn, s, rows, b.relo, a.rel_proj, b.tab);
        try kernels.attend(&f.m.attn, s, rows, lay_i, p.stride, b.parts.slots, b.qo, st.cache(sl), b.tab, b.parts.o, b.parts.ml, b.sink, b.gate, b.att);
        try f.dense(a.o, b.att, n, f.slot(r.w, b.ao, n), .f32);
    }

    /// One rank's experts partial of layer `i` over the picks in b.pick and b.wts, into its slot.
    fn moe(f: Forward, w: *const weights.Weights, i: usize, n: usize) !void {
        const c = &f.w.config;
        const h: usize = c.hidden;
        const b = f.b;
        const l = &w.layers[i];
        const out = f.slot(w, b.moe, n);
        var r0: usize = 0;
        while (r0 < n) : (r0 += expert_rows) {
            const rows = @min(expert_rows, n - r0);
            const caps: experts.Caps = .{ .gate = c.gate_softcap, .up = c.up_softcap };
            const sel = r0 * c.topk * 4;
            try experts.run(&f.m.experts, f.stream, l.experts, caps, l.bits, b.normed + r0 * h * 2, h, .bf16, b.pick + sel, b.wts + sel, rows, out + r0 * h * 4, b.exs);
        }
    }

    /// One rank's head over rows hr..n of b.normed into `logits` (fp32 [rows][its padded vocabulary span]).
    fn head(f: Forward, w: *const weights.Weights, hr: usize, n: usize, logits: u64) !void {
        const c = &f.w.config;
        const h: usize = c.hidden;
        const nv = w.share.vocab.len();
        try f.dense(w.head, f.b.normed + hr * h * 2, n - hr, logits + hr * nv * 4, .f32);
        try ops.softcap(&f.m.ops, f.stream, logits + hr * nv * 4, (n - hr) * nv, c.logit_softcap);
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

test {
    _ = @import("forward_test.zig");
}
