//! GLM-5.3-Flash on CUDA at TP=2: one rank's weights, one sequence's state, prompt chunks and serial greedy steps.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const load_ = @import("cuda_load.zig");
const lay = cuda.glm_layer;
const dsa = cuda.glm_dsa;
const head_ = cuda.glm_head;
const nccl = cuda.nccl;

pub const world = load_.world;
pub const prefill_rows = 2048; // decode.PREFILL_ROWS: rows of a prompt chunk
const index_eps = 1e-6; // the DSA indexer's key LayerNorm (sparse.py)

/// One rank's weights on the device (weights.load without the MTP layer).
pub const Weights = struct {
    cfg: config.Config,
    pool: load_.Pool,
    layers: []lay.LayerW,
    embed: u64,
    head: head_.HeadW,
    gpa: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, rank: u32) !Weights {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        const cfg = try config.parse(gpa, text);
        var pool: load_.Pool = .{ .d = d, .gpa = gpa };
        errdefer pool.deinit();
        var r = try load_.Reader.open(gpa, io, dir, rank, &pool);
        defer r.close();
        const layers = try gpa.alloc(lay.LayerW, cfg.layers);
        errdefer gpa.free(layers);
        for (layers, 0..) |*l, i| l.* = try r.layer(@intCast(i), &cfg);
        const g = try r.globals(&cfg);
        return .{ .cfg = cfg, .pool = pool, .layers = layers, .embed = g.embed, .head = g.head, .gpa = gpa };
    }

    pub fn deinit(w: *Weights) void {
        w.gpa.free(w.layers);
        w.pool.deinit();
    }

    /// The constants every layer reads (weights.Config; the DSA scales as forward.py computes them).
    pub fn layerConfig(w: *const Weights) lay.Config {
        const c = &w.cfg;
        return .{
            .eps = c.eps,
            .hc_eps = c.hc_eps,
            .hc_iters = c.sinkhorn,
            .lower = c.lower_bound,
            .limit = c.swiglu_limit,
            .routed_scale = c.routed_scale,
            .top_k = c.topk,
            .experts = c.experts,
            .d = c.hidden,
            .streams = c.hc,
            .dsa = .{
                .eps = c.eps,
                .index_eps = index_eps,
                .scale = @floatCast(std.math.pow(f64, @floatFromInt(c.nope), -0.5)),
                .index_scale = @floatCast(std.math.pow(f64, @floatFromInt(c.i_dim), -0.5)),
                .wscale = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.i_heads)))),
            },
        };
    }

    fn firstDsa(w: *const Weights) !*const dsa.DsaW {
        for (w.layers) |*l| if (l.mixer == .dsa) return &l.mixer.dsa;
        return error.GlmNoDsa;
    }
};

/// One forward's buffers for up to `rows` rows (forward.Buffers): the layers', the DSA block's, the head's.
const Bufs = struct {
    rows: usize,
    sc: lay.Scratch,
    dsc: dsa.Scratch,
    hs: head_.Scratch,
    x: cuda.DeviceBuffer, // [rows, S * D] bf16: the hyper-connection streams
    ids: cuda.DeviceBuffer, // [rows] u32

    fn init(d: *const cuda.Driver, s: cuda.Stream, w: *const Weights, rows: usize, prefill: bool, cap: usize) !Bufs {
        const lc = w.layerConfig();
        var sh = lay.Shape.of(&w.layers[0]);
        for (w.layers[1..]) |*l| sh = sh.widen(lay.Shape.of(l));
        var sc = try lay.Scratch.initFor(d, s, lc, sh, rows, world);
        errdefer sc.deinit();
        var dsc = try dsa.Scratch.init(d, try w.firstDsa(), rows, prefill, cap);
        errdefer dsc.deinit();
        var hs = try head_.Scratch.init(d, s, w.head, lc.d, rows);
        errdefer hs.deinit();
        var x = try cuda.DeviceBuffer.alloc(d, rows * @as(usize, lc.streams) * lc.d * 2);
        errdefer x.free();
        const ids = try cuda.DeviceBuffer.alloc(d, rows * 4);
        return .{ .rows = rows, .sc = sc, .dsc = dsc, .hs = hs, .x = x, .ids = ids };
    }

    fn deinit(b: *Bufs) void {
        b.ids.free();
        b.x.free();
        b.hs.deinit();
        b.dsc.deinit();
        b.sc.deinit();
    }
};

/// A KDA layer's state: the conv window and the recurrent state's two buffers (forward.State conv, rec).
const KdaSlot = struct { conv: u64, rec: [2]u64 };

/// One sequence on one rank: the weights' forward over committed caches, the two ranks in lockstep over NCCL.
pub const Engine = struct {
    d: *const cuda.Driver,
    w: *const Weights,
    k: lay.Kernels,
    s: cuda.Stream,
    lc: lay.Config,
    lib: *const nccl.Library,
    comm: nccl.Comm,
    cap: usize,
    dec: Bufs,
    pre: Bufs,
    state: cuda.DeviceBuffer, // every layer's caches: the KDA states first (reset zeroes them), then DSA's
    kda_bytes: usize,
    kda: []KdaSlot,
    dsa: []dsa.DsaState,
    pos_dev: cuda.DeviceBuffer, // forward.State.pos_dev: i32
    cand: cuda.DeviceBuffer, // [world, 1, 2] f32: every rank's top candidate
    gpa: std.mem.Allocator,
    graphs: [2]?cuda.graph.Exec = .{ null, null }, // a dense-window decode step for each KDA state parity
    cur: u1 = 0,
    pos: usize = 0,

    /// `cap`: cache slots; the DSA pool buckets equal the Python engine's while cap / 4 covers every bucket used.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const Weights, lib: *const nccl.Library, comm: nccl.Comm, cap: usize) !Engine {
        if (cap % 4 != 0 or cap < prefill_rows) return error.GlmCapacity;
        const c = &w.cfg;
        var e: Engine = undefined;
        e.d = d;
        e.w = w;
        e.lib = lib;
        e.comm = comm;
        e.cap = cap;
        e.gpa = gpa;
        e.cur = 0;
        e.pos = 0;
        e.graphs = .{ null, null };
        e.lc = w.layerConfig();
        e.k = try lay.Kernels.load(d);
        errdefer e.k.unload();
        e.s = try cuda.Stream.init(d, false);
        errdefer e.s.deinit();
        e.dec = try Bufs.init(d, e.s, w, 1, false, cap);
        errdefer e.dec.deinit();
        e.pre = try Bufs.init(d, e.s, w, prefill_rows, true, cap);
        errdefer e.pre.deinit();
        const n_kda = c.countKind(.kda);
        const n_dsa = c.countKind(.mla); // the backbone's (the MTP layer is not loaded)
        const heads: usize = c.kda_heads / world;
        const conv_b = (c.conv - 1) * 3 * heads * c.kda_dim * 2;
        const rec_b = heads * c.kda_dim * c.kda_dim * 4;
        e.kda_bytes = n_kda * (conv_b + 2 * rec_b);
        const ix: usize = c.i_dim * 2;
        const dsa_b = cap * c.kv_lora + cap * 4 + 2 * cap * ix + (cap / 4 + 2) * ix;
        e.state = try cuda.DeviceBuffer.alloc(d, e.kda_bytes + n_dsa * dsa_b);
        errdefer e.state.free();
        try e.state.fill8(0, e.s.handle);
        e.pos_dev = try cuda.DeviceBuffer.alloc(d, 4);
        errdefer e.pos_dev.free();
        e.cand = try cuda.DeviceBuffer.alloc(d, world * 2 * 4);
        errdefer e.cand.free();
        e.kda = try gpa.alloc(KdaSlot, n_kda);
        errdefer gpa.free(e.kda);
        e.dsa = try gpa.alloc(dsa.DsaState, n_dsa);
        errdefer gpa.free(e.dsa);
        for (w.layers) |*l| try reserve(&e.k.dense, l);
        var at = e.state.ptr;
        for (e.kda) |*sl| {
            sl.* = .{ .conv = at, .rec = .{ at + conv_b, at + conv_b + rec_b } };
            at += conv_b + 2 * rec_b;
        }
        for (e.dsa) |*st| {
            st.* = .{ .lc = at, .scales = at + cap * c.kv_lora, .ik = at + cap * (c.kv_lora + 4), .ig = at + cap * (c.kv_lora + 4) + cap * ix, .pk = at + cap * (c.kv_lora + 4) + 2 * cap * ix, .pos_dev = e.pos_dev.ptr, .cap = cap };
            at += dsa_b;
        }
        return e;
    }

    pub fn deinit(e: *Engine) void {
        for (&e.graphs) |*g| if (g.*) |*x| x.deinit();
        e.gpa.free(e.dsa);
        e.gpa.free(e.kda);
        e.cand.free();
        e.pos_dev.free();
        e.state.free();
        e.pre.deinit();
        e.dec.deinit();
        e.s.deinit();
        e.k.unload();
    }

    /// State.reset: KDA states zeroed, position 0 (the DSA caches keep their bytes; positions past pos are never read).
    pub fn reset(e: *Engine) !void {
        const kv: cuda.DeviceBuffer = .{ .d = e.d, .ptr = e.state.ptr, .len = e.kda_bytes, .borrowed = true };
        try kv.fill8(0, e.s.handle);
        e.cur = 0;
        e.pos = 0;
    }

    /// decode.prefill without drafts: the prompt in prompt chunks, then its first token.
    pub fn prefill(e: *Engine, prompt: []const u32) !u32 {
        if (prompt.len == 0) return error.GlmEmptyPrompt;
        try e.reset();
        var tok: u32 = undefined;
        var at: usize = 0;
        while (at < prompt.len) : (at += prefill_rows) tok = try e.forward(prompt[at..@min(prompt.len, at + prefill_rows)], true);
        return tok;
    }

    /// serial_decode's step, the next token; within the dense window a graph captured on first use (forward.py's).
    pub fn step(e: *Engine, tok: u32) !u32 {
        if (e.pos + 1 > dsa.dense_limit) return e.forward(&.{tok}, false);
        if (e.pos + 1 > e.cap) return error.GlmCapacity;
        try e.dec.ids.upload(0, std.mem.asBytes(&tok));
        try e.pos_dev.fill32(@intCast(e.pos), e.s.handle);
        if (e.graphs[e.cur] == null) e.graphs[e.cur] = try e.capture();
        try e.graphs[e.cur].?.launchOn(e.s);
        return e.finish(1);
    }

    /// The decode step's GPU work recorded once (nothing runs): position and parity come from the replay's state.
    fn capture(e: *Engine) !cuda.graph.Exec {
        try cuda.graph.beginCapture(e.s, .global);
        const recorded = e.body(&e.dec, 1, false, true);
        var g = try cuda.graph.endCapture(e.s);
        defer g.deinit();
        try recorded;
        const x = try g.instantiate();
        try x.upload(e.s);
        return x;
    }

    /// One layer of the forward on its state; a decode row's KDA layer commits right after (its rows are reused next).
    fn layer(e: *Engine, l: *const lay.LayerW, b: *Bufs, rows: usize, prompt: bool, capture_: bool, at: *[2]usize, g: lay.Gather) !void {
        switch (l.mixer) {
            .kda => |*kw| {
                const sl = e.kda[at[0]];
                at[0] += 1;
                const st: lay.KdaState = .{ .conv = sl.conv, .rec_in = sl.rec[e.cur], .rec_out = sl.rec[1 - e.cur] };
                const mode: lay.Mode = if (prompt) .{ .prompt = e.pos } else .decode;
                try lay.forward(&e.k, &e.s, e.lc, l, .{ .kda = .{ .st = st, .mode = mode } }, &b.sc, b.x.ptr, rows, g);
                if (!prompt) try lay.commit(&e.k, &e.s, kw, st, &b.sc, rows, rows);
            },
            .dsa => {
                const run: lay.DsaRun = .{ .st = e.dsa[at[1]], .sc = &b.dsc, .pos = e.pos, .capture = capture_ };
                at[1] += 1;
                try lay.forward(&e.k, &e.s, e.lc, l, .{ .dsa = run }, &b.sc, b.x.ptr, rows, g);
            },
        }
    }

    fn gather(ctx: *anyopaque, s: cuda.Stream, site: lay.Site, part: u64, gath: u64, rows: usize) anyerror!void {
        _ = site;
        const e: *Engine = @ptrCast(@alignCast(ctx));
        try e.lib.check(e.lib.api.ncclAllGather(part, gath, rows * e.lc.d, .f32, e.comm, s.handle), "ncclAllGather");
    }

    /// compute + sample + commit for `ids` at the current position; the greedy token of the last row.
    fn forward(e: *Engine, ids: []const u32, prompt: bool) !u32 {
        const b = if (prompt) &e.pre else &e.dec;
        const rows = ids.len;
        if (rows == 0 or rows > b.rows or (!prompt and rows != 1)) return error.GlmRows;
        if (e.pos + rows > e.cap) return error.GlmCapacity;
        try b.ids.upload(0, std.mem.sliceAsBytes(ids));
        try e.pos_dev.fill32(@intCast(e.pos), e.s.handle);
        try e.body(b, rows, prompt, false);
        return e.finish(rows);
    }

    /// The forward's GPU work from b.ids and pos_dev to every rank's candidate in e.cand; no host waits (capturable).
    fn body(e: *Engine, b: *Bufs, rows: usize, prompt: bool, capture_: bool) !void {
        const lc = e.lc;
        try e.k.glue.embedB16(e.s, b.ids.ptr, e.w.embed, b.x.ptr, rows, lc.d, lc.streams);
        const g: lay.Gather = .{ .ctx = e, .world = world, .run = gather };
        var at: [2]usize = .{ 0, 0 }; // the next KDA and DSA state
        for (e.w.layers) |*l| try e.layer(l, b, rows, prompt, capture_, &at, g);
        try head_.forward(&e.k.glue, &e.k.bmm, e.s, lc.eps, lc.d, lc.streams, e.w.head, &b.hs, b.x.ptr, rows, prompt);
        try e.lib.check(e.lib.api.ncclAllGather(b.hs.cand, e.cand.ptr, 2, .f32, e.comm, e.s.handle), "ncclAllGather");
    }

    /// sample_rows' greedy pick from the gathered candidates, then the commit's position and KDA parity.
    fn finish(e: *Engine, rows: usize) !u32 {
        var got: [world * 2]f32 = undefined;
        try e.s.synchronize();
        try e.cand.download(0, std.mem.sliceAsBytes(&got));
        var tok: [1]u32 = undefined;
        try head_.choose(&got, world, 1, &tok);
        e.cur = 1 - e.cur;
        e.pos += rows;
        return tok[0];
    }
};

/// The EXL3 dense scratch at every layer's widest, before any capture.
fn reserve(de: *cuda.exl3.Dense, l: *const lay.LayerW) !void {
    switch (l.mixer) {
        .kda => |k| for ([_]cuda.exl3.Weight{ k.qkv, k.o }) |w| try de.reserve(w),
        .dsa => |a| for ([_]cuda.exl3.Weight{ a.q_a, a.kv_a, a.q_b, a.o }) |w| try de.reserve(w),
    }
    switch (l.ffn) {
        .moe => |m| for ([_]cuda.exl3.Weight{ m.gate, m.up, m.down }) |w| try de.reserve(w),
        .mlp => |m| for ([_]cuda.exl3.Weight{ m.gate, m.up, m.down }) |w| try de.reserve(w),
    }
}
