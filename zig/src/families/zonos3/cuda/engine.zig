//! ZONOS3 Mini's backbone on one GPU: frame rows in, the next frame's logits out, one fp32 row path for all rows.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("../config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");

const D = cfg.dim;
const W = cfg.frame_width;
const Buf = cuda.DeviceBuffer;

/// A host's say on a load: an error refuses it.
pub const Gate = struct { ptr: *anyopaque, check: *const fn (*anyopaque, *const cuda.Context) anyerror!void };

pub const Options = struct {
    device: u32 = 0,
    max_ctx: usize = 4096, // positions (prompt rows + frames), at most the model's 32768
    max_rows: usize = 64, // rows a pass
    gate: ?Gate = null, // asked once the context is current, before anything is allocated on it

    pub fn validate(o: Options) !void {
        if (o.max_ctx < 2 or o.max_ctx > cfg.max_seqlen) return error.BadContext;
        if (o.max_rows == 0 or o.max_rows > 1024) return error.BadPrefillRows;
    }
};

/// A prompt row whose summed embedding a vector replaces (the speaker row, the condition rows).
pub const Replace = struct { row: usize, vec: [D]f32 };

/// Per-sublayer probe of a pass under the oracle's dump names, called with device fp32 rows [rows, width].
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize) anyerror!void,

    fn at(p: Probe, comptime f: []const u8, layer: usize, ptr: u64, rows: usize, width: usize) !void {
        var nb: [48]u8 = undefined;
        try p.f(p.ctx, try std.fmt.bufPrint(&nb, f, .{layer}), ptr, rows, width);
    }
};

/// One stream's caches and position: what `run` reads and advances.
pub const Lane = struct { kv: [cfg.layers][2]u64, conv: [cfg.layers]u64, gstate: [cfg.layers]u64, state: u64, position: usize = 0 };

/// One lane's next row in a pass over several lanes, and where its logits go.
pub const Steer = @import("../steering.zig").Steer;
pub const Probed = @import("../laci_probe.zig").Probed;
const probe_layers = @import("../laci_probe.zig").layers;
pub const Turn = struct { lane: usize, row: [W]i32, out: *[cfg.head_rows]f32, steer: []const Steer = &.{}, probe: ?*Probed = null };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    o: Options,
    c: cfg.Config,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    k: kern.Kernels,
    w: *wts.Weights,
    bufs: std.ArrayList(Buf),
    counted: ?usize, // set: allocations are counted, not made
    position: usize = 0,
    // state
    kv: [cfg.layers][2]u64, // attention layers: K, V fp32 [max_ctx][4][32]
    conv: [cfg.layers]u64, // GDN layers: the last 3 conv inputs, fp32 [1536][3]
    gstate: [cfg.layers]u64, // GDN layers: fp32 [4][128][128]
    state: u64, // the conv and recurrent states in one block (zeroed per request)
    state_bytes: usize,
    row_host: [][W]i32, // [R] a pass's rows with extra text ids set to the padding row
    lanes: std.ArrayList(Lane), // every stream's state; lanes[active] is stale while the fields above run it
    active: usize,
    // scratch (R = max_rows)
    rows: u64, // i32 [R][11]
    bank: u64, // fp32 [sources][R][D]
    mixed: u64, // [R][D]
    xn: u64, // [R][D]
    rn: u64, // [R][D] the router's norm
    proj: u64, // [R][4096]: the GDN projection, the dense FFN's two halves
    conv_out: u64, // [R][1536]
    gates: u64, // [R][4][2]
    o32: u64, // [R][512]
    y: u64, // [R][512]
    q: u64, // [R][512]
    kvraw: u64, // [R][256]
    gate: u64, // [R][16]
    act: u64, // [R][2048]
    rl: u64, // [R][64]
    picks: u64, // i32 [R][2]
    rw: u64, // [R][2]
    latent: u64, // [R][256]
    gu: u64, // [2R][3072]
    hid: u64, // [2R][1536]
    down: u64, // [2R][256]
    comb: u64, // [R][256]
    logits: u64, // [R][head_rows]
    row_up: u64, // [D] a replaced row on its way in
    steer: u64, // [R][2][D]: a steered row of a bank source, then its delta
    unit: u64, // [2], each 1.0: the weights that make z3_combine a sum
    probe_q: u64, // [R][attention layers][512]: probed rows' queries on their way out
    mask_kv: [2]u64, // [max_ctx][128] each: the keys and values a masked head sees, closed up
    mask_out: u64, // [512]: a row's attention over them

    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        try o.validate();
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.o = o;
        e.position = 0;
        e.lanes = .empty;
        e.active = 0;
        e.bufs = .empty;
        e.counted = null;
        e.c = try cfg.Config.load(gpa, io, model_dir);
        e.driver = try cuda.Driver.open();
        errdefer e.driver.close();
        e.ctx = try cuda.Context.init(&e.driver, std.math.cast(c_int, o.device) orelse return error.NoSuchCudaDevice);
        errdefer e.ctx.deinit();
        if (o.gate) |g| try g.check(g.ptr, &e.ctx);
        e.stream = try cuda.Stream.init(&e.driver, true);
        errdefer e.stream.deinit();
        e.k = try kern.Kernels.load(&e.driver);
        errdefer e.k.unload();
        e.w = try wts.load(gpa, io, &e.driver, model_dir, e.c);
        errdefer {
            e.w.deinit();
            gpa.destroy(e.w);
        }
        errdefer e.freeState();
        e.row_host = try gpa.alloc([W]i32, o.max_rows);
        errdefer gpa.free(e.row_host);
        try e.allocState();
        try e.put(e.unit, std.mem.asBytes(&[2]f32{ 1, 1 }));
        try e.reset();
        return e;
    }

    pub fn destroy(e: *Engine) void {
        e.stream.synchronize() catch {};
        e.freeState();
        e.w.deinit();
        e.gpa.destroy(e.w);
        e.gpa.free(e.row_host);
        e.k.unload();
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    fn freeState(e: *Engine) void {
        for (e.bufs.items) |*b| b.free();
        e.bufs.deinit(e.gpa);
        e.lanes.deinit(e.gpa);
    }

    fn alloc(e: *Engine, bytes: usize) !u64 {
        if (e.counted) |*n| {
            n.* += @max(bytes, 256);
            return 0;
        }
        var b = try Buf.alloc(&e.driver, @max(bytes, 256));
        errdefer b.free();
        try e.bufs.append(e.gpa, b);
        return b.ptr;
    }

    /// Device bytes of the state and workspaces `o` allocates: the allocation code itself, counting.
    pub fn stateBytes(o: Options) usize {
        var e: Engine = undefined;
        e.o = o;
        e.counted = 0;
        e.allocState() catch unreachable;
        return e.counted.?;
    }

    /// Device bytes of a second lane's state under `o`.
    pub fn laneBytes(o: Options) usize {
        var e: Engine = undefined;
        e.o = o;
        e.counted = 0;
        _ = e.allocLane() catch unreachable;
        return e.counted.?;
    }

    /// One more stream's state on the device; its index for `use`. `reset` zeroes the lane in use.
    pub fn addLane(e: *Engine) !usize {
        try e.lanes.ensureUnusedCapacity(e.gpa, 1);
        e.lanes.appendAssumeCapacity(try e.allocLane());
        return e.lanes.items.len - 1;
    }

    /// Runs lane `i` from here on: its caches and its position.
    pub fn use(e: *Engine, i: usize) void {
        if (i == e.active) return;
        e.lanes.items[e.active] = .{ .kv = e.kv, .conv = e.conv, .gstate = e.gstate, .state = e.state, .position = e.position };
        e.load(e.lanes.items[i]);
        e.active = i;
    }

    fn load(e: *Engine, l: Lane) void {
        e.kv = l.kv;
        e.conv = l.conv;
        e.gstate = l.gstate;
        e.state = l.state;
        e.position = l.position;
    }

    /// A guided request's two lanes when one request runs alone: lane 1 is made by the first call.
    pub fn twin(e: *Engine) !void {
        if (e.lanes.items.len < 2) _ = try e.addLane();
    }

    /// Exchanges lanes 0 and 1.
    pub fn swap(e: *Engine) void {
        e.use(e.active ^ 1);
    }

    fn allocLane(e: *Engine) !Lane {
        var l: Lane = undefined;
        l.position = 0;
        const kvb = e.o.max_ctx * cfg.kv_width * 4;
        const conv_b = cfg.gdn_conv * 3 * 4;
        const gs_b = cfg.gdn_heads * cfg.gdn_dim * cfg.gdn_dim * 4;
        var n_gdn: usize = 0;
        for (0..cfg.layers) |li| n_gdn += @intFromBool(cfg.usesGdn(li));
        e.state_bytes = n_gdn * (conv_b + gs_b);
        l.state = try e.alloc(e.state_bytes);
        var g: usize = 0;
        for (0..cfg.layers) |li| {
            if (cfg.usesGdn(li)) {
                l.kv[li] = .{ 0, 0 };
                l.conv[li] = l.state + g * (conv_b + gs_b);
                l.gstate[li] = l.conv[li] + conv_b;
                g += 1;
            } else {
                l.kv[li] = .{ try e.alloc(kvb), try e.alloc(kvb) };
                l.conv[li] = 0;
                l.gstate[li] = 0;
            }
        }
        return l;
    }

    fn allocState(e: *Engine) !void {
        const R = e.o.max_rows;
        const l = try e.allocLane();
        e.load(l);
        if (e.counted == null) try e.lanes.append(e.gpa, l);
        e.rows = try e.alloc(R * W * 4);
        e.bank = try e.alloc(cfg.sources * R * D * 4);
        e.mixed = try e.alloc(R * D * 4);
        e.xn = try e.alloc(R * D * 4);
        e.rn = try e.alloc(R * D * 4);
        e.proj = try e.alloc(R * 2 * cfg.ffn_hidden * 4);
        e.conv_out = try e.alloc(R * cfg.gdn_conv * 4);
        e.gates = try e.alloc(R * cfg.gdn_heads * 2 * 4);
        e.o32 = try e.alloc(R * cfg.gdn_inner * 4);
        e.y = try e.alloc(R * cfg.gdn_inner * 4);
        e.q = try e.alloc(R * cfg.attn_inner * 4);
        e.kvraw = try e.alloc(R * 2 * cfg.kv_width * 4);
        e.gate = try e.alloc(R * cfg.heads * 4);
        e.act = try e.alloc(R * cfg.ffn_hidden * 4);
        e.rl = try e.alloc(R * cfg.experts * 4);
        e.picks = try e.alloc(R * cfg.top_k * 4);
        e.rw = try e.alloc(R * cfg.top_k * 4);
        e.latent = try e.alloc(R * cfg.moe_latent * 4);
        e.gu = try e.alloc(R * cfg.top_k * 2 * cfg.moe_hidden * 4);
        e.hid = try e.alloc(R * cfg.top_k * cfg.moe_hidden * 4);
        e.down = try e.alloc(R * cfg.top_k * cfg.moe_latent * 4);
        e.comb = try e.alloc(R * cfg.moe_latent * 4);
        e.logits = try e.alloc(R * cfg.head_rows * 4);
        e.row_up = try e.alloc(D * 4);
        e.steer = try e.alloc(R * 2 * D * 4);
        e.unit = try e.alloc(2 * 4);
        e.probe_q = try e.alloc(R * probe_layers * cfg.attn_inner * 4);
        for (&e.mask_kv) |*m| m.* = try e.alloc(e.o.max_ctx * cfg.kv_width * 4);
        e.mask_out = try e.alloc(cfg.attn_inner * 4);
    }

    fn buf(e: *Engine, ptr: u64) Buf {
        return .{ .d = &e.driver, .ptr = ptr, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    fn put(e: *Engine, dst: u64, bytes: []const u8) !void {
        try e.buf(dst).uploadAsync(0, bytes, e.stream.handle);
    }

    pub fn download(e: *Engine, src: u64, out: []u8) !void {
        try e.buf(src).downloadAsync(0, out, e.stream.handle);
        try e.stream.synchronize();
    }

    /// A new request on the lane in use: zero recurrent states, position 0. Cache rows are written before read.
    pub fn reset(e: *Engine) !void {
        var b = e.buf(e.state);
        b.len = e.state_bytes;
        try b.fill8(0, e.stream.handle);
        try e.stream.synchronize();
        e.position = 0;
    }

    fn slot(e: *const Engine, source: usize) u64 {
        return e.bank + source * e.o.max_rows * D * 4;
    }

    /// The next rows at the current position; `logits` then holds the last row's (every row's under a one-pass probe).
    pub fn run(e: *Engine, rows: []const [W]i32, replaced: []const Replace, probe: ?Probe) !void {
        if (rows.len == 0) return error.NoRows;
        if (e.position + rows.len > e.o.max_ctx) return error.ContextFull;
        if (probe != null and rows.len > e.o.max_rows) return error.ProbeNeedsOnePass;
        for (rows) |r| {
            for (r[0..cfg.codebooks]) |code| if (code < 0 or code >= cfg.audio_vocab) return error.BadFrameRow;
            if (r[cfg.codebooks] < 0 or r[cfg.codebooks] >= e.w.textIds()) return error.BadFrameRow;
        }
        var at: usize = 0;
        while (at < rows.len) {
            const t = @min(e.o.max_rows, rows.len - at);
            try e.pass(rows[at..][0..t], at, replaced, at + t == rows.len, probe, null, &.{});
            at += t;
        }
    }

    /// One generated row on the lane in use with `steer` added to its bank sources (synth: a decode step's deltas).
    pub fn runSteered(e: *Engine, row: [W]i32, steer: []const Steer) !void {
        if (e.position + 1 > e.o.max_ctx) return error.ContextFull;
        for (row[0..cfg.codebooks]) |code| if (code < 0 or code >= cfg.audio_vocab) return error.BadFrameRow;
        if (row[cfg.codebooks] < 0 or row[cfg.codebooks] >= e.w.text_rows) return error.BadFrameRow;
        for (steer) |st| if (st.source >= cfg.sources) return error.NoSuchSource;
        try e.pass(&.{row}, 0, &.{}, true, null, null, steer);
    }

    /// One row of each of `turns`' lanes in one pass; each lane moves a position and its logits go to its `out`.
    pub fn runLanes(e: *Engine, turns: []const Turn) !void {
        if (turns.len == 0) return;
        if (turns.len > e.o.max_rows) return error.TooManyLanes;
        e.lanes.items[e.active].position = e.position;
        var rows: [64][W]i32 = undefined;
        if (turns.len > rows.len) return error.TooManyLanes;
        for (turns, rows[0..turns.len], 0..) |turn, *row, b| {
            if (turn.lane >= e.lanes.items.len) return error.NoSuchLane;
            for (turns[0..b]) |earlier| if (earlier.lane == turn.lane) return error.LaneTwice;
            if (e.lanes.items[turn.lane].position + 1 > e.o.max_ctx) return error.ContextFull;
            for (turn.row[0..cfg.codebooks]) |code| if (code < 0 or code >= cfg.audio_vocab) return error.BadFrameRow;
            if (turn.row[cfg.codebooks] < 0 or turn.row[cfg.codebooks] >= e.w.text_rows) return error.BadFrameRow;
            for (turn.steer) |st| if (st.source >= cfg.sources) return error.NoSuchSource;
            row.* = turn.row;
        }
        try e.pass(rows[0..turns.len], 0, &.{}, true, null, turns, &.{});
        for (turns, 0..) |turn, b| {
            try e.buf(e.logits + b * cfg.head_rows * 4).downloadAsync(0, std.mem.asBytes(turn.out), e.stream.handle);
            if (turn.probe) |p| try e.buf(e.probe_q + b * probe_layers * cfg.attn_inner * 4).downloadAsync(0, std.mem.asBytes(&p.q), e.stream.handle);
        }
        try e.stream.synchronize();
        for (turns) |turn| e.lanes.items[turn.lane].position += 1;
        e.position = e.lanes.items[e.active].position;
    }

    /// ACI (lm._masked_heads): turn `b`'s head again over the keys it may see, closed up into one cache.
    fn maskedHead(e: *Engine, l: *const Lane, li: usize, b: usize, m: @import("../laci_probe.zig").Mask, p: *const Probed) !void {
        const row = cfg.kv_width * 4;
        const after = p.text_start + p.text_len;
        if (m.hi >= p.text_len or m.lo > m.hi or after > l.position) return error.BadProbeWindow;
        const window = m.hi - m.lo + 1;
        const tail = l.position + 1 - after; // the row's own key is in the cache already
        for (e.mask_kv, l.kv[li]) |dst, src| {
            if (p.text_start > 0) try e.buf(dst).copyFrom(0, src, p.text_start * row, e.stream.handle);
            try e.buf(dst).copyFrom(p.text_start * row, src + (p.text_start + m.lo) * row, window * row, e.stream.handle);
            try e.buf(dst).copyFrom((p.text_start + window) * row, src + after * row, tail * row, e.stream.handle);
        }
        const qb = e.q + b * cfg.attn_inner * 4;
        try e.k.attention(e.stream, qb, e.mask_kv[0], e.mask_kv[1], e.gate + b * cfg.heads * 4, p.text_start + window + tail - 1, e.mask_out, 1);
        const head = m.head * cfg.head_dim * 4;
        try e.buf(e.y + b * cfg.attn_inner * 4).copyFrom(head, e.mask_out + head, cfg.head_dim * 4, e.stream.handle);
    }

    /// The keys of cache rows [start, start + out.len / 128) of attention layer `layer` on the lane in use.
    pub fn keys(e: *Engine, layer: usize, start: usize, out: []f32) !void {
        if (cfg.usesGdn(layer) or out.len % cfg.kv_width != 0 or start + out.len / cfg.kv_width > e.position) return error.NoSuchKeys;
        try e.download(e.kv[layer][0] + start * cfg.kv_width * 4, std.mem.sliceAsBytes(out));
    }

    /// Bank source `source` of this pass takes its rows' deltas: source + delta, one fp32 add (lm.forward's source()).
    fn steerSource(e: *Engine, source: usize, many: ?[]const Turn, lone: []const Steer) !void {
        const n = if (many) |turns| turns.len else 1;
        for (0..n) |b| for (if (many) |turns| turns[b].steer else lone) |st| {
            if (st.source != source) continue;
            const row = e.slot(source) + b * D * 4;
            const pair = e.steer + b * 2 * D * 4;
            try e.buf(pair).copyFrom(0, row, D * 4, e.stream.handle);
            try e.put(pair + D * 4, std.mem.asBytes(st.delta));
            try e.k.combine(e.stream, pair, e.unit, D, row, 1);
        };
    }

    /// `many`: a row of each turn's lane at its position; else rows on the lane in use (one, when `lone` steers).
    fn pass(e: *Engine, rows: []const [W]i32, first: usize, replaced: []const Replace, last: bool, probe: ?Probe, many: ?[]const Turn, lone: []const Steer) !void {
        const k = &e.k;
        const s = e.stream;
        const t = rows.len;
        const R = e.o.max_rows;
        const pos0 = e.position;
        // an extra text id (a phoneme, start or pause token) embeds as the padding row; its fp32 row comes below
        var extra = false;
        for (rows) |r| extra = extra or r[cfg.codebooks] >= e.w.text_rows;
        if (extra) {
            for (e.row_host[0..t], rows) |*to, from| {
                to.* = from;
                if (from[cfg.codebooks] >= e.w.text_rows) to[cfg.codebooks] = cfg.text_vocab;
            }
            try e.put(e.rows, std.mem.sliceAsBytes(e.row_host[0..t]));
        } else try e.put(e.rows, std.mem.sliceAsBytes(rows));
        try k.embed(s, e.rows, e.w.tables, cfg.audio_vocab, e.mixed, D, t);
        if (extra) for (rows, 0..) |r, i| {
            if (r[cfg.codebooks] < e.w.text_rows) continue;
            var vec: [D]f32 = undefined;
            e.w.extraRow(@intCast(r[cfg.codebooks]), &vec);
            try e.put(e.row_up, std.mem.asBytes(&vec));
            try e.buf(e.mixed).copyFrom(i * D * 4, e.row_up, D * 4, s.handle);
        };
        if (probe) |p| try p.f(p.ctx, "emb_sum", e.mixed, t, D);
        for (replaced) |r| if (r.row >= first and r.row < first + t) {
            try e.put(e.row_up, std.mem.asBytes(&r.vec));
            try e.buf(e.mixed).copyFrom((r.row - first) * D * 4, e.row_up, D * 4, s.handle);
        };
        if (probe) |p| try p.f(p.ctx, "emb_cond", e.mixed, t, D);
        try k.rmsnorm(s, e.mixed, D, 0, e.slot(0), D, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "emb_norm", e.slot(0), t, D);
        try e.steerSource(0, many, lone);
        const q = e.w.queries;
        for (&e.w.layers, 0..) |*L, li| {
            const attn_out = e.slot(2 * li + 1);
            const ffn_out = e.slot(2 * li + 2);
            try k.attnres(s, e.bank, 2 * li + 1, R, q + 2 * li * D * 2, e.mixed, D, cfg.eps, t);
            if (probe) |p| try p.at("attnres_attn_{d}", li, e.mixed, t, D);
            try k.rmsnorm(s, e.mixed, D, L.attention_norm, e.xn, D, D, cfg.eps, t);
            if (probe) |p| try p.at("attn_in_{d}", li, e.xn, t, D);
            switch (L.mix) {
                .gdn => |g| {
                    const ps = cfg.gdn_proj;
                    try k.linear(s, g.in_proj, e.xn, D, D, ps, e.proj, ps, t, .{});
                    if (many) |turns| {
                        for (turns, 0..) |turn, b| try k.convSilu(s, e.proj + b * ps * 4, ps, 1, g.conv, e.lanes.items[turn.lane].conv[li], e.conv_out + b * cfg.gdn_conv * 4, cfg.gdn_conv);
                    } else try k.convSilu(s, e.proj, ps, t, g.conv, e.conv[li], e.conv_out, cfg.gdn_conv);
                    if (probe) |p| try p.at("gdn_conv_{d}", li, e.conv_out, t, cfg.gdn_conv);
                    try k.gdnPrep(s, e.conv_out, e.proj, ps, 4 * cfg.gdn_inner, g.a_log, g.dt_bias, e.gates, t);
                    if (many) |turns| {
                        for (turns, 0..) |turn, b| try k.gdn(s, e.conv_out + b * cfg.gdn_conv * 4, e.gates + b * cfg.gdn_heads * 2 * 4, 1, e.lanes.items[turn.lane].gstate[li], e.o32 + b * cfg.gdn_inner * 4);
                    } else try k.gdn(s, e.conv_out, e.gates, t, e.gstate[li], e.o32);
                    if (probe) |p| try p.at("gdn_recurrent_{d}", li, e.o32, t, cfg.gdn_inner);
                    try k.gatedNorm(s, e.o32, e.proj, ps, cfg.gdn_conv, g.out_weight, e.y, t);
                    if (probe) |p| try p.at("gdn_gated_{d}", li, e.y, t, cfg.gdn_inner);
                    try k.linear(s, g.wo, e.y, cfg.gdn_inner, cfg.gdn_inner, D, attn_out, D, t, .{});
                },
                .attn => |a| {
                    try k.linear(s, a.wq, e.xn, D, D, cfg.attn_inner, e.q, cfg.attn_inner, t, .{});
                    try k.linear(s, a.wkv, e.xn, D, D, 2 * cfg.kv_width, e.kvraw, 2 * cfg.kv_width, t, .{});
                    try k.linear(s, a.gater, e.xn, D, D, cfg.heads, e.gate, cfg.heads, t, .{});
                    if (many) |turns| {
                        for (turns, 0..) |turn, b| {
                            const l = &e.lanes.items[turn.lane];
                            const qb = e.q + b * cfg.attn_inner * 4;
                            try k.attnPrep(s, qb, e.kvraw + b * 2 * cfg.kv_width * 4, a.temp, l.position, l.kv[li][0], l.kv[li][1], cfg.rope_theta, 1);
                            try k.attention(s, qb, l.kv[li][0], l.kv[li][1], e.gate + b * cfg.heads * 4, l.position, e.y + b * cfg.attn_inner * 4, 1);
                            if (turn.probe) |p| {
                                try e.buf(e.probe_q + (b * probe_layers + li / cfg.gdn_period) * cfg.attn_inner * 4).copyFrom(0, qb, cfg.attn_inner * 4, s.handle);
                                for (p.masks) |m| if (m.layer == li) try e.maskedHead(l, li, b, m, p);
                            }
                        }
                    } else {
                        try k.attnPrep(s, e.q, e.kvraw, a.temp, pos0, e.kv[li][0], e.kv[li][1], cfg.rope_theta, t);
                        try k.attention(s, e.q, e.kv[li][0], e.kv[li][1], e.gate, pos0, e.y, t);
                    }
                    try k.linear(s, a.wo, e.y, cfg.attn_inner, cfg.attn_inner, D, attn_out, D, t, .{});
                },
            }
            if (probe) |p| try p.at("attn_out_{d}", li, attn_out, t, D);
            try e.steerSource(2 * li + 1, many, lone);
            try k.attnres(s, e.bank, 2 * li + 2, R, q + (2 * li + 1) * D * 2, e.mixed, D, cfg.eps, t);
            if (probe) |p| try p.at("attnres_ffn_{d}", li, e.mixed, t, D);
            try k.rmsnorm(s, e.mixed, D, L.ffn_norm, e.xn, D, D, cfg.eps, t);
            if (probe) |p| try p.at("ffn_in_{d}", li, e.xn, t, D);
            switch (L.ffn) {
                .dense => |f| {
                    const hidden = cfg.ffn_hidden;
                    try k.linear(s, f.w_in, e.xn, D, D, 2 * hidden, e.proj, 2 * hidden, t, .{});
                    try k.mulSilu(s, e.proj, 2 * hidden, 0, hidden, hidden, e.act, t);
                    try k.linear(s, f.w_out, e.act, hidden, hidden, D, ffn_out, D, t, .{});
                },
                .moe => |m| {
                    const hidden = cfg.moe_hidden;
                    const latent = cfg.moe_latent;
                    try k.rmsnorm(s, e.xn, D, m.eda, e.rn, D, D, cfg.eps, t);
                    try k.linear(s, m.router, e.rn, D, D, cfg.experts, e.rl, cfg.experts, t, .{});
                    if (probe) |p| try p.at("router_logits_{d}", li, e.rl, t, cfg.experts);
                    try k.route(s, e.rl, m.bias, cfg.experts, e.picks, e.rw, t);
                    try k.linear(s, m.latent_down, e.xn, D, D, latent, e.latent, latent, t, .{});
                    try k.linear(s, m.gate_up, e.latent, latent, latent, 2 * hidden, e.gu, 2 * hidden, t * cfg.top_k, .{ .picks = e.picks, .pick_stride = 2 * hidden * latent, .xrep = cfg.top_k });
                    try k.mulSilu(s, e.gu, 2 * hidden, hidden, 0, hidden, e.hid, t * cfg.top_k);
                    try k.linear(s, m.down, e.hid, hidden, hidden, latent, e.down, latent, t * cfg.top_k, .{ .picks = e.picks, .pick_stride = latent * hidden });
                    try k.combine(s, e.down, e.rw, latent, e.comb, t);
                    try k.linear(s, m.latent_up, e.comb, latent, latent, D, ffn_out, D, t, .{});
                },
            }
            if (probe) |p| try p.at("ffn_out_{d}", li, ffn_out, t, D);
            try e.steerSource(2 * li + 2, many, lone);
        }
        try k.attnres(s, e.bank, cfg.sources, R, q + (cfg.sources - 1) * D * 2, e.mixed, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "attnres_final", e.mixed, t, D);
        try k.rmsnorm(s, e.mixed, D, e.w.out_norm, e.xn, D, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "out_norm", e.xn, t, D);
        if (probe != null or many != null) {
            try k.linear(s, e.w.head, e.xn, D, D, cfg.head_rows, e.logits, cfg.head_rows, t, .{});
            try k.softcap(s, e.logits, t * cfg.head_rows, cfg.softcap);
            if (probe) |p| try p.f(p.ctx, "logits", e.logits, t, cfg.head_rows);
        } else if (last) {
            try k.linear(s, e.w.head, e.xn + (t - 1) * D * 4, D, D, cfg.head_rows, e.logits, cfg.head_rows, 1, .{});
            try k.softcap(s, e.logits, cfg.head_rows, cfg.softcap);
        }
        if (many == null) e.position = pos0 + t;
    }

    /// The last row's logits [codebooks][audio_vocab] to the host.
    pub fn lastLogits(e: *Engine, out: *[cfg.head_rows]f32) !void {
        try e.download(e.logits, std.mem.asBytes(out));
    }
};
