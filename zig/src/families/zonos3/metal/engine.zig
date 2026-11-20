//! ZONOS3 Mini's backbone on Metal: frame rows in, the next frame's logits out, one fp32 row path for all rows.
const std = @import("std");
const cfg = @import("../config.zig");
const gpu = @import("gpu.zig");
const wts = @import("weights.zig");
const k = @import("kernels.zig");
const adp = @import("adapter.zig");
const rows_ = @import("engine_rows.zig");
const metallib = @import("zonos3_metallib").bytes;

const D = cfg.dim;
const W = cfg.frame_width;
const Gpu = gpu.Gpu;

/// A host's say on a load: an error refuses it.
pub const Gate = struct { ptr: *anyopaque, check: *const fn (*anyopaque, *const Gpu) anyerror!void };

pub const Options = struct {
    device: u32 = 0, // the Mac's one GPU
    max_ctx: usize = 4096, // positions (prompt rows + frames), at most the model's 32768
    max_rows: usize = 64, // rows a pass
    gate: ?Gate = null, // asked once the device is open, before anything is allocated on it
    adapter: []const u8 = "", // the release's speech-edit adapter file; empty: none, and no edit fills

    pub fn validate(o: Options) !void {
        if (o.device != 0) return error.NoSuchDevice;
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

pub const Steer = @import("../steering.zig").Steer;
pub const Probed = @import("../laci_probe.zig").Probed;
const probe_layers = @import("../laci_probe.zig").layers;

fn fac(p: adp.Pair) ?adp.Pair {
    return if (p.rank == 0) null else p;
}

/// One lane's next row in a pass over several lanes, and where its logits go.
pub const Turn = struct {
    lane: usize,
    row: [W]i32,
    out: *[cfg.head_rows]f32,
    steer: []const Steer = &.{},
    probe: ?*Probed = null,
    /// Set: the row runs on the edit adapter, and its span-end logits go here.
    end: ?*[cfg.codebooks]f32 = null,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    o: Options,
    c: cfg.Config,
    g: *Gpu,
    w: *wts.Weights,
    /// The speech-edit adapter, when the engine was made with one; `adapt` runs `run` and `runSteered` on it.
    adapter: ?*adp.Adapter = null,
    adapt: bool = false,
    last_row: usize = 0, // the row of the last pass whose logits `logits` holds first
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
    steer: u64, // [R][sources][D]: a pass's steering deltas, staged before it runs
    probe_q: u64, // [R][attention layers][512]: probed rows' queries on their way out
    mask_kv: [2]u64, // [max_ctx][128] each: the keys and values a masked head sees, closed up
    mask_out: u64, // [512]: a row's attention over them
    // with an adapter
    lo: u64, // [2R][64]: a row's down product
    head_lo: u64, // [R][64]: the head's, kept for the span-end logits
    pair: u64, // [R][2][head_rows]: a linear's rows, each followed by its shift
    ones: u64, // [2R][2], each 1.0

    /// The backbone on its own device.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        const g = try Gpu.open(gpa, metallib);
        errdefer g.close();
        const e = try createOn(gpa, io, g, model_dir, o);
        return e;
    }

    /// The backbone on `g`, which it then owns.
    pub fn createOn(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, model_dir: []const u8, o: Options) !*Engine {
        try o.validate();
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.o = o;
        e.g = g;
        e.position = 0;
        e.lanes = .empty;
        e.active = 0;
        e.adapter = null;
        e.adapt = false;
        e.last_row = 0;
        e.c = try cfg.Config.load(gpa, io, model_dir);
        if (o.gate) |gate| try gate.check(gate.ptr, g);
        e.w = try wts.load(gpa, io, g, model_dir, e.c);
        errdefer {
            e.w.deinit();
            gpa.destroy(e.w);
        }
        if (o.adapter.len != 0) {
            const model = try std.fs.path.join(gpa, &.{ model_dir, "model.pth" });
            defer gpa.free(model);
            e.adapter = try adp.load(gpa, io, g, o.adapter, model);
        }
        errdefer if (e.adapter) |a| {
            a.deinit();
            gpa.destroy(a);
        };
        if (e.adapter) |a| {
            if (a.markers.len != @import("../edit.zig").marker_rows * D) return error.UnexpectedAdapterMarkers;
            try e.w.addMarkers(a.markers);
        }
        e.row_host = try gpa.alloc([W]i32, o.max_rows);
        errdefer gpa.free(e.row_host);
        errdefer e.lanes.deinit(gpa);
        try e.allocState();
        if (e.adapter != null) @memset(std.mem.bytesAsSlice(f32, g.host(e.ones, o.max_rows * cfg.top_k * 2 * 4)), 1);
        try e.reset();
        return e;
    }

    pub fn destroy(e: *Engine) void {
        if (e.adapter) |a| {
            a.deinit();
            e.gpa.destroy(a);
        }
        e.w.deinit();
        e.gpa.destroy(e.w);
        e.gpa.free(e.row_host);
        e.lanes.deinit(e.gpa);
        e.g.close();
        e.gpa.destroy(e);
    }

    fn alloc(e: *Engine, bytes: usize) !u64 {
        return e.g.alloc(bytes);
    }

    /// Bytes of the state and workspaces `o` allocates: the allocation code itself, counting.
    pub fn stateBytes(o: Options) usize {
        var g: Gpu = undefined;
        g.counted = 0;
        var e: Engine = undefined;
        e.o = o;
        e.g = &g;
        e.allocState() catch unreachable;
        return g.counted.?;
    }

    /// Device bytes of a second lane's state under `o`.
    pub fn laneBytes(o: Options) usize {
        var g: Gpu = undefined;
        g.counted = 0;
        var e: Engine = undefined;
        e.o = o;
        e.g = &g;
        _ = e.allocLane() catch unreachable;
        return g.counted.?;
    }

    /// One more stream's state on the device; its index for `use`. `reset` zeroes the lane in use.
    pub fn addLane(e: *Engine) !usize {
        const owner = e.g.own(null);
        defer _ = e.g.own(owner);
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
        if (e.g.counted == null) try e.lanes.append(e.gpa, l);
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
        e.steer = try e.alloc(R * cfg.sources * D * 4);
        e.probe_q = try e.alloc(R * probe_layers * cfg.attn_inner * 4);
        for (&e.mask_kv) |*m| m.* = try e.alloc(e.o.max_ctx * cfg.kv_width * 4);
        e.mask_out = try e.alloc(cfg.attn_inner * 4);
        if (e.o.adapter.len == 0) return;
        e.lo = try e.alloc(R * cfg.top_k * adp.max_rank * 4);
        e.head_lo = try e.alloc(R * adp.max_rank * 4);
        e.pair = try e.alloc(R * 2 * cfg.head_rows * 4);
        e.ones = try e.alloc(R * cfg.top_k * 2 * 4);
    }

    pub fn slot(e: *const Engine, source: usize) u64 {
        return e.bank + source * e.o.max_rows * D * 4;
    }

    /// The next rows at the current position; `logits` then holds the last row's (every row's under a one-pass probe).
    pub fn run(e: *Engine, rows: []const [W]i32, replaced: []const Replace, probe: ?Probe) !void {
        if (rows.len == 0) return error.NoRows;
        if (e.position + rows.len > e.o.max_ctx) return error.ContextFull;
        if (probe != null and rows.len > e.o.max_rows) return error.ProbeNeedsOnePass;
        if (e.adapt and e.adapter == null) return error.NoAdapter;
        for (rows) |r| {
            for (r[0..cfg.codebooks]) |code| if (code < 0 or code >= e.audioIds()) return error.BadFrameRow;
            if (r[cfg.codebooks] < 0 or r[cfg.codebooks] >= e.w.textIds()) return error.BadFrameRow;
        }
        var at: usize = 0;
        while (at < rows.len) {
            const t = @min(e.o.max_rows, rows.len - at);
            try e.pass(rows[at..][0..t], at, replaced, at + t == rows.len, probe, null, &.{}, e.adapt);
            at += t;
        }
    }

    /// One generated row on the lane in use with `steer` added to its bank sources (synth: a decode step's deltas).
    pub fn runSteered(e: *Engine, row: [W]i32, steer: []const Steer) !void {
        if (e.position + 1 > e.o.max_ctx) return error.ContextFull;
        if (e.adapt and e.adapter == null) return error.NoAdapter;
        for (row[0..cfg.codebooks]) |code| if (code < 0 or code >= e.audioIds()) return error.BadFrameRow;
        if (row[cfg.codebooks] < 0 or row[cfg.codebooks] >= e.w.text_rows) return error.BadFrameRow;
        for (steer) |st| if (st.source >= cfg.sources) return error.NoSuchSource;
        try e.pass(&.{row}, 0, &.{}, true, null, null, steer, e.adapt);
    }

    /// Audio ids a row may hold: the model's, and the span-end id with the adapter.
    fn audioIds(e: *const Engine) i32 {
        return cfg.audio_vocab + @as(i32, @intFromBool(e.adapter != null));
    }

    /// One row of each turn's lane, logits to its `out`; lanes on the adapter run in a pass of their own.
    pub fn runLanes(e: *Engine, turns: []const Turn) !void {
        if (turns.len == 0) return;
        if (turns.len > e.o.max_rows or turns.len > 64) return error.TooManyLanes;
        e.lanes.items[e.active].position = e.position;
        var adapted: usize = 0;
        for (turns, 0..) |turn, b| {
            if (turn.lane >= e.lanes.items.len) return error.NoSuchLane;
            for (turns[0..b]) |earlier| if (earlier.lane == turn.lane) return error.LaneTwice;
            if (e.lanes.items[turn.lane].position + 1 > e.o.max_ctx) return error.ContextFull;
            if (turn.end != null and e.adapter == null) return error.NoAdapter;
            const ids = if (turn.end != null) e.audioIds() else cfg.audio_vocab;
            for (turn.row[0..cfg.codebooks]) |code| if (code < 0 or code >= ids) return error.BadFrameRow;
            if (turn.row[cfg.codebooks] < 0 or turn.row[cfg.codebooks] >= e.w.text_rows) return error.BadFrameRow;
            for (turn.steer) |st| if (st.source >= cfg.sources) return error.NoSuchSource;
            adapted += @intFromBool(turn.end != null);
        }
        if (adapted == 0 or adapted == turns.len) {
            try e.lanesPass(turns, adapted != 0);
        } else {
            var group: [64]Turn = undefined;
            for ([_]bool{ false, true }) |on| {
                var n: usize = 0;
                for (turns) |turn| if ((turn.end != null) == on) {
                    group[n] = turn;
                    n += 1;
                };
                try e.lanesPass(group[0..n], on);
            }
        }
        for (turns) |turn| e.lanes.items[turn.lane].position += 1;
        e.position = e.lanes.items[e.active].position;
    }

    fn lanesPass(e: *Engine, turns: []const Turn, adapt: bool) !void {
        var rows: [64][W]i32 = undefined;
        for (turns, rows[0..turns.len]) |turn, *row| row.* = turn.row;
        try e.pass(rows[0..turns.len], 0, &.{}, true, null, turns, &.{}, adapt);
        for (turns, 0..) |turn, b| {
            try e.g.get(e.logits + b * cfg.head_rows * 4, std.mem.asBytes(turn.out));
            if (turn.probe) |p| try e.g.get(e.probe_q + b * probe_layers * cfg.attn_inner * 4, std.mem.asBytes(&p.q));
        }
        if (adapt) for (turns, 0..) |turn, b| try rows_.endLogits(e, b, b, turn.end.?);
    }

    /// The last row's span-end logits, after a `run` or `runSteered` on the adapter.
    pub fn lastEnd(e: *Engine, out: *[cfg.codebooks]f32) !void {
        if (e.adapter == null) return error.NoAdapter;
        try rows_.endLogits(e, e.last_row, 0, out);
    }

    /// The keys of cache rows [start, start + out.len / 128) of attention layer `layer` on the lane in use.
    pub fn keys(e: *Engine, layer: usize, start: usize, out: []f32) !void {
        if (cfg.usesGdn(layer) or out.len % cfg.kv_width != 0 or start + out.len / cfg.kv_width > e.position) return error.NoSuchKeys;
        try e.download(e.kv[layer][0] + start * cfg.kv_width * 4, std.mem.sliceAsBytes(out));
    }

    /// `many`: a row of each turn's lane at its position; else rows on the lane in use (one, when `lone` steers).
    fn pass(e: *Engine, rows: []const [W]i32, first: usize, replaced: []const Replace, last: bool, probe: ?Probe, many: ?[]const Turn, lone: []const Steer, adapt: bool) !void {
        const g = e.g;
        const t = rows.len;
        const R = e.o.max_rows;
        const pos0 = e.position;
        // an id past the tables (an extra text token, the span-end code) embeds as padding; its fp32 row comes below
        var extra = false;
        for (rows) |r| extra = extra or r[cfg.codebooks] >= e.w.text_rows or std.mem.indexOfScalar(i32, r[0..cfg.codebooks], cfg.span_end_id) != null;
        if (extra) {
            for (e.row_host[0..t], rows) |*to, from| {
                to.* = from;
                if (from[cfg.codebooks] >= e.w.text_rows) to[cfg.codebooks] = cfg.text_vocab;
                for (to[0..cfg.codebooks]) |*code| if (code.* == cfg.span_end_id) {
                    code.* = cfg.audio_pad_id;
                };
            }
            try g.put(e.rows, std.mem.sliceAsBytes(e.row_host[0..t]));
        } else try g.put(e.rows, std.mem.sliceAsBytes(rows));
        try rows_.stageSteers(e, many, lone);
        k.embed(g, e.rows, e.w.tables, cfg.audio_vocab, e.mixed, D, t);
        if (extra) for (rows, 0..) |r, i| {
            var vec: [D]f32 = undefined;
            if (std.mem.indexOfScalar(i32, r[0..cfg.codebooks], cfg.span_end_id) != null) {
                try rows_.hostRow(e, r, &vec);
            } else if (r[cfg.codebooks] >= e.w.text_rows) {
                e.w.extraRow(@intCast(r[cfg.codebooks]), &vec);
            } else continue;
            try g.put(e.mixed + i * D * 4, std.mem.asBytes(&vec));
        };
        if (probe) |p| try p.f(p.ctx, "emb_sum", e.mixed, t, D);
        for (replaced) |r| if (r.row >= first and r.row < first + t) {
            try g.put(e.mixed + (r.row - first) * D * 4, std.mem.asBytes(&r.vec));
        };
        if (probe) |p| try p.f(p.ctx, "emb_cond", e.mixed, t, D);
        k.rmsnorm(g, e.mixed, D, 0, e.slot(0), D, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "emb_norm", e.slot(0), t, D);
        rows_.steerSource(e, 0, many, lone);
        const q = e.w.queries;
        for (&e.w.layers, 0..) |*L, li| {
            const A: adp.Layer = if (adapt) e.adapter.?.layers[li] else .{};
            const attn_out = e.slot(2 * li + 1);
            const ffn_out = e.slot(2 * li + 2);
            k.attnres(g, e.bank, 2 * li + 1, R, q + 2 * li * D * 2, e.mixed, D, cfg.eps, t);
            if (probe) |p| try p.at("attnres_attn_{d}", li, e.mixed, t, D);
            k.rmsnorm(g, e.mixed, D, L.attention_norm, e.xn, D, D, cfg.eps, t);
            if (probe) |p| try p.at("attn_in_{d}", li, e.xn, t, D);
            switch (L.mix) {
                .gdn => |gd| {
                    const ps = cfg.gdn_proj;
                    rows_.lin(e, fac(A.in_proj), e.lo, gd.in_proj, e.xn, D, ps, e.proj, t, .{});
                    if (many) |turns| {
                        for (turns, 0..) |turn, b| k.convSilu(g, e.proj + b * ps * 4, ps, 1, gd.conv, e.lanes.items[turn.lane].conv[li], e.conv_out + b * cfg.gdn_conv * 4, cfg.gdn_conv);
                    } else k.convSilu(g, e.proj, ps, t, gd.conv, e.conv[li], e.conv_out, cfg.gdn_conv);
                    if (probe) |p| try p.at("gdn_conv_{d}", li, e.conv_out, t, cfg.gdn_conv);
                    k.gdnPrep(g, e.conv_out, e.proj, ps, 4 * cfg.gdn_inner, gd.a_log, gd.dt_bias, e.gates, t);
                    if (many) |turns| {
                        for (turns, 0..) |turn, b| k.gdn(g, e.conv_out + b * cfg.gdn_conv * 4, e.gates + b * cfg.gdn_heads * 2 * 4, 1, e.lanes.items[turn.lane].gstate[li], e.o32 + b * cfg.gdn_inner * 4);
                    } else k.gdn(g, e.conv_out, e.gates, t, e.gstate[li], e.o32);
                    if (probe) |p| try p.at("gdn_recurrent_{d}", li, e.o32, t, cfg.gdn_inner);
                    k.gatedNorm(g, e.o32, e.proj, ps, cfg.gdn_conv, gd.out_weight, e.y, t);
                    if (probe) |p| try p.at("gdn_gated_{d}", li, e.y, t, cfg.gdn_inner);
                    rows_.lin(e, fac(A.wo), e.lo, gd.wo, e.y, cfg.gdn_inner, D, attn_out, t, .{});
                },
                .attn => |a| {
                    rows_.lin(e, fac(A.wq), e.lo, a.wq, e.xn, D, cfg.attn_inner, e.q, t, .{});
                    rows_.lin(e, fac(A.wkv), e.lo, a.wkv, e.xn, D, 2 * cfg.kv_width, e.kvraw, t, .{});
                    rows_.lin(e, fac(A.gater), e.lo, a.gater, e.xn, D, cfg.heads, e.gate, t, .{});
                    if (many) |turns| {
                        for (turns, 0..) |turn, b| {
                            const l = &e.lanes.items[turn.lane];
                            const qb = e.q + b * cfg.attn_inner * 4;
                            k.attnPrep(g, qb, e.kvraw + b * 2 * cfg.kv_width * 4, a.temp, l.position, l.kv[li][0], l.kv[li][1], cfg.rope_theta, 1);
                            k.attention(g, qb, l.kv[li][0], l.kv[li][1], e.gate + b * cfg.heads * 4, l.position, e.y + b * cfg.attn_inner * 4, 1);
                            if (turn.probe) |p| {
                                g.copy(e.probe_q + (b * probe_layers + li / cfg.gdn_period) * cfg.attn_inner * 4, qb, cfg.attn_inner * 4);
                                for (p.masks) |m| if (m.layer == li) try rows_.maskedHead(e, l, li, b, m, p);
                            }
                        }
                    } else {
                        k.attnPrep(g, e.q, e.kvraw, a.temp, pos0, e.kv[li][0], e.kv[li][1], cfg.rope_theta, t);
                        k.attention(g, e.q, e.kv[li][0], e.kv[li][1], e.gate, pos0, e.y, t);
                    }
                    rows_.lin(e, fac(A.wo), e.lo, a.wo, e.y, cfg.attn_inner, D, attn_out, t, .{});
                },
            }
            if (probe) |p| try p.at("attn_out_{d}", li, attn_out, t, D);
            rows_.steerSource(e, 2 * li + 1, many, lone);
            k.attnres(g, e.bank, 2 * li + 2, R, q + (2 * li + 1) * D * 2, e.mixed, D, cfg.eps, t);
            if (probe) |p| try p.at("attnres_ffn_{d}", li, e.mixed, t, D);
            k.rmsnorm(g, e.mixed, D, L.ffn_norm, e.xn, D, D, cfg.eps, t);
            if (probe) |p| try p.at("ffn_in_{d}", li, e.xn, t, D);
            switch (L.ffn) {
                .dense => |f| {
                    const hidden = cfg.ffn_hidden;
                    rows_.lin(e, fac(A.w_in), e.lo, f.w_in, e.xn, D, 2 * hidden, e.proj, t, .{});
                    k.mulSilu(g, e.proj, 2 * hidden, 0, hidden, hidden, e.act, t);
                    rows_.lin(e, fac(A.w_out), e.lo, f.w_out, e.act, hidden, D, ffn_out, t, .{});
                },
                .moe => |m| {
                    const hidden = cfg.moe_hidden;
                    const latent = cfg.moe_latent;
                    k.rmsnorm(g, e.xn, D, m.eda, e.rn, D, D, cfg.eps, t);
                    rows_.lin(e, fac(A.router), e.lo, m.router, e.rn, D, cfg.experts, e.rl, t, .{});
                    if (probe) |p| try p.at("router_logits_{d}", li, e.rl, t, cfg.experts);
                    k.route(g, e.rl, if (A.bias != 0) A.bias else m.bias, cfg.experts, e.picks, e.rw, t);
                    rows_.lin(e, fac(A.latent_down), e.lo, m.latent_down, e.xn, D, latent, e.latent, t, .{});
                    rows_.lin(e, fac(A.gate_up), e.lo, m.gate_up, e.latent, latent, 2 * hidden, e.gu, t * cfg.top_k, .{ .picks = e.picks, .pick_stride = 2 * hidden * latent, .xrep = cfg.top_k });
                    k.mulSilu(g, e.gu, 2 * hidden, hidden, 0, hidden, e.hid, t * cfg.top_k);
                    rows_.lin(e, fac(A.down), e.lo, m.down, e.hid, hidden, latent, e.down, t * cfg.top_k, .{ .picks = e.picks, .pick_stride = latent * hidden });
                    k.combine(g, e.down, e.rw, latent, e.comb, t);
                    rows_.lin(e, fac(A.latent_up), e.lo, m.latent_up, e.comb, latent, D, ffn_out, t, .{});
                },
            }
            if (probe) |p| try p.at("ffn_out_{d}", li, ffn_out, t, D);
            rows_.steerSource(e, 2 * li + 2, many, lone);
        }
        k.attnres(g, e.bank, cfg.sources, R, q + (cfg.sources - 1) * D * 2, e.mixed, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "attnres_final", e.mixed, t, D);
        k.rmsnorm(g, e.mixed, D, e.w.out_norm, e.xn, D, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "out_norm", e.xn, t, D);
        if (probe != null or many != null) {
            rows_.lin(e, if (adapt) fac(e.adapter.?.head) else null, e.head_lo, e.w.head, e.xn, D, cfg.head_rows, e.logits, t, .{});
            k.softcap(g, e.logits, t * cfg.head_rows, cfg.softcap);
            if (probe) |p| try p.f(p.ctx, "logits", e.logits, t, cfg.head_rows);
        } else if (last) {
            rows_.lin(e, if (adapt) fac(e.adapter.?.head) else null, e.head_lo, e.w.head, e.xn + (t - 1) * D * 4, D, cfg.head_rows, e.logits, 1, .{});
            k.softcap(g, e.logits, cfg.head_rows, cfg.softcap);
        }
        e.last_row = t - 1;
        if (many == null) e.position = pos0 + t;
    }

    /// The last row's logits [codebooks][audio_vocab] to the host.
    pub fn lastLogits(e: *Engine, out: *[cfg.head_rows]f32) !void {
        try e.g.get(e.logits, std.mem.asBytes(out));
    }

    pub fn download(e: *Engine, src: u64, out: []u8) !void {
        try e.g.get(src, out);
    }

    /// A new request on the lane in use: zero recurrent states, position 0. Cache rows are written before read.
    pub fn reset(e: *Engine) !void {
        try e.g.flush();
        @memset(e.g.host(e.state, e.state_bytes), 0);
        e.position = 0;
    }
};
