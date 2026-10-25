//! Drama 3 generation on Metal: prompt frame rows in chunks, then one frame a step, committed as it is encoded.
const std = @import("std");
const model = @import("../model.zig");
const cfg = @import("../config.zig");
const common = @import("../common.zig");
const gpu = @import("gpu.zig");
const wts = @import("weights.zig");
const k = @import("kernels.zig");
const fwd = @import("forward.zig");

const H = cfg.hidden;
const W = cfg.frame_width;

pub const Options = common.Options;
pub const Sampling = common.Sampling;
pub const Probe = common.Probe;
pub const nowNs = common.nowNs;
const ms = common.ms;

var peak_seen: u64 = 0;

/// The device's peak allocation in bytes so far (Metal's currentAllocatedSize at each wait).
pub fn peakBytes() u64 {
    return peak_seen;
}

pub const Engine = struct {
    gpa: std.mem.Allocator,
    owns_model: bool,
    state_start: usize,
    state_end: usize,
    io: std.Io,
    o: Options,
    c: cfg.Config,
    g: *gpu.Gpu,
    w: *wts.Weights,
    kv_bb: [cfg.layers][2]u64, // full-attention layers: K, V [max_ctx, 2, 256] bf16
    kv_tw: [cfg.tower_layers][2]u64,
    conv: [cfg.layers]u64, // GDN layers: [8192, 3] bf16
    gstate: [cfg.layers]u64, // GDN layers: [32, 128, 128] fp32
    state_bufs: [2]u64,
    state_bytes: [2]usize,
    pos: u64, // i32: the newest row's position
    pos_bb: u64, // i32: the next backbone row's position
    bb_input: u64, // i32 [11]
    t_xn: u64,
    t_tmp: u64,
    t_proj: u64,
    t_y: u64,
    t_q: u64,
    step: u64, // u64: the sampler's step
    count: u64, // i32: frames made
    window: u64, // i32 [16]
    cur_input: u64, // i32 [11]
    frame: u64, // i32 [11]
    frames_out: u64, // i32 [max_frames, 11]
    pframes: u64, // i32 [max_ctx, 11]
    x: u64,
    xn: u64,
    tmp: u64,
    proj: u64,
    conv_out: u64,
    o32: u64,
    y: u64,
    q: u64,
    sp: u64,
    sact: u64,
    shout: u64,
    picks: u64,
    rw: u64,
    gu: u64,
    down: u64,
    xq: u64,
    xsc: u64,
    gq: u64,
    gsc: u64,
    counts: u64, // i32 [256]: the prompt plan
    members: u64, // i32 [256, max_rows * 8]
    hfin: u64,
    th: [2]u64,
    mix: u64,
    mixed: u64,
    act: u64,
    hlast: u64,
    cand_logits: u64,
    code_logits: u64,
    rows_done: usize = 0,
    prompt_rows: usize = 0,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.owns_model = true;
        e.io = io;
        e.o = o;
        e.rows_done = 0;
        e.prompt_rows = 0;
        e.c = try cfg.Config.load(gpa, io, model_dir);
        e.g = try gpu.Gpu.open(gpa, @import("drama3_metallib").bytes);
        errdefer e.g.close();
        e.w = try wts.load(e.g, io, model_dir, e.c);
        errdefer {
            e.g.releaseFrom(1);
            e.w.deinit(io);
            gpa.destroy(e.w);
        }
        e.state_start = e.g.bufs.items.len;
        errdefer e.g.releaseFrom(e.state_start);
        try e.allocState(false);
        e.state_end = e.g.bufs.items.len;
        if (o.warmup) {
            var rows: [4][W]i32 = @splat(@splat(0));
            for (&rows, 0..) |*r, i| r[0] = @intCast(198 + i);
            try e.prefill(&rows, null);
            for (0..2) |_| try e.frameStep(.{}, true, null);
            try e.flush();
        }
        return e;
    }

    pub fn destroy(e: *Engine) void {
        e.g.flush() catch {};
        if (e.owns_model) {
            e.g.close();
            e.w.deinit(e.io);
            e.gpa.destroy(e.w);
        }
        e.gpa.destroy(e);
    }

    pub fn fork(base: *Engine) !*Engine {
        return base.forkState(false);
    }

    /// CFG's independent backbone state, without codec tower caches, output frames or tower scratch.
    pub fn forkBackbone(base: *Engine) !*Engine {
        return base.forkState(true);
    }

    fn forkState(base: *Engine, backbone_only: bool) !*Engine {
        const e = try base.gpa.create(Engine);
        errdefer base.gpa.destroy(e);
        e.* = base.*;
        e.owns_model = false;
        if (backbone_only) e.o.max_rows = 1;
        e.state_start = e.g.bufs.items.len;
        errdefer e.g.releaseFrom(e.state_start);
        try e.allocState(backbone_only);
        e.state_end = e.g.bufs.items.len;
        return e;
    }

    pub fn poison(e: *Engine) !void {
        try e.flush();
        for (e.g.bufs.items[e.state_start..e.state_end]) |buffer| @memset(buffer.contents()[0..buffer.length()], 0xa5);
    }

    fn alloc(e: *Engine, bytes: usize) !u64 {
        return e.g.alloc(bytes);
    }

    fn allocState(e: *Engine, backbone_only: bool) !void {
        const R = e.o.max_rows;
        const ctxn = e.o.max_ctx;
        const kvb = ctxn * cfg.kv_heads * cfg.head_dim * 2;
        for (0..cfg.layers) |li| e.kv_bb[li] = if (e.c.full_attention[li]) .{ try e.alloc(kvb), try e.alloc(kvb) } else .{ 0, 0 };
        for (0..cfg.tower_layers) |ti| e.kv_tw[ti] = if (backbone_only) .{ 0, 0 } else .{ try e.alloc(kvb), try e.alloc(kvb) };
        var n_gdn: usize = 0;
        for (e.c.full_attention) |f| n_gdn += @intFromBool(!f);
        const conv_b = cfg.conv_channels * 3 * 2;
        const gs_b = cfg.lin_v_heads * cfg.lin_dim * cfg.lin_dim * 4;
        e.state_bytes = .{ n_gdn * conv_b, n_gdn * gs_b };
        e.state_bufs = .{ try e.alloc(e.state_bytes[0]), try e.alloc(e.state_bytes[1]) };
        var gi: usize = 0;
        for (0..cfg.layers) |li| {
            if (e.c.full_attention[li]) {
                e.conv[li] = 0;
                e.gstate[li] = 0;
            } else {
                e.conv[li] = e.state_bufs[0] + gi * conv_b;
                e.gstate[li] = e.state_bufs[1] + gi * gs_b;
                gi += 1;
            }
        }
        inline for (.{ "pos", "pos_bb", "step", "count" }) |f| @field(e, f) = try e.alloc(8);
        e.bb_input = try e.alloc(W * 4);
        inline for (.{ "t_xn", "t_tmp", "mixed" }) |f| @field(e, f) = if (backbone_only) 0 else try e.alloc(R * H * 2);
        inline for (.{ "x", "xn", "tmp", "shout", "hfin" }) |f| @field(e, f) = try e.alloc(R * H * 2);
        e.t_proj = if (backbone_only) 0 else try e.alloc(R * 2 * cfg.tower_ffn * 2);
        inline for (.{ "t_y", "t_q" }) |f| @field(e, f) = if (backbone_only) 0 else try e.alloc(R * 4096 * 2);
        inline for (.{ "y", "q" }) |f| @field(e, f) = try e.alloc(R * 4096 * 2);
        e.window = try e.alloc(16 * 4);
        e.cur_input = try e.alloc(W * 4);
        e.frame = if (backbone_only) 0 else try e.alloc(W * 4);
        e.frames_out = if (backbone_only) 0 else try e.alloc(e.o.max_frames * W * 4);
        e.pframes = try e.alloc((ctxn + 1) * W * 4);
        e.proj = try e.alloc(R * cfg.gdn_proj * 2);
        e.conv_out = try e.alloc(R * cfg.conv_channels * 2);
        e.o32 = try e.alloc(R * 4096 * 4);
        e.sp = try e.alloc(R * cfg.moe_proj * 2);
        e.sact = try e.alloc(R * cfg.shared_width * 2);
        e.picks = try e.alloc(R * 8 * 4);
        e.rw = try e.alloc(R * 8 * 4);
        e.gu = try e.alloc(R * 8 * cfg.expert_width * 2);
        e.down = try e.alloc(R * 8 * H * 4);
        e.xq = try e.alloc(R * H * 2);
        e.xsc = try e.alloc(R * (H / 128) * 4);
        e.gq = try e.alloc(R * 8 * cfg.expert_width * 2);
        e.gsc = try e.alloc(R * 8 * (cfg.expert_width / 128) * 4);
        e.counts = try e.alloc(cfg.experts * 4);
        e.members = try e.alloc(cfg.experts * R * 8 * 4);
        e.th = if (backbone_only) .{ 0, 0 } else .{ try e.alloc(R * H * 2), try e.alloc(R * H * 2) };
        e.mix = if (backbone_only) 0 else try e.alloc(R * 2 * H * 2);
        e.act = if (backbone_only) 0 else try e.alloc(R * cfg.tower_ffn * 2);
        e.hlast = try e.alloc(H * 2);
        e.cand_logits = try e.alloc(e.w.n_cand * 4);
        e.code_logits = if (backbone_only) 0 else try e.alloc(cfg.codebook_size * 4);
    }

    pub fn flush(e: *Engine) !void {
        try e.g.flush();
        peak_seen = @max(peak_seen, e.g.peak);
    }

    pub fn put(e: *Engine, dst: u64, bytes: []const u8) !void {
        try e.g.put(dst, bytes);
    }

    pub fn download(e: *Engine, src: u64, out: []u8) !void {
        try e.g.get(src, out);
        peak_seen = @max(peak_seen, e.g.peak);
    }

    /// Device bytes for a tool (freed with the engine).
    pub fn scratch(e: *Engine, bytes: usize) !u64 {
        return e.alloc(bytes);
    }

    pub fn setPos(e: *Engine, p: usize) !void {
        const v: i32 = @intCast(p);
        try e.put(e.pos, std.mem.asBytes(&v));
    }

    /// Zero the recurrent states and frame counters; cache rows are written before they are read.
    pub fn reset(e: *Engine) !void {
        try e.flush();
        for (0..2) |i| @memset(e.g.host(e.state_bufs[i], e.state_bytes[i]), 0);
        @memset(e.g.host(e.count, 8), 0);
        @memset(e.g.host(e.step, 8), 0);
        e.rows_done = 0;
        e.prompt_rows = 0;
    }

    /// Run the prompt rows (int32 [n][11]) through the backbone and the tower, leaving the frame step ready.
    pub fn prefill(e: *Engine, rows: []const [W]i32, probe: ?Probe) !void {
        try e.startPrompt(rows);
        const R = if (probe != null) e.o.max_rows else 1;
        if (probe != null and rows.len > R) return error.ProbeNeedsOneChunk;
        var at: usize = 0;
        while (at < rows.len) : (at += R) try e.promptChunk(rows.len, at, @min(R, rows.len - at), probe);
        try e.finishPrompt(rows);
    }

    pub fn startPrompt(e: *Engine, rows: []const [W]i32) !void {
        try model.validate(rows, e.c.ids, e.o.max_ctx, 1);
        const n = rows.len;
        if (n < 2) return error.PromptTooShort;
        if (n + 2 > e.o.max_ctx) return error.PromptTooLong;
        try e.reset();
        try e.put(e.pframes, std.mem.sliceAsBytes(rows));
    }

    /// Backbone-only prompt chunk for the unconditional CFG sequence; no codec tower work.
    pub fn backboneChunk(e: *Engine, c0: usize, t: usize, probe: ?Probe) !void {
        if (t == 0 or t > e.o.max_rows) return error.BadPromptChunk;
        const g = e.g;
        var nb: [64]u8 = undefined;
        try e.setPos(c0);
        k.embedBackbone(g, e.pframes + c0 * W * 4, e.w.embed, e.w.cb_emb, e.x, t, H, e.c.ids.sem_begin, e.c.ids.sem_end, cfg.codebook_size);
        if (probe) |p| try p.f(p.ctx, "backbone_input_embeds", e.x, t, H, false);
        for (0..cfg.layers) |li| {
            fwd.backboneLayer(e, fwd.laneBB(e, e.pos), li, t);
            if (probe) |p| try p.f(p.ctx, std.fmt.bufPrint(&nb, "backbone_hidden.layer{d:0>2}", .{li}) catch unreachable, e.x, t, H, false);
        }
        k.rmsnorm(g, e.x, H, e.w.final_norm, e.hfin, H, t, H, cfg.eps, true);
        if (probe) |p| try p.f(p.ctx, "backbone_final_norm", e.hfin, t, H, false);
    }

    pub fn promptChunk(e: *Engine, n: usize, c0: usize, t: usize, probe: ?Probe) !void {
        if (t == 0 or t > e.o.max_rows or c0 >= n or t > n - c0) return error.BadPromptChunk;
        try e.backboneChunk(c0, t, probe);
        const g = e.g;
        // the tower over this chunk's rows that have a next prompt row
        const tt = @min(c0 + t, n - 1) - c0;
        if (tt > 0) try e.tower(tt, c0, e.hfin, probe);
        if (c0 + t == n) g.copy(e.hfin + (t - 1) * H * 2, e.hlast, H * 2);
    }

    pub fn finishPrompt(e: *Engine, rows: []const [W]i32) !void {
        const n = rows.len;
        // decode state: the last prompt row is the newest row, and the RAS window holds the last 16 main tokens
        try e.setPos(n - 1);
        try e.put(e.cur_input, std.mem.asBytes(&rows[n - 1]));
        var win: [16]i32 = @splat(-1);
        const m = @min(16, n);
        for (0..m) |i| win[16 - m + i] = rows[n - m + i][0];
        try e.put(e.window, std.mem.asBytes(&win));
        const one: u64 = 1;
        try e.put(e.step, std.mem.asBytes(&one));
        e.prompt_rows = n;
        e.rows_done = n;
    }

    /// Recurrent state only. The caller must invalidate it before its lane's live prefix KV changes.
    pub const Snapshot = @import("../reference_cache.zig").Snapshot;

    /// Captures into one reusable, GPU-owned buffer pair (including partial allocation on failure).
    pub fn snapshot(e: *Engine, saved: *Snapshot) !void {
        try saved.capture(e.g, e.state_bufs, e.state_bytes);
    }

    pub fn restore(e: *Engine, saved: *const Snapshot) !void {
        try saved.restore(e.g, e.state_bufs, e.state_bytes);
    }

    /// Starts a prompt whose first `start` rows are already in the engine's state; writes the rest, no reset.
    pub fn beginPromptFrom(e: *Engine, rows: []const [W]i32, start: usize) !void {
        try model.validate(rows, e.c.ids, e.o.max_ctx, 1);
        const n = rows.len;
        if (n < 2 or start == 0 or start >= n) return error.PromptTooShort;
        if (n + 2 > e.o.max_ctx) return error.PromptTooLong;
        try e.put(e.pframes, std.mem.sliceAsBytes(rows));
        // The suffix prefill/finishPrompt rebuilds all other decode state, including hlast and RAS.
        @memset(e.g.host(e.count, 8), 0);
        @memset(e.g.host(e.step, 8), 0);
        e.rows_done = start;
        e.prompt_rows = start;
    }

    fn tower(e: *Engine, tt: usize, c0: usize, hfin: u64, probe: ?Probe) !void {
        var hin = hfin;
        var nb: [64]u8 = undefined;
        for (0..cfg.tower_layers) |ti| {
            const hout = e.th[ti % 2];
            fwd.towerLayer(e, fwd.laneTW(e), ti, tt, hin, hout, e.pframes + (c0 + 1) * W * 4, W, e.pframes + c0 * W * 4);
            if (probe) |p| try p.f(p.ctx, std.fmt.bufPrint(&nb, "tower_prefill.layer{d:0>2}", .{ti}) catch unreachable, hout, tt, H, false);
            hin = hout;
        }
    }

    /// Parity: the tower prefill alone over rows 0..n-2 from host final-norm rows hfin, on prefill()'s prompt rows.
    pub fn towerOnly(e: *Engine, n: usize, hfin: []const u8, probe: ?Probe) !void {
        if (n > e.o.max_rows) return error.ProbeNeedsOneChunk;
        try e.put(e.hfin, hfin);
        try e.setPos(0);
        try e.tower(n - 1, 0, e.hfin, probe);
    }

    /// Teacher forcing for parity: each pick goes to `picks`, then `frame`'s value (device int32 [11]) replaces it.
    pub const Force = struct { frame: u64, picks: u64 };

    pub fn frameForced(e: *Engine, force: Force, probe: ?Probe) !void {
        try fwd.enqueueFrame(e, .{ .temperature = 0, .top_k = 1 }, probe, force);
    }

    /// One frame, committed to the GPU as soon as it is encoded (the next one encodes while it runs).
    pub fn frameStep(e: *Engine, smp: Sampling, graphs: bool, probe: ?Probe) !void {
        _ = graphs;
        try fwd.enqueueFrame(e, smp, probe, null);
        try e.g.commit();
    }

    pub const Result = struct {
        frames: [][W]i32,
        eos: bool,
        prefill_ms: f64,
        first_frame_ms: f64,
        decode_ms: f64,
    };

    /// Generate up to max_frames frames after the prompt rows, stopping at the end-of-speech frame.
    pub fn generate(e: *Engine, io: std.Io, rows: []const [W]i32, smp: Sampling, max_frames: usize, graphs: bool, sync_every: usize) !Result {
        const t0 = nowNs(io);
        try e.prefill(rows, null);
        try e.flush();
        const t1 = nowNs(io);
        const limit = @min(max_frames, e.o.max_frames, e.o.max_ctx - rows.len - 1);
        var made: usize = 0;
        var eos = false;
        var first_ms: f64 = 0;
        const every = @max(sync_every, 1);
        const host = try e.gpa.alloc([W]i32, limit);
        errdefer e.gpa.free(host);
        var checked: usize = 0;
        while (made < limit and !eos) {
            const burst = @min(if (made == 0) 1 else every, limit - made);
            for (0..burst) |_| try e.frameStep(smp, graphs, null);
            made += burst;
            try e.download(e.frames_out + checked * W * 4, std.mem.sliceAsBytes(host[checked..made]));
            if (checked == 0) first_ms = ms(t0, nowNs(io));
            for (host[checked..made], checked..) |f, i| if (f[0] == e.c.ids.eos) {
                eos = true;
                made = i;
                break;
            };
            checked = made;
        }
        const t2 = nowNs(io);
        return .{ .frames = try model.finishFrames(e.gpa, host, made), .eos = eos, .prefill_ms = ms(t0, t1), .first_frame_ms = first_ms, .decode_ms = ms(t1, t2) };
    }
};
