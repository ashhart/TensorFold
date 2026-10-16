//! Drama 3 generation on one GPU: prompt frame rows in chunks, then one frame a step replayed as a CUDA graph.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");
const fwd = @import("forward.zig");
const common = @import("common.zig");

pub const Options = common.Options;
pub const Sampling = common.Sampling;
pub const Probe = common.Probe;
pub const nowNs = common.nowNs;
const ms = common.ms;

const H = cfg.hidden;
const W = cfg.frame_width;

const Buf = cuda.DeviceBuffer;
pub const max_split_rows = 8; // decode rows the split-key attention takes

pub const Engine = struct {
    gpa: std.mem.Allocator,
    o: Options,
    c: cfg.Config,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    stream2: cuda.Stream, // the tower's codes 2..9 in a frame, beside the next backbone row
    ev_fork: cuda.Event,
    ev_join: cuda.Event,
    k: kern.Kernels,
    w: *wts.Weights,
    bufs: std.ArrayList(Buf),
    // state
    kv_bb: [cfg.layers][2]u64, // full-attention layers: K, V [max_ctx, 2, 256]
    kv_tw: [cfg.tower_layers][2]u64,
    conv: [cfg.layers]u64, // GDN layers: [8192, 3] bf16
    gstate: [cfg.layers]u64, // GDN layers: [32, 128, 128] fp32
    state_bufs: [2]u64, // conv / gstate blocks (zeroed per request)
    state_bytes: [2]usize,
    // device scalars
    pos: u64, // i32: the newest row's position
    pos_bb: u64, // i32: the next backbone row's position (pos + 1 inside a frame)
    bb_input: u64, // i32 [11]: the next backbone row's frame (main, code 0, code 1)
    t_xn: u64, // the tower lane's scratch
    t_tmp: u64,
    t_proj: u64,
    t_y: u64,
    t_q: u64,
    pml: [2]u64, // split-key attention partials, a lane each
    pacc: [2]u64,
    step: u64, // u64: the sampler's step
    count: u64, // i32: frames made
    window: u64, // i32 [16]
    cur_input: u64, // i32 [11]: the frame the backbone takes next (at pos)
    frame: u64, // i32 [11]: the frame being sampled
    frames_out: u64, // i32 [max_frames, 11]
    pframes: u64, // i32 [max_ctx, 11]: the prompt rows
    // scratch (rows R = max_rows)
    x: u64, // residual stream [R, H] bf16
    xn: u64,
    tmp: u64, // [R, H]
    proj: u64, // [R, 12352] bf16
    conv_out: u64, // [R, 8192]
    o32: u64, // [R, 4096] fp32 (GDN core out)
    y: u64, // [R, 4096] bf16 (gated norm / attention out)
    q: u64, // [R, 16, 256]
    qg: u64,
    og: u64,
    scores: u64, // [2, 8 * attn_chunk, max_ctx] fp32
    vf: u64, // V cache rows in fp32 [max_ctx, 512] for the prompt PV GEMM
    sp: u64, // [R, 1288]
    sact: u64, // [R, 512]
    shout: u64, // [R, H]
    picks: u64, // i32 [R * 8]
    rw: u64, // f32 [R * 8]
    gu: u64, // [R * 8, 512] bf16
    down: u64, // [R * 8, H] fp32
    xq: u64, // [R, H] bf16 (e4m3 values)
    xsc: u64, // [R, H / 128] fp32
    gq: u64, // [R * 8, 512]
    gsc: u64, // [R * 8, 4]
    plan: cuda.grouped.Plan,
    hfin: u64, // [R, H] final-norm hidden
    th: [2]u64, // tower hidden ping-pong [R, H]
    mix: u64, // [R, 2H]
    mixed: u64, // [R, H]
    act: u64, // [R, 6144]
    hlast: u64, // [H] the newest row's final-norm hidden
    cand_logits: u64, // f32 [4097]
    code_logits: u64, // f32 [4096]
    // frame graph
    graph: ?cuda.graph.Exec = null,
    graph_sampling: Sampling = .{},
    sampling: Sampling = .{},
    rows_done: usize = 0, // positions in the cache (prompt rows; the frame step advances on the device)
    prompt_rows: usize = 0,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.o = o;
        e.graph = null;
        e.sampling = .{};
        e.graph_sampling = .{};
        e.rows_done = 0;
        e.prompt_rows = 0;
        e.bufs = .empty;
        e.c = try cfg.Config.load(gpa, io, model_dir);
        e.driver = try cuda.Driver.open();
        e.ctx = try cuda.Context.init(&e.driver, 0);
        e.stream = try cuda.Stream.init(&e.driver, true);
        e.stream2 = try cuda.Stream.init(&e.driver, true);
        e.ev_fork = try cuda.Event.init(&e.driver, false);
        e.ev_join = try cuda.Event.init(&e.driver, false);
        e.k = try kern.Kernels.load(&e.driver, &e.ctx);
        e.k.pdl = o.pdl and try e.ctx.capability() >= 90; // programmatic dependent launch needs sm_90 up
        e.w = try wts.load(gpa, io, &e.driver, model_dir, e.c, .{ .threads = o.threads });
        try e.allocState();
        if (o.warmup) {
            var rows: [4][W]i32 = @splat(@splat(0));
            for (&rows, 0..) |*r, i| r[0] = @intCast(198 + i);
            try e.prefill(&rows, null);
            for (0..2) |_| try e.frameStep(.{}, true, null);
            try e.stream.synchronize();
        }
        return e;
    }

    pub fn destroy(e: *Engine) void {
        if (e.graph) |*g| g.deinit();
        for (e.bufs.items) |*b| b.free();
        e.bufs.deinit(e.gpa);
        e.w.deinit();
        e.gpa.destroy(e.w);
        e.k.unload();
        e.ev_fork.deinit();
        e.ev_join.deinit();
        e.stream2.deinit();
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    fn alloc(e: *Engine, bytes: usize) !u64 {
        const b = try Buf.alloc(&e.driver, @max(bytes, 256));
        try e.bufs.append(e.gpa, b);
        return b.ptr;
    }

    fn allocState(e: *Engine) !void {
        const R = e.o.max_rows;
        const ctxn = e.o.max_ctx;
        const kvb = ctxn * cfg.kv_heads * cfg.head_dim * 2;
        for (0..cfg.layers) |li| {
            if (e.c.full_attention[li]) {
                e.kv_bb[li] = .{ try e.alloc(kvb), try e.alloc(kvb) };
            } else e.kv_bb[li] = .{ 0, 0 };
        }
        for (0..cfg.tower_layers) |ti| e.kv_tw[ti] = .{ try e.alloc(kvb), try e.alloc(kvb) };
        var n_gdn: usize = 0;
        for (e.c.full_attention) |f| n_gdn += @intFromBool(!f);
        const conv_b = cfg.conv_channels * 3 * 2;
        const gs_b = cfg.lin_v_heads * cfg.lin_dim * cfg.lin_dim * 4;
        e.state_bytes = .{ n_gdn * conv_b, n_gdn * gs_b };
        e.state_bufs = .{ try e.alloc(e.state_bytes[0]), try e.alloc(e.state_bytes[1]) };
        var g: usize = 0;
        for (0..cfg.layers) |li| {
            if (e.c.full_attention[li]) {
                e.conv[li] = 0;
                e.gstate[li] = 0;
            } else {
                e.conv[li] = e.state_bufs[0] + g * conv_b;
                e.gstate[li] = e.state_bufs[1] + g * gs_b;
                g += 1;
            }
        }
        e.pos = try e.alloc(8);
        e.pos_bb = try e.alloc(8);
        e.bb_input = try e.alloc(W * 4);
        e.t_xn = try e.alloc(R * H * 2);
        e.t_tmp = try e.alloc(R * H * 2);
        e.t_proj = try e.alloc(R * cfg.gdn_proj * 2);
        e.t_y = try e.alloc(R * 4096 * 2);
        e.t_q = try e.alloc(R * 4096 * 2);
        for (0..2) |li| {
            e.pml[li] = try e.alloc(max_split_rows * 16 * kern.Kernels.attn_splits * 2 * 4);
            e.pacc[li] = try e.alloc(max_split_rows * 16 * kern.Kernels.attn_splits * 256 * 4);
        }
        e.step = try e.alloc(8);
        e.count = try e.alloc(8);
        e.window = try e.alloc(16 * 4);
        e.cur_input = try e.alloc(W * 4);
        e.frame = try e.alloc(W * 4);
        e.frames_out = try e.alloc(e.o.max_frames * W * 4);
        e.pframes = try e.alloc((ctxn + 1) * W * 4);
        e.x = try e.alloc(R * H * 2);
        e.xn = try e.alloc(R * H * 2);
        e.tmp = try e.alloc(R * H * 2);
        e.proj = try e.alloc(R * cfg.gdn_proj * 2);
        e.conv_out = try e.alloc(R * cfg.conv_channels * 2);
        e.o32 = try e.alloc(R * 4096 * 4);
        e.y = try e.alloc(R * 4096 * 2);
        e.q = try e.alloc(R * 4096 * 2);
        e.qg = try e.alloc(R * 4096 * 2);
        e.og = try e.alloc(R * 4096 * 4);
        const ac = @min(e.o.attn_chunk, R);
        e.scores = try e.alloc(2 * 8 * ac * (ctxn + 8) * 4);
        e.vf = try e.alloc((ctxn + 8) * 512 * 4);
        e.sp = try e.alloc(R * cfg.moe_proj * 2);
        e.sact = try e.alloc(R * cfg.shared_width * 2);
        e.shout = try e.alloc(R * H * 2);
        e.picks = try e.alloc(R * 8 * 4);
        e.rw = try e.alloc(R * 8 * 4);
        e.gu = try e.alloc(R * 8 * cfg.expert_width * 2);
        e.down = try e.alloc(R * 8 * H * 4);
        e.xq = try e.alloc(R * H * 2);
        e.xsc = try e.alloc(R * (H / 128) * 4);
        e.gq = try e.alloc(R * 8 * cfg.expert_width * 2);
        e.gsc = try e.alloc(R * 8 * (cfg.expert_width / 128) * 4);
        const pairs = R * 8;
        const items = cuda.grouped.maxItems(pairs, cfg.experts, 16);
        const nblk = (pairs + 1023) / 1024;
        e.plan = .{ .members = try e.alloc(pairs * 4), .items = try e.alloc(items * 3 * 4), .counts = try e.alloc(16), .rank = try e.alloc(pairs * 4), .hist = try e.alloc(nblk * cfg.experts * 4) };
        e.hfin = try e.alloc(R * H * 2);
        e.th = .{ try e.alloc(R * H * 2), try e.alloc(R * H * 2) };
        e.mix = try e.alloc(R * 2 * H * 2);
        e.mixed = try e.alloc(R * H * 2);
        e.act = try e.alloc(R * cfg.tower_ffn * 2);
        e.hlast = try e.alloc(H * 2);
        e.cand_logits = try e.alloc(e.w.n_cand * 4);
        e.code_logits = try e.alloc(cfg.codebook_size * 4);
    }

    pub fn buf(ptr: u64) Buf {
        return .{ .d = undefined, .ptr = ptr, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    pub fn put(e: *Engine, dst: u64, bytes: []const u8) !void {
        var b = buf(dst);
        b.d = &e.driver;
        try b.uploadAsync(0, bytes, e.stream.handle);
    }

    fn get(e: *Engine, src: u64, out: []u8) !void {
        var b = buf(src);
        b.d = &e.driver;
        try b.downloadAsync(0, out, e.stream.handle);
        try e.stream.synchronize();
    }

    pub fn setPos(e: *Engine, p: usize) !void {
        const v: i32 = @intCast(p);
        try e.put(e.pos, std.mem.asBytes(&v));
        try e.stream.synchronize();
    }

    /// Zero the recurrent states and frame counters; cache rows are written before they are read.
    pub fn reset(e: *Engine) !void {
        for (0..2) |i| {
            var b = buf(e.state_bufs[i]);
            b.d = &e.driver;
            b.len = e.state_bytes[i];
            try b.fill8(0, e.stream.handle);
        }
        const zero: [2]u64 = .{ 0, 0 };
        try e.put(e.count, std.mem.asBytes(&zero));
        try e.put(e.step, std.mem.asBytes(&zero));
        e.rows_done = 0;
        e.prompt_rows = 0;
        try e.stream.synchronize();
    }

    /// Run the prompt rows (int32 [n][11]) through the backbone and the tower, leaving the frame step ready.
    pub fn prefill(e: *Engine, rows: []const [W]i32, probe: ?Probe) !void {
        const n = rows.len;
        if (n < 2) return error.PromptTooShort;
        if (n + 2 > e.o.max_ctx) return error.PromptTooLong;
        try e.reset();
        try e.put(e.pframes, std.mem.sliceAsBytes(rows));
        const k = &e.k;
        const s = e.stream;
        const R = e.o.max_rows;
        if (probe != null and n > R) return error.ProbeNeedsOneChunk;
        var c0: usize = 0;
        while (c0 < n) : (c0 += R) {
            const t = @min(R, n - c0);
            try e.setPos(c0);
            try k.embedBackbone(s, e.pframes + c0 * W * 4, e.w.embed, e.w.cb_emb, e.x, t, H, e.c.ids.sem_begin, e.c.ids.sem_end, cfg.codebook_size);
            if (probe) |p| try p.f(p.ctx, "backbone_input_embeds", e.x, t, H, false);
            for (0..cfg.layers) |li| {
                try fwd.backboneLayer(e, fwd.laneBB(e, e.pos), li, t, c0);
                if (probe) |p| {
                    var nb: [64]u8 = undefined;
                    try p.f(p.ctx, try std.fmt.bufPrint(&nb, "backbone_hidden.layer{d:0>2}", .{li}), e.x, t, H, false);
                }
            }
            try k.rmsnorm(s, e.x, H, e.w.final_norm, e.hfin, H, t, H, cfg.eps, true);
            if (probe) |p| try p.f(p.ctx, "backbone_final_norm", e.hfin, t, H, false);
            // the tower over this chunk's rows that have a next prompt row
            const tt = @min(c0 + t, n - 1) - c0;
            if (tt > 0) {
                var hin = e.hfin;
                for (0..cfg.tower_layers) |ti| {
                    const hout = e.th[ti % 2];
                    try fwd.towerLayer(e, fwd.laneTW(e, e.stream), ti, tt, c0, hin, hout, e.pframes + (c0 + 1) * W * 4, W, e.pframes + c0 * W * 4);
                    if (probe) |p| {
                        var nb: [64]u8 = undefined;
                        try p.f(p.ctx, try std.fmt.bufPrint(&nb, "tower_prefill.layer{d:0>2}", .{ti}), hout, tt, H, false);
                    }
                    hin = hout;
                }
            }
            if (c0 + t == n) try e.copyRow(e.hfin + (t - 1) * H * 2, e.hlast, H * 2);
        }
        // decode state: the last prompt row is the newest row, and the RAS window holds the last 16 main tokens
        try e.setPos(n - 1);
        try e.put(e.cur_input, std.mem.asBytes(&rows[n - 1]));
        var win: [16]i32 = @splat(-1);
        const m = @min(16, n);
        for (0..m) |i| win[16 - m + i] = rows[n - m + i][0];
        try e.put(e.window, std.mem.asBytes(&win));
        const one: u64 = 1;
        try e.put(e.step, std.mem.asBytes(&one));
        try e.stream.synchronize();
        e.prompt_rows = n;
        e.rows_done = n;
    }

    /// Parity: the tower prefill alone over rows 0..n-2 from host final-norm rows hfin, on prefill()'s prompt rows.
    pub fn towerOnly(e: *Engine, n: usize, hfin: []const u8, probe: ?Probe) !void {
        if (n > e.o.max_rows) return error.ProbeNeedsOneChunk;
        try e.put(e.hfin, hfin);
        try e.setPos(0);
        var hin = e.hfin;
        for (0..cfg.tower_layers) |ti| {
            const hout = e.th[ti % 2];
            try fwd.towerLayer(e, fwd.laneTW(e, e.stream), ti, n - 1, 0, hin, hout, e.pframes + W * 4, W, e.pframes);
            if (probe) |p| {
                var nb: [64]u8 = undefined;
                try p.f(p.ctx, try std.fmt.bufPrint(&nb, "tower_prefill.layer{d:0>2}", .{ti}), hout, n - 1, H, false);
            }
            hin = hout;
        }
    }

    pub fn copyRow(e: *Engine, src: u64, dst: u64, bytes: usize) !void {
        var b = buf(dst);
        b.d = &e.driver;
        try b.copyFrom(0, src, bytes, e.stream.handle);
    }

    /// Teacher forcing for parity: each pick goes to `picks`, then `frame`'s value (device int32 [11]) replaces it.
    pub const Force = struct { frame: u64, picks: u64 };

    /// One teacher-forced greedy frame, eager, with an optional probe of its logits.
    pub fn frameForced(e: *Engine, force: Force, probe: ?Probe) !void {
        try fwd.enqueueFrame(e, .{ .temperature = 0, .top_k = 1 }, probe, force);
    }

    /// One frame (eager, or replaying the captured graph when `graphs`).
    pub fn frameStep(e: *Engine, smp: Sampling, graphs: bool, probe: ?Probe) !void {
        if (!graphs or probe != null) return fwd.enqueueFrame(e, smp, probe, null);
        if (e.graph == null or !std.meta.eql(e.graph_sampling, smp)) {
            if (e.graph) |*g| g.deinit();
            e.graph = null;
            try cuda.graph.beginCapture(e.stream, .thread_local);
            fwd.enqueueFrame(e, smp, null, null) catch |er| {
                var g = cuda.graph.endCapture(e.stream) catch return er;
                g.deinit();
                return er;
            };
            var g = try cuda.graph.endCapture(e.stream);
            defer g.deinit();
            e.graph = try g.instantiate();
            e.graph_sampling = smp;
        }
        try e.graph.?.launchOn(e.stream);
    }

    pub const Result = struct {
        frames: [][W]i32, // without the end-of-speech frame
        eos: bool,
        prefill_ms: f64,
        first_frame_ms: f64, // from the request's start to the first frame on the host
        decode_ms: f64,
    };

    /// Generate up to max_frames frames after the prompt rows, stopping at the end-of-speech frame.
    pub fn generate(e: *Engine, io: std.Io, rows: []const [W]i32, smp: Sampling, max_frames: usize, graphs: bool, sync_every: usize) !Result {
        const t0 = nowNs(io);
        try e.prefill(rows, null);
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
            try e.get(e.frames_out + checked * W * 4, std.mem.sliceAsBytes(host[checked..made]));
            if (checked == 0) first_ms = ms(t0, nowNs(io));
            for (host[checked..made], checked..) |f, i| if (f[0] == e.c.ids.eos) {
                eos = true;
                made = i;
                break;
            };
            checked = made;
        }
        const t2 = nowNs(io);
        return .{ .frames = host[0..made], .eos = eos, .prefill_ms = ms(t0, t1), .first_frame_ms = first_ms, .decode_ms = ms(t1, t2) };
    }

    pub fn download(e: *Engine, src: u64, out: []u8) !void {
        return e.get(src, out);
    }

    /// Device bytes for a tool (freed with the engine).
    pub fn scratch(e: *Engine, bytes: usize) !u64 {
        return e.alloc(bytes);
    }
};

/// The device's peak allocation in bytes so far (the runtime's own count).
pub fn peakBytes() u64 {
    return cuda.usage(false).peak;
}
