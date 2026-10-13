//! One GLM-5.3-Flash layer on one rank (forward.py layer_forward): HC glue, the KDA or DSA mixer, MoE or dense MLP.
const std = @import("std");
const driver = @import("driver.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const exl3 = @import("exl3.zig");
const exl3e = @import("exl3_experts.zig");
const glue_ = @import("glm_glue.zig");
const glue2_ = @import("glm_glue2.zig");
const bmm_ = @import("glm_bmm.zig");
const kda_ = @import("glm_kda.zig");
const chunk_ = @import("glm_kda_chunk.zig");
const latent_ = @import("glm_latent.zig");
const sparse_ = @import("glm_sparse.zig");
const dsa_ = @import("glm_dsa.zig");
const Stream = stream_.Stream;
const B16 = bmm_.B16;

/// Every kernel family a layer launches, loaded once.
pub const Kernels = struct {
    d: *const driver.Driver,
    glue: glue_.Glue,
    glue2: glue2_.Glue2,
    bmm: bmm_.Bmm,
    dense: exl3.Dense,
    kda: kda_.Kda,
    chunk: chunk_.Chunk,
    experts: exl3e.Experts,
    latent: latent_.Latent,
    sparse: sparse_.Sparse,

    pub fn load(d: *const driver.Driver) !Kernels {
        var glue = try glue_.Glue.load(d);
        errdefer glue.unload();
        var glue2 = try glue2_.Glue2.load(d);
        errdefer glue2.unload();
        var bmm = try bmm_.Bmm.load(d);
        errdefer bmm.unload();
        var dense = try exl3.Dense.load(d);
        errdefer dense.unload();
        var kda = try kda_.Kda.load(d);
        errdefer kda.unload();
        var chunk = try chunk_.Chunk.load(d);
        errdefer chunk.unload();
        var experts = try exl3e.Experts.load(d);
        errdefer experts.unload();
        var latent = try latent_.Latent.load(d);
        errdefer latent.unload();
        const sparse = try sparse_.Sparse.load(d);
        return .{ .d = d, .glue = glue, .glue2 = glue2, .bmm = bmm, .dense = dense, .kda = kda, .chunk = chunk, .experts = experts, .latent = latent, .sparse = sparse };
    }

    pub fn unload(k: *Kernels) void {
        k.sparse.unload();
        k.latent.unload();
        k.experts.unload();
        k.chunk.unload();
        k.kda.unload();
        k.dense.unload();
        k.bmm.unload();
        k.glue2.unload();
        k.glue.unload();
    }

    /// The DSA block's view of these kernels.
    pub fn dsa(k: *Kernels) dsa_.Kernels {
        return .{ .glue2 = &k.glue2, .bmm = &k.bmm, .dense = &k.dense, .latent = &k.latent, .sparse = &k.sparse };
    }
};

/// The model's constants a layer reads (weights.Config).
pub const Config = struct {
    eps: f32,
    hc_eps: f32,
    hc_iters: u32,
    lower: f32,
    limit: f32,
    routed_scale: f32,
    top_k: u32,
    experts: u32,
    d: u32 = 4096,
    streams: u32 = 4,
    dsa: ?dsa_.Config = null, // a DSA layer's constants
};

/// One hyper-connection site: fn [24, S * D] fp32, base [24] fp32, scale [3] fp32.
pub const Hc = struct { fn_w: u64, base: u64, scale: u64 };

/// The rank's KDA weights: fused q|k|v EXL3, the bf16 pair and b, conv taps, decays, gated norm, o_proj's rows.
pub const KdaW = struct {
    qkv: exl3.Weight,
    fab: B16, // f_a | g_a | b stacked by rows: one matmul writes p's columns fa_off.. (the three are adjacent)
    fb: B16,
    gb: B16,
    o: exl3.Weight,
    conv: u64, // bf16 [3 H 128, 4]
    a_log: u64, // fp32 [H]
    dt_bias: u64, // fp32 [H 128]
    norm: u64, // bf16 [128]
    heads: u32,
    fa_off: u32, // p's columns: q|k|v, then f_a, g_a, b
    ga_off: u32,
    b_off: u32,
    p_stride: u32,
};

/// The rank's MoE: router [E, D] bf16 and bias [E] fp32, routed EXL3 experts, the shared expert's three linears.
pub const MoeW = struct { router: B16, bias: u64, experts: exl3e.Layer, gate: exl3.Weight, up: exl3.Weight, down: exl3.Weight };

/// A dense layer's MLP (layers 0-2): gate and up (the rank's columns), down (the rank's rows, an fp32 partial).
pub const MlpW = struct { gate: exl3.Weight, up: exl3.Weight, down: exl3.Weight };

/// The layer's mixer: the KDA block's weights or the DSA block's.
pub const Mixer = union(enum) { kda: KdaW, dsa: dsa_.DsaW };

/// The layer's feed-forward: routed experts plus the shared one, or a dense SwiGLU MLP.
pub const Ffn = union(enum) { moe: MoeW, mlp: MlpW };

pub const LayerW = struct { attn_hc: Hc, ffn_hc: Hc, in_norm: u64, post_norm: u64, mixer: Mixer, ffn: Ffn };

/// This layer's KDA state for one forward: the conv window [3, 3 H 128] bf16, the fp32 state read and written.
pub const KdaState = struct { conv: u64, rec_in: u64, rec_out: u64 };

/// Decode windows run the serial chain (commit shifts the conv later); prompt chunks the chunked form at `pos`.
pub const Mode = union(enum) { decode, prompt: u64 };

/// A DSA call: the layer's caches, its buffers (dsa.Scratch, sized by the engine's rows) and the host position.
pub const DsaRun = struct { st: dsa_.DsaState, sc: *const dsa_.Scratch, pos: usize, capture: bool = false };

/// The mixer's state for one forward, matching the layer's Mixer.
pub const MixState = union(enum) { kda: struct { st: KdaState, mode: Mode }, dsa: DsaRun };

pub const Site = enum { attn, ffn };

/// Fills gath [world, rows, D] fp32 with every rank's partial in rank order (forward.gather: NCCL, or a test's).
pub const Gather = struct {
    ctx: *anyopaque,
    world: u32,
    run: *const fn (ctx: *anyopaque, s: Stream, site: Site, part: u64, gath: u64, rows: usize) anyerror!void,
};

const Lay = struct {
    at: u64,
    fn take(l: *Lay, bytes: usize) u64 {
        const r = l.at;
        l.at += (bytes + 255) / 256 * 256;
        return r;
    }
};

/// What a Scratch holds: the KDA mixer's buffers (null: none), the routed experts' (null: none), the widest SwiGLU.
pub const Shape = struct {
    kda: ?*const KdaW = null,
    experts: ?exl3e.Layer = null,
    width: u32,

    /// One layer's needs.
    pub fn of(w: *const LayerW) Shape {
        return .{
            .kda = if (w.mixer == .kda) &w.mixer.kda else null,
            .experts = if (w.ffn == .moe) w.ffn.moe.experts else null,
            .width = switch (w.ffn) {
                .moe => |m| m.gate.n,
                .mlp => |m| m.gate.n,
            },
        };
    }

    /// Both shapes' needs: one Scratch then serves layers of either (every KDA layer has the same heads).
    pub fn widen(a: Shape, b: Shape) Shape {
        return .{ .kda = a.kda orelse b.kda, .experts = a.experts orelse b.experts, .width = @max(a.width, b.width) };
    }
};

/// A forward's buffers for up to `rows` rows (forward.Buffers' slice for the layers of a Shape), one allocation.
pub const Scratch = struct {
    buf: memory.DeviceBuffer,
    rows: usize,
    world: u32,
    normed: u64,
    xs: u64,
    post: u64,
    comb: u64,
    hcpart: u64,
    p: u64, // [rows, p_stride] bf16: the decode window's st.proj rows, the prompt chunk's b.kproj
    qkv: u64,
    fab: u64, // [rows, fab.n] bf16: f_a | g_a | b
    ka: u64,
    kg: u64,
    mpart: u64, // split-K partials of the bf16 matmuls and the router (one at a time)
    kout: u64,
    kda: kda_.Scratch,
    part: u64,
    gath: u64,
    mlog: u64,
    pick: u64,
    wts: u64,
    ey: u64, // [rows, slots, D] fp32: routed slots, then the shared expert in the last
    esc: ?exl3e.Scratch, // the routed experts' (null: the Shape has none)
    esc_len: usize,
    width: usize, // the widest SwiGLU the next buffers hold
    smg: u64,
    smu: u64,
    sgu: u64,
    sact: u64,
    sxs: u64,
    sy: u64,

    fn place(l: *Lay, c: Config, sh: Shape, rows: usize, world: u32) !Scratch {
        const d: usize = c.d;
        // a DSA layer takes no KDA buffers (their sizes are 0; dsa.Scratch holds its own); a dense MLP none of MoE's
        const kd = sh.kda;
        const per: usize = if (kd) |v| @as(usize, v.heads) * kda_.d_k else 0;
        const heads: usize = if (kd) |v| v.heads else 0;
        const p_stride: usize = if (kd) |v| v.p_stride else 0;
        const slots: usize = c.top_k + 1;
        const ne: usize = if (sh.experts != null) c.experts else 0;
        const er: usize = if (sh.experts != null) rows else 0;
        const sw: usize = sh.width;
        var mp: usize = @as(usize, bmm_.router_ks) * rows * ne * 4;
        if (kd) |v| for ([_]B16{ v.fab, v.fb, v.gb }) |q| {
            mp = @max(mp, bmm_.Bmm.partBytes(q, rows));
        };
        var sc: Scratch = undefined;
        sc.rows = rows;
        sc.world = world;
        sc.normed = l.take(rows * d * 2);
        sc.xs = l.take(rows * d / 64 * 4);
        sc.post = l.take(rows * c.streams * 4);
        sc.comb = l.take(rows * c.streams * c.streams * 4);
        sc.hcpart = l.take(rows * 16 * 32 * 4);
        sc.p = l.take(rows * p_stride * 2);
        sc.qkv = l.take(rows * @as(usize, if (kd) |v| v.qkv.n else 0) * 2);
        sc.fab = l.take(rows * @as(usize, if (kd) |v| v.fab.n else 0) * 2);
        sc.ka = l.take(rows * per * 2);
        sc.kg = l.take(rows * per * 2);
        sc.mpart = l.take(mp);
        sc.kout = l.take(rows * per * 2);
        sc.kda = .{ .k = l.take(rows * per * 4), .v = l.take(rows * per * 2), .g = l.take(rows * per * 4), .b = l.take(rows * heads * 4), .q = l.take(rows * per * 4), .y = l.take(rows * per * 2) };
        sc.part = l.take(rows * d * 4);
        sc.gath = l.take(@as(usize, world) * rows * d * 4);
        sc.mlog = l.take(rows * ne * 4);
        sc.pick = l.take(er * slots * 4);
        sc.wts = l.take(er * slots * 4);
        sc.ey = l.take(er * slots * d * 4);
        sc.esc = null;
        sc.esc_len = 0;
        if (sh.experts) |ex| {
            sc.esc_len = try exl3e.scratchBytes(ex, rows, slots);
            sc.esc = try exl3e.scratch(l.take(sc.esc_len), ex, rows, slots);
        }
        sc.width = sw;
        sc.smg = l.take(rows * sw * 2);
        sc.smu = l.take(rows * sw * 2);
        sc.sgu = l.take(rows * 2 * sw * 2);
        sc.sact = l.take(rows * sw * 2);
        sc.sxs = l.take(rows * sw / 64 * 4);
        sc.sy = l.take(er * d * 4);
        if (rows == 1) { // one row: the copies into p, sgu and ey's shared slot become the outputs themselves
            if (kd) |v| {
                sc.qkv = sc.p;
                sc.fab = sc.p + @as(u64, v.fa_off) * 2;
            }
            sc.smg = sc.sgu;
            sc.smu = sc.sgu + sw * 2;
            if (er > 0) sc.sy = sc.ey + (slots - 1) * d * 4;
        }
        return sc;
    }

    /// Zeroed buffers (torch.zeros' experts scratch, its members list at -1); `world` ranks' partials gathered.
    pub fn initFor(d: *const driver.Driver, s: Stream, c: Config, sh: Shape, rows: usize, world: u32) !Scratch {
        var probe: Lay = .{ .at = 0 };
        _ = try place(&probe, c, sh, rows, world);
        var buf = try memory.DeviceBuffer.alloc(d, probe.at);
        errdefer buf.free();
        try buf.fill8(0, s.handle);
        var l: Lay = .{ .at = buf.ptr };
        var sc = try place(&l, c, sh, rows, world);
        sc.buf = buf;
        if (sc.esc) |esc| {
            const mb: memory.DeviceBuffer = .{ .d = d, .ptr = esc.members, .len = esc.xg + sc.esc_len - esc.members, .borrowed = true };
            try mb.fill8(0xff, s.handle);
        }
        return sc;
    }

    /// One layer's buffers (Shape.of).
    pub fn init(d: *const driver.Driver, s: Stream, c: Config, w: *const LayerW, rows: usize, world: u32) !Scratch {
        return initFor(d, s, c, Shape.of(w), rows, world);
    }

    pub fn deinit(sc: *Scratch) void {
        sc.buf.free();
    }
};

/// rows rows of `bytes` each from src (row stride ss bytes) to dst (row stride ds bytes): torch's strided copy_.
fn copyRows(d: *const driver.Driver, s: Stream, dst: u64, ds: usize, src: u64, ss: usize, bytes: usize, rows: usize) !void {
    if (dst == src and (rows == 1 or ds == ss)) return; // a one-row Scratch's aliases
    for (0..rows) |r| {
        const b: memory.DeviceBuffer = .{ .d = d, .ptr = dst + r * ds, .len = bytes, .borrowed = true };
        try b.copyFrom(0, src + r * ss, bytes, s.handle);
    }
}

/// glue.hc_pre: normed, xs, post and comb of the S streams x [rows, S * D] at one site.
pub fn hcPre(k: *Kernels, s: Stream, c: Config, h: Hc, norm: u64, x: u64, sc: *const Scratch, rows: usize) !void {
    try k.glue.hcPartial(s, x, h.fn_w, sc.hcpart, rows);
    try k.glue.hcFinish(s, x, sc.hcpart, h.base, h.scale, norm, sc.normed, sc.xs, sc.post, sc.comb, c.eps, c.hc_eps, c.hc_iters, rows, c.d);
}

/// kda_block: p's projections, the chain into kout and rec_out (a prompt chunk commits its conv), o_proj's partial.
pub fn kdaBlock(k: *Kernels, s: *Stream, c: Config, w: *const KdaW, st: KdaState, sc: *const Scratch, rows: usize, mode: Mode) !void {
    if (rows == 0 or rows > sc.rows) return error.GlmLayerRows;
    const ps: usize = @as(usize, w.p_stride) * 2;
    try k.dense.prompt(s.*, w.qkv, sc.normed, .bf16, rows, sc.qkv, .bf16);
    try copyRows(k.d, s.*, sc.p, ps, sc.qkv, @as(usize, w.qkv.n) * 2, @as(usize, w.qkv.n) * 2, rows);
    if (w.fa_off + w.fab.n != w.p_stride) return error.GlmLayerShape;
    try k.bmm.matmul(s.*, w.fab, sc.normed, c.d, rows, sc.fab, false, sc.mpart);
    try copyRows(k.d, s.*, sc.p + @as(u64, w.fa_off) * 2, ps, sc.fab, @as(usize, w.fab.n) * 2, @as(usize, w.fab.n) * 2, rows);
    try k.bmm.matmul(s.*, w.fb, sc.p + @as(u64, w.fa_off) * 2, w.p_stride, rows, sc.ka, false, sc.mpart);
    try k.bmm.matmul(s.*, w.gb, sc.p + @as(u64, w.ga_off) * 2, w.p_stride, rows, sc.kg, false, sc.mpart);
    const per: i64 = @as(i64, w.heads) * kda_.d_k;
    const r32: u32 = @intCast(rows);
    const b_off: i32 = @intCast(w.b_off);
    switch (mode) {
        .decode => if (rows >= 64)
            try k.kda.chainWide(s, w.heads, sc.p, w.p_stride, b_off, sc.ka, per, sc.kg, per, st.conv, w.conv, st.rec_in, w.a_log, w.dt_bias, w.norm, c.eps, c.lower, r32, sc.kout, st.rec_out, sc.kda)
        else
            try k.kda.chain(s, w.heads, sc.p, w.p_stride, b_off, sc.ka, per, sc.kg, per, st.conv, w.conv, st.rec_in, w.a_log, w.dt_bias, w.norm, c.eps, c.lower, r32, sc.kout, st.rec_out, sc.kda),
        .prompt => |pos| {
            const off: u32 = @intCast(pos % chunk_.sub_chunk);
            try k.chunk.chain(s, w.heads, sc.p, w.p_stride, b_off, sc.ka, per, sc.kg, per, st.conv, w.conv, st.rec_in, w.a_log, w.dt_bias, w.norm, c.eps, c.lower, r32, off, sc.kout, st.rec_out);
            try shiftConv(k, s, w, st, sc, rows);
        },
    }
    try k.dense.prompt(s.*, w.o, sc.kout, .bf16, rows, sc.part, .f32);
}

/// _shift_conv for this layer: the window becomes rows keep .. keep + 2 of [old window; p rows].
fn shiftConv(k: *Kernels, s: *Stream, w: *const KdaW, st: KdaState, sc: *const Scratch, keep: usize) !void {
    const ch: u32 = 3 * w.heads * kda_.d_k;
    try k.kda.convShift(s, st.conv, sc.p, 1, @intCast(keep), 3 * @as(i64, ch), @as(i64, @intCast(sc.rows)) * w.p_stride, w.p_stride, ch);
}

/// commit(rows, keep) after a decode window: dropped rows replayed out of rec_out, then the conv window shifted.
pub fn commit(k: *Kernels, s: *Stream, w: *const KdaW, st: KdaState, sc: *const Scratch, rows: usize, keep: usize) !void {
    if (keep < 1 or keep > rows or rows > sc.rows) return error.GlmLayerKeep;
    if (keep < rows) try k.kda.replay(s, w.heads, st.rec_in, sc.kda, @intCast(keep), st.rec_out);
    try shiftConv(k, s, w, st, sc, keep);
}

/// A SwiGLU MLP on normed (shared expert or dense layer): EXL3 gate and up, the fused pair, down into out fp32.
fn swigluMlp(k: *Kernels, s: Stream, c: Config, gate: exl3.Weight, up: exl3.Weight, down: exl3.Weight, sc: *const Scratch, rows: usize, out: u64) !void {
    const sw: usize = gate.n;
    if (sw > sc.width or up.n != sw or down.k != sw) return error.GlmLayerWidth;
    try k.dense.prompt(s, gate, sc.normed, .bf16, rows, sc.smg, .bf16);
    try k.dense.prompt(s, up, sc.normed, .bf16, rows, sc.smu, .bf16);
    try copyRows(k.d, s, sc.sgu, 4 * sw, sc.smg, 2 * sw, 2 * sw, rows);
    try copyRows(k.d, s, sc.sgu + 2 * sw, 4 * sw, sc.smu, 2 * sw, 2 * sw, rows);
    try k.glue.swiglu(s, sc.sgu, sc.sact, sc.sxs, c.limit, rows, sw);
    try k.dense.prompt(s, down, sc.sact, .bf16, rows, out, .f32);
}

/// moe_block on the EXL3 route: router, select, routed per-slot y, the shared expert into the last slot, combine.
pub fn moeBlock(k: *Kernels, s: Stream, c: Config, w: *const MoeW, sc: *const Scratch, rows: usize) !void {
    if (rows == 0 or rows > sc.rows) return error.GlmLayerRows;
    const esc = sc.esc orelse return error.GlmLayerScratch;
    const d: usize = c.d;
    const slots: usize = c.top_k + 1;
    if (w.router.n != c.experts or w.router.k != d) return error.GlmLayerShape;
    try k.bmm.router(s, sc.normed, c.d, w.router, rows, sc.mlog, sc.mpart);
    try k.glue.topk(s, sc.mlog, w.bias, sc.pick, sc.wts, c.routed_scale, rows, c.experts, c.top_k);
    try k.experts.runPerSlot(s, w.experts, sc.normed, d, .bf16, sc.pick, rows, sc.ey, esc);
    try swigluMlp(k, s, c, w.gate, w.up, w.down, sc, rows, sc.sy);
    try copyRows(k.d, s, sc.ey + (slots - 1) * d * 4, slots * d * 4, sc.sy, d * 4, d * 4, rows);
    try k.glue2.combineF32(s, sc.ey, sc.wts, sc.part, rows, d, slots);
}

/// mlp_block: a dense layer's MLP, the rank's fp32 partial into part.
pub fn mlpBlock(k: *Kernels, s: Stream, c: Config, w: *const MlpW, sc: *const Scratch, rows: usize) !void {
    if (rows == 0 or rows > sc.rows) return error.GlmLayerRows;
    try swigluMlp(k, s, c, w.gate, w.up, w.down, sc, rows, sc.part);
}

/// hc_post of the gathered partials (rank 0's first) into x in place.
fn post(k: *Kernels, s: Stream, c: Config, x: u64, sc: *const Scratch, g: Gather, site: Site, rows: usize) !void {
    if (g.world != sc.world) return error.GlmLayerWorld;
    try g.run(g.ctx, s, site, sc.part, sc.gath, rows);
    try k.glue.hcPost(s, x, x, sc.gath, sc.post, sc.comb, @intCast(rows * c.d), g.world, rows, c.d, c.streams);
}

/// layer_forward: x [rows, S * D] bf16 in place; a KDA state per its mode (a decode window commits later), or DSA.
pub fn forward(k: *Kernels, s: *Stream, c: Config, w: *const LayerW, mix: MixState, sc: *const Scratch, x: u64, rows: usize, g: Gather) !void {
    if (!std.mem.eql(u8, @tagName(w.mixer), @tagName(mix))) return error.GlmLayerMixer;
    try hcPre(k, s.*, c, w.attn_hc, w.in_norm, x, sc, rows);
    switch (mix) {
        .kda => |m| try kdaBlock(k, s, c, &w.mixer.kda, m.st, sc, rows, m.mode),
        .dsa => |m| {
            const dc = c.dsa orelse return error.GlmLayerConfig;
            try dsa_.dsaBlock(k.dsa(), s, dc, &w.mixer.dsa, m.st, m.sc, sc.normed, sc.part, rows, m.pos, m.capture);
        },
    }
    try post(k, s.*, c, x, sc, g, .attn, rows);
    try hcPre(k, s.*, c, w.ffn_hc, w.post_norm, x, sc, rows);
    switch (w.ffn) {
        .moe => |*m| try moeBlock(k, s.*, c, m, sc, rows),
        .mlp => |*m| try mlpBlock(k, s.*, c, m, sc, rows),
    }
    try post(k, s.*, c, x, sc, g, .ffn, rows);
}
