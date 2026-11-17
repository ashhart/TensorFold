//! ZONOS3 Mini's backbone on Metal: frame rows in, the next frame's logits out, one fp32 row path for all rows.
const std = @import("std");
const cfg = @import("../config.zig");
const gpu = @import("gpu.zig");
const wts = @import("weights.zig");
const k = @import("kernels.zig");
const metallib = @import("zonos3_metallib").bytes;

const D = cfg.dim;
const W = cfg.frame_width;
const Gpu = gpu.Gpu;

pub const Options = struct {
    max_ctx: usize = 4096, // positions (prompt rows + frames), at most the model's 32768
    max_rows: usize = 64, // rows a pass

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

pub const Engine = struct {
    gpa: std.mem.Allocator,
    o: Options,
    c: cfg.Config,
    g: *Gpu,
    w: *wts.Weights,
    weight_slots: usize, // the device's buffers before the state's
    position: usize = 0,
    // state
    kv: [cfg.layers][2]u64, // attention layers: K, V fp32 [max_ctx][4][32]
    conv: [cfg.layers]u64, // GDN layers: the last 3 conv inputs, fp32 [1536][3]
    gstate: [cfg.layers]u64, // GDN layers: fp32 [4][128][128]
    state: u64, // the conv and recurrent states in one block (zeroed per request)
    state_bytes: usize,
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
        e.c = try cfg.Config.load(gpa, io, model_dir);
        e.w = try wts.load(gpa, io, g, model_dir, e.c);
        errdefer {
            e.w.deinit();
            gpa.destroy(e.w);
        }
        e.weight_slots = g.slots();
        errdefer g.releaseFrom(e.weight_slots);
        try e.allocState();
        try e.reset();
        return e;
    }

    pub fn destroy(e: *Engine) void {
        e.w.deinit();
        e.gpa.destroy(e.w);
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

    fn allocState(e: *Engine) !void {
        const R = e.o.max_rows;
        const kvb = e.o.max_ctx * cfg.kv_width * 4;
        const conv_b = cfg.gdn_conv * 3 * 4;
        const gs_b = cfg.gdn_heads * cfg.gdn_dim * cfg.gdn_dim * 4;
        var n_gdn: usize = 0;
        for (0..cfg.layers) |li| n_gdn += @intFromBool(cfg.usesGdn(li));
        e.state_bytes = n_gdn * (conv_b + gs_b);
        e.state = try e.alloc(e.state_bytes);
        var g: usize = 0;
        for (0..cfg.layers) |li| {
            if (cfg.usesGdn(li)) {
                e.kv[li] = .{ 0, 0 };
                e.conv[li] = e.state + g * (conv_b + gs_b);
                e.gstate[li] = e.conv[li] + conv_b;
                g += 1;
            } else {
                e.kv[li] = .{ try e.alloc(kvb), try e.alloc(kvb) };
                e.conv[li] = 0;
                e.gstate[li] = 0;
            }
        }
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
            if (r[cfg.codebooks] < 0 or r[cfg.codebooks] >= e.w.text_rows) return error.BadFrameRow;
        }
        var at: usize = 0;
        while (at < rows.len) {
            const t = @min(e.o.max_rows, rows.len - at);
            try e.pass(rows[at..][0..t], at, replaced, at + t == rows.len, probe);
            at += t;
        }
    }

    fn pass(e: *Engine, rows: []const [W]i32, first: usize, replaced: []const Replace, last: bool, probe: ?Probe) !void {
        const g = e.g;
        const t = rows.len;
        const R = e.o.max_rows;
        const pos0 = e.position;
        try g.put(e.rows, std.mem.sliceAsBytes(rows));
        k.embed(g, e.rows, e.w.tables, cfg.audio_vocab, e.mixed, D, t);
        if (probe) |p| try p.f(p.ctx, "emb_sum", e.mixed, t, D);
        for (replaced) |r| if (r.row >= first and r.row < first + t) {
            try g.put(e.mixed + (r.row - first) * D * 4, std.mem.asBytes(&r.vec));
        };
        if (probe) |p| try p.f(p.ctx, "emb_cond", e.mixed, t, D);
        k.rmsnorm(g, e.mixed, D, 0, e.slot(0), D, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "emb_norm", e.slot(0), t, D);
        const q = e.w.queries;
        for (&e.w.layers, 0..) |*L, li| {
            const attn_out = e.slot(2 * li + 1);
            const ffn_out = e.slot(2 * li + 2);
            k.attnres(g, e.bank, 2 * li + 1, R, q + 2 * li * D * 2, e.mixed, D, cfg.eps, t);
            if (probe) |p| try p.at("attnres_attn_{d}", li, e.mixed, t, D);
            k.rmsnorm(g, e.mixed, D, L.attention_norm, e.xn, D, D, cfg.eps, t);
            if (probe) |p| try p.at("attn_in_{d}", li, e.xn, t, D);
            switch (L.mix) {
                .gdn => |gd| {
                    const ps = cfg.gdn_proj;
                    k.linear(g, gd.in_proj, e.xn, D, D, ps, e.proj, ps, t, .{});
                    k.convSilu(g, e.proj, ps, t, gd.conv, e.conv[li], e.conv_out, cfg.gdn_conv);
                    if (probe) |p| try p.at("gdn_conv_{d}", li, e.conv_out, t, cfg.gdn_conv);
                    k.gdnPrep(g, e.conv_out, e.proj, ps, 4 * cfg.gdn_inner, gd.a_log, gd.dt_bias, e.gates, t);
                    k.gdn(g, e.conv_out, e.gates, t, e.gstate[li], e.o32);
                    if (probe) |p| try p.at("gdn_recurrent_{d}", li, e.o32, t, cfg.gdn_inner);
                    k.gatedNorm(g, e.o32, e.proj, ps, cfg.gdn_conv, gd.out_weight, e.y, t);
                    if (probe) |p| try p.at("gdn_gated_{d}", li, e.y, t, cfg.gdn_inner);
                    k.linear(g, gd.wo, e.y, cfg.gdn_inner, cfg.gdn_inner, D, attn_out, D, t, .{});
                },
                .attn => |a| {
                    k.linear(g, a.wq, e.xn, D, D, cfg.attn_inner, e.q, cfg.attn_inner, t, .{});
                    k.linear(g, a.wkv, e.xn, D, D, 2 * cfg.kv_width, e.kvraw, 2 * cfg.kv_width, t, .{});
                    k.linear(g, a.gater, e.xn, D, D, cfg.heads, e.gate, cfg.heads, t, .{});
                    k.attnPrep(g, e.q, e.kvraw, a.temp, pos0, e.kv[li][0], e.kv[li][1], cfg.rope_theta, t);
                    k.attention(g, e.q, e.kv[li][0], e.kv[li][1], e.gate, pos0, e.y, t);
                    k.linear(g, a.wo, e.y, cfg.attn_inner, cfg.attn_inner, D, attn_out, D, t, .{});
                },
            }
            if (probe) |p| try p.at("attn_out_{d}", li, attn_out, t, D);
            k.attnres(g, e.bank, 2 * li + 2, R, q + (2 * li + 1) * D * 2, e.mixed, D, cfg.eps, t);
            if (probe) |p| try p.at("attnres_ffn_{d}", li, e.mixed, t, D);
            k.rmsnorm(g, e.mixed, D, L.ffn_norm, e.xn, D, D, cfg.eps, t);
            if (probe) |p| try p.at("ffn_in_{d}", li, e.xn, t, D);
            switch (L.ffn) {
                .dense => |f| {
                    const hidden = cfg.ffn_hidden;
                    k.linear(g, f.w_in, e.xn, D, D, 2 * hidden, e.proj, 2 * hidden, t, .{});
                    k.mulSilu(g, e.proj, 2 * hidden, 0, hidden, hidden, e.act, t);
                    k.linear(g, f.w_out, e.act, hidden, hidden, D, ffn_out, D, t, .{});
                },
                .moe => |m| {
                    const hidden = cfg.moe_hidden;
                    const latent = cfg.moe_latent;
                    k.rmsnorm(g, e.xn, D, m.eda, e.rn, D, D, cfg.eps, t);
                    k.linear(g, m.router, e.rn, D, D, cfg.experts, e.rl, cfg.experts, t, .{});
                    if (probe) |p| try p.at("router_logits_{d}", li, e.rl, t, cfg.experts);
                    k.route(g, e.rl, m.bias, cfg.experts, e.picks, e.rw, t);
                    k.linear(g, m.latent_down, e.xn, D, D, latent, e.latent, latent, t, .{});
                    k.linear(g, m.gate_up, e.latent, latent, latent, 2 * hidden, e.gu, 2 * hidden, t * cfg.top_k, .{ .picks = e.picks, .pick_stride = 2 * hidden * latent, .xrep = cfg.top_k });
                    k.mulSilu(g, e.gu, 2 * hidden, hidden, 0, hidden, e.hid, t * cfg.top_k);
                    k.linear(g, m.down, e.hid, hidden, hidden, latent, e.down, latent, t * cfg.top_k, .{ .picks = e.picks, .pick_stride = latent * hidden });
                    k.combine(g, e.down, e.rw, latent, e.comb, t);
                    k.linear(g, m.latent_up, e.comb, latent, latent, D, ffn_out, D, t, .{});
                },
            }
            if (probe) |p| try p.at("ffn_out_{d}", li, ffn_out, t, D);
        }
        k.attnres(g, e.bank, cfg.sources, R, q + (cfg.sources - 1) * D * 2, e.mixed, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "attnres_final", e.mixed, t, D);
        k.rmsnorm(g, e.mixed, D, e.w.out_norm, e.xn, D, D, cfg.eps, t);
        if (probe) |p| try p.f(p.ctx, "out_norm", e.xn, t, D);
        if (probe != null) {
            k.linear(g, e.w.head, e.xn, D, D, cfg.head_rows, e.logits, cfg.head_rows, t, .{});
            k.softcap(g, e.logits, t * cfg.head_rows, cfg.softcap);
            if (probe) |p| try p.f(p.ctx, "logits", e.logits, t, cfg.head_rows);
        } else if (last) {
            k.linear(g, e.w.head, e.xn + (t - 1) * D * 4, D, D, cfg.head_rows, e.logits, cfg.head_rows, 1, .{});
            k.softcap(g, e.logits, cfg.head_rows, cfg.softcap);
        }
        e.position = pos0 + t;
    }

    /// The last row's logits [codebooks][audio_vocab] to the host.
    pub fn lastLogits(e: *Engine, out: *[cfg.head_rows]f32) !void {
        try e.g.get(e.logits, std.mem.asBytes(out));
    }

    pub fn download(e: *Engine, src: u64, out: []u8) !void {
        try e.g.get(src, out);
    }

    /// A new request: zero recurrent states, position 0. Cache rows are written before they are read.
    pub fn reset(e: *Engine) !void {
        try e.g.flush();
        @memset(e.g.host(e.state, e.state_bytes), 0);
        e.position = 0;
    }
};
