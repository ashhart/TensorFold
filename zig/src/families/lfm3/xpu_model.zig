//! LFM3 forward on the Intel GPU: MLX 4-bit weights on the device, a conv state and KV rings, windows of rows, no DPAS.

const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("config.zig");
const names = @import("names.zig");
const rt = xpu.rt;
const ld = xpu.loader;

const Buf = rt.Buffer;
const H: u32 = cfg.hidden;
const FF: u32 = cfg.intermediate;
const V: u32 = cfg.vocab;
const NQ: u32 = cfg.query_heads;
const NKV: u32 = cfg.kv_heads;
const HD: u32 = cfg.head_dim;
const QKV: u32 = (NQ + 2 * NKV) * HD;
const chunk: u32 = 128; // keys a split-K work-group scans
const argmax_parts: u32 = 64;
/// Without DPAS (Gen12LP) prompt projections take nem_prow_gen12: 4 outputs a sub-group, as 8 x 8 spills on Gen12LP.
const gen12 = !xpu.kernels.dpas;
const tile_rows: u32 = 8;
const tile_outs: u32 = if (gen12) 4 else 8;

pub const Table = struct { w: Buf, s: Buf, b: Buf };
const Conv = struct { in: Table, out: Table, cw: Buf, state: Buf };
const Attn = struct { q: Table, k: Table, v: Table, g: Table, o: Table, qn: Buf, kn: Buf, kc: Buf, vc: Buf, ring: u32, window: u32 };
const Mixer = union(enum) { conv: Conv, attn: Attn };
const Layer = struct { op_norm: Buf, post_op: Buf, ffn_norm: Buf, post_ffn: Buf, w1: Table, w3: Table, w2: Table, mixer: Mixer };

const Kernels = struct {
    embed: rt.Kernel,
    qmv: rt.Kernel,
    head: rt.Kernel,
    am_part: rt.Kernel,
    am_fin: rt.Kernel,
    pdense: rt.Kernel,
    rms: rt.Kernel,
    pan: rt.Kernel,
    conv: rt.Kernel,
    rope: rt.Kernel,
    vst: rt.Kernel,
    part: rt.Kernel,
    merge: rt.Kernel,
    swiglu: rt.Kernel,
};

/// Device scratch for a window of up to `max_rows` rows.
const Scratch = struct { ids: Buf, x: Buf, xn: Buf, d: Buf, proj: Buf, qkv: Buf, g: Buf, q: Buf, att: Buf, a: Buf, b: Buf, act: Buf, po: Buf, pm: Buf, pl: Buf, logits: Buf, am_scratch: Buf, am_out: Buf };

/// Called after each layer (index) and after the final norm (-1) with the window's rows, bf16 [n][2048].
pub const Hook = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque, layer: i32, rows: []const u16) void };

fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

/// Inverse RoPE frequencies as upstream builds them (fp32): theta^(-2i/128) for i < angles, then zeros (zero-tail).
pub fn invFreq(theta: f32, partial: f32, out: *[HD / 2]f32) void {
    const angles: usize = @intFromFloat(partial * @as(f32, HD) / 2);
    for (out, 0..) |*f, i| f.* = if (i < angles) 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, HD)) else 0;
}

pub const Model = struct {
    r: *rt.Runtime,
    layers: []Layer,
    emb: Table,
    norm_f: Buf,
    inv_full: Buf,
    inv_slide: Buf,
    k: Kernels,
    s: Scratch,
    host_rows: []u16,
    cap: u32,
    max_rows: u32,
    maxc: u32,
    pos: u32 = 0,
    device_bytes: u64 = 0,

    /// The affine tensors of module `m` (in layer `layer`, or a model tensor) by names.zig in namespace `ns`.
    fn table(l: *ld.Loader, ns: names.Namespace, layer: ?usize, m: names.Module) !Table {
        var a: [160]u8 = undefined;
        var b: [176]u8 = undefined;
        const base = try names.moduleName(&a, ns, layer, m);
        return .{
            .w = try l.load(try std.fmt.bufPrint(&b, "{s}.weight", .{base})),
            .s = try l.load(try std.fmt.bufPrint(&b, "{s}.scales", .{base})),
            .b = try l.load(try std.fmt.bufPrint(&b, "{s}.biases", .{base})),
        };
    }

    fn vec(l: *ld.Loader, ns: names.Namespace, layer: ?usize, m: names.Module) !Buf {
        var a: [160]u8 = undefined;
        return l.load(try names.moduleName(&a, ns, layer, m));
    }

    /// The checkpoint's namespace (bare `model.` or wrapped `language_model.model.`) from the embedding it holds.
    fn namespace(l: *const ld.Loader) !names.Namespace {
        var a: [64]u8 = undefined;
        var b: [64]u8 = undefined;
        return names.detect(l.map.contains(try names.embedName(&a, .bare)), l.map.contains(try names.embedName(&b, .wrapped)));
    }

    /// Uploads the checkpoint: `cap` positions on full-attention layers, windows of at most `max_rows` rows.
    pub fn load(gpa: std.mem.Allocator, r: *rt.Runtime, l: *ld.Loader, c: *const cfg.Config, cap: u32, max_rows: u32) !Model {
        if (max_rows == 0 or max_rows > 256 or cap < max_rows) return error.Invalid;
        var bm = try r.module(xpu.kernels.basic);
        var am = try r.module(xpu.kernels.attn);
        var pm = try r.module(if (gen12) xpu.kernels.nem_prow_gen12 else xpu.kernels.nem_prow);
        var fm = try r.module(xpu.kernels.lfm3);
        const k: Kernels = .{
            .embed = try bm.kernel("embed4", .{ 64, 1, 1 }),
            .qmv = try am.kernel("qmv4_bf", .{ 64, 1, 1 }),
            .head = try am.kernel("qmv4_f32", .{ 64, 1, 1 }),
            .am_part = try am.kernel("argmax_partial", .{ 256, 1, 1 }),
            .am_fin = try am.kernel("argmax_final", .{ 256, 1, 1 }),
            .pdense = try pm.kernel("pdense", .{ 16 * (16 / tile_outs), 1, 1 }),
            .rms = try fm.kernel("lfm3_rmsnorm", .{ 256, 1, 1 }),
            .pan = try fm.kernel("lfm3_post_add_norm", .{ 256, 1, 1 }),
            .conv = try fm.kernel("lfm3_conv_rows", .{ 64, 1, 1 }),
            .rope = try fm.kernel("lfm3_qk_norm_rope", .{ 64, 1, 1 }),
            .vst = try fm.kernel("lfm3_v_store", .{ 64, 1, 1 }),
            .part = try fm.kernel("lfm3_attn_partial", .{ 64, 1, 1 }),
            .merge = try fm.kernel("lfm3_attn_merge_gate", .{ 16, 1, 1 }),
            .swiglu = try fm.kernel("lfm3_swiglu", .{ 64, 1, 1 }),
        };
        const maxc = (cap + chunk - 1) / chunk;
        const w: usize = max_rows;
        const s: Scratch = .{
            .ids = try l.empty(w * 4),
            .x = try l.empty(w * H * 2),
            .xn = try l.empty(w * H * 2),
            .d = try l.empty(w * H * 2),
            .proj = try l.empty(w * 3 * H * 2),
            .qkv = try l.empty(w * QKV * 2),
            .g = try l.empty(w * NQ * 2),
            .q = try l.empty(w * NQ * HD * 2),
            .att = try l.empty(w * NQ * HD * 2),
            .a = try l.empty(w * FF * 2),
            .b = try l.empty(w * FF * 2),
            .act = try l.empty(w * FF * 2),
            .po = try l.empty(w * NQ * maxc * HD * 4),
            .pm = try l.empty(w * NQ * maxc * 4),
            .pl = try l.empty(w * NQ * maxc * 4),
            .logits = try l.empty(@as(usize, V) * 4),
            .am_scratch = try l.empty(argmax_parts * 8),
            .am_out = try l.empty(4),
        };
        const ns = try namespace(l);
        const layers = try gpa.alloc(Layer, cfg.layers);
        for (layers, 0..) |*ly, i| {
            const kind = c.layer_types[i];
            ly.* = .{
                .op_norm = try vec(l, ns, i, .operator_norm),
                .post_op = try vec(l, ns, i, .post_operator_norm),
                .ffn_norm = try vec(l, ns, i, .ffn_norm),
                .post_ffn = try vec(l, ns, i, .post_ffn_norm),
                .w1 = try table(l, ns, i, .w1),
                .w3 = try table(l, ns, i, .w3),
                .w2 = try table(l, ns, i, .w2),
                .mixer = if (kind == .conv) .{ .conv = .{
                    .in = try table(l, ns, i, .conv_in),
                    .out = try table(l, ns, i, .conv_out),
                    .cw = try vec(l, ns, i, .conv),
                    .state = try l.zeros(6 * H * 2),
                } } else blk: {
                    const ring: u32 = if (kind == .sliding_attention) cfg.sliding_window + max_rows else cap;
                    break :blk .{ .attn = .{
                        .q = try table(l, ns, i, .q_proj),
                        .k = try table(l, ns, i, .k_proj),
                        .v = try table(l, ns, i, .v_proj),
                        .g = try table(l, ns, i, .gate_proj),
                        .o = try table(l, ns, i, .o_proj),
                        .qn = try vec(l, ns, i, .q_norm),
                        .kn = try vec(l, ns, i, .k_norm),
                        .kc = try l.empty(@as(usize, ring) * NKV * HD * 2),
                        .vc = try l.empty(@as(usize, ring) * NKV * HD * 2),
                        .ring = ring,
                        .window = if (kind == .sliding_attention) cfg.sliding_window else 0,
                    } };
                },
            };
        }
        var inv: [HD / 2]f32 = undefined;
        const inv_full = try l.empty(HD / 2 * 4);
        invFreq(c.full_rotary.theta, c.full_rotary.partial, &inv);
        try r.upload(inv_full, std.mem.sliceAsBytes(&inv));
        const inv_slide = try l.empty(HD / 2 * 4);
        invFreq(c.sliding_rotary.theta, c.sliding_rotary.partial, &inv);
        try r.upload(inv_slide, std.mem.sliceAsBytes(&inv));
        try r.sync();
        return .{
            .r = r,
            .layers = layers,
            .emb = try table(l, ns, null, .embed_tokens),
            .norm_f = try vec(l, ns, null, .embedding_norm),
            .inv_full = inv_full,
            .inv_slide = inv_slide,
            .k = k,
            .s = s,
            .host_rows = try gpa.alloc(u16, w * H),
            .cap = cap,
            .max_rows = max_rows,
            .maxc = maxc,
            .device_bytes = l.total,
        };
    }

    /// Back to position 0: the conv states are zeroed; KV slots are only read once written.
    pub fn reset(m: *Model) !void {
        const zero: [6 * 256]u16 = @splat(0);
        for (m.layers) |ly| switch (ly.mixer) {
            .conv => |c| {
                var off: usize = 0;
                while (off < 6 * H * 2) : (off += zero.len * 2) {
                    try m.r.upload(.{ .rt = c.state.rt, .ptr = @ptrFromInt(@intFromPtr(c.state.ptr.?) + off), .len = zero.len * 2 }, std.mem.sliceAsBytes(&zero));
                }
            },
            .attn => {},
        };
        try m.r.sync();
        m.pos = 0;
    }

    /// y[r * y_stride + y_off + o] = x[r] . W[o] for n rows: the decode matvec for one row, pdense for a window.
    fn proj(m: *Model, t: Table, x: Buf, y: Buf, in_dim: u32, rows: u32, n: u32, y_stride: u32, y_off: u32) !void {
        if (n == 1) return run(&m.k.qmv, .{ (rows + 3) / 4, 1, 1 }, .{ t.w, t.s, t.b, x, y, in_dim, @as(u32, 0), y_off, rows });
        try run(&m.k.pdense, .{ (rows + 15) / 16, (n + tile_rows - 1) / tile_rows, 1 }, .{ t.w, t.s, t.b, x, y, in_dim, rows, n, y_stride, y_off, @as(u32, 0) });
    }

    fn mixer(m: *Model, ly: *const Layer, n: u32) !void {
        const s = &m.s;
        switch (ly.mixer) {
            .conv => |c| {
                try m.proj(c.in, s.xn, s.proj, H, 3 * H, n, 3 * H, 0);
                try run(&m.k.conv, .{ H / 64, 1, 1 }, .{ s.proj, 3 * H, c.state, c.state, c.cw, s.att, n, H });
                try m.proj(c.out, s.att, s.d, H, H, n, H, 0);
            },
            .attn => |a| {
                try m.proj(a.q, s.xn, s.qkv, H, NQ * HD, n, QKV, 0);
                try m.proj(a.k, s.xn, s.qkv, H, NKV * HD, n, QKV, NQ * HD);
                try m.proj(a.v, s.xn, s.qkv, H, NKV * HD, n, QKV, (NQ + NKV) * HD);
                try m.proj(a.g, s.xn, s.g, H, NQ, n, NQ, 0);
                const inv = if (a.window == 0) m.inv_full else m.inv_slide;
                try run(&m.k.rope, .{ NQ, n, 1 }, .{ s.qkv, QKV, @as(u32, 0), s.q, NQ * HD, a.qn, inv, m.pos, @as(u32, 0), cfg.eps });
                try run(&m.k.rope, .{ NKV, n, 1 }, .{ s.qkv, QKV, NQ * HD, a.kc, NKV * HD, a.kn, inv, m.pos, a.ring, cfg.eps });
                try run(&m.k.vst, .{ NKV * HD / 64, n, 1 }, .{ s.qkv, QKV, (NQ + NKV) * HD, a.vc, NKV * HD, m.pos, a.ring });
                const span = if (a.window == 0) m.pos + n else @min(m.pos + n, a.window);
                const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
                try run(&m.k.part, .{ NKV, (span + chunk - 1) / chunk, n }, .{ s.q, a.kc, a.vc, s.po, s.pm, s.pl, m.pos, chunk, a.ring, a.window, NKV, m.maxc, scale });
                try run(&m.k.merge, .{ NQ, n, 1 }, .{ s.po, s.pm, s.pl, s.g, s.att, m.pos, chunk, a.window, NQ, m.maxc });
                try m.proj(a.o, s.att, s.d, NQ * HD, H, n, H, 0);
            },
        }
    }

    fn hook(m: *Model, h: ?Hook, layer: i32, b: Buf, n: u32) !void {
        const k = h orelse return;
        const rows = m.host_rows[0 .. n * H];
        try m.r.sync();
        try m.r.download(std.mem.sliceAsBytes(rows), b);
        try m.r.sync();
        k.f(k.ctx, layer, rows);
    }

    /// Runs the tokens as one window at m.pos (1 .. max_rows rows); `logits`: the last row's logits and argmax too.
    pub fn forward(m: *Model, ids: []const u32, logits: bool, h: ?Hook) !void {
        const n: u32 = @intCast(ids.len);
        if (n == 0 or n > m.max_rows) return error.Invalid;
        if (m.pos + n > m.cap) return error.ContextFull;
        const s = &m.s;
        try m.r.upload(s.ids, std.mem.sliceAsBytes(ids));
        try run(&m.k.embed, .{ n, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, s.ids, s.x, H });
        try run(&m.k.rms, .{ n, 1, 1 }, .{ s.x, m.layers[0].op_norm, s.xn, H, cfg.eps });
        for (m.layers, 0..) |*ly, i| {
            try m.mixer(ly, n);
            try run(&m.k.pan, .{ n, 1, 1 }, .{ s.d, s.x, ly.post_op, ly.ffn_norm, s.xn, H, cfg.eps });
            try m.proj(ly.w1, s.xn, s.a, H, FF, n, FF, 0);
            try m.proj(ly.w3, s.xn, s.b, H, FF, n, FF, 0);
            try run(&m.k.swiglu, .{ (n * FF + 63) / 64, 1, 1 }, .{ s.a, s.b, s.act, n * FF });
            try m.proj(ly.w2, s.act, s.d, FF, H, n, H, 0);
            const next = if (i + 1 < m.layers.len) m.layers[i + 1].op_norm else m.norm_f;
            try run(&m.k.pan, .{ n, 1, 1 }, .{ s.d, s.x, ly.post_ffn, next, s.xn, H, cfg.eps });
            try m.hook(h, @intCast(i), s.x, n);
        }
        try m.hook(h, -1, s.xn, n);
        m.pos += n;
        if (!logits) return;
        try run(&m.k.head, .{ V / 4, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, s.xn, s.logits, H, (n - 1) * H, @as(u32, 0), V });
        const per = (V + argmax_parts - 1) / argmax_parts;
        try run(&m.k.am_part, .{ argmax_parts, 1, 1 }, .{ s.logits, s.am_scratch, V, per });
        try run(&m.k.am_fin, .{ 1, 1, 1 }, .{ s.am_scratch, s.am_out, argmax_parts });
    }

    /// Prefill of a prompt in windows of max_rows; the last window leaves its logits and argmax.
    pub fn prefill(m: *Model, ids: []const u32, h: ?Hook) !void {
        var i: usize = 0;
        while (i < ids.len) {
            const n = @min(ids.len - i, m.max_rows);
            try m.forward(ids[i .. i + n], i + n == ids.len, h);
            i += n;
        }
    }

    /// The greedy token of the last forward with logits.
    pub fn argmax(m: *Model) !u32 {
        var out: [1]i32 = undefined;
        try m.r.sync();
        try m.r.download(std.mem.sliceAsBytes(&out), m.s.am_out);
        try m.r.sync();
        return @intCast(out[0]);
    }

    pub fn fetchLogits(m: *Model, out: []f32) !void {
        try m.r.sync();
        try m.r.download(std.mem.sliceAsBytes(out[0..V]), m.s.logits);
        try m.r.sync();
    }
};
