//! Qwen3.8-27B weights onto the device: int4 projections (MLX affine 4/64 or GPTQ 4/128), sharded across TP={1,2,4}.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const rope_table = @import("rope_table.zig");
const Checkpoint = dev.Checkpoint;
const D = q.D;
const Dev = dev.Dev;
const LAYERS = q.LAYERS;
const alloc = q.alloc;
const isAttention = q.isAttention;
const kvBytes = fwd.kvBytes;
const off = dev.off;

/// An int4 projection in the GEMM layout (words, fp16 scales, biases); `wide` picks the kernel family, `ilv` gate|up.
pub const Proj = struct { w: *anyopaque, s: *anyopaque, bg: *anyopaque, n: u32, k: u32, wide: bool, gs: u32 = 64, ilv: bool = false };

pub const PartSlice = struct {
    suffix: []const u8,
    k_off: u32 = 0,
    k_len: u32 = 0, // 0 => full K
    n_off: u32 = 0,
    n_len: u32 = 0, // 0 => full N
};

pub const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    attn: bool,
    mixer_in: Proj, // gdn.in or attn.qkv
    mixer_out: Proj, // out_proj or o_proj
    gu: Proj,
    down: Proj,
    // GDN
    ab: ?*anyopaque = null, // GPTQ: the a and b gates, fp16 [2 * gv, 5120] (dense, as stored), into P columns ga …
    conv_w: ?*anyopaque = null,
    alog: ?*anyopaque = null,
    dt: ?*anyopaque = null,
    gnorm: ?*anyopaque = null,
    rec: ?*anyopaque = null, // fp32 [gv, 128, 128]
    conv: [2]?*anyopaque = .{ null, null }, // fp16 [3, nconv], ping-pong
    conv_cur: u1 = 0,
    // a verified block's GDN inputs and its entry state, to replay the kept rows (Runner.commit)
    rec_save: ?*anyopaque = null,
    stash: ?*anyopaque = null, // per-row stash: q, k, v, g, β and the qkv rows
    // attention
    qn: ?*anyopaque = null,
    kn: ?*anyopaque = null,
    kc: ?*anyopaque = null, // bf16 [MAX_CTX, akv, 256]
    vc: ?*anyopaque = null,
    // rec, conv, conv_cur, kc and vc above are the selected slot's (Runner.select); each slot's own are kept here
    slot_rec: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_save: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_conv: [MAX_SLOTS][2]?*anyopaque = @splat(.{ null, null }),
    slot_cur: [MAX_SLOTS]u1 = @splat(0),
    slot_kc: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_vc: [MAX_SLOTS]?*anyopaque = @splat(null),
};

/// Sequences a runner holds at most (one a stream of the lane host).
pub const MAX_SLOTS = 16;

pub const Model = struct {
    embed_w: *anyopaque, // MLX: words [V, D/8] ; GPTQ: the fp16 table [V, D]
    embed_s: ?*anyopaque = null, // MLX only
    embed_b: ?*anyopaque = null,
    layers: [LAYERS]Layer,
    norm: *anyopaque,
    head: Proj,
    invf: *anyopaque,
};

pub fn bf16Tensor(d: *Dev, c: *const Checkpoint, name: []const u8) !*anyopaque {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
    return d.upload(t.bytes);
}

/// Fused projection from MLX parts (optionally sliced on N and/or K) stacked on N, repacked on the GPU.
pub fn projectionSliced(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const PartSlice, wide: bool, min_n: u32, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p.suffix }));
        const full_n: u32 = @intCast(w.shape[0]);
        const full_k: u32 = @intCast(w.shape[1] * 8);
        const nl = if (p.n_len == 0) full_n else p.n_len;
        const kl = if (p.k_len == 0) full_k else p.k_len;
        if (p.n_off + nl > full_n or p.k_off + kl > full_k or p.k_off % 64 != 0 or kl % 64 != 0) return error.UnexpectedShape;
        n_total += nl;
        k = kl;
    }
    const n_real = n_total;
    n_total = (@max(n_total, min_n) + 255) / 256 * 256;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, k / 64) * n_total * 2), .n = n_total, .k = k, .wide = wide, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.w, @as(u32, 0), @as(usize, k / 8) * n_total * 4);
        try d.fill(out.s, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p.suffix }));
        const s = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p.suffix }));
        const b = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.biases", .{ prefix, p.suffix }));
        const full_n: u32 = @intCast(w.shape[0]);
        const full_k: u32 = @intCast(w.shape[1] * 8);
        const n: u32 = if (p.n_len == 0) full_n else p.n_len;
        const w_bytes = @as(usize, n) * (k / 8) * 4;
        const s_bytes = @as(usize, n) * (k / 64) * 2;
        const sw = stage;
        const ss = off(stage, w_bytes);
        const sb = off(ss, s_bytes);

        if (p.k_off == 0 and k == full_k) {
            const w_row = @as(usize, full_k / 8) * 4;
            const s_row = @as(usize, full_k / 64) * 2;
            try d.copyIn(sw, w.bytes[@as(usize, p.n_off) * w_row ..][0..w_bytes]);
            try d.copyIn(ss, s.bytes[@as(usize, p.n_off) * s_row ..][0..s_bytes]);
            try d.copyIn(sb, b.bytes[@as(usize, p.n_off) * s_row ..][0..s_bytes]);
        } else {
            const hw = try alloc.alloc(u8, w_bytes);
            defer alloc.free(hw);
            const hs = try alloc.alloc(u8, s_bytes);
            defer alloc.free(hs);
            const hb = try alloc.alloc(u8, s_bytes);
            defer alloc.free(hb);
            const full_wr = @as(usize, full_k / 8) * 4;
            const sub_wr = @as(usize, k / 8) * 4;
            const w_koff = @as(usize, p.k_off / 8) * 4;
            const full_sr = @as(usize, full_k / 64) * 2;
            const sub_sr = @as(usize, k / 64) * 2;
            const s_koff = @as(usize, p.k_off / 64) * 2;
            for (0..n) |nr| {
                @memcpy(hw[nr * sub_wr ..][0..sub_wr], w.bytes[(p.n_off + nr) * full_wr + w_koff ..][0..sub_wr]);
                @memcpy(hs[nr * sub_sr ..][0..sub_sr], s.bytes[(p.n_off + nr) * full_sr + s_koff ..][0..sub_sr]);
                @memcpy(hb[nr * sub_sr ..][0..sub_sr], b.bytes[(p.n_off + nr) * full_sr + s_koff ..][0..sub_sr]);
            }
            try d.copyIn(sw, hw);
            try d.copyIn(ss, hs);
            try d.copyIn(sb, hb);
        }
        try d.launch(d.k.repack, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ sw, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// Fused projection from GPTQ symmetric 4-bit parts (optionally sliced on N and/or K) stacked on N.
pub fn projectionGptqSliced(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const PartSlice, wide: bool, min_n: u32, ilv: bool, stage: *anyopaque) !Proj {
    const g = q.GS;
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p.suffix }));
        const full_k: u32 = @intCast(w.shape[0] * 8);
        const full_n: u32 = @intCast(w.shape[1]);
        const nl = if (p.n_len == 0) full_n else p.n_len;
        const kl = if (p.k_len == 0) full_k else p.k_len;
        if (p.n_off + nl > full_n or p.k_off + kl > full_k or p.k_off % g != 0 or kl % g != 0) return error.UnexpectedShape;
        n_total += nl;
        k = kl;
    }
    const n_real = n_total;
    n_total = (@max(n_total, min_n) + 255) / 256 * 256;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / g) * n_total * 2), .bg = try d.alloc_(@as(usize, k / g) * n_total * 2), .n = n_total, .k = k, .wide = wide, .gs = g, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.w, @as(u32, 0), @as(usize, k / 8) * n_total * 4);
        try d.fill(out.s, @as(u16, 0), @as(usize, k / g) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / g) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p.suffix }));
        const sc = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p.suffix }));
        const full_k: u32 = @intCast(w.shape[0] * 8);
        const full_n: u32 = @intCast(w.shape[1]);
        if (!std.mem.eql(u8, sc.dtype, "F16") or sc.shape[0] * g != full_k) return error.UnexpectedGptqLayout;
        const n: u32 = if (p.n_len == 0) full_n else p.n_len;
        const w_bytes = @as(usize, k / 8) * n * 4;
        const s_bytes = @as(usize, k / g) * n * 2;
        const ss = off(stage, w_bytes);

        if (p.n_off == 0 and n == full_n) {
            const w_off = @as(usize, p.k_off / 8) * full_n * 4;
            const s_off = @as(usize, p.k_off / g) * full_n * 2;
            try d.copyIn(stage, w.bytes[w_off..][0..w_bytes]);
            try d.copyIn(ss, sc.bytes[s_off..][0..s_bytes]);
        } else {
            const hw = try alloc.alloc(u8, w_bytes);
            defer alloc.free(hw);
            const hs = try alloc.alloc(u8, s_bytes);
            defer alloc.free(hs);
            const full_wr = @as(usize, full_n) * 4;
            const sub_wr = @as(usize, n) * 4;
            const w_noff = @as(usize, p.n_off) * 4;
            for (0..k / 8) |kr| {
                const src_r = @as(usize, p.k_off / 8) + kr;
                @memcpy(hw[kr * sub_wr ..][0..sub_wr], w.bytes[src_r * full_wr + w_noff ..][0..sub_wr]);
            }
            const full_sr = @as(usize, full_n) * 2;
            const sub_sr = @as(usize, n) * 2;
            const s_noff = @as(usize, p.n_off) * 2;
            for (0..k / g) |gr| {
                const src_g = @as(usize, p.k_off / g) + gr;
                @memcpy(hs[gr * sub_sr ..][0..sub_sr], sc.bytes[src_g * full_sr + s_noff ..][0..sub_sr]);
            }
            try d.copyIn(stage, hw);
            try d.copyIn(ss, hs);
        }
        try d.launch(d.k.repack_gptq, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ stage, ss, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, @intCast(g)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// An fp16 [N_full, K] weight (rows [n_off .. n_off + n_len)) quantized at load to MLX affine 4/64 in the GEMM layout.
pub fn projectionQuantizedSliced(d: *Dev, c: *const Checkpoint, name: []const u8, n_off: u32, n_len: u32, wide: bool, stage: *anyopaque) !Proj {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const full_n: u32 = @intCast(t.shape[0]);
    const k: u32 = @intCast(t.shape[1]);
    const n: u32 = if (n_len == 0) full_n else n_len;
    if (n_off + n > full_n or n % 256 != 0) return error.UnexpectedShape;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n * 4), .s = try d.alloc_(@as(usize, k / 64) * n * 2), .bg = try d.alloc_(@as(usize, k / 64) * n * 2), .n = n, .k = k, .wide = wide };
    const rows: u32 = 32768;
    var n0: u32 = 0;
    while (n0 < n) : (n0 += rows) {
        const r = @min(rows, n - n0);
        const raw = stage;
        const mw = off(raw, @as(usize, r) * k * 2);
        const ms = off(mw, @as(usize, r) * (k / 8) * 4);
        const mb = off(ms, @as(usize, r) * (k / 64) * 2);
        try d.copyIn(raw, t.bytes[@as(usize, n_off + n0) * k * 2 ..][0 .. @as(usize, r) * k * 2]);
        try d.launch(d.k.quantize_mlx, .{ .x = (r + 15) / 16, .y = k / 64 }, .{ raw, @as(i32, @intCast(r)), @as(i32, @intCast(k)), mw, ms, mb });
        try d.launch(d.k.repack, .{ .x = (r + 15) / 16, .y = k / 8 }, .{ mw, ms, mb, out.w, out.s, out.bg, @as(i32, @intCast(r)), @as(i32, @intCast(k)), @as(i32, @intCast(n)), @as(i32, @intCast(n0)), @as(i32, 0) });
        try d.sync();
    }
    return out;
}

fn convertVecSlice(bytes: []const u8, dtype: []const u8, centred: bool, dst: []u16) !void {
    if (std.mem.eql(u8, dtype, "BF16")) {
        @memcpy(std.mem.sliceAsBytes(dst), bytes);
        return;
    }
    if (!std.mem.eql(u8, dtype, "F16") and !std.mem.eql(u8, dtype, "F32")) return error.UnexpectedDtype;
    if (std.mem.eql(u8, dtype, "F32")) {
        for (dst, 0..) |*o, i| {
            var v: f32 = @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
            if (centred) v += 1.0;
            o.* = toBf16(v);
        }
        return;
    }
    for (dst, 0..) |*o, i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
        var v: f32 = h;
        if (centred) v += 1.0;
        o.* = toBf16(v);
    }
}

/// A small weight (optionally sliced `[elem_off .. elem_off + elem_len)`) to bf16 on the device.
pub fn vecTensorSliced(d: *Dev, c: *const Checkpoint, name: []const u8, centred: bool, elem_off: usize, elem_len: usize) !*anyopaque {
    const t = try c.get(name);
    const bpe: usize = if (std.mem.eql(u8, t.dtype, "F32")) 4 else 2;
    const total_elems = t.bytes.len / bpe;
    const len = if (elem_len == 0) total_elems else elem_len;
    if (elem_off + len > total_elems) return error.UnexpectedShape;
    const slice = t.bytes[elem_off * bpe .. (elem_off + len) * bpe];
    if (std.mem.eql(u8, t.dtype, "BF16") and !centred) return d.upload(slice);
    const out = try alloc.alloc(u16, len);
    defer alloc.free(out);
    try convertVecSlice(slice, t.dtype, centred, out);
    return d.upload(std.mem.sliceAsBytes(out));
}

pub fn vecTensor(d: *Dev, c: *const Checkpoint, name: []const u8, centred: bool) !*anyopaque {
    return vecTensorSliced(d, c, name, centred, 0, 0);
}

/// Uploads the 3-segment `(Q | K | V)` causal conv1d weights `[nconv, 1, 4]` for `rank`.
fn convWeightRank(d: *Dev, c: *const Checkpoint, name: []const u8, rank: u32, sh: q.TpShape) !*anyopaque {
    const t = try c.get(name);
    const bpe: usize = if (std.mem.eql(u8, t.dtype, "F32")) 4 else 2;
    if (t.bytes.len != 10240 * 4 * bpe) return error.UnexpectedShape;
    const out = try alloc.alloc(u16, @as(usize, sh.nconv) * 4);
    defer alloc.free(out);
    const q_off: usize = @as(usize, rank * sh.nqk) * 4;
    const q_len: usize = @as(usize, sh.nqk) * 4;
    const k_off: usize = @as(usize, 2048 + rank * sh.nqk) * 4;
    const v_off: usize = @as(usize, 4096 + rank * sh.vd) * 4;
    const v_len: usize = @as(usize, sh.vd) * 4;
    try convertVecSlice(t.bytes[q_off * bpe .. (q_off + q_len) * bpe], t.dtype, false, out[0..q_len]);
    try convertVecSlice(t.bytes[k_off * bpe .. (k_off + q_len) * bpe], t.dtype, false, out[q_len .. 2 * q_len]);
    try convertVecSlice(t.bytes[v_off * bpe .. (v_off + v_len) * bpe], t.dtype, false, out[2 * q_len .. 2 * q_len + v_len]);
    return d.upload(std.mem.sliceAsBytes(out));
}

fn toBf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// Loads rank `rank` of `q.TP` from a symmetric 4-bit GPTQ checkpoint (g128).
pub fn loadGptqRank(d: *Dev, c: *const Checkpoint, rank: u32, vocab_off: u32, vocab_len: u32) !Model {
    const sh = q.tpShape();
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e = try c.get("model.embed_tokens.weight");
    if (!std.mem.eql(u8, e.dtype, "F16")) return error.UnexpectedDtype;
    if (e.bytes.len != q.VOCAB * D * 2) return error.UnexpectedShape;
    m.embed_w = try d.alloc_(e.bytes.len);
    try d.copyIn(m.embed_w, e.bytes);
    m.embed_s = null;
    m.embed_b = null;
    m.norm = try vecTensor(d, c, "model.norm.weight", true);
    m.head = try projectionQuantizedSliced(d, c, "lm_head.weight", vocab_off, vocab_len, true, stage);
    const invf = rope_table.target();
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, sx: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, sx }) catch unreachable;
            }
        }.n;
        l.* = .{
            .in_norm = try vecTensor(d, c, f(p, "input_layernorm.weight"), true),
            .post_norm = try vecTensor(d, c, f(p, "post_attention_layernorm.weight"), true),
            .attn = isAttention(i),
            .mixer_in = undefined,
            .mixer_out = undefined,
            .gu = try projectionGptqSliced(d, c, f(p, "mlp."), &.{
                .{ .suffix = "gate_proj", .n_off = rank * sh.ff, .n_len = sh.ff },
                .{ .suffix = "up_proj", .n_off = rank * sh.ff, .n_len = sh.ff },
            }, true, 0, true, stage),
            .down = try projectionGptqSliced(d, c, f(p, "mlp."), &.{
                .{ .suffix = "down_proj", .k_off = rank * sh.ff, .k_len = sh.ff },
            }, false, 0, false, stage),
        };
        if (l.attn) {
            l.mixer_in = try projectionGptqSliced(d, c, f(p, "self_attn."), &.{
                .{ .suffix = "q_proj", .n_off = rank * sh.aq * 512, .n_len = sh.aq * 512 },
                .{ .suffix = "k_proj", .n_off = rank * sh.akv * 256, .n_len = sh.akv * 256 },
                .{ .suffix = "v_proj", .n_off = rank * sh.akv * 256, .n_len = sh.akv * 256 },
            }, true, 0, false, stage);
            l.mixer_out = try projectionGptqSliced(d, c, f(p, "self_attn."), &.{
                .{ .suffix = "o_proj", .k_off = rank * sh.od, .k_len = sh.od },
            }, false, 0, false, stage);
            l.qn = try vecTensor(d, c, f(p, "self_attn.q_norm.weight"), true);
            l.kn = try vecTensor(d, c, f(p, "self_attn.k_norm.weight"), true);
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try projectionGptqSliced(d, c, f(p, "linear_attn."), &.{
                .{ .suffix = "in_proj_qkv", .n_off = rank * sh.nqk, .n_len = sh.nqk },
                .{ .suffix = "in_proj_qkv", .n_off = 2048 + rank * sh.nqk, .n_len = sh.nqk },
                .{ .suffix = "in_proj_qkv", .n_off = 4096 + rank * sh.vd, .n_len = sh.vd },
                .{ .suffix = "in_proj_z", .n_off = rank * sh.vd, .n_len = sh.vd },
            }, true, sh.gdn_in, false, stage);
            l.mixer_out = try projectionGptqSliced(d, c, f(p, "linear_attn."), &.{
                .{ .suffix = "out_proj", .k_off = rank * sh.vd, .k_len = sh.vd },
            }, false, 0, false, stage);
            const ta = try c.get(f(p, "linear_attn.in_proj_a.weight"));
            const tb = try c.get(f(p, "linear_attn.in_proj_b.weight"));
            if (!std.mem.eql(u8, ta.dtype, "F16") or ta.bytes.len != 48 * D * 2 or tb.bytes.len != 48 * D * 2) return error.UnexpectedShape;
            const ab_part = @as(usize, sh.gv) * D * 2;
            const ab_off = @as(usize, rank) * ab_part;
            l.ab = try d.alloc_(2 * ab_part);
            try d.copyIn(l.ab.?, ta.bytes[ab_off .. ab_off + ab_part]);
            try d.copyIn(off(l.ab.?, ab_part), tb.bytes[ab_off .. ab_off + ab_part]);
            l.conv_w = try convWeightRank(d, c, f(p, "linear_attn.conv1d.weight"), rank, sh);
            l.alog = try vecTensorSliced(d, c, f(p, "linear_attn.A_log"), false, rank * sh.gv, sh.gv);
            l.dt = try vecTensorSliced(d, c, f(p, "linear_attn.dt_bias"), false, rank * sh.gv, sh.gv);
            l.gnorm = try vecTensor(d, c, f(p, "linear_attn.norm.weight"), false);
            const rec_bytes = @as(usize, sh.gv) * 128 * 128 * 4;
            const conv_bytes = 3 * @as(usize, sh.nconv) * 2;
            l.rec = try d.alloc_(rec_bytes);
            l.rec_save = try d.alloc_(rec_bytes);
            l.stash = try d.alloc_(sh.stash);
            try d.fill(l.rec.?, @as(u32, 0), rec_bytes);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(conv_bytes);
                try d.fill(cv.*.?, @as(u16, 0), conv_bytes);
            }
        }
        if (rank == 0 and i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20;
    return m;
}

pub fn loadRank(d: *Dev, c: *const Checkpoint, rank: u32, vocab_off: u32, vocab_len: u32) !Model {
    if (c.has("in_proj_qkv.qweight")) {
        q.GS = 128;
        return loadGptqRank(d, c, rank, vocab_off, vocab_len);
    }
    q.GS = 64;
    const sh = q.tpShape();
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e_w = try c.get("model.embed_tokens.weight");
    const e_s = try c.get("model.embed_tokens.scales");
    const e_b = try c.get("model.embed_tokens.biases");
    if (e_w.bytes.len != q.VOCAB * (D / 8) * 4 or e_s.bytes.len != q.VOCAB * (D / 64) * 2 or e_b.bytes.len != e_s.bytes.len) return error.UnexpectedShape;
    m.embed_w = try d.upload(e_w.bytes);
    m.embed_s = try bf16Tensor(d, c, "model.embed_tokens.scales");
    m.embed_b = try bf16Tensor(d, c, "model.embed_tokens.biases");
    m.norm = try bf16Tensor(d, c, "model.norm.weight");
    m.head = try projectionSliced(d, c, "lm_head", &.{.{ .suffix = "", .n_off = vocab_off, .n_len = vocab_len }}, true, 0, false, stage);
    const invf = rope_table.target();
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, s: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, s }) catch unreachable;
            }
        }.n;
        l.* = .{
            .in_norm = try bf16Tensor(d, c, f(p, "input_layernorm.weight")),
            .post_norm = try bf16Tensor(d, c, f(p, "post_attention_layernorm.weight")),
            .attn = isAttention(i),
            .mixer_in = undefined,
            .mixer_out = undefined,
            .gu = try projectionSliced(d, c, f(p, "mlp."), &.{
                .{ .suffix = "gate_proj", .n_off = rank * sh.ff, .n_len = sh.ff },
                .{ .suffix = "up_proj", .n_off = rank * sh.ff, .n_len = sh.ff },
            }, true, 0, true, stage),
            .down = try projectionSliced(d, c, f(p, "mlp."), &.{
                .{ .suffix = "down_proj", .k_off = rank * sh.ff, .k_len = sh.ff },
            }, false, 0, false, stage),
        };
        if (l.attn) {
            l.mixer_in = try projectionSliced(d, c, f(p, "self_attn."), &.{
                .{ .suffix = "q_proj", .n_off = rank * sh.aq * 512, .n_len = sh.aq * 512 },
                .{ .suffix = "k_proj", .n_off = rank * sh.akv * 256, .n_len = sh.akv * 256 },
                .{ .suffix = "v_proj", .n_off = rank * sh.akv * 256, .n_len = sh.akv * 256 },
            }, true, 0, false, stage);
            l.mixer_out = try projectionSliced(d, c, f(p, "self_attn."), &.{
                .{ .suffix = "o_proj", .k_off = rank * sh.od, .k_len = sh.od },
            }, false, 0, false, stage);
            l.qn = try bf16Tensor(d, c, f(p, "self_attn.q_norm.weight"));
            l.kn = try bf16Tensor(d, c, f(p, "self_attn.k_norm.weight"));
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try projectionSliced(d, c, f(p, "linear_attn."), &.{
                .{ .suffix = "in_proj_qkv", .n_off = rank * sh.nqk, .n_len = sh.nqk },
                .{ .suffix = "in_proj_qkv", .n_off = 2048 + rank * sh.nqk, .n_len = sh.nqk },
                .{ .suffix = "in_proj_qkv", .n_off = 4096 + rank * sh.vd, .n_len = sh.vd },
                .{ .suffix = "in_proj_z", .n_off = rank * sh.vd, .n_len = sh.vd },
                .{ .suffix = "in_proj_a", .n_off = rank * sh.gv, .n_len = sh.gv },
                .{ .suffix = "in_proj_b", .n_off = rank * sh.gv, .n_len = sh.gv },
            }, true, sh.gdn_in, false, stage);
            l.mixer_out = try projectionSliced(d, c, f(p, "linear_attn."), &.{
                .{ .suffix = "out_proj", .k_off = rank * sh.vd, .k_len = sh.vd },
            }, false, 0, false, stage);
            l.conv_w = try convWeightRank(d, c, f(p, "linear_attn.conv1d.weight"), rank, sh);
            l.alog = try vecTensorSliced(d, c, f(p, "linear_attn.A_log"), false, rank * sh.gv, sh.gv);
            l.dt = try vecTensorSliced(d, c, f(p, "linear_attn.dt_bias"), false, rank * sh.gv, sh.gv);
            l.gnorm = try bf16Tensor(d, c, f(p, "linear_attn.norm.weight"));
            const rec_bytes = @as(usize, sh.gv) * 128 * 128 * 4;
            const conv_bytes = 3 * @as(usize, sh.nconv) * 2;
            l.rec = try d.alloc_(rec_bytes);
            l.rec_save = try d.alloc_(rec_bytes);
            l.stash = try d.alloc_(sh.stash);
            try d.fill(l.rec.?, @as(u32, 0), rec_bytes);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(conv_bytes);
                try d.fill(cv.*.?, @as(u16, 0), conv_bytes);
            }
        }
        if (rank == 0 and i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20;
    return m;
}

pub fn load(d: *Dev, c: *const Checkpoint) !Model {
    return loadRank(d, c, 0, 0, q.VOCAB);
}
