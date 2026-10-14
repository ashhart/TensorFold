//! Qwen3.8-27B weights onto the device: int4 projections (MLX affine 4/64 or GPTQ 4/128) and the layer table.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const Checkpoint = dev.Checkpoint;
const D = q.D;
const Dev = dev.Dev;
const GDN_IN = q.GDN_IN;
const LAYERS = q.LAYERS;
const STASH = q.STASH;
const alloc = q.alloc;
const isAttention = q.isAttention;
const kvBytes = fwd.kvBytes;
const off = dev.off;

/// An int4 projection in the GEMM layout (words, fp16 scales, biases); `wide` picks the kernel family, `ilv` gate|up.
pub const Proj = struct { w: *anyopaque, s: *anyopaque, bg: *anyopaque, n: u32, k: u32, wide: bool, gs: u32 = 64, ilv: bool = false };

pub const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    attn: bool,
    mixer_in: Proj, // gdn.in or attn.qkv
    mixer_out: Proj, // out_proj or o_proj
    gu: Proj,
    down: Proj,
    // GDN
    ab: ?*anyopaque = null, // GPTQ: the a and b gates, fp16 [96, 5120] (dense, as stored), into P columns 16 384 …
    conv_w: ?*anyopaque = null,
    alog: ?*anyopaque = null,
    dt: ?*anyopaque = null,
    gnorm: ?*anyopaque = null,
    rec: ?*anyopaque = null, // fp32 [48, 128, 128]
    conv: [2]?*anyopaque = .{ null, null }, // fp16 [3, 10240], ping-pong
    conv_cur: u1 = 0,
    // a verified block's GDN inputs and its entry state, to replay the kept rows (Runner.commit)
    rec_save: ?*anyopaque = null,
    stash: ?*anyopaque = null, // per-row stash: q, k, v, g, β and the qkv rows
    // attention
    qn: ?*anyopaque = null,
    kn: ?*anyopaque = null,
    kc: ?*anyopaque = null, // bf16 [MAX_CTX, 4, 256]
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

/// Fused projection from MLX parts stacked on N, repacked on the GPU through a raw staging area.
pub fn projection(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        n_total += @intCast(w.shape[0]);
        k = @intCast(w.shape[1] * 8);
    }
    // N padded to the GEMM tile: the padding columns have zero scales and biases, so they come out as zeros.
    const n_real = n_total;
    n_total = (n_total + 255) / 256 * 256;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, k / 64) * n_total * 2), .n = n_total, .k = k, .wide = wide, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.s, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        const s = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        const b = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.biases", .{ prefix, p }));
        const n: u32 = @intCast(w.shape[0]);
        const sw = stage;
        const ss = off(stage, w.bytes.len);
        const sb = off(ss, s.bytes.len);
        try d.copyIn(sw, w.bytes);
        try d.copyIn(ss, s.bytes);
        try d.copyIn(sb, b.bytes);
        try d.launch(d.k.repack, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ sw, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// Fused projection from GPTQ symmetric 4-bit parts stacked on N, at least `min_n` columns (the extra ones zero).
pub fn projectionGptq(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, min_n: u32, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p }));
        n_total += @intCast(w.shape[1]);
        k = @intCast(w.shape[0] * 8);
    }
    const n_real = n_total;
    n_total = (@max(n_total, min_n) + 255) / 256 * 256;
    const g = q.GS;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / g) * n_total * 2), .bg = try d.alloc_(@as(usize, k / g) * n_total * 2), .n = n_total, .k = k, .wide = wide, .gs = g, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.w, @as(u32, 0), @as(usize, k / 8) * n_total * 4);
        try d.fill(out.s, @as(u16, 0), @as(usize, k / g) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / g) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p }));
        const sc = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        if (!std.mem.eql(u8, sc.dtype, "F16") or sc.shape[0] * g != k) return error.UnexpectedGptqLayout;
        const n: u32 = @intCast(w.shape[1]);
        const ss = off(stage, w.bytes.len);
        try d.copyIn(stage, w.bytes);
        try d.copyIn(ss, sc.bytes);
        try d.launch(d.k.repack_gptq, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ stage, ss, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, @intCast(g)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// An fp16 [N, K] weight quantized at load to MLX affine 4/64 in the GEMM layout, in row blocks via staging.
pub fn projectionQuantized(d: *Dev, c: *const Checkpoint, name: []const u8, wide: bool, stage: *anyopaque) !Proj {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const n: u32 = @intCast(t.shape[0]);
    const k: u32 = @intCast(t.shape[1]);
    if (n % 256 != 0) return error.UnexpectedShape;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n * 4), .s = try d.alloc_(@as(usize, k / 64) * n * 2), .bg = try d.alloc_(@as(usize, k / 64) * n * 2), .n = n, .k = k, .wide = wide };
    const rows: u32 = 32768;
    var n0: u32 = 0;
    while (n0 < n) : (n0 += rows) {
        const r = @min(rows, n - n0);
        const raw = stage;
        const mw = off(raw, @as(usize, r) * k * 2);
        const ms = off(mw, @as(usize, r) * (k / 8) * 4);
        const mb = off(ms, @as(usize, r) * (k / 64) * 2);
        try d.copyIn(raw, t.bytes[@as(usize, n0) * k * 2 ..][0 .. @as(usize, r) * k * 2]);
        try d.launch(d.k.quantize_mlx, .{ .x = (r + 15) / 16, .y = k / 64 }, .{ raw, @as(i32, @intCast(r)), @as(i32, @intCast(k)), mw, ms, mb });
        try d.launch(d.k.repack, .{ .x = (r + 15) / 16, .y = k / 8 }, .{ mw, ms, mb, out.w, out.s, out.bg, @as(i32, @intCast(r)), @as(i32, @intCast(k)), @as(i32, @intCast(n)), @as(i32, @intCast(n0)), @as(i32, 0) });
        try d.sync();
    }
    return out;
}

/// A small weight to bf16 on the device; `centred`: a GPTQ RMSNorm weight stored as gamma − 1 gets its 1 back.
pub fn vecTensor(d: *Dev, c: *const Checkpoint, name: []const u8, centred: bool) !*anyopaque {
    const t = try c.get(name);
    if (std.mem.eql(u8, t.dtype, "BF16")) return d.upload(t.bytes);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const n = t.bytes.len / 2;
    const out = try alloc.alloc(u16, n);
    defer alloc.free(out);
    for (out, 0..) |*o, i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
        var v: f32 = h;
        if (centred) v += 1.0;
        const u: u32 = @bitCast(v);
        o.* = @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    return d.upload(std.mem.sliceAsBytes(out));
}

/// Qwen3.8-27B from a symmetric 4-bit GPTQ checkpoint (g128): int4 projections, fp16 embedding, head quantized to 4/64.
pub fn loadGptq(d: *Dev, c: *const Checkpoint) !Model {
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e = try c.get("model.embed_tokens.weight");
    if (!std.mem.eql(u8, e.dtype, "F16")) return error.UnexpectedDtype;
    m.embed_w = try d.alloc_(e.bytes.len);
    try d.copyIn(m.embed_w, e.bytes);
    m.embed_s = null;
    m.embed_b = null;
    m.norm = try vecTensor(d, c, "model.norm.weight", true);
    m.head = try projectionQuantized(d, c, "lm_head.weight", true, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, sx: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, sx }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try vecTensor(d, c, f(p, "input_layernorm.weight"), true), .post_norm = try vecTensor(d, c, f(p, "post_attention_layernorm.weight"), true), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projectionGptq(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, 0, true, stage), .down = try projectionGptq(d, c, f(p, "mlp."), &.{"down_proj"}, false, 0, false, stage) };
        if (l.attn) {
            l.mixer_in = try projectionGptq(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, 0, false, stage);
            l.mixer_out = try projectionGptq(d, c, f(p, "self_attn."), &.{"o_proj"}, false, 0, false, stage);
            l.qn = try vecTensor(d, c, f(p, "self_attn.q_norm.weight"), true);
            l.kn = try vecTensor(d, c, f(p, "self_attn.k_norm.weight"), true);
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try projectionGptq(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z" }, true, GDN_IN, false, stage);
            l.mixer_out = try projectionGptq(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, 0, false, stage);
            const ta = try c.get(f(p, "linear_attn.in_proj_a.weight"));
            const tb = try c.get(f(p, "linear_attn.in_proj_b.weight"));
            if (!std.mem.eql(u8, ta.dtype, "F16") or ta.bytes.len != 48 * D * 2 or tb.bytes.len != 48 * D * 2) return error.UnexpectedShape;
            l.ab = try d.alloc_(96 * D * 2);
            try d.copyIn(l.ab.?, ta.bytes);
            try d.copyIn(off(l.ab.?, 48 * D * 2), tb.bytes);
            l.conv_w = try vecTensor(d, c, f(p, "linear_attn.conv1d.weight"), false);
            l.alog = try vecTensor(d, c, f(p, "linear_attn.A_log"), false);
            l.dt = try vecTensor(d, c, f(p, "linear_attn.dt_bias"), false);
            l.gnorm = try vecTensor(d, c, f(p, "linear_attn.norm.weight"), false);
            l.rec = try d.alloc_(48 * 128 * 128 * 4);
            l.rec_save = try d.alloc_(48 * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * 10240 * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    return m;
}

pub fn load(d: *Dev, c: *const Checkpoint) !Model {
    if (c.has("in_proj_qkv.qweight")) {
        q.GS = 128;
        return loadGptq(d, c);
    }
    q.GS = 64;
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e_w = try c.get("model.embed_tokens.weight");
    m.embed_w = try d.upload(e_w.bytes);
    m.embed_s = try bf16Tensor(d, c, "model.embed_tokens.scales");
    m.embed_b = try bf16Tensor(d, c, "model.embed_tokens.biases");
    m.norm = try bf16Tensor(d, c, "model.norm.weight");
    m.head = try projection(d, c, "lm_head", &.{""}, true, false, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, s: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, s }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try bf16Tensor(d, c, f(p, "input_layernorm.weight")), .post_norm = try bf16Tensor(d, c, f(p, "post_attention_layernorm.weight")), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projection(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, true, stage), .down = try projection(d, c, f(p, "mlp."), &.{"down_proj"}, false, false, stage) };
        if (l.attn) {
            l.mixer_in = try projection(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, false, stage);
            l.mixer_out = try projection(d, c, f(p, "self_attn."), &.{"o_proj"}, false, false, stage);
            l.qn = try bf16Tensor(d, c, f(p, "self_attn.q_norm.weight"));
            l.kn = try bf16Tensor(d, c, f(p, "self_attn.k_norm.weight"));
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try projection(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b" }, true, false, stage);
            l.mixer_out = try projection(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, false, stage);
            l.conv_w = try bf16Tensor(d, c, f(p, "linear_attn.conv1d.weight"));
            l.alog = try bf16Tensor(d, c, f(p, "linear_attn.A_log"));
            l.dt = try bf16Tensor(d, c, f(p, "linear_attn.dt_bias"));
            l.gnorm = try bf16Tensor(d, c, f(p, "linear_attn.norm.weight"));
            l.rec = try d.alloc_(48 * 128 * 128 * 4);
            l.rec_save = try d.alloc_(48 * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * 10240 * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    return m;
}
