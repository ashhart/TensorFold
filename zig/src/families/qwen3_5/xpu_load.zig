//! Qwen3.8 checkpoint loaders: MLX affine 4-bit and EXL3 trellis layouts behind the same Linear/GdnW/AttnW/MlpW values.

const std = @import("std");
const ld = @import("xpu").loader;
const exl3 = @import("xpu").exl3;
const cq = @import("xpu_config.zig");
const qb = @import("xpu_blocks.zig");
const al = @import("xpu_attn_long.zig");
const qg = @import("xpu_gguf.zig");
const gg = @import("xpu").ggml;

const Buf = qb.Buf;
const Table = qb.Linear;
const mxb = @import("xpu_mlx4b.zig");
const hidden = qb.hidden;
const inter = qb.inter;
const heads = qb.heads;
const kv_dim = qb.kv_dim;
const q_dim = qb.q_dim;
const head_dim = qb.head_dim;
const gdn_qkv = qb.gdn_qkv;
const gdn_v = qb.gdn_v;
const gdn_heads = qb.gdn_heads;

extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;

/// Chooses the weight format from the checkpoint and, for EXL3, builds the shared engine in `ops`; call first.
pub fn attach(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, cfg: cq.Config) !void {
    if (l.gguf) {
        const set = try gpa.create(gg.Set);
        set.* = try gg.Set.init(ops.r);
        ops.gg = set;
        ops.tiled = true;
        ops.f32_norm = std.c.getenv("ARC_F32_NORM") != null; // opt-in: float32 norm math instead of the bf16 rounding
        return;
    }
    if (!l.map.contains("lm_head.trellis")) {
        if (cfg.quantization == null) {
            std.log.err("checkpoint is neither EXL3 (no lm_head.trellis) nor MLX quantized (no quantization in config.json)", .{});
            return error.UnsupportedFormat;
        }
        return;
    }
    const m = try gpa.create(@import("xpu").rt.Module);
    m.* = try ops.r.module(@import("xpu").kernels.exl3_mul1);
    const e = try gpa.create(qb.Exl);
    // largest split-K partials over the Qwen3.8-27B shapes (K, N); max K 17408 for the rotated input
    const shapes = [_][2]u32{ .{ 5120, 17408 }, .{ 17408, 5120 }, .{ 5120, 12288 }, .{ 5120, 1024 }, .{ 6144, 5120 }, .{ 5120, 10240 }, .{ 5120, 6144 }, .{ 5120, cfg.text_config.vocab_size }, .{ 10240, 5120 } };
    var zn: u64 = 0;
    for (shapes) |s| zn = @max(zn, exl3.zn(s[0], s[1]));
    e.* = .{ .eng = try exl3.Engine.init(ops.r, m), .scratch = try exl3.Scratch.init(ops.r, 16, 17408, zn) };
    ops.ex = e;
}

/// Tensor bytes into host memory (EXL3 repacking and uploading happen from there).
fn readHost(gpa: std.mem.Allocator, l: *ld.Loader, name: []const u8, dtype: []const u8, bytes: u64) ![]u8 {
    try need(l, name, dtype, bytes);
    const inf = try l.info(name);
    const buf = try gpa.alloc(u8, inf.len);
    var done: usize = 0;
    while (done < buf.len) {
        const n = pread(inf.fd, buf.ptr + done, buf.len - done, @intCast(inf.off + done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
    return buf;
}

/// An RMSNorm weight: bf16 as stored for MLX; EXL3 keeps the zero-centred (1 + w) form, so it is shifted by one here.
pub fn normW(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, name: []const u8, count: u64) !Buf {
    return normWs(gpa, ops.ex != null, l, name, count);
}

/// `normW` with the convention chosen by the caller (shift: EXL3's zero-centred form).
pub fn normWs(gpa: std.mem.Allocator, shift: bool, l: *ld.Loader, name: []const u8, count: u64) !Buf {
    if (!shift) return plain(l, name, count);
    const raw = try readHost(gpa, l, name, "BF16", count * 2);
    defer gpa.free(raw);
    const v = std.mem.bytesAsSlice(u16, raw);
    for (v) |*e| {
        const f: f32 = @as(f32, @bitCast(@as(u32, e.*) << 16)) + 1.0;
        const u: u32 = @bitCast(f);
        e.* = @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    const b = try l.empty(raw.len);
    try l.r.upload(b, raw);
    try l.r.sync();
    return b;
}

/// Checkpoint tensor name prefix of the language model.
fn base(ops: *qb.Ops) []const u8 {
    return if (ops.ex != null) "model.language_model" else "language_model.model";
}

/// An EXL3 projection [rows][in]: trellis, suh, svh (fp16), mul1 marker; the trellis size gives the bit width.
pub fn exl3Proj(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, prefix: []const u8, rows: u32, in: u32) !Table {
    return exl3ProjE(gpa, &ops.ex.?.eng, l, prefix, rows, in);
}

/// `exl3Proj` on an explicit engine (the MTP head loaded next to a non-EXL3 target).
pub fn exl3ProjE(gpa: std.mem.Allocator, eng: *exl3.Engine, l: *ld.Loader, prefix: []const u8, rows: u32, in: u32) !Table {
    return exl3ProjCut(gpa, eng, l, prefix, rows, in, rows);
}

/// `exl3ProjE` keeping only the first `keep` output rows (a multiple of 128: the output Hadamard works on 128-blocks).
pub fn exl3ProjCut(gpa: std.mem.Allocator, eng: *exl3.Engine, l: *ld.Loader, prefix: []const u8, rows: u32, in: u32, keep: u32) !Table {
    if (keep > rows or keep % 128 != 0) return error.Invalid;
    var nb: [200]u8 = undefined;
    const marker = try std.fmt.bufPrint(&nb, "{s}.mul1", .{prefix});
    if (!l.map.contains(marker)) {
        std.log.err("{s}: not a mul1 EXL3 tensor (marker missing); only the mul1 codebook is wired", .{prefix});
        return error.UnsupportedFormat;
    }
    const tn = try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix});
    const inf = try l.info(tn);
    const per_bit: u64 = @as(u64, in / 16) * (rows / 16) * 32; // bytes of the trellis at one bit per weight
    if (!std.mem.eql(u8, inf.dtype, "I16") or inf.len % per_bit != 0) {
        std.log.err("{s}: trellis {s} {d} bytes does not fit [{d}, {d}]", .{ tn, inf.dtype, inf.len, in, rows });
        return error.UnexpectedTensor;
    }
    const bits: u32 = @intCast(inf.len / per_bit);
    var trellis = try readHost(gpa, l, tn, "I16", inf.len);
    defer gpa.free(trellis);
    if (keep < rows) { // [K/16][N/16][16 bits] int16: the first keep/16 column tiles of every k tile row
        const tw: usize = @as(usize, bits) * 32;
        const nt = rows / 16;
        const kt = in / 16;
        const nk = keep / 16;
        const cut = try gpa.alloc(u8, @as(usize, kt) * nk * tw);
        for (0..kt) |t| @memcpy(cut[t * nk * tw ..][0 .. nk * tw], trellis[t * nt * tw ..][0 .. nk * tw]);
        gpa.free(trellis);
        trellis = cut;
    }
    const suh = try readHost(gpa, l, try std.fmt.bufPrint(&nb, "{s}.suh", .{prefix}), "F16", @as(u64, in) * 2);
    defer gpa.free(suh);
    const svh = try readHost(gpa, l, try std.fmt.bufPrint(&nb, "{s}.svh", .{prefix}), "F16", @as(u64, rows) * 2);
    defer gpa.free(svh);
    const layer = try gpa.create(exl3.Layer);
    layer.* = try exl3.Layer.init(eng, .{ .k = in, .n = keep, .k2 = 2 * bits, .cb = .mul1, .trellis = trellis, .suh = suh, .svh = svh[0 .. @as(usize, keep) * 2] });
    l.total += layer.cols.len + layer.suh.len + layer.svh.len;
    return .{ .format = .exl3, .w = layer.cols, .s = layer.suh, .b = layer.svh, .rows = keep, .in = in, .ex = layer };
}

/// A projection in whichever format the checkpoint uses.
fn proj(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, prefix: []const u8, rows: u32, in: u32) !Table {
    return if (ops.ex != null) exl3Proj(gpa, ops, l, prefix, rows, in) else tableB(gpa, l, prefix, rows, in);
}

/// An unquantized fp16 projection (EXL3 keeps in_proj_a/in_proj_b as fp16 [rows][in]).
fn f16Proj(gpa: std.mem.Allocator, l: *ld.Loader, prefix: []const u8, rows: u32, in: u32) !Table {
    var nb: [200]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, "{s}.weight", .{prefix});
    try need(l, name, "F16", @as(u64, rows) * in * 2);
    _ = gpa;
    return .{ .format = .f16, .w = try l.load(name), .s = undefined, .b = undefined, .rows = rows, .in = in };
}

/// The small a/b gate projections: fp16 in EXL3, 4-bit affine in MLX.
fn gateProj(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, prefix: []const u8) !Table {
    return if (ops.ex != null) f16Proj(gpa, l, prefix, gdn_heads, hidden) else tableB(gpa, l, prefix, gdn_heads, hidden);
}

pub fn loadEmbed(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, vocab: u32) !Table {
    if (l.gguf) return qg.loadEmbed(l, vocab);
    var nb: [200]u8 = undefined;
    if (ops.ex == null) return table(gpa, l, try std.fmt.bufPrint(&nb, "{s}.embed_tokens", .{base(ops)}), vocab, hidden);
    const name = try std.fmt.bufPrint(&nb, "{s}.embed_tokens.weight", .{base(ops)});
    try need(l, name, "BF16", @as(u64, vocab) * hidden * 2);
    return .{ .format = .bf16, .w = try l.load(name), .s = undefined, .b = undefined, .rows = vocab, .in = hidden };
}

pub fn loadHead(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, vocab: u32) !Table {
    if (l.gguf) return qg.loadHead(l, vocab);
    return proj(gpa, ops, l, if (ops.ex != null) "lm_head" else "language_model.lm_head", vocab, hidden);
}

pub fn loadFinalNorm(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader) !Buf {
    if (l.gguf) return qg.loadFinalNorm(gpa, ops, l);
    var nb: [200]u8 = undefined;
    return normW(gpa, ops, l, try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{base(ops)}), hidden);
}

fn need(l: *ld.Loader, name: []const u8, dtype: []const u8, bytes: u64) !void {
    const inf = try l.info(name);
    if (!std.mem.eql(u8, inf.dtype, dtype) or inf.len != bytes) {
        std.log.err("{s}: {s} {d} bytes, expected {s} {d}", .{ name, inf.dtype, inf.len, dtype, bytes });
        return error.UnexpectedTensor;
    }
}

/// A 4-bit affine projection (words, scales, biases); the byte sizes pin bits = 4 and group size = 64 for [rows][in].
pub fn table(gpa: std.mem.Allocator, l: *ld.Loader, prefix: []const u8, rows: u32, in: u32) !Table {
    const n: u64 = @as(u64, rows) * in;
    var t: Table = .{ .w = undefined, .s = undefined, .b = undefined, .rows = rows, .in = in };
    inline for (.{ .{ "weight", "U32", n / 2 }, .{ "scales", "BF16", n / 32 }, .{ "biases", "BF16", n / 32 } }) |f| {
        const name = try std.fmt.allocPrint(gpa, "{s}.{s}", .{ prefix, f[0] });
        defer gpa.free(name);
        try need(l, name, f[1], f[2]);
        @field(t, if (f[0][0] == 'w') "w" else if (f[0][0] == 's') "s" else "b") = try l.load(name);
    }
    return t;
}

/// MLX projections are repacked block-interleaved at load (one summation order for all row counts); MLX4_BLOCK=0: rows.
fn blockMode() bool {
    const v = std.c.getenv("MLX4_BLOCK") orelse return true;
    return v[0] != '0';
}

/// `table`, repacked block-interleaved unless MLX4_BLOCK=0 and the shape allows (rows % 16, in % 256).
fn tableB(gpa: std.mem.Allocator, l: *ld.Loader, prefix: []const u8, rows: u32, in: u32) !Table {
    if (!blockMode() or rows % 16 != 0 or in % 256 != 0) return table(gpa, l, prefix, rows, in);
    var t: Table = .{ .w = undefined, .s = undefined, .b = undefined, .rows = rows, .in = in, .block = true };
    const n: u64 = @as(u64, rows) * in;
    var host: [3][]u8 = undefined;
    var out: [3][]u8 = undefined;
    const sizes = [3]u64{ n / 2, n / 32, n / 32 };
    inline for (.{ "weight", "scales", "biases" }, 0..) |f, i| {
        const name = try std.fmt.allocPrint(gpa, "{s}.{s}", .{ prefix, f });
        defer gpa.free(name);
        try need(l, name, if (i == 0) "U32" else "BF16", sizes[i]);
        host[i] = try gpa.alloc(u8, sizes[i]);
        out[i] = try gpa.alloc(u8, sizes[i]);
        const inf = try l.info(name);
        try ld.readExact(inf.fd, host[i], inf.off);
    }
    mxb.repack(in, rows, host[0], host[1], host[2], out[0], out[1], out[2]);
    for (0..3) |i| {
        const b = try l.empty(out[i].len);
        var off: usize = 0;
        while (off < out[i].len) {
            const c = @min(out[i].len - off, l.stage.len);
            @memcpy(l.stage[0..c], out[i][off..][0..c]);
            try l.r.upload(.{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = c }, l.stage[0..c]);
            try l.r.sync();
            off += c;
        }
        gpa.free(host[i]);
        gpa.free(out[i]);
        if (i == 0) t.w = b else if (i == 1) t.s = b else t.b = b;
    }
    return t;
}

fn plain(l: *ld.Loader, name: []const u8, count: u64) !Buf {
    try need(l, name, "BF16", count * 2);
    return l.load(name);
}

/// A small tensor widened to fp32 on the device (the checkpoint stores it bf16 or fp32).
fn wide(l: *ld.Loader, name: []const u8, count: u64) !Buf {
    const inf = try l.info(name);
    if (std.mem.eql(u8, inf.dtype, "F32") and inf.len == count * 4) return l.load(name);
    try need(l, name, "BF16", count * 2);
    return l.loadF32(name);
}

fn fmt(buf: []u8, comptime f: []const u8, a: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, f, a);
}

pub fn loadGdn(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, i: usize) !qb.GdnW {
    if (l.gguf) return qg.loadGdn(gpa, ops, l, i);
    var nb: [128]u8 = undefined;
    const p = try fmt(&nb, "{s}.layers.{d}", .{ base(ops), i });
    var pre: [128]u8 = undefined;
    const lp = try fmt(&pre, "{s}.linear_attn", .{p});
    var b1: [160]u8 = undefined;
    var b2: [160]u8 = undefined;
    return .{
        .norm = try normW(gpa, ops, l, try fmt(&b1, "{s}.input_layernorm.weight", .{p}), hidden),
        .qkv = try proj(gpa, ops, l, try fmt(&b1, "{s}.in_proj_qkv", .{lp}), gdn_qkv, hidden),
        .z = try proj(gpa, ops, l, try fmt(&b1, "{s}.in_proj_z", .{lp}), gdn_v, hidden),
        .b = try gateProj(gpa, ops, l, try fmt(&b1, "{s}.in_proj_b", .{lp})),
        .a = try gateProj(gpa, ops, l, try fmt(&b1, "{s}.in_proj_a", .{lp})),
        .out = try proj(gpa, ops, l, try fmt(&b1, "{s}.out_proj", .{lp}), hidden, gdn_v),
        .conv = try plain(l, try fmt(&b2, "{s}.conv1d.weight", .{lp}), @as(u64, gdn_qkv) * 4),
        .a_log = try wide(l, try fmt(&b1, "{s}.A_log", .{lp}), gdn_heads),
        .dt_bias = try wide(l, try fmt(&b2, "{s}.dt_bias", .{lp}), gdn_heads),
        .gnorm = try wide16(l, try fmt(&b1, "{s}.norm.weight", .{lp})),
        .cstate = try l.zeros(3 * gdn_qkv * 2),
        .sstate = try l.zeros(@as(usize, gdn_heads) * 128 * 128 * 4),
    };
}

/// The gated-norm weight is bf16 (128 values), used as-is by the kernel.
fn wide16(l: *ld.Loader, name: []const u8) !Buf {
    try need(l, name, "BF16", 128 * 2);
    return l.load(name);
}

pub fn loadAttn(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, i: usize, cap: u32) !qb.AttnW {
    if (l.gguf) return qg.loadAttn(gpa, ops, l, i, cap);
    var nb: [128]u8 = undefined;
    const p = try fmt(&nb, "{s}.layers.{d}", .{ base(ops), i });
    var b1: [160]u8 = undefined;
    var b2: [160]u8 = undefined;
    const kv = al.cacheBytes(cap);
    return .{
        .norm = try normW(gpa, ops, l, try fmt(&b1, "{s}.input_layernorm.weight", .{p}), hidden),
        .q = try proj(gpa, ops, l, try fmt(&b1, "{s}.self_attn.q_proj", .{p}), 2 * q_dim, hidden),
        .k = try proj(gpa, ops, l, try fmt(&b1, "{s}.self_attn.k_proj", .{p}), kv_dim, hidden),
        .v = try proj(gpa, ops, l, try fmt(&b1, "{s}.self_attn.v_proj", .{p}), kv_dim, hidden),
        .o = try proj(gpa, ops, l, try fmt(&b1, "{s}.self_attn.o_proj", .{p}), hidden, q_dim),
        .qn = try normW(gpa, ops, l, try fmt(&b1, "{s}.self_attn.q_norm.weight", .{p}), head_dim),
        .kn = try normW(gpa, ops, l, try fmt(&b2, "{s}.self_attn.k_norm.weight", .{p}), head_dim),
        .kc = try l.zeros(kv),
        .vc = try l.zeros(kv),
    };
}

pub fn loadMlp(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, i: usize) !qb.MlpW {
    if (l.gguf) return qg.loadMlp(gpa, ops, l, i);
    var nb: [128]u8 = undefined;
    const p = try fmt(&nb, "{s}.layers.{d}", .{ base(ops), i });
    var b1: [160]u8 = undefined;
    return .{
        .norm = try normW(gpa, ops, l, try fmt(&b1, "{s}.post_attention_layernorm.weight", .{p}), hidden),
        .gate = try proj(gpa, ops, l, try fmt(&b1, "{s}.mlp.gate_proj", .{p}), inter, hidden),
        .up = try proj(gpa, ops, l, try fmt(&b1, "{s}.mlp.up_proj", .{p}), inter, hidden),
        .down = try proj(gpa, ops, l, try fmt(&b1, "{s}.mlp.down_proj", .{p}), hidden, inter),
    };
}

