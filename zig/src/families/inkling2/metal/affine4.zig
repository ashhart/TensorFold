//! Inkling 2 Mini MLX 4-bit affine group-64 weights, validation, host conversions, and Metal dispatches.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const mlx_affine = @import("core").quant.mlx_affine;

pub const Ref = @import("inkling2_metal_ref").Ref;
pub const ref = @import("affine4_ref.zig");

pub const hidden: u32 = 2304;
pub const group_size: u32 = 64;
pub const bits: u8 = 4;
pub const nvfp4_global_denom: f32 = 2688.0; // 6.0 (E2M1 max) * 448.0 (E4M3 max)

pub const Affine4Tensor = struct {
    weight: ck.Tensor,
    scales: ck.Tensor,
    biases: ck.Tensor,
    rows: u32,
    cols: u32,
};

pub const DenseWeight = union(enum) {
    bf16: ck.Tensor,
    affine4: Affine4Tensor,
};

pub const EmbedArgs = extern struct { rows: u32 };
pub const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
pub const LogitsArgs = extern struct { rows: u32, vocab: u32 };

fn TensorOf(comptime C: type) type {
    const T = if (@typeInfo(C) == .pointer) @typeInfo(C).pointer.child else C;
    return @typeInfo(@typeInfo(@TypeOf(T.get)).@"fn".return_type.?).error_union.payload;
}

/// Validate a 4-bit affine group-64 tensor triple via `core/quant/mlx_affine.zig`.
pub fn readTriple(c: anytype, module: []const u8, rows: u32, cols: u32) !mlx_affine.Triple(TensorOf(@TypeOf(c))) {
    var buf: [320]u8 = undefined;
    const w = try c.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
    const s = try c.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{module}));
    const z = try c.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{module}));
    return mlx_affine.check(w, s, z, .{ .n = rows, .k = cols, .bits = bits, .group = group_size });
}

/// Validate a BF16 dense tensor while refusing any stray `.scales` or `.biases` affine companion.
pub fn readBf16NoCompanions(c: anytype, module: []const u8, rows: u32, cols: u32) !TensorOf(@TypeOf(c)) {
    var buf: [320]u8 = undefined;
    if (c.has(try std.fmt.bufPrint(&buf, "{s}.scales", .{module})) or
        c.has(try std.fmt.bufPrint(&buf, "{s}.biases", .{module})))
        return error.UnexpectedAffineCompanion;
    const w = try c.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
    mlx_affine.require(w, .bf16, &.{ rows, cols }) catch return error.UnexpectedInkling2Tensor;
    return w;
}

pub fn loadAffine4(c: *const ck.Checkpoint, comptime fmt: []const u8, args: anytype, rows: u32, cols: u32) !Affine4Tensor {
    var buf: [256]u8 = undefined;
    const module = try std.fmt.bufPrint(&buf, fmt, args);
    const t = readTriple(c, module, rows, cols) catch |err| {
        std.log.err("inkling2: {s} is not a valid MLX 4-bit affine g64 [{d}, {d}] projection ({s})", .{ module, rows, cols, @errorName(err) });
        return err;
    };
    return .{ .weight = t.weight, .scales = t.scales, .biases = t.biases, .rows = rows, .cols = cols };
}

pub fn loadBf16Dense(c: *const ck.Checkpoint, comptime fmt: []const u8, args: anytype, rows: u32, cols: u32) !DenseWeight {
    var buf: [256]u8 = undefined;
    const module = try std.fmt.bufPrint(&buf, fmt, args);
    const w = readBf16NoCompanions(c, module, rows, cols) catch |err| {
        if (err == error.UnexpectedAffineCompanion)
            std.log.err("inkling2: unexpected affine companion (.scales/.biases) on BF16 module {s}", .{module});
        return err;
    };
    return .{ .bf16 = w };
}

pub fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Alignment-safe conversion of a little-endian FP32 tensor slice to BF16 (lossless for BF16-origin MLX casts).
pub fn convertF32BytesToBf16(src_bytes: []const u8, dst: []u16) void {
    std.debug.assert(src_bytes.len == dst.len * 4);
    for (dst, 0..) |*out, i| {
        const bits_u32 = std.mem.readInt(u32, src_bytes[i * 4 ..][0..4], .little);
        out.* = toBf16(@bitCast(bits_u32));
    }
}

/// Exact MLX 4-bit affine g64 dequantize to BF16: `bf16(float(q) * float(scale) + float(bias))`.
pub fn dequantAffine4ToBf16(w_bytes: []const u8, s_bytes: []const u8, b_bytes: []const u8, rows: usize, cols: usize, dst: []u16) void {
    std.debug.assert(cols % group_size == 0);
    const words_per_row = cols / 8;
    const groups_per_row = cols / group_size;
    std.debug.assert(w_bytes.len == rows * words_per_row * 4);
    std.debug.assert(s_bytes.len == rows * groups_per_row * 2);
    std.debug.assert(b_bytes.len == rows * groups_per_row * 2);
    std.debug.assert(dst.len == rows * cols);
    for (0..rows) |r| {
        for (0..groups_per_row) |g| {
            const gi = (r * groups_per_row + g) * 2;
            const sc = fromBf16(std.mem.readInt(u16, s_bytes[gi..][0..2], .little));
            const bi = fromBf16(std.mem.readInt(u16, b_bytes[gi..][0..2], .little));
            for (0..8) |wi| {
                const off = (r * words_per_row + g * 8 + wi) * 4;
                const word = std.mem.readInt(u32, w_bytes[off..][0..4], .little);
                const col0 = g * group_size + wi * 8;
                inline for (0..8) |nib| {
                    const q: f32 = @floatFromInt((word >> @intCast(nib * 4)) & 0xf);
                    dst[r * cols + col0 + nib] = toBf16(q * sc + bi);
                }
            }
        }
    }
}

/// Split MLX `message_input.query` [12, 4, 2304] into `cross_queries` (q, mlp) and `cross_kv_queries` (k, v).
pub fn splitMessageQueries(src_bytes: []const u8, cross_q: []f32, cross_kv: []f32) void {
    const h: usize = hidden;
    std.debug.assert(src_bytes.len == 12 * 4 * h * 4);
    std.debug.assert(cross_q.len == 24 * h and cross_kv.len == 24 * h);
    for (0..12) |core| {
        const readRow = struct {
            fn copy(bytes: []const u8, core_i: usize, role: usize, dst: []f32) void {
                const base = (core_i * 4 + role) * hidden * 4;
                for (dst, 0..) |*v, d| {
                    const u = std.mem.readInt(u32, bytes[base + d * 4 ..][0..4], .little);
                    v.* = @bitCast(u);
                }
            }
        }.copy;
        readRow(src_bytes, core, 0, cross_q[(2 * core) * h ..][0..h]);
        readRow(src_bytes, core, 3, cross_q[(2 * core + 1) * h ..][0..h]);
        readRow(src_bytes, core, 1, cross_kv[(2 * core) * h ..][0..h]);
        readRow(src_bytes, core, 2, cross_kv[(2 * core + 1) * h ..][0..h]);
    }
}

pub const Pipelines = struct {
    embed_norm: mtl.Pipeline,
    dense: mtl.Pipeline,
    logits: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, src.affine4);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const opts: mtl.CompileOptions = .{ .language = mtl.CompileOptions.mlx().language, .math_mode = .safe, .math_functions = .precise };
        const lib = try mtl.Library.fromSource(device, text, opts);
        defer lib.deinit();
        const embed_norm = try mtl.Pipeline.init(device, lib, "tf_ink2_embed_norm_q4", false);
        errdefer embed_norm.deinit();
        if (embed_norm.maxThreads() < 256) return error.Inkling2Threads;
        const dense_pipe = try mtl.Pipeline.init(device, lib, "tf_ink2_dense_q4", false);
        errdefer dense_pipe.deinit();
        if (dense_pipe.maxThreads() < 256) return error.Inkling2Threads;
        const logits_pipe = try mtl.Pipeline.init(device, lib, "tf_ink2_logits_q4", false);
        errdefer logits_pipe.deinit();
        if (logits_pipe.maxThreads() < 256) return error.Inkling2Threads;
        return .{ .embed_norm = embed_norm, .dense = dense_pipe, .logits = logits_pipe };
    }

    pub fn deinit(p: Pipelines) void {
        p.logits.deinit();
        p.dense.deinit();
        p.embed_norm.deinit();
    }
};

fn refOf(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

pub fn encodeEmbedNorm(comptime Encoder: type, p: anytype, e: Encoder, ids: Ref, weight: Ref, scales: Ref, biases: Ref, norm_w: Ref, out: Ref, rows: u32) void {
    if (rows == 0) return;
    e.setPipeline(p.embed_norm);
    for ([_]Ref{ ids, weight, scales, biases, norm_w, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(EmbedArgs{ .rows = rows }, 6);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn encodeDense(comptime Encoder: type, p: anytype, e: Encoder, x: Ref, weight: Ref, scales: Ref, biases: Ref, out: Ref, args: DenseArgs) void {
    if (args.rows == 0 or args.n == 0) return;
    std.debug.assert(args.k > 0 and args.k % group_size == 0 and args.ldy >= args.n);
    e.setPipeline(p.dense);
    for ([_]Ref{ x, weight, scales, biases, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(args, 5);
    e.dispatchGroups(mtl.Size.of((args.n + 7) / 8, (args.rows + 3) / 4, 1), mtl.Size.of(256, 1, 1));
}

pub fn encodeLogits(comptime Encoder: type, p: anytype, e: Encoder, h: Ref, weight: Ref, scales: Ref, biases: Ref, out: Ref, rows: u32, vocab: u32) void {
    if (rows == 0 or vocab == 0) return;
    e.setPipeline(p.logits);
    for ([_]Ref{ h, weight, scales, biases, out }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(LogitsArgs{ .rows = rows, .vocab = vocab }, 5);
    e.dispatchGroups(mtl.Size.of((vocab + 7) / 8, (rows + 3) / 4, 1), mtl.Size.of(256, 1, 1));
}

pub fn embedNormRefs(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, weight: Ref, scales: Ref, biases: Ref, norm_w: Ref, out: Ref, rows: u32) void {
    encodeEmbedNorm(mtl.ComputeEncoder, p, e, ids, weight, scales, biases, norm_w, out, rows);
}

pub fn denseRefs(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, weight: Ref, scales: Ref, biases: Ref, out: Ref, args: DenseArgs) void {
    encodeDense(mtl.ComputeEncoder, p, e, x, weight, scales, biases, out, args);
}

pub fn logitsRefs(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, weight: Ref, scales: Ref, biases: Ref, out: Ref, rows: u32, vocab: u32) void {
    encodeLogits(mtl.ComputeEncoder, p, e, h, weight, scales, biases, out, rows, vocab);
}

pub fn embedNorm(p: *const Pipelines, e: mtl.ComputeEncoder, ids: Ref, w: Affine4Tensor, norm_w: Ref, out: Ref, rows: u32) void {
    std.debug.assert(w.cols == hidden);
    embedNormRefs(p, e, ids, refOf(w.weight), refOf(w.scales), refOf(w.biases), norm_w, out, rows);
}

pub fn dense(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, w: Affine4Tensor, out: Ref, rows: u32) void {
    denseRefs(p, e, x, refOf(w.weight), refOf(w.scales), refOf(w.biases), out, .{ .rows = rows, .n = w.rows, .k = w.cols, .ldy = w.rows });
}

pub fn logits(p: *const Pipelines, e: mtl.ComputeEncoder, h: Ref, w: Affine4Tensor, out: Ref, rows: u32, vocab: u32) void {
    std.debug.assert(w.cols == hidden and vocab <= w.rows);
    logitsRefs(p, e, h, refOf(w.weight), refOf(w.scales), refOf(w.biases), out, rows, vocab);
}
