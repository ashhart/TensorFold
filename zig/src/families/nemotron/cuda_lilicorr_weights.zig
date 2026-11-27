//! The LiLiCorr weights on the device: bf16 linears affine-4; nvfp4 by a proven dequant-then-requant recipe.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const engine = @import("cuda_engine.zig");
const nvfp4 = @import("cuda_lilicorr_nvfp4.zig");

const Affine4 = cuda.qlinear.Affine4;
const QLinear = cuda.qlinear.Weight;
const Affine4Kernels = cuda.qlinear.Affine4Kernels;
const DeviceBuffer = cuda.DeviceBuffer;
const Engine = engine.Engine;

pub const Layer = struct {
    input_norm: u64, // [hidden] bf16
    post_norm: u64, // [hidden] bf16
    q: QLinear, // [q_width, hidden] checkpoint-bf16, TRUE bf16 (16c: the affine-4 requant priced ~9%/projection)
    k: QLinear, // [kv_width, hidden]
    v: QLinear, // [kv_width, hidden]
    o: QLinear, // [hidden, q_width]
    q_norm: u64, // [head_dim] bf16
    k_norm: u64, // [head_dim] bf16
    sink: u64, // [q heads] bf16 — the draft attention's per-head sink bias
    gate: QLinear, // [intermediate, hidden] checkpoint-NATIVE nvfp4, direct (16c-2)
    up: QLinear, // [intermediate, hidden]
    down: QLinear, // [hidden, intermediate]
    attn_conv: QLinear, // kernel_projection [dynamic, hidden] checkpoint-bf16, TRUE bf16
    attn_base: u64, // base_kernel [2, 2, hidden] bf16
    mlp_conv: QLinear,
    mlp_base: u64,
};

pub const Weights = struct {
    fc: Affine4, // [hidden, taps * hidden] (nvfp4 in the checkpoint, requantized)
    hidden_norm: u64,
    final_norm: u64,
    layers: []Layer,
    arena: DeviceBuffer,
    arena_at: usize = 0,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        w.arena.free();
        gpa.free(w.layers);
    }

    pub fn alloc(w: *Weights, bytes: usize) !u64 {
        const at = aligned(w.arena_at);
        if (at + bytes > w.arena.len) return error.ArenaExhausted;
        w.arena_at = at + bytes;
        return w.arena.ptr + at;
    }
};

/// The drafter's widths the loader needs; taps = the config's tap count, dynamic = 1024 on the Bolt drafter.
pub const Shape = struct { hidden: usize, intermediate: usize, layers: usize, q_width: usize, kv_width: usize, head_dim: usize, dynamic: usize, taps: usize };

pub fn packedBytes(n: usize, k: usize) usize {
    const l = Affine4.layout(n, k);
    return aligned(l.words) + 2 * aligned(l.scales);
}

pub fn rawBytes(n: usize, k: usize) usize {
    return n * k / 2 + 2 * (n * k / 64 * 2);
}

/// 16c: a TRUE-bf16 lane weight's arena bytes (the padded-rows bf16 table).
pub fn bf16Bytes(n: usize, k: usize) usize {
    return aligned(cuda.modelopt.padded(n) * k * 2);
}

/// 16c-2: a checkpoint-native nvfp4 lane weight's arena bytes (padded codes + scale tiles).
pub fn nvfp4Bytes(n: usize, k: usize) usize {
    return aligned(cuda.modelopt.padded(n) * k / 2) + aligned(cuda.modelopt.padded(n) * k / 16);
}

/// The largest projection `load` stages: the staging buffer it allocates and frees after.
pub fn stageBytes(s: Shape) usize {
    var most: usize = 0;
    const shapes = [_][2]usize{
        .{ s.hidden, s.taps * s.hidden }, .{ s.q_width, s.hidden },
        .{ s.kv_width, s.hidden },        .{ s.hidden, s.q_width },
        .{ s.intermediate, s.hidden },    .{ s.hidden, s.intermediate },
        .{ s.dynamic, s.hidden },
    };
    for (shapes) |nk| {
        most = @max(most, rawBytes(nk[0], nk[1]));
        most = @max(most, nk[0] * nk[1] * 2); // a bf16 lane weight stages its raw rows (16c)
    }
    return most;
}

fn aligned(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}

/// Device bytes of everything `load` packs (norms and vectors are the only plain bf16).
pub fn total(s: Shape) usize {
    var t = packedBytes(s.hidden, s.taps * s.hidden) + 2 * aligned(s.hidden * 2);
    t += s.layers * (2 * aligned(s.hidden * 2) + bf16Bytes(s.q_width, s.hidden) +
        2 * bf16Bytes(s.kv_width, s.hidden) + bf16Bytes(s.hidden, s.q_width) +
        2 * aligned(s.head_dim * 2) + aligned(s.q_width / s.head_dim * 2) +
        2 * nvfp4Bytes(s.intermediate, s.hidden) + nvfp4Bytes(s.hidden, s.intermediate) +
        2 * bf16Bytes(s.dynamic, s.hidden) + 2 * aligned(4 * s.hidden * 2));
    return t;
}

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

// round to nearest even, as the device does
const toBf16 = core.bf16_round.bits;

/// Affine-4, group 64: scale bf16((max - min) / 15), bias bf16(min), nibbles round((w - bias) / scale), low first.
fn quantize(gpa: std.mem.Allocator, w: []align(1) const u16, n: usize, k: usize) !struct { words: []u32, scales: []u16, biases: []u16 } {
    const groups = k / 64;
    const words = try gpa.alloc(u32, n * k / 8);
    errdefer gpa.free(words);
    const scales = try gpa.alloc(u16, n * groups);
    errdefer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * groups);
    errdefer gpa.free(biases);
    for (0..n) |r| {
        for (0..groups) |g| {
            const row = w[r * k + g * 64 ..][0..64];
            var lo: f32 = std.math.inf(f32);
            var hi: f32 = -std.math.inf(f32);
            for (row) |h| {
                const x = bf16(h);
                lo = @min(lo, x);
                hi = @max(hi, x);
            }
            const scale = toBf16((hi - lo) / 15.0);
            const bias = toBf16(lo);
            scales[r * groups + g] = scale;
            biases[r * groups + g] = bias;
            const s = bf16(scale);
            const b = bf16(bias);
            const inv = if (s == 0) 0.0 else 1.0 / s;
            for (0..8) |j| {
                var word: u32 = 0;
                for (0..8) |i| {
                    const q = std.math.clamp(@round((bf16(row[j * 8 + i]) - b) * inv), 0.0, 15.0);
                    word |= @as(u32, @intFromFloat(q)) << @intCast(4 * i);
                }
                words[(r * k + g * 64) / 8 + j] = word;
            }
        }
    }
    return .{ .words = words, .scales = scales, .biases = biases };
}

/// 16c-2: a checkpoint-native nvfp4 projection as a lane weight directly (no requant round-trip, ~14% measured).
fn linearNvfp4Direct(e: *Engine, stage: *DeviceBuffer, w: *Weights, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) !QLinear {
    var buf: [160]u8 = undefined;
    const cw = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}));
    if (cw.dtype != .u8 or cw.rank != 2 or cw.dim(0) != n or cw.dim(1) != k / 2) return error.UnexpectedTensor;
    const sw = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{name}), .f8_e4m3, &.{ n, k / 16 });
    const t2 = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{name}));
    if (t2.dtype != .f32 or t2.bytes.len != 4) return error.UnexpectedTensor;
    const scale_2: f32 = @bitCast(std.mem.readInt(u32, t2.bytes[0..4], .little));
    const codes = n * k / 2;
    if (codes + sw.bytes.len > stage.len) return error.StageTooSmall;
    try stage.upload(0, cw.bytes);
    try stage.upload(codes, sw.bytes);
    const words = try w.alloc(cuda.modelopt.padded(n) * k / 2);
    const tiles = try w.alloc(cuda.modelopt.padded(n) * k / 16);
    const m = e.k.modelopt orelse return error.ModeloptNotLoaded;
    try m.pack.fp4Lane(e.stream, stage.ptr, stage.ptr + codes, words, tiles, n, k);
    try e.stream.synchronize();
    return .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, scale_2, n, k) };
}

/// The requant path: a checkpoint nvfp4 projection host-dequantized (the bit-exact recipe), then rounded to bf16.
fn nvfp4Rows(gpa: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) ![]u16 {
    var buf: [160]u8 = undefined;
    const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}));
    if (w.dtype != .u8 or w.rank != 2 or w.dim(0) != n or w.dim(1) != k / 2) return error.UnexpectedTensor;
    const scales = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{name}), .f8_e4m3, &.{ n, k / 16 });
    const t2 = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{name}));
    if (t2.dtype != .f32 or t2.bytes.len != 4) return error.UnexpectedTensor;
    const scale_2: f32 = @bitCast(std.mem.readInt(u32, t2.bytes[0..4], .little));
    const dst = try gpa.alloc(f32, n * k);
    defer gpa.free(dst);
    try nvfp4.dequantNvfp4(w.bytes, n, k, scales.bytes, scale_2, dst);
    const out = try gpa.alloc(u16, n * k);
    errdefer gpa.free(out);
    for (dst, 0..) |x, i| out[i] = toBf16(x);
    return out;
}

/// A bf16 (n, k) checkpoint tensor quantized and packed into `w`'s arena through `stage`.
fn linear(gpa: std.mem.Allocator, e: *Engine, a: *const Affine4Kernels, stage: *DeviceBuffer, w: *Weights, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) !Affine4 {
    const t = try ck.expect(name, .bf16, &.{ n, k });
    return packRows(gpa, e, a, stage, w, std.mem.bytesAsSlice(u16, t.bytes), n, k);
}

/// 16c: a checkpoint-bf16 projection as a TRUE-bf16 lane weight, NO affine-4 requant (priced ~9%/projection).
fn linearBf16(e: *Engine, stage: *DeviceBuffer, w: *Weights, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) !QLinear {
    const t = try ck.expect(name, .bf16, &.{ n, k });
    if (t.bytes.len > stage.len) return error.StageTooSmall;
    try stage.upload(0, t.bytes);
    const out = try w.alloc(bf16Bytes(n, k));
    const m = e.k.modelopt orelse return error.ModeloptNotLoaded;
    try m.pack.bf16Lane(e.stream, stage.ptr, out, n, k);
    try e.stream.synchronize();
    return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
}

/// The nvfp4 projection's requant: dequantize, then the same affine-4 packing.
fn linearNvfp4(gpa: std.mem.Allocator, e: *Engine, a: *const Affine4Kernels, stage: *DeviceBuffer, w: *Weights, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) !Affine4 {
    const rows = try nvfp4Rows(gpa, ck, name, n, k);
    defer gpa.free(rows);
    return packRows(gpa, e, a, stage, w, rows, n, k);
}

fn packRows(gpa: std.mem.Allocator, e: *Engine, a: *const Affine4Kernels, stage: *DeviceBuffer, w: *Weights, rows: []align(1) const u16, n: usize, k: usize) !Affine4 {
    if (rows.len != n * k) return error.UnexpectedTensor;
    const q = try quantize(gpa, rows, n, k);
    defer {
        gpa.free(q.words);
        gpa.free(q.scales);
        gpa.free(q.biases);
    }
    const parts = [_][]const u8{ std.mem.sliceAsBytes(q.words), std.mem.sliceAsBytes(q.scales), std.mem.sliceAsBytes(q.biases) };
    var raw: [3]u64 = undefined;
    var at: usize = 0;
    for (parts, &raw) |p, *r| {
        if (at + p.len > stage.len) return error.StageTooSmall;
        try stage.upload(at, p);
        r.* = stage.ptr + at;
        at += p.len;
    }
    const l = Affine4.layout(n, k);
    const out: Affine4 = .{ .w = try w.alloc(l.words), .s = try w.alloc(l.scales), .b = try w.alloc(l.scales), .n = n, .k = k, .npad = l.npad };
    try a.pack(e.stream, raw[0], raw[1], raw[2], out);
    try e.stream.synchronize();
    return out;
}

fn vector(w: *Weights, ck: *core.Checkpoint, name: []const u8, shape: []const usize) !u64 {
    const t = try ck.expect(name, .bf16, shape);
    const p = try w.alloc(t.bytes.len);
    try w.arena.upload(p - w.arena.ptr, t.bytes);
    return p;
}

/// The drafter's weights: a = the driver's affine-4 pack/decode instance (from the family's loaded modules).
pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *Engine, a: *const Affine4Kernels, dir: []const u8, s: Shape) !Weights {
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    var w: Weights = .{ .fc = undefined, .hidden_norm = 0, .final_norm = 0, .layers = &.{}, .arena = try DeviceBuffer.alloc(e.ctx.d, total(s)) };
    errdefer w.arena.free();
    try w.arena.fill8(0, null);
    var stage = try DeviceBuffer.alloc(e.ctx.d, stageBytes(s));
    defer stage.free();
    w.layers = try gpa.alloc(Layer, s.layers);
    errdefer gpa.free(w.layers);
    w.fc = try linearNvfp4(gpa, e, a, &stage, &w, &ck, "fc", s.hidden, s.taps * s.hidden);
    w.hidden_norm = try vector(&w, &ck, "hidden_norm.weight", &.{s.hidden});
    w.final_norm = try vector(&w, &ck, "norm.weight", &.{s.hidden});
    for (w.layers, 0..) |*l, i| {
        var buf: [96]u8 = undefined;
        const Nm = struct {
            buf: *[96]u8,
            i: usize,
            fn of(self: @This(), suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(self.buf, "layers.{d}.{s}", .{ self.i, suffix });
            }
        };
        const nm: Nm = .{ .buf = &buf, .i = i };
        l.* = .{
            .input_norm = try vector(&w, &ck, try nm.of("input_layernorm.weight"), &.{s.hidden}),
            .post_norm = try vector(&w, &ck, try nm.of("post_attention_layernorm.weight"), &.{s.hidden}),
            .q = try linearBf16(e, &stage, &w, &ck, try nm.of("self_attn.q_proj.weight"), s.q_width, s.hidden),
            .k = try linearBf16(e, &stage, &w, &ck, try nm.of("self_attn.k_proj.weight"), s.kv_width, s.hidden),
            .v = try linearBf16(e, &stage, &w, &ck, try nm.of("self_attn.v_proj.weight"), s.kv_width, s.hidden),
            .o = try linearBf16(e, &stage, &w, &ck, try nm.of("self_attn.o_proj.weight"), s.hidden, s.q_width),
            .q_norm = try vector(&w, &ck, try nm.of("self_attn.q_norm.weight"), &.{s.head_dim}),
            .k_norm = try vector(&w, &ck, try nm.of("self_attn.k_norm.weight"), &.{s.head_dim}),
            .sink = try vector(&w, &ck, try nm.of("self_attn.attention_sink_bias"), &.{s.q_width / s.head_dim}),
            .gate = try linearNvfp4Direct(e, &stage, &w, &ck, try nm.of("mlp.gate_proj"), s.intermediate, s.hidden),
            .up = try linearNvfp4Direct(e, &stage, &w, &ck, try nm.of("mlp.up_proj"), s.intermediate, s.hidden),
            .down = try linearNvfp4Direct(e, &stage, &w, &ck, try nm.of("mlp.down_proj"), s.hidden, s.intermediate),
            .attn_conv = try linearBf16(e, &stage, &w, &ck, try nm.of("attention_conv.kernel_projection.weight"), s.dynamic, s.hidden),
            .attn_base = try vector(&w, &ck, try nm.of("attention_conv.base_kernel"), &.{ 2, 2, s.hidden }),
            .mlp_conv = try linearBf16(e, &stage, &w, &ck, try nm.of("mlp_conv.kernel_projection.weight"), s.dynamic, s.hidden),
            .mlp_base = try vector(&w, &ck, try nm.of("mlp_conv.base_kernel"), &.{ 2, 2, s.hidden }),
        };
    }
    return w;
}

test "the byte accounting covers the loader's own staging and packing" {
    const bolt: Shape = .{ .hidden = 4096, .intermediate = 12288, .layers = 5, .q_width = 4096, .kv_width = 256, .head_dim = 128, .dynamic = 1024, .taps = 6 };
    try std.testing.expect(stageBytes(bolt) <= total(bolt));
    try std.testing.expect(total(bolt) > 100 << 20); // the body's real size order (5 layers of 4096-wide affine-4)
    // a packed projection's bytes match the raw staging it came from (both aligned here)
    try std.testing.expectEqual(packedBytes(4096, 4096), rawBytes(4096, 4096));
}

test "quantize maps a flat group to scale 0, bias the value, zero nibbles" {
    const w: [64]u16 = @splat(0x3f80);
    const q = try quantize(std.testing.allocator, &w, 1, 64);
    defer {
        std.testing.allocator.free(q.words);
        std.testing.allocator.free(q.scales);
        std.testing.allocator.free(q.biases);
    }
    try std.testing.expectEqual(@as(u16, 0), q.scales[0]);
    try std.testing.expectEqual(@as(u16, 0x3f80), q.biases[0]);
    try std.testing.expectEqual(@as(u32, 0), q.words[0]);
}
