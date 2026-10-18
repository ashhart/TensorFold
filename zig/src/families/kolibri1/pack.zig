//! One layer in the device layouts, packed on the host: lane FP8 tiles, stacked experts (shared last), fp32 norms.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const names = @import("names.zig");

const Tensor = core.checkpoint.Tensor;
const fp8 = cuda.fp8;
const fx8 = cuda.experts.Fp8Experts;

/// A block-FP8 projection in the lane's layout (cuda/fp8.zig): codes [npad/64][k/64][8][32][2][8], fp32 scale tiles.
pub const Linear = struct { w8: []u8, bs: []f32, n: usize, k: usize };

pub const Layer = struct {
    input_norm: []f32,
    qkv: Linear, // q, k and v rows stacked, as attention reads them in one call
    q_norm: []f32,
    k_norm: []f32,
    o: Linear,
    post_attn_norm: []f32,
    pre_moe_norm: []f32,
    router: []u8, // [E, D] bf16 as stored
    bias: []f32,
    up: []u32, // experts.Layer.up (fp8g): [E][NI/32][D/32][2][32][2][4]
    down: []u32, // [E][D/32][NI/32][1][32][2][4]
    up_scale: []f32, // [E][2][NI/128][D/128] (gate, up)
    down_scale: []f32, // [E][1][D/128][NI/128]

    pub fn deinit(l: *Layer, gpa: std.mem.Allocator) void {
        inline for (.{ "input_norm", "q_norm", "k_norm", "post_attn_norm", "pre_moe_norm", "bias", "up_scale", "down_scale" }) |f|
            gpa.free(@field(l, f));
        for ([_]Linear{ l.qkv, l.o }) |x| {
            gpa.free(x.w8);
            gpa.free(x.bs);
        }
        gpa.free(l.router);
        gpa.free(l.up);
        gpa.free(l.down);
        l.* = undefined;
    }
};

/// `.float()` of a bf16 tensor: exact.
pub fn widen(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try gpa.alloc(f32, t.bytes.len / 2);
    for (out, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16);
    return out;
}

/// fp32 values of a tensor whose bytes may sit at any offset of the mapped file.
fn floats(dst: []f32, t: Tensor) void {
    for (dst, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
}

/// Row-stacked block-FP8 parts as one projection: codes and scales concatenated by rows, then packed.
fn linear(gpa: std.mem.Allocator, parts: []const names.Fp8) !Linear {
    const k = parts[0].w.dim(1);
    var n: usize = 0;
    var blocks: usize = 0;
    for (parts) |p| {
        if (p.w.dim(1) != k or p.w.dim(0) % names.block != 0) return error.UnexpectedTensor;
        n += p.w.dim(0);
        blocks += p.s.numel();
    }
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    const inv = try gpa.alloc(f32, blocks);
    defer gpa.free(inv);
    var at: usize = 0;
    var sat: usize = 0;
    for (parts) |p| {
        @memcpy(codes[at..][0..p.w.bytes.len], p.w.bytes);
        at += p.w.bytes.len;
        floats(inv[sat..][0..p.s.numel()], p.s);
        sat += p.s.numel();
    }
    const npad = fp8.padded(n);
    const w8 = try gpa.alloc(u8, npad * k);
    errdefer gpa.free(w8);
    fp8.packCodes(w8, codes, n, k);
    const bs = try gpa.alloc(f32, npad * (k / fp8.group));
    fp8.packScales(bs, inv, n, k);
    return .{ .w8 = w8, .bs = bs, .n = n, .k = k };
}

/// Experts as cuda/experts.zig's fp8g layer takes them: gate and up interleaved, down alone, scales gate then up.
fn experts(gpa: std.mem.Allocator, l: *Layer, ex: []const names.Expert, c: Config) !void {
    const ni = c.moe_width;
    const d = c.hidden;
    const one = ni * d;
    const sb = (ni / names.block) * (d / names.block);
    l.up = try gpa.alloc(u32, ex.len * one / 2);
    l.down = try gpa.alloc(u32, ex.len * one / 4);
    l.up_scale = try gpa.alloc(f32, ex.len * 2 * sb);
    l.down_scale = try gpa.alloc(f32, ex.len * sb);
    for (ex, 0..) |e, i| {
        try fx8.packGateUp(gpa, l.up[i * one / 2 ..][0 .. one / 2], e.gate.w.bytes, e.up.w.bytes, ni, d);
        fx8.packOne(l.down[i * one / 4 ..][0 .. one / 4], e.down.w.bytes, d, ni);
        floats(l.up_scale[2 * i * sb ..][0..sb], e.gate.s);
        floats(l.up_scale[(2 * i + 1) * sb ..][0..sb], e.up.s);
        floats(l.down_scale[i * sb ..][0..sb], e.down.s);
    }
}

/// The packed layer for one mapped layer.
pub fn layer(gpa: std.mem.Allocator, m: names.Layer, c: Config) !Layer {
    var l: Layer = undefined;
    l.qkv = try linear(gpa, &.{ m.q, m.k, m.v });
    l.o = try linear(gpa, &.{m.o});
    l.input_norm = try widen(gpa, m.input_norm);
    l.q_norm = try widen(gpa, m.q_norm);
    l.k_norm = try widen(gpa, m.k_norm);
    l.post_attn_norm = try widen(gpa, m.post_attn_norm);
    l.pre_moe_norm = try widen(gpa, m.pre_moe_norm);
    l.bias = try widen(gpa, m.bias);
    l.router = try gpa.dupe(u8, m.router.bytes);
    try experts(gpa, &l, m.experts, c);
    return l;
}

const testing = std.testing;

fn tensor(dtype: core.safetensors.DType, shape: []const usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = undefined, .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

test "q, k and v stack by rows into one projection, codes and scales alike" {
    const a = testing.allocator;
    const k = 128;
    var codes: [3][128 * k]u8 = undefined;
    var inv: [3][1]f32 = .{ .{2}, .{3}, .{5} };
    for (&codes, 0..) |*c, p| for (c, 0..) |*b, i| {
        b.* = @truncate(i * 13 + p * 7 + 1);
    };
    var parts: [3]names.Fp8 = undefined;
    for (&parts, 0..) |*p, i| p.* = .{ .w = tensor(.f8_e4m3, &.{ 128, k }, &codes[i]), .s = tensor(.f32, &.{ 1, 1 }, std.mem.asBytes(&inv[i])) };
    const l = try linear(a, &parts);
    defer a.free(l.w8);
    defer a.free(l.bs);
    try testing.expectEqual(@as(usize, 384), l.n);
    const all = try a.alloc(u8, 3 * 128 * k);
    defer a.free(all);
    for (codes, 0..) |c, i| @memcpy(all[i * 128 * k ..][0 .. 128 * k], &c);
    const want = try a.alloc(u8, 384 * k);
    defer a.free(want);
    fp8.packCodes(want, all, 384, k);
    try testing.expectEqualSlices(u8, want, l.w8);
    var ws: [384 * 2]f32 = undefined;
    fp8.packScales(&ws, &.{ 2, 3, 5 }, 384, k);
    try testing.expectEqualSlices(f32, &ws, l.bs);
}

test "bf16 widens to fp32 exactly, and other dtypes are refused" {
    const bits = [_]u16{ 0x3f80, 0xc000, 0x0001 };
    const v = try widen(testing.allocator, tensor(.bf16, &.{3}, std.mem.sliceAsBytes(&bits)));
    defer testing.allocator.free(v);
    try testing.expectEqual(@as(f32, 1), v[0]);
    try testing.expectEqual(@as(f32, -2), v[1]);
    try testing.expectEqual(@as(u32, 0x00010000), @as(u32, @bitCast(v[2])));
    try testing.expectError(error.UnexpectedTensor, widen(testing.allocator, tensor(.f32, &.{1}, std.mem.sliceAsBytes(&bits)[0..4])));
}
