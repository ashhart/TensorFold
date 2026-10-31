//! The host's part of an Idol DINOv3 encode: the RoPE tables and the normalized patches, rounded as torch rounds them.
const std = @import("std");
const cfg = @import("dino_config.zig");

/// fp32 to bf16 bits, ties to even, as torch's .to(bfloat16) rounds (NaN stays NaN).
pub fn bits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// The fp32 value of bf16 bits.
pub fn value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// ImageEncoder's ImageNet mean and std, fp32 as torch.tensor makes them.
const mean = [3]f32{ 0.485, 0.456, 0.406 };
const std_ = [3]f32{ 0.229, 0.224, 0.225 };

/// transformers' inv_freq: 1 / 100 ** arange(0, 1, 4 / 64) in fp32 (generated, not the checkpoint's bf16 periods).
pub fn invFreq() [cfg.rope_freqs]f32 {
    var out: [cfg.rope_freqs]f32 = undefined;
    for (&out, 0..) |*o, i| {
        const e = @as(f64, @floatFromInt(i)) * 4.0 / cfg.head_dim;
        const p: f32 = @floatCast(std.math.pow(f64, cfg.rope_theta, e));
        o.* = 1.0 / p;
    }
    return out;
}

/// cos and sin [4096, 64] bf16 bits for the patches: angles 2 pi y f, 2 pi x f (fp32, as torch's ops run), tiled twice.
pub fn rope(cos: []u16, sin: []u16) void {
    std.debug.assert(cos.len == cfg.patches * cfg.head_dim and sin.len == cos.len);
    const inv = invFreq();
    const two_pi: f32 = @floatCast(2.0 * std.math.pi);
    for (0..cfg.grid) |h| for (0..cfg.grid) |w| {
        const p = h * cfg.grid + w;
        // patch centers in [-1, 1]: 2 ((i + 0.5) / 64) - 1, exact in fp32
        const at = [2]f32{ 2.0 * ((@as(f32, @floatFromInt(h)) + 0.5) / cfg.grid) - 1.0, 2.0 * ((@as(f32, @floatFromInt(w)) + 0.5) / cfg.grid) - 1.0 };
        for (0..cfg.head_dim) |j| {
            const axis = (j % 32) / cfg.rope_freqs;
            const a: f32 = (two_pi * at[axis]) * inv[j % cfg.rope_freqs];
            const x: f64 = a;
            cos[p * cfg.head_dim + j] = bits(@floatCast(@cos(x)));
            sin[p * cfg.head_dim + j] = bits(@floatCast(@sin(x)));
        }
    };
}

/// The patch embedding's input [4096, 768] bf16 bits from pixels [3, 1024, 1024] in [0, 1]: (x - mean) / std in fp32.
pub fn patchRows(pixels: []const f32, out: []u16) void {
    std.debug.assert(pixels.len == 3 * cfg.image * cfg.image and out.len == cfg.patches * cfg.patch_k);
    const P = cfg.patch;
    for (0..cfg.grid) |ph| for (0..cfg.grid) |pw| {
        const row = out[(ph * cfg.grid + pw) * cfg.patch_k ..][0..cfg.patch_k];
        for (0..3) |c| for (0..P) |kh| {
            const src = pixels[(c * cfg.image + ph * P + kh) * cfg.image + pw * P ..][0..P];
            for (src, 0..) |v, kw| row[(c * P + kh) * P + kw] = bits((v - mean[c]) / std_[c]);
        };
    };
}

test "the RoPE tables: the first patch's angles, the axes' halves and the tiling" {
    const a = std.testing.allocator;
    const cos = try a.alloc(u16, cfg.patches * cfg.head_dim);
    defer a.free(cos);
    const sin = try a.alloc(u16, cfg.patches * cfg.head_dim);
    defer a.free(sin);
    rope(cos, sin);
    const inv = invFreq();
    try std.testing.expectEqual(@as(f32, 1), inv[0]);
    try std.testing.expectApproxEqRel(@as(f32, 1.0 / 74.989421), inv[15], 1e-6);
    // patch (0, 0) sits at (-63/64, -63/64): its y and x halves agree, and the second 32 repeat the first
    const y0: f32 = @floatCast(@cos(@as(f64, (@as(f32, @floatCast(2.0 * std.math.pi)) * (-63.0 / 64.0)) * inv[0])));
    try std.testing.expectEqual(bits(y0), cos[0]);
    for (0..32) |j| try std.testing.expectEqual(cos[j], cos[32 + j]);
    for (0..16) |j| try std.testing.expectEqual(sin[j], sin[16 + j]);
    // patch (0, 1): x moves, y does not
    try std.testing.expectEqual(cos[0], cos[cfg.head_dim]);
    try std.testing.expect(cos[16] != cos[cfg.head_dim + 16]);
}

test "bf16 rounding is to nearest, ties to even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bits(1.0 + 0x1p-8));
    try std.testing.expectEqual(@as(u16, 0x3f82), bits(1.0 + 3 * 0x1p-8));
    try std.testing.expectEqual(@as(f32, -2.5), value(bits(-2.5)));
}

test "a patch row takes its 16 x 16 pixels channel by channel, normalized" {
    const a = std.testing.allocator;
    const px = try a.alloc(f32, 3 * cfg.image * cfg.image);
    defer a.free(px);
    for (px, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 251)) / 255.0;
    const out = try a.alloc(u16, cfg.patches * cfg.patch_k);
    defer a.free(out);
    patchRows(px, out);
    // patch (2, 5), channel 1, row 3, column 7
    const src = px[(1 * cfg.image + 2 * 16 + 3) * cfg.image + 5 * 16 + 7];
    try std.testing.expectEqual(bits((src - 0.456) / 0.224), out[(2 * cfg.grid + 5) * cfg.patch_k + (1 * 16 + 3) * 16 + 7]);
}
