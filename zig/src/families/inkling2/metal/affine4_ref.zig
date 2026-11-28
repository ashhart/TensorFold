//! Named scalar exact-arithmetic baselines for Inkling 2 Mini MLX 4-bit affine g64 Metal kernels.
const std = @import("std");

pub const hidden: usize = 2304;
pub const words_hidden: usize = hidden / 8;
pub const groups_hidden: usize = hidden / 64;
pub const group_size: usize = 64;
pub const eps: f32 = 1e-6;
pub const logit_cap: f32 = 30.0;
pub const tanh_sat: f32 = 9.010913347;

pub fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Exact 5-step binary tree reduction matching `ink2_q4_warp_sum` (`simd_shuffle_down` 16, 8, 4, 2, 1).
pub fn warpSum(lanes: [32]f32) f32 {
    var v = lanes;
    var delta: usize = 16;
    while (delta > 0) : (delta >>= 1) {
        for (0..delta) |i| v[i] = v[i] + v[i + delta];
    }
    return v[0];
}

/// Pure IEEE-754 binary32 tanh matching `ink2_q4_tanh` (0 ULP across Apple M3/M5 and CPU).
pub fn exactTanh(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    const ax = @abs(x);
    if (ax >= tanh_sat) return std.math.copysign(@as(f32, 1.0), x);
    if (ax < 0.000244140625) return x;
    const u: f32 = x * 0.0625;
    const uu: f32 = u * u;
    var cf: f32 = 11.0;
    cf = 9.0 + (uu / cf);
    cf = 7.0 + (uu / cf);
    cf = 5.0 + (uu / cf);
    cf = 3.0 + (uu / cf);
    var t: f32 = u / (1.0 + (uu / cf));
    for (0..4) |_| {
        t = (2.0 * t) / (1.0 + t * t);
    }
    return t;
}

/// Exact softcap30 matching `ink2_q4_softcap30`: BF16 round of dot, then `30 * exactTanh(v / 30)`.
pub fn softcap30(dot: f32) f32 {
    const v = fromBf16(toBf16(dot));
    return logit_cap * exactTanh(v / logit_cap);
}

inline fn readU32(bytes: []const u8, idx: usize) u32 {
    return std.mem.readInt(u32, bytes[idx * 4 ..][0..4], .little);
}

inline fn readU16(bytes: []const u8, idx: usize) u16 {
    return std.mem.readInt(u16, bytes[idx * 2 ..][0..2], .little);
}

/// Named scalar baseline for `tf_ink2_embed_norm_q4`: exact dequant, 256-thread tree reduction, and RMSNorm.
pub fn embedNormBytes(
    ids: []const u32,
    w_bytes: []const u8,
    s_bytes: []const u8,
    b_bytes: []const u8,
    norm_bytes: []const u8,
    out_bf16: []u16,
) void {
    std.debug.assert(out_bf16.len == ids.len * hidden);
    for (ids, 0..) |id, row| {
        const w_base = @as(usize, id) * words_hidden;
        const g_base = @as(usize, id) * groups_hidden;
        var vals: [hidden]f32 = undefined;
        var red: [8]f32 = undefined;
        for (0..8) |warp| {
            var lane_parts: [32]f32 = undefined;
            for (0..32) |lane| {
                const t = warp * 32 + lane;
                var part: f32 = 0.0;
                for (0..9) |k| {
                    const d = t + k * 256;
                    const word = readU32(w_bytes, w_base + (d >> 3));
                    const q: f32 = @floatFromInt((word >> @intCast((d & 7) * 4)) & 0xf);
                    const g = d >> 6;
                    const sc = fromBf16(readU16(s_bytes, g_base + g));
                    const bi = fromBf16(readU16(b_bytes, g_base + g));
                    const v = fromBf16(toBf16(q * sc + bi));
                    vals[d] = v;
                    part = part + v * v;
                }
                lane_parts[lane] = part;
            }
            red[warp] = warpSum(lane_parts);
        }
        var total: f32 = 0.0;
        for (red) |r| total = total + r;
        const rms: f32 = @sqrt((total / @as(f32, @floatFromInt(hidden))) + eps);
        const inverse: f32 = 1.0 / rms;
        for (0..hidden) |d| {
            const nw = fromBf16(readU16(norm_bytes, d));
            out_bf16[row * hidden + d] = toBf16(nw * (vals[d] * inverse));
        }
    }
}

/// Named scalar baseline for `tf_ink2_dense_q4`: exact dequant, pair accumulation, and 32-lane tree sum.
pub fn denseBytes(
    x_bf16: []const u16,
    w_bytes: []const u8,
    s_bytes: []const u8,
    b_bytes: []const u8,
    out_bf16: []u16,
    rows: usize,
    n: usize,
    k: usize,
    ldy: usize,
) void {
    std.debug.assert(k > 0 and k % group_size == 0 and ldy >= n);
    const groups = k / group_size;
    const words_per_row = k / 8;
    for (0..rows) |r| {
        const x_row = x_bf16[r * k ..][0..k];
        for (0..n) |col| {
            const w_base = col * words_per_row;
            const g_base = col * groups;
            var lane_acc: [32]f32 = @splat(0.0);
            for (0..groups) |g| {
                const sc = fromBf16(readU16(s_bytes, g_base + g));
                const bi = fromBf16(readU16(b_bytes, g_base + g));
                for (0..32) |lane| {
                    const word = readU32(w_bytes, w_base + g * 8 + (lane >> 2));
                    const sh: u5 = @intCast((lane & 3) * 8);
                    const dw0: f32 = @as(f32, @floatFromInt((word >> sh) & 0xf)) * sc + bi;
                    const dw1: f32 = @as(f32, @floatFromInt((word >> (sh + 4)) & 0xf)) * sc + bi;
                    const k0 = g * 64 + lane * 2;
                    const p0: f32 = fromBf16(x_row[k0]) * dw0;
                    const p1: f32 = fromBf16(x_row[k0 + 1]) * dw1;
                    lane_acc[lane] = (lane_acc[lane] + p0) + p1;
                }
            }
            out_bf16[r * ldy + col] = toBf16(warpSum(lane_acc));
        }
    }
}

/// Named scalar baseline for `tf_ink2_logits_q4` over vocab columns `[v_start, v_end)`.
pub fn logitsRangeBytes(
    h_bf16: []const u16,
    w_bytes: []const u8,
    s_bytes: []const u8,
    b_bytes: []const u8,
    out_f32: []f32,
    rows: usize,
    vocab: usize,
    v_start: usize,
    v_end: usize,
) void {
    for (0..rows) |r| {
        const h_row = h_bf16[r * hidden ..][0..hidden];
        for (v_start..v_end) |v| {
            const w_base = v * words_hidden;
            const g_base = v * groups_hidden;
            var lane_acc: [32]f32 = @splat(0.0);
            for (0..groups_hidden) |g| {
                const sc = fromBf16(readU16(s_bytes, g_base + g));
                const bi = fromBf16(readU16(b_bytes, g_base + g));
                for (0..32) |lane| {
                    const word = readU32(w_bytes, w_base + g * 8 + (lane >> 2));
                    const sh: u5 = @intCast((lane & 3) * 8);
                    const lw0: f32 = @as(f32, @floatFromInt((word >> sh) & 0xf)) * sc + bi;
                    const lw1: f32 = @as(f32, @floatFromInt((word >> (sh + 4)) & 0xf)) * sc + bi;
                    const k0 = g * 64 + lane * 2;
                    const p0: f32 = fromBf16(h_row[k0]) * lw0;
                    const p1: f32 = fromBf16(h_row[k0 + 1]) * lw1;
                    lane_acc[lane] = (lane_acc[lane] + p0) + p1;
                }
            }
            out_f32[r * vocab + v] = softcap30(warpSum(lane_acc));
        }
    }
}

/// Named scalar baseline for `tf_ink2_logits_q4`: exact dequant, pair acc, tree sum, and softcap30.
pub fn logitsBytes(
    h_bf16: []const u16,
    w_bytes: []const u8,
    s_bytes: []const u8,
    b_bytes: []const u8,
    out_f32: []f32,
    rows: usize,
    vocab: usize,
) void {
    logitsRangeBytes(h_bf16, w_bytes, s_bytes, b_bytes, out_f32, rows, vocab, 0, vocab);
}

test "exactTanh matches binary64 tanh within 4 ULP across all 65536 BF16 softcap30 inputs" {
    const t = std.testing;
    var max_tanh_ulp: u32 = 0;
    var max_cap_ulp: u32 = 0;
    for (0..65536) |raw| {
        const bf: u16 = @intCast(raw);
        const v = fromBf16(bf);
        const got = softcap30(v);
        if (std.math.isNan(v)) {
            try t.expect(std.math.isNan(got));
            continue;
        }
        if (std.math.isInf(v)) {
            try t.expectEqual(@as(u32, @bitCast(std.math.copysign(logit_cap, v))), @as(u32, @bitCast(got)));
            continue;
        }
        if (v == 0.0) {
            try t.expectEqual(@as(u32, @bitCast(v)), @as(u32, @bitCast(got)));
            continue;
        }
        const x: f32 = v / logit_cap;
        const got_t = exactTanh(x);
        const want_t: f32 = @floatCast(std.math.tanh(@as(f64, x)));
        const ta: u32 = @bitCast(got_t);
        const tb: u32 = @bitCast(want_t);
        try t.expectEqual(ta >> 31, tb >> 31);
        max_tanh_ulp = @max(max_tanh_ulp, if (ta >= tb) ta - tb else tb - ta);
        if (@abs(v) >= 1e-36) {
            const want: f32 = @floatCast(30.0 * std.math.tanh(@as(f64, v) / 30.0));
            const a_bits: u32 = @bitCast(got);
            const b_bits: u32 = @bitCast(want);
            try t.expectEqual(a_bits >> 31, b_bits >> 31);
            max_cap_ulp = @max(max_cap_ulp, if (a_bits >= b_bits) a_bits - b_bits else b_bits - a_bits);
        }
    }
    try t.expect(max_tanh_ulp <= 4);
    try t.expect(max_cap_ulp <= 5);
}
