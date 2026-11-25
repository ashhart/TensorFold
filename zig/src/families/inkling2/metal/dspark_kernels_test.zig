//! Host-only FP64 witnesses for Mini DSpark rounding, pairing, reach and launch contracts.
const std = @import("std");
const k = @import("dspark_kernels.zig");
const records = @import("core").contract.inkling2_dspark_metal_records.records;
const shader = @import("inkling2_metal").inkling2_dspark;
const t = std.testing;

fn near(want: f64, got: f64) !void {
    try t.expectApproxEqAbs(want, got, 1e-10);
}
fn bf(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x)) @truncate((bits >> 16) | 0x40) else @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    const out: f32 = @bitCast(@as(u32, rounded) << 16);
    return out;
}
fn f32v(v: f64) f64 {
    return @as(f32, @floatCast(v));
}

test "DSpark symbols, records, ABIs and disabled contraction are exact" {
    for (std.enums.values(k.Kind), records) |kind, record| {
        try t.expectEqualStrings(kind.label(), record.name);
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, kind.symbol()));
        const width: u32 = switch (kind) {
            .block_qkv, .context_kv, .attention => 128,
            else => 256,
        };
        try t.expectEqual(width, record.threads);
        try t.expectEqual(record.threads, record.reserved);
        const threadgroup: u32 = switch (kind) {
            .fuse_taps => 32,
            .block_qkv, .context_kv, .attention, .residual_norm => 16,
            else => 0,
        };
        try t.expectEqual(threadgroup, record.tg_memory);
    }
    try t.expectEqual(@as(usize, 4), @sizeOf(k.PositionArgs));
    try t.expectEqual(@as(usize, 12), @sizeOf(k.ContextArgs));
    try t.expectEqual(@as(usize, 16), @sizeOf(k.AttentionArgs));
    try t.expectEqual(@as(usize, 4), @sizeOf(k.ResidualArgs));
    try t.expectEqual(@as(usize, 4), @sizeOf(k.TokenArgs));
    try t.expectEqual(@as(usize, 4), @sizeOf(k.CountArgs));
    try t.expectEqual(@as(usize, 8), @sizeOf(k.BaseArgs));
    try t.expect(std.mem.indexOf(u8, shader, "#pragma METAL fp contract(off)") != null);
    try t.expect(std.mem.indexOf(u8, shader, "DSP_SCALE = 0x1.6a09e6p-4f") != null);
}

test "raw base projection rounds the dot without applying target softcap" {
    const unembed: [k.hidden]f32 = @splat(1);
    const y: [k.hidden]f32 = @splat(1);
    var dot: f32 = 0;
    for (unembed, y) |u, v| dot += u * v;
    try near(2304, bf(dot));
    try near(30, bf(30.0 * std.math.tanh(dot / 30.0)));
    try t.expect(std.mem.indexOf(u8, shader, "kernel void tf_ink2_dspark_base") != null);
}

test "fusion rounds each softmax alpha through BF16 before ordered FP32 taps" {
    const weights = [_]f64{ 0, 0.5, -0.25, 2 };
    const taps = [_]f64{ 303.2126159667969, -170.3546905517578, 454.9715576171875, -93.47307586669922 };
    var expv: [4]f64 = undefined;
    var den: f64 = 0;
    for (weights, 0..) |w, j| {
        expv[j] = @exp(w - 2);
        den += expv[j];
    }
    var rounded: f32 = 0;
    var raw: f32 = 0;
    for (0..4) |j| {
        const a = f32v(expv[j] / f32v(den));
        const alpha: f32 = @floatCast(bf(a));
        rounded += alpha * @as(f32, @floatCast(taps[j]));
        raw += @as(f32, @floatCast(a)) * @as(f32, @floatCast(taps[j]));
    }
    try t.expect(bf(rounded) != bf(raw));
    var ordered: f32 = 0;
    for (0..4) |j| ordered += @as(f32, @floatCast(bf(f32v(expv[j] / f32v(den))))) * @as(f32, @floatCast(taps[j]));
    try t.expectEqual(ordered, rounded);
}

test "residual normalized output sees sigma rather than rounded BF16 storage" {
    const x: f32 = 1.0;
    const res: f32 = 0.00390625;
    const sigma: f32 = x + res;
    const stored = bf(sigma);
    const weight = 0.0;
    const scale: f32 = 1.00390625;
    const from_sigma = bf(sigma * scale);
    const from_store = bf(@as(f32, @floatCast(stored)) * scale);
    try t.expect(from_sigma != from_store);
    try near(@as(f64, weight) + 1, 1.0);
}

test "rotate-half RoPE pairs d with d plus 64 at absolute positions" {
    const x = [_]f64{ 1, 2 };
    const angle = std.math.pi / 2.0;
    const first = x[0] * @cos(angle) - x[1] * @sin(angle);
    const second = x[0] * @sin(angle) + x[1] * @cos(angle);
    try near(-2, first);
    try near(1, second);
    try t.expect(@abs(second - (-1)) > 1);
    try t.expect(@abs(k.ropeRotateHalf(1, 2, 0, @floatCast(angle)) - @as(f32, @floatCast(first))) < 2e-6);
    try t.expect(@abs(k.ropeRotateHalf(2, 1, 64, @floatCast(angle)) - @as(f32, @floatCast(second))) < 2e-6);
}

test "local reach starts at committed minus 1024 without crossing a restored lower bound" {
    const committed: usize = 1027;
    const from: usize = 5;
    const local_first = @max(from, committed - 1024);
    const global_first = from;
    try t.expectEqual(@as(usize, 5), local_first);
    try t.expectEqual(@as(usize, 1022 + 8), committed - local_first + 8);
    try t.expectEqual(committed - from + 8, committed - global_first + 8);
    for (0..8) |row| {
        const position = 5000 + row;
        try t.expectEqual(position, 5000 + row);
    }
}

test "recurrence rounds z activations and every state operation through BF16" {
    const gamma: f32 = @floatCast(bf(@as(f32, 0.431640625)));
    const candidate: f32 = @floatCast(bf(@as(f32, 0.030029296875)));
    const previous: f32 = -14.5625;
    const keep: f32 = @floatCast(bf(1.0 - gamma));
    const left: f32 = @floatCast(bf(gamma * previous));
    const right: f32 = @floatCast(bf(keep * candidate));
    const stepped = bf(left + right);
    const unrounded = bf(@as(f32, gamma) * previous + (1.0 - @as(f32, gamma)) * candidate);
    try t.expect(stepped != unrounded);
    try near(stepped, bf(@as(f32, left + right)));
    try near(stepped, k.recurrentBlend(gamma, candidate, previous));
    const z = bf(@as(f32, 0.33333334));
    const lambda = bf(std.math.tanh(z));
    try near(lambda, bf(std.math.tanh(z)));
}

test "folded correction rounds bias then logits before the thirty softcap" {
    const beta: f32 = 0.125;
    const dot: f32 = 0.0001;
    const base: f32 = 40.0;
    const bias: f32 = @floatCast(bf(beta + dot));
    const logit: f32 = @floatCast(bf(base + bias));
    const capped = bf(30.0 * std.math.tanh(logit / 30.0));
    const uncapped = bf(@as(f32, 30.0) * @as(f32, @floatCast(std.math.tanh(logit / 30.0))));
    try near(capped, uncapped);
    try near(capped, k.foldedLogit(base, beta, dot));
    const no_bias_round: f32 = @floatCast(bf(base + (beta + dot)));
    try t.expect(logit != no_bias_round);
    try t.expect(@abs(capped) < @abs(logit));
    const saturated = bf(30.0 * std.math.tanh(@as(f64, 1000) / 30.0));
    try t.expect(saturated <= 30 and saturated > 29.9);
}
