//! Host-only M3 format, layout, BF16-rounding and dispatch-geometry tests.
const std = @import("std");
const p = @import("experts_plan.zig");

fn bf16Chain(terms: []const f32, stride: usize, first: usize) f32 {
    var sum: f32 = 0;
    var i = first;
    while (i < terms.len) : (i += stride) sum = p.roundBf16(sum + p.roundBf16(terms[i]));
    return sum;
}

fn bf16Halves(terms: []const f32) f32 {
    if (terms.len == 0) return 0;
    if (terms.len == 1) return p.roundBf16(terms[0]);
    const mid = terms.len / 2;
    return p.roundBf16(bf16Halves(terms[0..mid]) + bf16Halves(terms[mid..]));
}

fn bf16Strided(terms: []const f32) f32 {
    var lanes: [32]f32 = @splat(0);
    for (&lanes, 0..) |*lane, i| lane.* = bf16Chain(terms, 32, i);
    return bf16Halves(&lanes);
}

test "Mini layer formats and dimensions are a clean cut at layer 20" {
    for (0..20) |layer| try std.testing.expectEqual(p.Format.nvfp4, try p.formatForLayer(@intCast(layer)));
    for (20..24) |layer| try std.testing.expectEqual(p.Format.mxfp8, try p.formatForLayer(@intCast(layer)));
    try std.testing.expectError(error.BadExpertLayer, p.formatForLayer(24));
    try std.testing.expectEqual(@as(u32, 2304), p.hidden);
    try std.testing.expectEqual(@as(u32, 1152), p.intermediate);
    try std.testing.expectEqual(@as(u32, 128), p.expert_count);
    try std.testing.expectEqual(@as(u32, 8), p.top_k);
}

test "every E2M1 code decodes low/high nibble and signed zero" {
    const magnitude = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    for (0..16) |code| {
        const want = if (code & 8 != 0) -magnitude[code & 7] else magnitude[code & 7];
        try std.testing.expectEqual(want, p.e2m1(@intCast(code)));
    }
    const byte: u8 = 0xa7; // low 7 = 6, high 10 = -1
    try std.testing.expectEqual(@as(f32, 6), p.nvWeight(byte, 0, 0x38, 1));
    try std.testing.expectEqual(@as(f32, -1), p.nvWeight(byte, 1, 0x38, 1));
    try std.testing.expectEqual(@as(f32, 0), p.e2m1(8));
}

test "all E4M3 values match the independent half-bit decode" {
    for (0..256) |raw| {
        const code: u8 = @intCast(raw);
        const h: f16 = @bitCast(@as(u16, code & 0x7f) << 7);
        const magnitude = @as(f32, h) * 256;
        const want = if ((code & 0x80) != 0) -magnitude else magnitude;
        const got = p.e4m3(code);
        // E4M3FN reserves 0x7f for NaN; the half-bit helper alone does not encode that case.
        if ((code & 0x7f) == 0x7f) {
            try std.testing.expect(std.math.isNan(got));
        } else try std.testing.expectEqual(want, got);
    }
}

test "NVFP4 globals round to BF16 before one FP32 division" {
    const amax: f32 = @bitCast(@as(u32, 0x3f80_8001));
    const den: f32 = @bitCast(@as(u32, 0x4534_0001));
    const rounded_den = p.roundBf16(den);
    try std.testing.expect(den != rounded_den);
    try std.testing.expectEqual(p.roundBf16(amax) / rounded_den, p.phi(amax, den));
    try std.testing.expect(p.phi(amax, den) != amax / den); // A2: the reference rounds both loader inputs.
    const raw = p.e2m1(7) * p.e4m3(0x40) * p.phi(amax, den);
    try std.testing.expectEqual(p.roundBf16(raw), p.nvWeight(7, 0, 0x40, p.phi(amax, den)));
}

test "every MXFP8 value and UE8M0 exponent is exactly BF16 decoded" {
    for ([_]u8{ 0, 1, 126, 127, 128, 129, 254, 255 }) |scale| for (0..256) |raw| {
        const code: u8 = @intCast(raw);
        const want = std.math.ldexp(p.e4m3(code), @as(i32, scale) - 127);
        if (std.math.isNan(want)) {
            try std.testing.expect(std.math.isNan(p.mxWeight(code, scale)));
        } else try std.testing.expectEqual(p.roundBf16(want), p.mxWeight(code, scale));
    };
}

test "GU BF16 round precedes cap sigmoid cap and final BF16 activation" {
    const raw_gate: f32 = -10;
    const raw_up: f32 = -9.4;
    const g = p.roundBf16(raw_gate);
    const u = p.roundBf16(raw_up);
    const capped_g: f64 = 4 * std.math.tanh(@as(f64, g) / 4);
    const sigmoid: f64 = 1 / (1 + @exp(-@as(f64, g)));
    const capped_u: f64 = 25 * std.math.tanh(@as(f64, u) / 25);
    const wide = capped_g * sigmoid * capped_u;
    try std.testing.expectEqual(@as(f32, @floatCast(@import("bf16_round").bf16(wide))), p.swiglu(g, u));
    try std.testing.expect(p.swiglu(raw_gate, raw_up) != p.swiglu(g, u));
}

test "combine route control overflows slot order while ascending experts remain finite" {
    const values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const weights = [_]f32{ 0x1p127, 0x1p126, -0x1p127 / 3.0, 0.25, -0.2, 1.0 / 3.0, -2.0 / 7.0, 0.375 };
    var terms: [8]f32 = undefined;
    for (&terms, values, weights) |*term, value, weight| term.* = value * weight;
    var route_sum: f32 = 0;
    for (terms) |term| route_sum += term;
    try std.testing.expect(std.math.isInf(route_sum) and route_sum > 0);

    const ids = [_]i32{ 2, 0, 1, 7, 3, 6, 4, 5 };
    var expert_sum: f32 = 0;
    for (0..8) |expert| for (0..8) |slot| if (ids[slot] == expert) {
        expert_sum += terms[slot];
    };
    try std.testing.expect(std.math.isFinite(expert_sum) and expert_sum > 0);
}

test "eight-term combine stress separates FP32 and BF16 in every contract order" {
    const terms = [_]f32{ 1024, 1, 1, 1, 1, 1, 1, -1029 };
    var forward: f32 = 0;
    var reverse: f32 = 0;
    for (terms) |term| forward += term;
    for (1..terms.len + 1) |n| reverse += terms[terms.len - n];
    try std.testing.expectEqual(@as(f32, 1), forward);
    try std.testing.expectEqual(@as(f32, 1), reverse);

    const reduced = [_]f32{ bf16Chain(&terms, 1, 0), bf16Halves(&terms), bf16Strided(&terms) };
    for (reduced) |sum| try std.testing.expectEqual(@as(f32, -8), sum);
    const exact_allowance = (3 * 0x1p-24 + 0x1p-8) + 0x1p-38;
    try std.testing.expect(9.0 / exact_allowance >= 100);
}

test "launch geometry is one thread an output over every route pair and validates block/layout edges" {
    const nv = p.Call{ .rows = 3, .k = 2304, .n = 1152, .format = .nvfp4 };
    try p.check(nv);
    try std.testing.expectEqual(@as(u32, 24), p.geometry(nv).groups);
    try std.testing.expectEqual(@as(u32, 36), p.geometry(nv).blocks);
    try std.testing.expectEqual(@as(u32, 32), p.geometry(nv).threads);
    try std.testing.expectEqual(@as(u32, 3), p.combineGeometry(nv).groups);
    try std.testing.expectEqual(@as(u32, 36), p.combineGeometry(nv).blocks);
    const mx = p.Call{ .rows = 1, .k = 1152, .n = 2304, .format = .mxfp8 };
    try p.check(mx);
    var bad = nv;
    bad.k = 17;
    try std.testing.expectError(error.BadExpertBlock, p.check(bad));
    bad = nv;
    bad.topk = 129;
    try std.testing.expectError(error.BadExpertShape, p.check(bad));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(p.MatrixArgs));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(p.DecodeArgs));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(p.MatrixArgs, "k"));
    try std.testing.expectEqual(@as(usize, 12), @offsetOf(p.MatrixArgs, "experts"));
}

test "grouped expert tile upper bound, buffer sizing and ABI layouts hold across all row tiers" {
    const expected = [_][2]u32{ .{ 1, 8 }, .{ 2, 16 }, .{ 4, 32 }, .{ 8, 64 }, .{ 16, 128 }, .{ 32, 144 }, .{ 64, 176 } };
    for (expected) |pair| {
        const c = p.Call{ .rows = pair[0], .k = 2304, .n = 1152, .format = .nvfp4 };
        try p.checkGrouped(c);
        try std.testing.expectEqual(pair[1], p.maxTiles(c));
        try std.testing.expectEqual(pair[1], p.groupedGeometry(c).groups);
    }
    try std.testing.expectEqual(@as(u32, 5), p.maxTiles(.{ .rows = 8, .k = 64, .n = 65, .experts = 4, .topk = 2, .format = .mxfp8 }));
    try std.testing.expectEqual(@as(usize, 4100), p.groupBytes(64, 8));
    try std.testing.expectError(error.BadExpertShape, p.checkGrouped(.{ .rows = 65, .k = 2304, .n = 1152, .format = .nvfp4 }));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(p.GroupArgs));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(p.GroupedMatrixArgs));
}
