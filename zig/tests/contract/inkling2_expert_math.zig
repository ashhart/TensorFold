//! Independent numeric oracle for packed experts: numerical FP4/FP8 grids, scalar FP32 quantizer decisions,
//! FP64 products/sums and the Mini grouped-BF16 epilogue. This code does not emulate Metal's lanes for normal fixtures.
const std = @import("std");
const u = @import("inkling2_util.zig");

pub const Format = enum {
    nvfp4,
    mxfp8,

    pub fn block(f: Format) usize {
        return if (f == .nvfp4) 16 else 32;
    }
};

pub fn fp32(x: f64) f64 {
    const r: f32 = @floatCast(x);
    return r;
}

const fp4_grid = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

pub fn e2m1(bits: u8) f64 {
    const mag = fp4_grid[bits & 7];
    return if (bits & 8 == 0) mag else -mag;
}

pub fn e4m3(bits: u8) f64 {
    const v = switch (bits & 127) {
        127 => std.math.nan(f64),
        0...7 => @as(f64, @floatFromInt(bits & 7)) / 512,
        else => std.math.ldexp(@as(f64, @floatFromInt(8 + (bits & 7))), @as(i32, (bits & 127) / 8) - 10),
    };
    return if (bits & 128 == 0) v else -v;
}

pub fn e8m0(bits: u8) f64 {
    return if (bits == 255) std.math.nan(f64) else std.math.ldexp(@as(f64, 1), @as(i32, bits) - 127);
}

fn nearest(value: f64, comptime decode: fn (u8) f64, max_code: u8, sign: u8) u8 {
    const v = @abs(value);
    var lo: u8 = 0;
    var hi: u8 = max_code;
    // Locate the two nearest representable values by numeric distance, not by FP32 representation bits.
    while (hi - lo > 1) {
        const m = lo + (hi - lo) / 2;
        if (decode(m) < v) lo = m else hi = m;
    }
    const dl = v - decode(lo);
    const dh = decode(hi) - v;
    const mag = if (dh < dl or (dh == dl and hi % 2 == 0)) hi else lo;
    return mag | (if (@as(u64, @bitCast(value)) >> 63 != 0) sign else 0);
}

fn pairwise16(values: *[16]f64) f64 {
    var stride: usize = 1;
    while (stride < 16) : (stride *= 2) {
        var i: usize = 0;
        while (i < 16) : (i += 2 * stride) values[i] = fp32(values[i] + values[i + stride]);
    }
    return values[0];
}

fn nvCandidate(x: []const u16, row_max: f64, block_max: f64, divisor: f64, codes: *[16]u8) struct { u8, f64 } {
    const fmax: f64 = std.math.floatMax(f32);
    const encoder = @min(fp32(1536.0 / row_max), fmax);
    const decoder = fp32(1.0 / encoder);
    const scale = nearest(fp32(block_max * fp32(encoder * fp32(1.0 / divisor))), e4m3, 126, 128);
    const s = e4m3(scale);
    const inverse = @min(fp32(1.0 / fp32(s * decoder)), fmax);
    const outer = fp32(row_max * fp32(1.0 / 1536.0));
    var errors: [16]f64 = undefined;
    for (x, &errors, codes) |word, *diff, *bits| {
        const v = u.val(word);
        bits.* = nearest(fp32(v * inverse), e2m1, 7, 8);
        diff.* = @abs(fp32(fp32((e2m1(bits.*) * s) * outer) - v));
    }
    return .{ scale, pairwise16(&errors) };
}

fn mxBlock(x: []const u16, bm: f64, codes: []u8) u8 {
    const raw = fp32(bm * fp32(1.0 / 448.0));
    // Decompose the numeric value; a fraction of one half means `raw` is exactly a power of two.
    const parts = std.math.frexp(raw);
    const exponent = std.math.clamp(parts.exponent - @intFromBool(parts.significand == 0.5), -127, 127);
    const scale: u8 = if (raw <= 0x1p-127) 0 else @intCast(exponent + 127);
    for (x, codes) |w, *out| out.* = nearest(fp32(u.val(w) / e8m0(scale)), e4m3, 126, 128);
    return scale;
}

/// Codes are unpacked on purpose (compare low/high nibbles separately when reading the GPU).
pub fn quant(format: Format, x: []const u16, codes: []u8, scales: []u8) f64 {
    var row_max: f64 = 0;
    for (x) |w| {
        const v = u.val(w);
        if (!std.math.isFinite(v)) {
            @memset(codes, 0);
            @memset(scales, if (format == .nvfp4) 0 else 255);
            return std.math.nan(f64);
        }
        row_max = @max(row_max, @abs(v));
    }
    const block = format.block();
    for (scales, 0..) |*scale, b| {
        const row = x[b * block ..][0..block];
        const dst = codes[b * block ..][0..block];
        var bm: f64 = 0;
        for (row) |w| bm = @max(bm, @abs(u.val(w)));
        switch (format) {
            .nvfp4 => {
                var at6: [16]u8 = undefined;
                var at4: [16]u8 = undefined;
                const s6, const e6 = nvCandidate(row, row_max, bm, 6, &at6);
                const s4, const e4 = nvCandidate(row, row_max, bm, 4, &at4);
                @memcpy(dst, if (e4 < e6) &at4 else &at6);
                scale.* = if (e4 < e6) s4 else s6;
            },
            .mxfp8 => scale.* = mxBlock(row, bm, dst),
        }
    }
    return if (format == .nvfp4) row_max else 1;
}

pub fn dotTerm(format: Format, x: u8, xs: u8, w: u8, ws: u8) f64 {
    return switch (format) {
        .nvfp4 => e2m1(x) * e4m3(xs) * e2m1(w) * e4m3(ws),
        .mxfp8 => e4m3(x) * e8m0(xs) * e4m3(w) * e8m0(ws),
    };
}

pub fn dotOutput(format: Format, raw: f64, x_max: f64, w_max: f64, den: f64) f64 {
    if (format == .mxfp8) return u.bf(raw);
    const outer = fp32(fp32(x_max * w_max) * fp32(1.0 / fp32(1536.0 * den)));
    return u.bf(fp32(u.bf(raw) * outer));
}

/// Use FP64 transcendental functions and round each FP32 operation before the separate BF16 store.
pub fn swiglu(g: f64, x: f64, gamma: f32, gate_cap: f64, up_cap: f64) f64 {
    const sigmoid = fp32(1 / fp32(1 + fp32(@exp(-g))));
    const gate = fp32(gate_cap * fp32(std.math.tanh(fp32(g / gate_cap))));
    const up = fp32(up_cap * fp32(std.math.tanh(fp32(x / up_cap))));
    return fp32(fp32(fp32(gate * sigmoid) * up) * gamma);
}

test "Mini encodings distinguish signed nibbles and the smallest UE8M0 scale" {
    try std.testing.expectEqual(@as(f64, -0.5), e2m1(9));
    try std.testing.expectEqual(@as(f64, 448), e4m3(126));
    try std.testing.expectEqual(@as(f64, 0x1p-127), e8m0(0));
    try std.testing.expect(std.math.isNan(e8m0(255)));
    try std.testing.expectEqual(@as(u8, 0), nearest(0.25, e2m1, 7, 8));
    try std.testing.expectEqual(@as(u8, 2), nearest(0.75, e2m1, 7, 8));
    try std.testing.expectEqual(@as(u8, 0x80), nearest(-0.0001, e4m3, 126, 128));
}
