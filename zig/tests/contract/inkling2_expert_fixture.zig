//! Sparse-but-dimensioned checkpoint tensors for the expert probes. All tensors use original names and wire layout.
const std = @import("std");
const tf = @import("tensorfold");
const ck = tf.checkpoint_metal;
const cfg = tf.inkling2_metal.config;
const u = @import("inkling2_util.zig");
const rig_ = @import("rig.zig");
const math = @import("inkling2_expert_math.zig");
pub const Format = math.Format;
pub const experts = 9;
pub const topk = 8;
pub const gate_cap = 4;
pub const up_cap = 25;
pub const hidden = 2304;
pub const inter = 1152;

pub fn configuration(shape: u.c.Shape) cfg.Config {
    var c: cfg.Config = undefined;
    c.text.hidden = @intCast(shape.k);
    c.text.expert_inter = @intCast(shape.n);
    c.text.layers = 1;
    c.text.experts = experts;
    c.text.topk = topk;
    c.text.gate_cap = gate_cap;
    c.text.up_cap = up_cap;
    return c;
}

pub const Index = struct { at: usize, code: u8, scale: u8 };

pub fn support(f: Format, fill: u.g.Fill, gate_up: bool, index: usize, expert: usize, k: usize, out: *[4]Index) []const Index {
    const nv = f == .nvfp4;
    const one: u8 = if (nv) 2 else 0x38;
    const scale_one: u8 = if (nv) 0x38 else 127;
    if (fill == .stress) {
        if (gate_up) {
            out[0] = .{ .at = 0, .code = one, .scale = scale_one };
            return out[0..1];
        }
        // Large terms cancel in FP32; FP16 and BF16 lose a small term before cancellation.
        const b = f.block();
        const large: u8 = if (nv) 7 else one;
        const big_scale: u8 = if (nv) 0x78 else 137;
        const little: u8 = if (nv) 1 else one;
        const small_scale: u8 = if (nv) 0x18 else 122;
        out.* = .{
            .{ .at = 0, .code = large, .scale = big_scale },
            .{ .at = b, .code = little, .scale = small_scale },
            .{ .at = 2 * b, .code = if (nv) large | 8 else large | 128, .scale = big_scale },
            .{ .at = 32 * b, .code = little, .scale = small_scale },
        };
        return out;
    }
    if (fill == .nonfinite and expert >= 7) {
        out[0] = .{ .at = 0, .code = one, .scale = scale_one };
        if (gate_up) {
            if (expert == 8) {
                // NVFP4: an E4M3 NaN scale. MXFP8: a zero code in this block times exp2(128).
                out[0].scale = if (nv) 0x7f else 255;
            } else if (!nv) {
                // UE8M0=0, but the decoded BF16 weight is normal (2^-119), as is the input.
                out[0] = .{ .at = 0, .code = 0x78, .scale = 0 };
            }
        }
        return out[0..1];
    }
    const j = if (gate_up) index / 2 else index;
    const parity = if (gate_up) index % 2 else 0;
    if (expert < 3 and j == 0) {
        const negative = !gate_up and expert == 1;
        const sign: u8 = if (nv) 8 else 128;
        out[0] = .{ .at = 0, .code = one | (if (negative) sign else 0), .scale = scale_one };
        return out[0..1];
    }
    if (nv and expert == 3 and j == 0) {
        // Rounding 26/21 to 79/64 crosses a BF16 midpoint in the down output.
        out[0] = .{ .at = 0, .code = if (gate_up) 7 else 2, .scale = if (gate_up) 0x7e else 0x7d };
        return out[0..1];
    }
    // Nonfinite inputs are planted in the last two columns, where every stored weight is zero.
    const usable = k - @as(usize, if (fill == .nonfinite) 2 else 0);
    const at = switch (index % 11) {
        0 => 0,
        1 => usable - 2,
        else => 2 * ((j * 37 + expert * 19 + parity * 23) % (usable / 2)),
    };
    const code4 = [_]u8{ 1, 2, 3, 7, 4, 6, 14, 13 };
    const code8 = [_]u8{ 0x38, 0x40, 0xab, 0x3e, 0x54, 0xc3, 0x01, 0xbe };
    const scales4 = if (expert < 3) [_]u8{ 0x20, 0x30, 0x38, 0x40 } else [_]u8{ 0x50, 0x68, 0x78, 0x7e };
    const scales8 = [_]u8{ 125, 126, 127, 128 };
    const code = (if (nv) code4 else code8)[(j + expert + 3 * parity) % 8];
    const scale = (if (nv) scales4 else scales8)[(j + expert + parity) % 4];
    // Alternate single low/high codes, with some gate/up rows exercising both nibbles of one byte.
    const adjacent = gate_up and index % 3 == 0;
    out[0] = .{ .at = at + @intFromBool(!adjacent and index % 2 != 0), .code = code, .scale = scale };
    if (!adjacent) return out[0..1];
    out[1] = .{ .at = at + 1, .code = code ^ @as(u8, if (nv) 8 else 128), .scale = scale };
    return out[0..2];
}

pub fn denominator(gate_up: bool) f32 {
    return if (gate_up) 2693 else 923;
}

pub fn wAmax(fill: u.g.Fill, gate_up: bool, e: usize) f32 {
    if (e < 3 or (fill == .nonfinite and e >= 7)) return if (gate_up) 2688 else 924;
    return (if (gate_up) @as(f32, 1.003) else 1.253) + @as(f32, @floatFromInt(e)) * 0.5;
}

fn name(cp: *ck.Checkpoint, r: rig_.Ref, dtype: ck.DType, dims: []const usize, comptime part: []const u8, comptime suffix: []const u8) !void {
    var t: ck.Tensor = .{ .buffer = r.buf, .offset = r.off, .dtype = dtype, .rank = dims.len, .bytes = dtype.size() };
    for (dims, 0..) |v, i| {
        t.shape[i] = v;
        t.bytes *= v;
    }
    try cp.set("model.llm.layers.0.mlp.experts." ++ part ++ suffix, t);
}

fn matrix(allocator: std.mem.Allocator, rig: *rig_.Rig, cp: *ck.Checkpoint, f: Format, fill: u.g.Fill, n: usize, k: usize, comptime gate_up: bool) !void {
    const nv = f == .nvfp4;
    const block = f.block();
    const pitch = if (nv) k / 2 else k;
    const scale_pitch = k / block;
    const vals = try allocator.alloc(u8, experts * n * pitch);
    defer allocator.free(vals);
    const scales = try allocator.alloc(u8, experts * n * scale_pitch);
    defer allocator.free(scales);
    @memset(vals, 0);
    @memset(scales, if (nv) 0x38 else 127);
    for (0..experts) |e| for (0..n) |i| {
        const weight = vals[(e * n + i) * pitch ..][0..pitch];
        const scale = scales[(e * n + i) * scale_pitch ..][0..scale_pitch];
        var buf: [4]Index = undefined;
        for (support(f, fill, gate_up, i, e, k, &buf)) |term| {
            if (nv) {
                const sh: u3 = @intCast((term.at % 2) * 4);
                weight[term.at / 2] |= term.code << sh;
            } else weight[term.at] = term.code;
            scale[term.at / block] = term.scale;
        }
    };
    const part = if (gate_up) "w13" else "w2";
    try name(cp, try rig.input(vals), if (nv) .u8 else .f8_e4m3, &.{ experts, n, pitch }, part, "_weight");
    try name(cp, try rig.input(scales), if (nv) .f8_e4m3 else .u8, &.{ experts, n, scale_pitch }, part, "_weight_scale");
    if (nv) {
        var peaks: [experts]f32 = undefined;
        for (&peaks, 0..) |*v, e| v.* = wAmax(fill, gate_up, e);
        try name(cp, try rig.input(std.mem.asBytes(&peaks)), .f32, &.{experts}, part, "_weight_amax");
        const den = denominator(gate_up);
        try name(cp, try rig.input(std.mem.asBytes(&den)), .f32, &.{1}, part, "_weight_outer_scale_denominator");
    }
}

/// Every packed byte live: random codes, scales in a mild range, so FP32 sums are inexact and order-sensitive.
fn denseMatrix(allocator: std.mem.Allocator, rig: *rig_.Rig, cp: *ck.Checkpoint, f: Format, random: std.Random, n: usize, k: usize, comptime gate_up: bool) !void {
    const nv = f == .nvfp4;
    const pitch = if (nv) k / 2 else k;
    const scale_pitch = k / f.block();
    const vals = try allocator.alloc(u8, experts * n * pitch);
    defer allocator.free(vals);
    const scales = try allocator.alloc(u8, experts * n * scale_pitch);
    defer allocator.free(scales);
    random.bytes(vals);
    for (scales) |*v| v.* = if (nv) 0x30 + random.uintLessThan(u8, 0x28) else 118 + random.uintLessThan(u8, 12);
    const part = if (gate_up) "w13" else "w2";
    try name(cp, try rig.input(vals), if (nv) .u8 else .f8_e4m3, &.{ experts, n, pitch }, part, "_weight");
    try name(cp, try rig.input(scales), if (nv) .f8_e4m3 else .u8, &.{ experts, n, scale_pitch }, part, "_weight_scale");
    if (nv) {
        var peaks: [experts]f32 = undefined;
        for (&peaks, 0..) |*v, e| v.* = wAmax(.random, gate_up, e);
        try name(cp, try rig.input(std.mem.asBytes(&peaks)), .f32, &.{experts}, part, "_weight_amax");
        const den = denominator(gate_up);
        try name(cp, try rig.input(std.mem.asBytes(&den)), .f32, &.{1}, part, "_weight_outer_scale_denominator");
    }
}

pub fn dense(a: std.mem.Allocator, rig: *rig_.Rig, cp: *ck.Checkpoint, f: Format, seed: u64, shape: u.c.Shape) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    try denseMatrix(a, rig, cp, f, prng.random(), shape.n * 2, shape.k, true);
    try denseMatrix(a, rig, cp, f, prng.random(), shape.k, shape.n, false);
}

pub fn tensors(a: std.mem.Allocator, rig: *rig_.Rig, cp: *ck.Checkpoint, f: Format, fill: u.g.Fill, shape: u.c.Shape) !void {
    try matrix(a, rig, cp, f, fill, shape.n * 2, shape.k, true);
    try matrix(a, rig, cp, f, fill, shape.k, shape.n, false);
}

pub fn rows(f: Format, shape: u.c.Shape, fill: u.g.Fill, seed: u64, x: []u16, ids: []u32, weights: []f32) void {
    var random = std.Random.DefaultPrng.init(seed);
    const rng = random.random();
    for (0..shape.rows) |r| {
        const row = x[r * shape.k ..][0..shape.k];
        u.fillDyadic(rng, row, 31, 16);
        row[row.len - 2] = u.word(1.25);
        row[row.len - 1] = u.word(-1.5);
        for (0..topk) |s| {
            ids[r * topk + s] = @intCast((r * 3 + s * 5) % experts);
            weights[r * topk + s] = @as(f32, @floatFromInt(1 + (s * 3 + r) % 7)) / 13;
        }
        if (r % 2 == 0) inactive(ids, weights, r * topk + topk - 1);
    }
    switch (fill) {
        .random => {
            x[0] = u.word(1);
            for (0..topk) |i| inactive(ids, weights, i);
            // Expert order cancels the large terms first; slot order loses expert 2's result.
            ids[0..3].* = .{ 2, 0, 1 };
            weights[0..3].* = .{ 1, 0x1p27, 0x1p27 };
            if (shape.rows > 1) {
                x[shape.k] = u.word(1);
                for (topk..2 * topk) |i| inactive(ids, weights, i);
                // The equal and opposite BF16 down rows differ only by a small FP32 factor.
                ids[topk..][0..2].* = .{ 0, 1 };
                weights[topk..][0..2].* = .{ 1, 1 + 0x1p-10 };
            }
            if (shape.rows > 2 and f == .nvfp4) {
                // Expert 3's gate/up weights are 2.5; this input rounds their projections to 1.
                x[2 * shape.k] = u.word(0.4);
                for (2 * topk..3 * topk) |i| inactive(ids, weights, i);
                ids[2 * topk] = 3;
                weights[2 * topk] = 1;
            }
        },
        .stress => {
            @memset(x, 0);
            for (0..ids.len) |i| inactive(ids, weights, i);
            for (0..shape.rows) |r| {
                x[r * shape.k] = u.word(1);
                ids[r * topk] = 0;
                weights[r * topk] = 1;
            }
        },
        .nonfinite => {
            for (0..ids.len) |i| inactive(ids, weights, i);
            for (0..4) |r| {
                x[(r + 1) * shape.k - 1] = ([_]u16{ u.nan_bits, u.pinf_bits, u.ninf_bits, u.pinf_bits })[r];
                ids[r * topk + r] = 0;
                weights[r * topk + r] = 1;
            }
            x[4 * shape.k - 2] = u.ninf_bits;
            // One all-inactive token ignores even NaN input and route weights.
            x[4 * shape.k] = u.nan_bits;
            for (5..8) |r| {
                @memset(x[r * shape.k ..][0..shape.k], 0);
                x[r * shape.k] = u.word(if (r == 5 and f == .mxfp8) 0x1p119 else 1);
                ids[r * topk + 1] = if (r == 7) 0 else @intCast(r + 2);
                weights[r * topk + 1] = if (r == 7) std.math.nan(f32) else 1;
            }
        },
    }
}

fn inactive(ids: []u32, weights: []f32, i: usize) void {
    ids[i] = if (i % 2 == 0) experts else std.math.maxInt(u32);
    weights[i] = std.math.nan(f32);
}
