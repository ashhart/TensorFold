//! LFM3-Omni's Metal convolutions and gathers: the gated short conv, its state, the encoder's strided conv, embeddings.
const std = @import("std");
const mtl = @import("metal");
const k = @import("metal_omni_common.zig");

const c = k.c;
const R = k.R;
const Probe = k.Probe;

fn plantRows(x: []u16, rows: usize, d: usize) void {
    const at = [_][2]usize{ .{ 0, d / 3 }, .{ 1, d / 2 }, .{ 2, d - 1 }, .{ 3, 0 }, .{ 3, d - 1 } };
    const put = [_]u16{ k.nan16, k.inf16, k.ninf16, k.inf16, k.ninf16 };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * d + p[1]] = v;
    };
}

/// An odd integer in [47, 61], whose products with another pass 2^11 (fp16's last exact integer).
fn big(rng: std.Random, signed: bool) f32 {
    const v: f32 = @floatFromInt(47 + 2 * rng.uintLessThan(u32, 8));
    return if (signed and rng.boolean()) -v else v;
}

/// An odd integer in [1, top] (top odd), with a random sign when `signed`.
fn odd(rng: std.Random, top: u32, signed: bool) f32 {
    const v: f32 = @floatFromInt(2 * rng.uintLessThan(u32, (top + 1) / 2) + 1);
    return if (signed and rng.boolean()) -v else v;
}

// --- the gated short conv: om_short_conv (a few rows, the state updated), om_short_conv_rows and om_conv_state

const ConvArgs = extern struct { ld: i32, ldy: i32, T: i32, D: i32, K: i32 };

/// B | C | x rows, taps, state; stress: B = C = 1, x and state h, taps (w, -w - 2) pairs then K: partials past 2^11.
fn convInputs(p: *Probe, rng: std.Random) !void {
    const s = p.shape;
    const d = s.n;
    const kw = s.k;
    const bcx = try p.alloc(u16, 0, s.rows * 3 * d);
    const w = try p.alloc(u16, 1, d * kw);
    const state = try p.alloc(u16, 2, d * (kw - 1));
    if (p.fill == .stress) {
        for (0..d) |ch| {
            const h = c.toBf16(big(rng, true));
            for (0..s.rows) |t| {
                bcx[t * 3 * d + ch] = c.toBf16(1);
                bcx[t * 3 * d + d + ch] = c.toBf16(1);
                bcx[t * 3 * d + 2 * d + ch] = h;
            }
            for (0..kw - 1) |j| state[ch * (kw - 1) + j] = h;
            var j: usize = 0;
            while (j + 1 < kw) : (j += 2) {
                const v = big(rng, false);
                w[ch * kw + j] = c.toBf16(v);
                w[ch * kw + j + 1] = c.toBf16(-v - 2);
            }
            w[ch * kw + kw - 1] = c.toBf16(@floatFromInt(kw));
        }
        return;
    }
    k.randomBf16(rng, bcx, -2, 2);
    k.randomBf16(rng, w, -1, 1);
    k.randomBf16(rng, state, -2, 2);
    // the last rows, which the state keeps
    const r4 = @min(s.rows, 4);
    if (p.fill == .nonfinite) plantRows(bcx[(s.rows - r4) * 3 * d ..], r4, 3 * d);
}

fn convPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.k < 2 or s.k > 8 or s.k % 2 == 0 or s.n == 0) return false;
    try convInputs(p, rng);
    _ = try p.alloc(u16, 3, s.rows * s.n);
    try p.in(0);
    try p.in(1);
    if (p.variant == 0) try p.out(2, true) else try p.in(2);
    try p.out(3, false);
    return true;
}

fn convLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const a = ConvArgs{ .ld = k.i32_(3 * s.n), .ldy = k.i32_(s.n), .T = k.i32_(s.rows), .D = k.i32_(s.n), .K = k.i32_(s.k) };
    p.bind(e, &.{ 0, 1, 2, 3 }, a);
    if (p.variant == 0) Probe.go(e, .{ k.blocks(s.n, 128), 1, 1 }, 128) else Probe.go(e, .{ k.blocks(s.rows * s.n, 256), 1, 1 }, 256);
}

fn convCount(p: *Probe) usize {
    return p.shape.rows * p.shape.n + if (p.variant == 0) p.shape.n * (p.shape.k - 1) else 0;
}

fn bx(p: *Probe, t: usize, ch: usize) R {
    const d = p.shape.n;
    const row = p.get(u16, 0)[t * 3 * d ..];
    return R.bf16(row[ch]).mul(R.bf16(row[2 * d + ch])).bf();
}

/// Both forms: the taps over the state then B x rows, summed in order; y = bf16(C bf16(acc)).
fn convEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const s = p.shape;
    const d = s.n;
    const kw = s.k;
    const w = p.get(u16, 1);
    const state = p.get(u16, 2);
    const ys = if (p.variant == 0) out[d * (kw - 1) ..] else out;
    var terms: [8]R = undefined;
    for (0..d) |ch| {
        for (0..s.rows) |t| {
            for (0..kw) |j| {
                const src = @as(isize, @intCast(t + j)) - @as(isize, @intCast(kw - 1));
                const v = if (src >= 0) bx(p, @intCast(src), ch) else R.bf16(state[ch * (kw - 1) + @as(usize, @intCast(@as(isize, @intCast(kw - 1)) + src))]);
                terms[j] = R.bf16(w[ch * kw + j]).mulExact(v);
            }
            const acc = k.sum(mode, p.scratch, terms[0..kw], kw + 1).bf();
            ys[t * d + ch] = R.bf16(p.get(u16, 0)[t * 3 * d + d + ch]).mul(acc).bf();
        }
        if (p.variant == 0) for (0..kw - 1) |j| {
            const src = @as(isize, @intCast(s.rows + j)) - @as(isize, @intCast(kw - 1));
            out[ch * (kw - 1) + j] = if (src >= 0) bx(p, @intCast(src), ch) else R.bf16(state[ch * (kw - 1) + j + s.rows]);
        };
    }
}

const conv_def: k.Def = .{ .prep = convPrep, .launch = convLaunch, .count = convCount, .eval = convEval };

fn convDepth(s: c.Shape) usize {
    return s.k + 1;
}

fn stateDef() k.Def {
    return .{ .prep = statePrep, .launch = stateLaunch, .count = stateCount, .eval = stateEval };
}

fn statePrep(p: *Probe, rng: std.Random) !bool {
    if (p.shape.k < 2 or p.shape.k > 8) return false;
    try convInputs(p, rng);
    try p.in(0);
    try p.out(2, true);
    return true;
}

fn stateLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    p.bind(e, &.{ 0, 2 }, ConvArgs{ .ld = k.i32_(3 * s.n), .ldy = k.i32_(s.n), .T = k.i32_(s.rows), .D = k.i32_(s.n), .K = k.i32_(s.k) });
    Probe.go(e, .{ k.blocks(s.n, 128), 1, 1 }, 128);
}

fn stateCount(p: *Probe) usize {
    return p.shape.n * (p.shape.k - 1);
}

fn stateEval(p: *Probe, _: k.Mode, out: []R) !void {
    const s = p.shape;
    const kw = s.k;
    const state = p.get(u16, 2);
    for (0..s.n) |ch| for (0..kw - 1) |j| {
        const src = @as(isize, @intCast(s.rows + j)) - @as(isize, @intCast(kw - 1));
        out[ch * (kw - 1) + j] = if (src >= 0) bx(p, @intCast(src), ch) else R.bf16(state[ch * (kw - 1) + j + s.rows]);
    };
}

const state_def = stateDef();

// --- om_conv2d_s2: the encoder's stride-2 3x3 convs; shape rows C, n H, k W; variant 0 conv_in (+SiLU), 1 depthwise

const Conv2dArgs = extern struct { CIN: i32, H: i32, W: i32, C: i32, H2: i32, W2: i32, silu: i32 };

fn c2Prep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.rows == 0 or s.n < 2 or s.k < 2) return false;
    const planes: usize = if (p.variant == 0) 1 else s.rows;
    const x = try p.alloc(u16, 0, planes * s.n * s.k);
    const w = try p.alloc(u16, 1, s.rows * 9);
    const bias = try p.alloc(u16, 2, s.rows);
    _ = try p.alloc(u16, 3, s.rows * ((s.n + 1) / 2) * ((s.k + 1) / 2));
    if (p.fill == .stress) {
        // every plane one odd integer, taps (w, -w - 2) pairs then 9: interior sums come to the plane's value
        for (0..planes) |pl| @memset(x[pl * s.n * s.k ..][0 .. s.n * s.k], c.toBf16(big(rng, true)));
        for (0..s.rows) |ch| {
            var j: usize = 0;
            while (j < 8) : (j += 2) {
                const v = big(rng, false);
                w[ch * 9 + j] = c.toBf16(v);
                w[ch * 9 + j + 1] = c.toBf16(-v - 2);
            }
            w[ch * 9 + 8] = c.toBf16(9);
        }
        @memset(bias, 0);
    } else {
        k.randomBf16(rng, x, -2, 2);
        k.randomBf16(rng, w, -1, 1);
        k.randomBf16(rng, bias, -0.5, 0.5);
        if (p.fill == .nonfinite) plantRows(x, @min(planes, 4) * s.n, s.k);
    }
    for (0..3) |i| try p.in(i);
    try p.out(3, false);
    return true;
}

fn c2Launch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const h2 = (s.n + 1) / 2;
    const w2 = (s.k + 1) / 2;
    p.bind(e, &.{ 0, 1, 2, 3 }, Conv2dArgs{ .CIN = if (p.variant == 0) 1 else k.i32_(s.rows), .H = k.i32_(s.n), .W = k.i32_(s.k), .C = k.i32_(s.rows), .H2 = k.i32_(h2), .W2 = k.i32_(w2), .silu = @intFromBool(p.variant == 0) });
    Probe.go(e, .{ k.blocks(s.rows * h2 * w2, 256), 1, 1 }, 256);
}

fn c2Count(p: *Probe) usize {
    return p.shape.rows * ((p.shape.n + 1) / 2) * ((p.shape.k + 1) / 2);
}

fn c2Eval(p: *Probe, mode: k.Mode, out: []R) !void {
    const s = p.shape;
    const h2 = (s.n + 1) / 2;
    const w2 = (s.k + 1) / 2;
    const x = p.get(u16, 0);
    const w = p.get(u16, 1);
    var terms: [9]R = undefined;
    for (0..s.rows) |ch| for (0..h2) |h| for (0..w2) |col| {
        const plane = x[(if (p.variant == 0) 0 else ch) * s.n * s.k ..];
        var n: usize = 0;
        for (0..3) |kh| for (0..3) |kc| {
            const ih = @as(isize, @intCast(2 * h + kh)) - 1;
            const iw = @as(isize, @intCast(2 * col + kc)) - 1;
            if (ih < 0 or ih >= s.n or iw < 0 or iw >= s.k) continue;
            terms[n] = R.bf16(w[ch * 9 + kh * 3 + kc]).mulExact(R.bf16(plane[@as(usize, @intCast(ih)) * s.k + @as(usize, @intCast(iw))]));
            n += 1;
        };
        var v = k.sum(mode, p.scratch, terms[0..n], 10).add(R.bf16(p.get(u16, 2)[ch])).bf();
        if (p.variant == 0) v = k.siluR(v).bf();
        out[(ch * h2 + h) * w2 + col] = v;
    };
}

const c2_def: k.Def = .{ .prep = c2Prep, .launch = c2Launch, .count = c2Count, .eval = c2Eval };

fn depth10(_: c.Shape) usize {
    return 10;
}

// --- gathers: om_embed (ids < 0 leave a row), om_embed_codes (12 codebooks summed), om_depth_in

const EmbedArgs = extern struct { T: i32, D: i32, NQ: i32, size: i32 };
const nq = 12;
const size = 2017;
const embed_rows = 4096;

fn embedPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0 or s.rows == 0) return false;
    const table = try p.alloc(u16, 0, embed_rows * s.n);
    const ids = try p.alloc(i32, 1, s.rows);
    const y = try p.alloc(u16, 2, s.rows * s.n);
    k.randomBf16(rng, table, -1, 1);
    k.randomBf16(rng, y, -1, 1);
    for (ids, 0..) |*id, t| id.* = if (t % 5 == 3) -1 else @intCast(rng.uintLessThan(u32, embed_rows));
    if (p.fill == .nonfinite) {
        ids[0] = 7;
        ids[1] = 9;
        plantRows(table[7 * s.n ..][0 .. 3 * s.n], 3, s.n);
    }
    try p.in(0);
    try p.in(1);
    try p.out(2, true);
    return true;
}

fn embedLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    p.bind(e, &.{ 0, 1, 2 }, EmbedArgs{ .T = k.i32_(s.rows), .D = k.i32_(s.n), .NQ = 0, .size = 0 });
    Probe.go(e, .{ k.blocks(s.rows * s.n, 256), 1, 1 }, 256);
}

fn rowsN(p: *Probe) usize {
    return p.shape.rows * p.shape.n;
}

fn embedEval(p: *Probe, _: k.Mode, out: []R) !void {
    const d = p.shape.n;
    for (p.get(i32, 1), 0..) |id, t| for (0..d) |i| {
        out[t * d + i] = R.bf16(if (id >= 0) p.get(u16, 0)[@as(usize, @intCast(id)) * d + i] else p.get(u16, 2)[t * d + i]);
    };
}

const embed_def: k.Def = .{ .prep = embedPrep, .launch = embedLaunch, .count = rowsN, .eval = embedEval };

/// Codebook tables; stress: a big multiple of 64, an odd integer, then their negations less a little: small sums.
fn codesPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0 or s.rows == 0 or s.k != nq) return false;
    const table = try p.alloc(u16, 0, nq * size * s.n);
    const codes = try p.alloc(i32, 1, s.rows * nq);
    _ = try p.alloc(u16, 2, s.rows * s.n);
    for (codes) |*x| x.* = @intCast(rng.uintLessThan(u32, size));
    if (p.fill == .stress) {
        @memset(table, 0);
        for (0..s.n) |col| {
            const top: f32 = @floatFromInt(64 * (64 + rng.uintLessThan(u32, 192)));
            const o = odd(rng, 255, true);
            const small: f32 = @floatFromInt(rng.uintLessThan(u32, 9));
            for (0..size) |code| {
                table[(0 * size + code) * s.n + col] = c.toBf16(top);
                table[(1 * size + code) * s.n + col] = c.toBf16(o);
                table[(2 * size + code) * s.n + col] = c.toBf16(-top);
                table[(3 * size + code) * s.n + col] = c.toBf16(small - o);
            }
        }
    } else {
        k.randomBf16(rng, table, -1, 1);
        if (p.fill == .nonfinite) {
            for (0..nq) |q| codes[q] = 5;
            plantRows(table[(2 * size + 5) * s.n ..][0..s.n], 1, s.n);
            table[(7 * size + 5) * s.n + 1] = k.inf16;
            table[(9 * size + 5) * s.n + 2] = k.ninf16;
            table[(4 * size + 5) * s.n + 3] = k.inf16;
            table[(8 * size + 5) * s.n + 3] = k.ninf16;
        }
    }
    try p.in(0);
    try p.in(1);
    try p.out(2, false);
    return true;
}

fn codesLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    p.bind(e, &.{ 0, 1, 2 }, EmbedArgs{ .T = k.i32_(s.rows), .D = k.i32_(s.n), .NQ = nq, .size = size });
    Probe.go(e, .{ k.blocks(s.rows * s.n, 256), 1, 1 }, 256);
}

fn codesEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const d = p.shape.n;
    const codes = p.get(i32, 1);
    var terms: [nq]R = undefined;
    for (0..p.shape.rows) |t| for (0..d) |i| {
        for (&terms, 0..) |*x, q| x.* = R.bf16(p.get(u16, 0)[(@as(usize, @intCast(codes[t * nq + q])) + q * size) * d + i]);
        out[t * d + i] = k.sum(mode, p.scratch, &terms, nq).bf();
    };
}

const codes_def: k.Def = .{ .prep = codesPrep, .launch = codesLaunch, .count = rowsN, .eval = codesEval };

fn depthNq(_: c.Shape) usize {
    return nq;
}

fn dinPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.rows == 0 or s.rows > nq or s.n == 0) return false;
    const din = try p.alloc(u16, 0, s.rows * s.n);
    const mask = try p.alloc(u16, 1, s.n);
    const tables = try p.alloc(u16, 2, nq * size * s.n);
    const codes = try p.alloc(i32, 3, s.rows);
    const masked = try p.alloc(i32, 4, s.rows);
    _ = try p.alloc(u16, 5, s.rows * s.n);
    k.randomBf16(rng, din, -2, 2);
    k.randomBf16(rng, mask, -1, 1);
    k.randomBf16(rng, tables, -1, 1);
    for (codes, masked) |*x, *m| {
        x.* = @intCast(rng.uintLessThan(u32, size));
        m.* = @intFromBool(rng.boolean());
    }
    if (p.fill == .nonfinite) plantRows(din, s.rows, s.n);
    for (0..5) |i| try p.in(i);
    try p.out(5, false);
    return true;
}

fn dinLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1, 2, 3, 4, 5 }, EmbedArgs{ .T = 0, .D = k.i32_(p.shape.n), .NQ = k.i32_(p.shape.rows), .size = size });
    Probe.go(e, .{ k.blocks(p.shape.rows * p.shape.n, 256), 1, 1 }, 256);
}

fn dinEval(p: *Probe, _: k.Mode, out: []R) !void {
    const d = p.shape.n;
    for (0..p.shape.rows) |q| for (0..d) |i| {
        const e = if (p.get(i32, 4)[q] != 0) p.get(u16, 1)[i] else p.get(u16, 2)[(q * size + @as(usize, @intCast(p.get(i32, 3)[q]))) * d + i];
        out[q * d + i] = R.bf16(p.get(u16, 0)[q * d + i]).add(R.bf16(e)).bf();
    };
}

const din_def: k.Def = .{ .prep = dinPrep, .launch = dinLaunch, .count = rowsN, .eval = dinEval };

// --- moves: om_copy_rows (shape rows T, n the width, k the source's row stride), om_repeat_rows (k the factor)

const CopyArgs = extern struct { T: i32, n: i32, lds: i32, ldd: i32 };

fn copyPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.k < s.n or s.n == 0) return false;
    const src = try p.alloc(u16, 0, s.rows * s.k);
    const dst = try p.alloc(u16, 1, s.rows * (s.n + 8));
    k.randomBf16(rng, src, -2, 2);
    k.randomBf16(rng, dst, -2, 2);
    if (p.fill == .nonfinite) plantRows(src, s.rows, s.k);
    try p.in(0);
    try p.out(1, true);
    return true;
}

fn copyLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    p.bind(e, &.{ 0, 1 }, CopyArgs{ .T = k.i32_(s.rows), .n = k.i32_(s.n), .lds = k.i32_(s.k), .ldd = k.i32_(s.n + 8) });
    Probe.go(e, .{ k.blocks(s.rows * s.n, 256), 1, 1 }, 256);
}

fn copyCount(p: *Probe) usize {
    return p.shape.rows * (p.shape.n + 8);
}

fn copyEval(p: *Probe, _: k.Mode, out: []R) !void {
    const s = p.shape;
    for (out, p.get(u16, 1)) |*o, v| o.* = R.bf16(v);
    for (0..s.rows) |t| for (0..s.n) |i| {
        out[t * (s.n + 8) + i] = R.bf16(p.get(u16, 0)[t * s.k + i]);
    };
}

const copy_def: k.Def = .{ .prep = copyPrep, .launch = copyLaunch, .count = copyCount, .eval = copyEval };

const RepeatArgs = extern struct { n: i32, D: i32, rep: i32 };

fn repPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.k == 0 or s.n == 0) return false;
    const x = try p.alloc(u16, 0, s.rows * s.n);
    _ = try p.alloc(u16, 1, s.rows * s.k * s.n);
    k.randomBf16(rng, x, -2, 2);
    if (p.fill == .nonfinite) plantRows(x, s.rows, s.n);
    try p.in(0);
    try p.out(1, false);
    return true;
}

fn repLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    p.bind(e, &.{ 0, 1 }, RepeatArgs{ .n = k.i32_(s.rows), .D = k.i32_(s.n), .rep = k.i32_(s.k) });
    Probe.go(e, .{ k.blocks(s.rows * s.k * s.n, 256), 1, 1 }, 256);
}

fn repCount(p: *Probe) usize {
    return p.shape.rows * p.shape.k * p.shape.n;
}

fn repEval(p: *Probe, _: k.Mode, out: []R) !void {
    const s = p.shape;
    for (out, 0..) |*o, i| o.* = R.bf16(p.get(u16, 0)[(i / s.n / s.k) * s.n + i % s.n]);
}

const rep_def: k.Def = .{ .prep = repPrep, .launch = repLaunch, .count = repCount, .eval = repEval };

fn none(name: []const u8, served: []const c.Shape, in_place: bool) c.Spec {
    return .{ .name = name, .accum = .none, .depth = k.noSums, .ops = 4, .out = .bf16, .in_place = in_place, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums" } };
}

pub const entries = [_]k.Entry{
    .{ .def = &conv_def, .spec = .{ .name = "metal lfm3 omni om_short_conv", .accum = .f32, .depth = convDepth, .ops = 2, .out = .bf16, .in_place = true, .tiles = .{ .{ .size = 4, .min = 1 }, .{ .size = 128 }, .{} }, .served = &.{ .{ .rows = 1, .n = 2048, .k = 7 }, .{ .rows = 4, .n = 512, .k = 7 }, .{ .rows = 1, .n = 1024, .k = 7 } } }, .file = .omni, .pipe = "om_short_conv", .control = .{ "const float bx = rb(f(row[d]) * f(row[2 * D + d]));", "const float bx = (f(row[d]) * f(row[2 * D + d]));" }, .what = "drops bf16(B x)" },
    .{ .def = &conv_def, .spec = .{ .name = "metal lfm3 omni om_short_conv_rows", .accum = .f32, .depth = convDepth, .ops = 2, .out = .bf16, .tiles = .{ .{ .size = 64 }, .{}, .{} }, .served = &.{ .{ .rows = 94, .n = 768, .k = 3 }, .{ .rows = 40, .n = 2048, .k = 7 }, .{ .rows = 48, .n = 512, .k = 7 } } }, .file = .omni, .pipe = "om_short_conv_rows", .variant = 1, .control = .{ "Y[(long)t * a.ldy + d] = b(f(BCx[(long)t * a.ld + D + d]) * rb(acc));", "Y[(long)t * a.ldy + d] = b(f(BCx[(long)t * a.ld + D + d]) * (acc));" }, .what = "drops bf16(acc)" },
    .{ .def = &state_def, .spec = none("metal lfm3 omni om_conv_state", &.{ .{ .rows = 94, .n = 768, .k = 3 }, .{ .rows = 40, .n = 2048, .k = 7 }, .{ .rows = 5, .n = 512, .k = 7 } }, true), .file = .omni, .pipe = "om_conv_state", .control = .{ "STATE[d * (K - 1) + j] = STATE[d * (K - 1) + j + T];", "STATE[d * (K - 1) + j] = STATE[d * (K - 1) + j + T - 1];" }, .what = "shifts the kept state by one row less" },
    .{ .def = &c2_def, .spec = .{ .name = "metal lfm3 omni om_conv2d_s2 conv_in, SiLU", .accum = .f32, .depth = depth10, .ops = 8, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 256, .n = 128, .k = 201 }, .{ .rows = 16, .n = 128, .k = 64 } } }, .file = .omni, .pipe = "om_conv2d_s2", .variant = 0, .control = .{ "float v = rb(acc + f(Bias[c]));", "float v = (acc + f(Bias[c]));" }, .what = "drops bf16(conv + bias) before SiLU" },
    .{ .def = &c2_def, .spec = .{ .name = "metal lfm3 omni om_conv2d_s2 depthwise", .accum = .f32, .depth = depth10, .ops = 4, .out = .bf16, .served = &.{ .{ .rows = 256, .n = 64, .k = 101 }, .{ .rows = 256, .n = 32, .k = 51 } } }, .file = .omni, .pipe = "om_conv2d_s2", .variant = 1, .control = .{ "const int ih = 2 * h - 1 + kh;", "const int ih = 2 * h + kh;" }, .what = "drops the top padding row" },
    .{ .def = &embed_def, .spec = none("metal lfm3 omni om_embed", &.{ .{ .rows = 37, .n = 2048, .k = 1 }, .{ .rows = 1, .n = 2048, .k = 1 } }, true), .file = .omni, .pipe = "om_embed", .control = .{ "if (id >= 0) Y[i] = TABLE[(long)id * a.D + c];", "if (id >= 0) Y[i] = TABLE[(long)id * a.D + (c ^ 1)];" }, .what = "swaps column pairs" },
    .{ .def = &codes_def, .spec = .{ .name = "metal lfm3 omni om_embed_codes", .accum = .f32, .depth = depthNq, .out = .bf16, .served = &.{ .{ .rows = 1, .n = 2048, .k = nq }, .{ .rows = 5, .n = 2048, .k = nq } } }, .file = .omni, .pipe = "om_embed_codes", .control = .{ "for (int q = 0; q < a.NQ; ++q) s += f(TABLE[", "for (int q = 0; q < a.NQ; ++q) s = rb(s) + f(TABLE[" }, .what = "rounds each partial sum to bf16" },
    .{ .def = &din_def, .spec = none("metal lfm3 omni om_depth_in", &.{.{ .rows = nq, .n = 1024, .k = 1 }}, false), .file = .omni, .pipe = "om_depth_in", .control = .{ "X[i] = b(f(DIN[i]) + e);", "X[i] = b(f(DIN[i]) + rb(e * 1.0078125f));" }, .what = "scales the slot's embedding" },
    .{ .def = &copy_def, .spec = none("metal lfm3 omni om_copy_rows", &.{ .{ .rows = 12, .n = 1024, .k = 1024 }, .{ .rows = 6, .n = 512, .k = 776 } }, true), .file = .omni, .pipe = "om_copy_rows", .control = .{ "D[(long)t * a.ldd + c] = S[(long)t * a.lds + c];", "D[(long)t * a.ldd + c] = S[(long)t * a.lds + (c ^ 1)];" }, .what = "swaps column pairs" },
    .{ .def = &rep_def, .spec = none("metal lfm3 omni om_repeat_rows", &.{ .{ .rows = 64, .n = 512, .k = 6 }, .{ .rows = 1, .n = 512, .k = 6 } }, false), .file = .omni, .pipe = "om_repeat_rows", .control = .{ "Y[i] = X[(r / a.rep) * a.D + i % a.D];", "Y[i] = X[(r % a.n) * a.D + i % a.D];" }, .what = "repeats the rows in turn" },
};
