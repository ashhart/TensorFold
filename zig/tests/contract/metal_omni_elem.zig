//! LFM3-Omni's Metal elementwise kernels: activations, casts, the codec's FSQ and spectra, log-mel, word moves.
const std = @import("std");
const mtl = @import("metal");
const k = @import("metal_omni_common.zig");

const c = k.c;
const R = k.R;
const Probe = k.Probe;

/// NaN, +Inf, -Inf at three places, and both infinities at two more (indexes past the end are skipped).
fn plant16(x: []u16) void {
    const n = x.len;
    for ([_]usize{ n / 3, n / 2, n - 1, 1, n / 5 }, [_]u16{ k.nan16, k.inf16, k.ninf16, k.inf16, k.ninf16 }) |i, v| x[i] = v;
}

fn plant32(x: []f32) void {
    const n = x.len;
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    for ([_]usize{ n / 3, n / 2, n - 1, 1, n / 5 }, [_]f32{ nan, inf, -inf, inf, -inf }) |i, v| x[i] = v;
}

fn none(name: []const u8, out: c.Elem, served: []const c.Shape) c.Spec {
    return .{ .name = name, .accum = .none, .depth = k.noSums, .ops = 8, .out = out, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums" } };
}

fn flat(p: *Probe) usize {
    return p.shape.rows * p.shape.n;
}

// --- om_swiglu: rows of 2F, halves (variant 0) or (a, c) pairs (1)

const SwiArgs = extern struct { T: i32, F: i32, inter: i32 };

fn swiPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0) return false;
    const ac = try p.alloc(u16, 0, s.rows * 2 * s.n);
    _ = try p.alloc(u16, 1, s.rows * s.n);
    k.randomBf16(rng, ac, -6, 6);
    if (p.fill == .nonfinite) plant16(ac[0 .. 2 * s.n * @min(s.rows, 4)]);
    try p.in(0);
    try p.out(1, false);
    return true;
}

fn swiLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1 }, SwiArgs{ .T = k.i32_(p.shape.rows), .F = k.i32_(p.shape.n), .inter = @intCast(p.variant) });
    Probe.go(e, .{ k.blocks(flat(p), 256), 1, 1 }, 256);
}

fn swiEval(p: *Probe, _: k.Mode, out: []R) !void {
    const f = p.shape.n;
    const ac = p.get(u16, 0);
    for (out, 0..) |*o, i| {
        const t = i / f;
        const col = i % f;
        const ia = if (p.variant == 1) t * 2 * f + 2 * col else t * 2 * f + col;
        const ic = if (p.variant == 1) ia + 1 else ia + f;
        o.* = k.siluR(R.bf16(ac[ia])).bf().mul(R.bf16(ac[ic])).bf();
    }
}

const swi_def: k.Def = .{ .prep = swiPrep, .launch = swiLaunch, .count = flat, .eval = swiEval };

// --- om_bias_act: act(bf16(A + bias)) from fp32 products; variant 0 column bias, 1 row bias and SiLU, 2 GELU

const BiasArgs = extern struct { rows: i32, cols: i32, per_row: i32, act: i32, has_bias: i32 };

fn biasPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0) return false;
    const a = try p.alloc(f32, 0, s.rows * s.n);
    const bias = try p.alloc(u16, 1, if (p.variant == 1) s.rows else s.n);
    _ = try p.alloc(u16, 2, s.rows * s.n);
    for (a) |*x| x.* = (rng.float(f32) * 2 - 1) * 4;
    k.randomBf16(rng, bias, -1, 1);
    if (p.fill == .nonfinite) plant32(a[0 .. s.n * @min(s.rows, 4)]);
    try p.in(0);
    try p.in(1);
    try p.out(2, false);
    return true;
}

fn biasLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1, 2 }, BiasArgs{ .rows = k.i32_(p.shape.rows), .cols = k.i32_(p.shape.n), .per_row = @intFromBool(p.variant == 1), .act = @intCast(p.variant), .has_bias = 1 });
    Probe.go(e, .{ k.blocks(flat(p), 256), 1, 1 }, 256);
}

fn biasEval(p: *Probe, _: k.Mode, out: []R) !void {
    const n = p.shape.n;
    for (out, p.get(f32, 0), 0..) |*o, a, i| {
        const b = R.bf16(p.get(u16, 1)[if (p.variant == 1) i / n else i % n]);
        const v = R.of(a).add(b).bf();
        o.* = (switch (p.variant) {
            1 => k.siluR(v),
            2 => k.geluR(v),
            else => v,
        }).bf();
    }
}

const bias_def: k.Def = .{ .prep = biasPrep, .launch = biasLaunch, .count = flat, .eval = biasEval };

// --- casts: om_f32_bf16, om_bf16_f32; shape rows 1, n elements

fn castPrep(p: *Probe, rng: std.Random) !bool {
    const n = flat(p);
    if (n == 0) return false;
    if (p.variant == 0) {
        const x = try p.alloc(f32, 0, n);
        for (x, 0..) |*v, i| v.* = (rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 1e5) else 1);
        if (p.fill == .nonfinite) plant32(x);
        _ = try p.alloc(u16, 1, n);
    } else {
        const x = try p.alloc(u16, 0, n);
        k.randomBf16(rng, x, -3, 3);
        if (p.fill == .nonfinite) plant16(x);
        _ = try p.alloc(f32, 1, n);
    }
    try p.in(0);
    try p.out(1, false);
    return true;
}

fn castLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1 }, @as(u32, @intCast(flat(p))));
    Probe.go(e, .{ k.blocks(flat(p), 256), 1, 1 }, 256);
}

fn castEval(p: *Probe, _: k.Mode, out: []R) !void {
    for (out, 0..) |*o, i| o.* = if (p.variant == 0) R.of(p.get(f32, 0)[i]).bf() else R.bf16(p.get(u16, 0)[i]);
}

const cast_def: k.Def = .{ .prep = castPrep, .launch = castLaunch, .count = flat, .eval = castEval };

// --- om_fsq_decode: codes [n, 12] to latents [n, 48], levels 8 7 6 6; integer inputs, so a stated rule

const FsqArgs = extern struct { n: i32, NQ: i32 };

fn fsqPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n != 12) return false;
    const codes = try p.alloc(i32, 0, s.rows * 12);
    _ = try p.alloc(u16, 1, s.rows * 48);
    for (codes, 0..) |*x, i| x.* = if (i < 2016) @intCast(i) else @intCast(rng.uintLessThan(u32, 2016));
    try p.in(0);
    try p.out(1, false);
    return true;
}

fn fsqLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1 }, FsqArgs{ .n = k.i32_(p.shape.rows), .NQ = 12 });
    Probe.go(e, .{ k.blocks(p.shape.rows * 12, 128), 1, 1 }, 128);
}

fn fsqCount(p: *Probe) usize {
    return p.shape.rows * 48;
}

fn fsqEval(p: *Probe, _: k.Mode, out: []R) !void {
    const levels = [4]i32{ 8, 7, 6, 6 };
    const basis = [4]i32{ 1, 8, 56, 336 };
    for (p.get(i32, 0), 0..) |code, i| for (0..4) |j| {
        const hf = @divTrunc(levels[j], 2);
        const digit = @mod(@divTrunc(code, basis[j]), levels[j]) - hf;
        out[i * 4 + j] = R.of(@floatFromInt(digit)).div(R.of(@floatFromInt(hf))).bf();
    };
}

const fsq_def: k.Def = .{ .prep = fsqPrep, .launch = fsqLaunch, .count = fsqCount, .eval = fsqEval, .check = k.checkReplay };

// --- the codec's spectra: om_polar (rows of log-magnitude | phase to re | im, fp32)

const PolarArgs = extern struct { T: i32, NB: i32 };

fn polarPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0) return false;
    const x = try p.alloc(u16, 0, s.rows * 2 * s.n);
    _ = try p.alloc(f32, 1, s.rows * 2 * s.n);
    for (0..s.rows) |t| {
        k.randomBf16(rng, x[t * 2 * s.n ..][0..s.n], -6, 3);
        k.randomBf16(rng, x[t * 2 * s.n + s.n ..][0..s.n], -4, 4);
    }
    if (p.fill == .nonfinite) plant16(x[0 .. 2 * s.n * @min(s.rows, 4)]);
    try p.in(0);
    try p.out(1, false);
    return true;
}

fn polarLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1 }, PolarArgs{ .T = k.i32_(p.shape.rows), .NB = k.i32_(p.shape.n) });
    Probe.go(e, .{ k.blocks(flat(p), 256), 1, 1 }, 256);
}

fn polarCount(p: *Probe) usize {
    return 2 * flat(p);
}

fn cosf(x: f64) f64 {
    return @cos(x);
}
fn sinf(x: f64) f64 {
    return @sin(x);
}

fn polarEval(p: *Probe, _: k.Mode, out: []R) !void {
    const nb = p.shape.n;
    const x = p.get(u16, 0);
    for (0..p.shape.rows) |t| for (0..nb) |i| {
        const m = R.bf16(x[t * 2 * nb + i]).up(k.exp, k.fn_ulps);
        const a = R.bf16(x[t * 2 * nb + nb + i]);
        out[t * 2 * nb + i] = m.mul(a.lip(cosf, 1, k.fn_ulps));
        out[t * 2 * nb + nb + i] = m.mul(a.lip(sinf, 1, k.fn_ulps));
    };
}

const polar_def: k.Def = .{ .prep = polarPrep, .launch = polarLaunch, .count = polarCount, .eval = polarEval };

// --- the log-mel front end: om_mel_frames, om_power, om_logmel (fp32 throughout)

const MelArgs = extern struct { len: i32, nf: i32, nfft: i32, hop: i32 };

/// Shape rows the frames, n the FFT's length, k the hop; the clip ends inside the last frame.
fn framesPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0 or s.k == 0) return false;
    const len = (s.rows - 1) * s.k + s.n / 2;
    const x = try p.alloc(f32, 0, len + s.n);
    const win = try p.alloc(f32, 1, s.n);
    _ = try p.alloc(f32, 2, s.rows * s.n);
    for (x) |*v| v.* = rng.float(f32) * 2 - 1;
    for (win, 0..) |*v, i| v.* = @floatCast(0.5 - 0.5 * @cos(2 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(s.n))));
    if (p.fill == .nonfinite) plant32(x[0..len]);
    for (0..2) |i| try p.in(i);
    try p.out(2, false);
    return true;
}

fn framesLen(p: *const Probe) usize {
    return (p.shape.rows - 1) * p.shape.k + p.shape.n / 2;
}

fn framesLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    p.bind(e, &.{ 0, 1, 2 }, MelArgs{ .len = k.i32_(framesLen(p)), .nf = k.i32_(s.rows), .nfft = k.i32_(s.n), .hop = k.i32_(s.k) });
    Probe.go(e, .{ k.blocks(s.rows * s.n, 256), 1, 1 }, 256);
}

fn framesEval(p: *Probe, _: k.Mode, out: []R) !void {
    const s = p.shape;
    for (out, 0..) |*o, i| {
        const at = (i / s.n) * s.k + i % s.n;
        o.* = if (at < framesLen(p)) R.of(p.get(f32, 0)[at]).mul(R.of(p.get(f32, 1)[i % s.n])) else R.of(0);
    }
}

const frames_def: k.Def = .{ .prep = framesPrep, .launch = framesLaunch, .count = flat, .eval = framesEval };

const PowerArgs = extern struct { nf: i32, bins: i32 };

/// Spectra rows re | im; stress: integers below 2^10, so both squares and their sum are exact in fp32.
fn powerPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0) return false;
    const sp = try p.alloc(f32, 0, s.rows * 2 * s.n);
    _ = try p.alloc(f32, 1, s.rows * s.n);
    for (sp) |*v| v.* = if (p.fill == .stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 2047))) - 1023) else (rng.float(f32) * 2 - 1) * 30;
    if (p.fill == .nonfinite) plant32(sp[0 .. 2 * s.n * @min(s.rows, 4)]);
    try p.in(0);
    try p.out(1, false);
    return true;
}

fn powerLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1 }, PowerArgs{ .nf = k.i32_(p.shape.rows), .bins = k.i32_(p.shape.n) });
    Probe.go(e, .{ k.blocks(flat(p), 256), 1, 1 }, 256);
}

fn sqrt(x: f64) f64 {
    return @sqrt(x);
}

fn powerEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const nb = p.shape.n;
    const sp = p.get(f32, 0);
    var t: [2]R = undefined;
    for (out, 0..) |*o, i| {
        const re = R.of(sp[(i / nb) * 2 * nb + i % nb]);
        const im = R.of(sp[(i / nb) * 2 * nb + nb + i % nb]);
        t = .{ re.mulExact(re), im.mulExact(im) };
        const m = k.sum(mode, p.scratch, &t, 3).fp(0).up(sqrt, k.sqrt_ulps);
        o.* = m.mul(m);
    }
}

fn depth3(_: c.Shape) usize {
    return 3;
}

const power_def: k.Def = .{ .prep = powerPrep, .launch = powerLaunch, .count = flat, .eval = powerEval };

const LogmelArgs = extern struct { nf: i32, mels: i32, floor_: f32 };

/// Mel rows [nf, mels] with zeros and values under 1e-12 among them (the floor's side).
fn logmelPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0) return false;
    const m = try p.alloc(f32, 0, s.rows * s.n);
    _ = try p.alloc(f32, 1, s.rows * s.n);
    for (m, 0..) |*v, i| v.* = switch (i % 17) {
        3 => 0,
        5 => 1e-14,
        else => std.math.exp(rng.float(f32) * 30 - 20),
    };
    if (p.fill == .nonfinite) plant32(m);
    try p.in(0);
    try p.out(1, false);
    return true;
}

fn logmelLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1 }, LogmelArgs{ .nf = k.i32_(p.shape.rows), .mels = k.i32_(p.shape.n), .floor_ = -12 });
    Probe.go(e, .{ k.blocks(flat(p), 256), 1, 1 }, 256);
}

fn log10(x: f64) f64 {
    return std.math.log10(x);
}

/// The floor as metal::max takes it: a NaN logarithm gives the floor.
fn logmelEval(p: *Probe, _: k.Mode, out: []R) !void {
    const s = p.shape;
    for (p.get(f32, 0), 0..) |m, i| {
        var l = R.of(m).up(log10, k.fn_ulps);
        if (std.math.isNan(l.v)) l = R.of(-12) else if (l.v < -12) l = R.of(-12) else if (l.lo < -12) l.lo = -12;
        out[(i % s.n) * s.rows + i / s.n] = l.add(R.of(4)).div(R.of(4));
    }
}

const logmel_def: k.Def = .{ .prep = logmelPrep, .launch = logmelLaunch, .count = flat, .eval = logmelEval, .check = k.checkReplay };

// --- word moves: om_set_words (bytes from the arguments), om_zero_words, om_copy_words; raw, checked byte for byte

fn wordsPrep(p: *Probe, rng: std.Random) !bool {
    const n = p.shape.n;
    if (n == 0 or (p.variant == 0 and n > 1024)) return false;
    const src = try p.alloc(u32, 0, n);
    const dst = try p.alloc(u32, 1, n + 8);
    for (src) |*x| x.* = rng.int(u32);
    for (dst) |*x| x.* = rng.int(u32);
    try p.in(0);
    try p.out(1, true);
    return true;
}

fn wordsLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const n: u32 = @intCast(p.shape.n);
    e.setPipeline(p.pipe);
    switch (p.variant) {
        0 => {
            e.setBuffer(p.refs[1].buf, p.refs[1].off, 0);
            e.setBytes(p.host[0], 1);
            e.setBytes(std.mem.asBytes(&n), 2);
        },
        1 => {
            e.setBuffer(p.refs[1].buf, p.refs[1].off, 0);
            e.setBytes(std.mem.asBytes(&n), 1);
        },
        else => {
            e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
            e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
            e.setBytes(std.mem.asBytes(&n), 2);
        },
    }
    Probe.go(e, .{ k.blocks(n, 256), 1, 1 }, 256);
}

fn wordsCount(p: *Probe) usize {
    return (p.shape.n + 8) * 4;
}

fn wordsEval(_: *Probe, _: k.Mode, out: []R) !void {
    @memset(out, R.of(0));
}

fn wordsCheck(p: *Probe, _: k.g.Gate, out: []const u8) !?[]const u8 {
    const n = p.shape.n;
    const want = try p.a.dupe(u32, p.get(u32, 1));
    defer p.a.free(want);
    for (want[0..n], p.get(u32, 0)) |*w, s| w.* = if (p.variant == 1) 0 else s;
    return k.sameBytes(out, std.mem.sliceAsBytes(want));
}

const words_def: k.Def = .{ .prep = wordsPrep, .launch = wordsLaunch, .count = wordsCount, .eval = wordsEval, .check = wordsCheck };

fn raw(name: []const u8, served: []const c.Shape) c.Spec {
    return .{ .name = name, .accum = .none, .depth = k.noSums, .out = .raw, .in_place = true, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = served, .nan = .{ .stated = "moves 32-bit words: every byte as it was" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums" } };
}

pub const entries = [_]k.Entry{
    .{ .def = &swi_def, .spec = none("metal lfm3 omni om_swiglu halves", .bf16, &.{ .{ .rows = 94, .n = 1792, .k = 1 }, .{ .rows = 40, .n = 7680, .k = 1 } }), .file = .omni, .pipe = "om_swiglu", .variant = 0, .control = .{ "Y[i] = b(rb(x / (1.0f + precise::exp(-x))) * f(AC[ic]));", "Y[i] = b((x / (1.0f + precise::exp(-x))) * f(AC[ic]));" }, .what = "drops bf16(silu(a))" },
    .{ .def = &swi_def, .spec = none("metal lfm3 omni om_swiglu pairs", .bf16, &.{ .{ .rows = 40, .n = 7680, .k = 1 }, .{ .rows = 48, .n = 2304, .k = 1 } }), .file = .omni, .pipe = "om_swiglu", .variant = 1, .control = .{ "const long ic = a.inter ? ia + 1 : ia + a.F;", "const long ic = a.inter ? ia : ia + a.F;" }, .what = "reads a for c in the pairs layout" },
    .{ .def = &bias_def, .spec = none("metal lfm3 omni om_bias_act bias", .bf16, &.{ .{ .rows = 94, .n = 768, .k = 1 }, .{ .rows = 64, .n = 1282, .k = 1 } }), .file = .omni, .pipe = "om_bias_act", .variant = 0, .control = .{ "  v = rb(v);\n  if (a.act == 1)", "  v = rb(v * 1.0078125f);\n  if (a.act == 1)" }, .what = "scales the sum" },
    .{ .def = &bias_def, .spec = none("metal lfm3 omni om_bias_act row bias, SiLU", .bf16, &.{.{ .rows = 256, .n = 1632, .k = 1 }}), .file = .omni, .pipe = "om_bias_act", .variant = 1, .control = .{ "float v = A[i] + (a.has_bias ? f(Bias[a.per_row ? r : c]) : 0.0f);", "float v = A[i] + (a.has_bias ? f(Bias[a.per_row ? c % a.rows : c]) : 0.0f);" }, .what = "takes the row bias by column" },
    .{ .def = &bias_def, .spec = none("metal lfm3 omni om_bias_act GELU", .bf16, &.{.{ .rows = 94, .n = 2048, .k = 1 }}), .file = .omni, .pipe = "om_bias_act", .variant = 2, .control = .{ "else if (a.act == 2) v = 0.5f * v * (1.0f + erf_f(v * 0.70710678118654752f));", "else if (a.act == 2) v = 0.5f * v * (1.0f + rb(erf_f(v * 0.70710678118654752f)));" }, .what = "adds bf16(erf)" },
    .{ .def = &cast_def, .spec = none("metal lfm3 omni om_f32_bf16", .bf16, &.{ .{ .rows = 1, .n = 128 * 3000, .k = 1 }, .{ .rows = 1, .n = 1000, .k = 1 } }), .file = .omni, .pipe = "om_f32_bf16", .variant = 0, .control = .{ "if (gid < n) Y[gid] = b(X[gid]);", "if (gid < n) Y[gid] = bf(as_type<uint>(X[gid]) >> 16);" }, .what = "truncates instead of rounding" },
    .{ .def = &cast_def, .spec = none("metal lfm3 omni om_bf16_f32", .f32, &.{ .{ .rows = 1, .n = 12 * 2017, .k = 1 }, .{ .rows = 1, .n = 1000, .k = 1 } }), .file = .omni, .pipe = "om_bf16_f32", .variant = 1, .control = .{ "if (gid < n) Y[gid] = f(X[gid]);", "if (gid < n) Y[gid] = f(X[gid ^ 1]);" }, .what = "swaps neighbours" },
    .{ .def = &fsq_def, .spec = .{ .name = "metal lfm3 omni om_fsq_decode", .accum = .none, .depth = k.noSums, .out = .bf16, .served = &.{ .{ .rows = 200, .n = 12, .k = 1 }, .{ .rows = 1, .n = 12, .k = 1 } }, .nan = .{ .stated = "integer codes in: no non-finite input; every latent inside its replay's range" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums" } }, .file = .omni, .pipe = "om_fsq_decode", .control = .{ "const int levels[4] = {8, 7, 6, 6};", "const int levels[4] = {8, 7, 6, 5};" }, .what = "a wrong level" },
    .{ .def = &polar_def, .spec = none("metal lfm3 omni om_polar", .f32, &.{ .{ .rows = 48, .n = 641, .k = 1 }, .{ .rows = 6, .n = 641, .k = 1 } }), .file = .omni, .pipe = "om_polar", .control = .{ "S[t * 2 * a.NB + k] = m * precise::cos(ph);", "S[t * 2 * a.NB + k] = rb(m) * precise::cos(ph);" }, .what = "adds bf16(magnitude)" },
    .{ .def = &frames_def, .spec = none("metal lfm3 omni om_mel_frames", .f32, &.{.{ .rows = 300, .n = 400, .k = 160 }}), .file = .gemm, .pipe = "om_mel_frames", .control = .{ "FR[i] = s < a.len ? X[s] * WIN[n] : 0.0f;", "FR[i] = s < a.len ? X[s] * WIN[n ^ 1] : 0.0f;" }, .what = "swaps window taps" },
    .{ .def = &power_def, .spec = .{ .name = "metal lfm3 omni om_power", .accum = .f32, .depth = depth3, .ops = 6, .out = .f32, .served = &.{.{ .rows = 300, .n = 201, .k = 2 }} }, .file = .gemm, .pipe = "om_power", .control = .{ "P[i] = m * m;", "P[i] = rb(m) * m;" }, .what = "adds bf16(magnitude)" },
    .{ .def = &logmel_def, .spec = .{ .name = "metal lfm3 omni om_logmel", .accum = .none, .depth = k.noSums, .out = .f32, .served = &.{.{ .rows = 300, .n = 128, .k = 1 }}, .nan = .{ .stated = "metal::max takes the floor for a NaN logarithm; infinities as float64" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums" } }, .file = .gemm, .pipe = "om_logmel", .control = .{ "(max(precise::log10(M[i]), a.floor_) + 4.0f) / 4.0f", "(precise::log10(M[i]) + 4.0f) / 4.0f" }, .what = "drops the floor" },
    .{ .def = &words_def, .spec = raw("metal lfm3 omni om_set_words", &.{ .{ .rows = 1, .n = 36, .k = 1 }, .{ .rows = 1, .n = 1024, .k = 1 } }), .file = .omni, .pipe = "om_set_words", .variant = 0, .control = .{ "if (gid < n) D[gid] = V[gid];", "if (gid < n) D[gid] = V[gid / 2];" }, .what = "reads every other word" },
    .{ .def = &words_def, .spec = raw("metal lfm3 omni om_zero_words", &.{ .{ .rows = 1, .n = 36, .k = 1 }, .{ .rows = 1, .n = 3000, .k = 1 } }), .file = .omni, .pipe = "om_zero_words", .variant = 1, .control = .{ "if (gid < n) D[gid] = 0;", "if (gid + 1 < n) D[gid] = 0;" }, .what = "leaves the last word" },
    .{ .def = &words_def, .spec = raw("metal lfm3 omni om_copy_words", &.{ .{ .rows = 1, .n = 36, .k = 1 }, .{ .rows = 1, .n = 3000, .k = 1 } }), .file = .omni, .pipe = "om_copy_words", .variant = 2, .control = .{ "if (gid < n) D[gid] = S[gid];", "if (gid < n) D[gid] = S[gid] ^ (gid == n - 1 ? 1u : 0u);" }, .what = "flips a bit of the last word" },
};
