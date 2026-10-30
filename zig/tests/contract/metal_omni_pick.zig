//! LFM3-Omni's Metal picks: the greedy text pick's two stages, the depth decoder's code picks and a frame's end.
const std = @import("std");
const mtl = @import("metal");
const k = @import("metal_omni_common.zig");

const c = k.c;
const R = k.R;
const Probe = k.Probe;
const nq = 12;
const size = 2017;
const eos = 2016;
const hist_cap = 64;
const none_at: i32 = 0x7fffffff;

/// A logit with few distinct values (ties everywhere), NaN and infinities sprinkled when asked.
fn logit(rng: std.Random, fill: k.g.Fill) u16 {
    if (fill == .nonfinite and rng.uintLessThan(u32, 400) < 3) return ([3]u16{ k.nan16, k.inf16, k.ninf16 })[rng.uintLessThan(u32, 3)];
    const v = @as(f32, @floatFromInt(rng.uintLessThan(u32, 61))) / 8 - 4;
    return if (v == 0 and rng.boolean()) 0x8000 else c.toBf16(v);
}

/// Whether `pick` is a valid greedy choice: none surely larger, lowest id on a sure tie, never NaN, else 2^31 - 1.
fn valid(pick: i32, lo: []const f64, hi: []const f64, skip: []const bool) bool {
    var any = false;
    for (lo, skip) |l, s| any = any or (!s and !std.math.isNan(l));
    if (!any) return pick == none_at;
    if (pick < 0 or pick >= lo.len) return false;
    const p: usize = @intCast(pick);
    if (skip[p] or std.math.isNan(lo[p])) return false;
    for (lo, hi, skip, 0..) |l, h, s, j| {
        if (j == p or s or std.math.isNan(l)) continue;
        if (l > hi[p]) return false;
        if (j < p and l == h and lo[p] == hi[p] and l == hi[p]) return false;
    }
    return true;
}

// --- om_argmax_part (shape n logits, k a part's width) and om_argmax_final (n parts)

const PartArgs = extern struct { N: i32, ns: i32, per: i32 };
const sup = [4]i32{ 125031, 125032, 125030, 125029 };

fn partPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n == 0 or s.k == 0) return false;
    const parts = k.blocks(s.n, s.k);
    const x = try p.alloc(u16, 0, s.n);
    const ids = try p.alloc(i32, 1, 4);
    _ = try p.alloc(f32, 2, parts);
    _ = try p.alloc(i32, 3, parts);
    for (x) |*v| v.* = logit(rng, p.fill);
    // the largest logits on the excluded ids, and a part left with nothing but excluded and NaN ids
    @memcpy(ids, &sup);
    for (sup) |id| if (id < s.n) {
        x[@intCast(id)] = c.toBf16(9);
    };
    if (p.fill == .nonfinite and s.n >= 2 * s.k) @memset(x[s.k .. 2 * s.k], k.nan16);
    try p.in(0);
    try p.in(1);
    try p.out(2, false);
    try p.out(3, false);
    return true;
}

fn partLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    p.bind(e, &.{ 0, 1, 2, 3 }, PartArgs{ .N = k.i32_(s.n), .ns = 4, .per = k.i32_(s.k) });
    Probe.go(e, .{ k.blocks(s.n, s.k), 1, 1 }, 256);
}

fn partCount(p: *Probe) usize {
    return 8 * k.blocks(p.shape.n, p.shape.k);
}

fn zeroEval(_: *Probe, _: k.Mode, out: []R) !void {
    @memset(out, R.of(0));
}

fn partCheck(p: *Probe, _: k.g.Gate, out: []const u8) !?[]const u8 {
    const s = p.shape;
    const parts = k.blocks(s.n, s.k);
    const x = p.get(u16, 0);
    const pv = std.mem.bytesAsSlice(f32, out[0 .. 4 * parts]);
    const pi = std.mem.bytesAsSlice(i32, out[4 * parts ..]);
    var lo: [4096]f64 = undefined;
    var skip: [4096]bool = undefined;
    for (0..parts) |g| {
        const a = g * s.k;
        const b = @min(s.n, a + s.k);
        for (a..b) |i| {
            lo[i - a] = c.fromBf16(x[i]);
            skip[i - a] = std.mem.indexOfScalar(i32, &sup, @intCast(i)) != null;
        }
        const pick = if (pi[g] == none_at) none_at else pi[g] - @as(i32, @intCast(a));
        if (!valid(pick, lo[0 .. b - a], lo[0 .. b - a], skip[0 .. b - a])) return "a part's pick is not its lowest-id largest";
        const want: f32 = if (pi[g] == none_at) -std.math.inf(f32) else c.fromBf16(x[@intCast(pi[g])]);
        if (@as(u32, @bitCast(pv[g])) != @as(u32, @bitCast(want))) return "a part's value is not its pick's logit";
    }
    return null;
}

const part_def: k.Def = .{ .prep = partPrep, .launch = partLaunch, .count = partCount, .eval = zeroEval, .check = partCheck };

fn finalPrep(p: *Probe, rng: std.Random) !bool {
    const n = p.shape.n;
    if (n == 0) return false;
    const pv = try p.alloc(f32, 0, n);
    const pi = try p.alloc(i32, 1, n);
    _ = try p.alloc(i32, 2, 1);
    for (pv, pi, 0..) |*v, *i, j| {
        v.* = c.fromBf16(logit(rng, if (p.fill == .nonfinite) .nonfinite else .random));
        i.* = @intCast(j * 1024 + rng.uintLessThan(u32, 1024));
    }
    // the same best in two parts of one lane, the lower id first, and in a third part of another lane
    if (n > 40) {
        pv[1] = 9;
        pv[33] = 9;
        pv[n - 1] = 9;
        pi[1] = 5;
        pi[33] = 2000;
        pi[n - 1] = 7;
    }
    try p.in(0);
    try p.in(1);
    try p.out(2, false);
    return true;
}

fn finalLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1, 2 }, @as(i32, k.i32_(p.shape.n)));
    Probe.go(e, .{ 1, 1, 1 }, 32);
}

fn four(_: *Probe) usize {
    return 4;
}

fn finalCheck(p: *Probe, _: k.g.Gate, out: []const u8) !?[]const u8 {
    const pv = p.get(f32, 0);
    const pi = p.get(i32, 1);
    var best = -std.math.inf(f32);
    var at: i32 = none_at;
    for (pv, pi) |v, i| if (v > best or (v == best and i < at)) {
        best = v;
        at = i;
    };
    return k.sameBytes(out, std.mem.asBytes(&at));
}

const final_def: k.Def = .{ .prep = finalPrep, .launch = finalLaunch, .count = four, .eval = zeroEval, .check = finalCheck };

// --- om_depth_pick: shape rows the pass's picks, n the codebook's size, k the window's frames; variant 1 sampled

const PickArgs = extern struct { p0: i32, hn: i32, penalty: f32, temp: f32, seed: u64, frame: i32, NQ: i32, size: i32, pad: i32 = 0 };
const penalty: f32 = 1.2;
const temp: f32 = 0.4;
const seed: u64 = 1003;
const frame = 37;
const p0 = 4;

fn mix64(z0: u64) u64 {
    var z = z0 +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn pickPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.n != size or s.rows == 0 or p0 + s.rows > nq or s.k > hist_cap) return false;
    const log = try p.alloc(u16, 0, nq * size);
    const frame_ = try p.alloc(i32, 1, 3 * nq);
    const hist = try p.alloc(i32, 2, hist_cap * nq);
    for (log) |*v| v.* = logit(rng, p.fill);
    for (0..nq) |q| frame_[q] = @intCast(q);
    rng.shuffle(i32, frame_[0..nq]);
    for (frame_[nq .. 2 * nq]) |*x| x.* = 0;
    for (frame_[2 * nq ..]) |*x| x.* = 1;
    for (hist) |*h| h.* = @intCast(rng.uintLessThan(u32, 64));
    // a repeated code among each codebook's largest logits, and exact ties on unrepeated codes
    for (0..nq) |q| {
        log[q * size + 3] = c.toBf16(5);
        log[q * size + 70] = c.toBf16(5);
        log[q * size + 90] = c.toBf16(5);
    }
    try p.in(0);
    try p.in(2);
    try p.out(1, true);
    return true;
}

fn pickLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const codes = Probe.at(p.refs[1], nq * 4);
    const masked = Probe.at(p.refs[1], 2 * nq * 4);
    e.setPipeline(p.pipe);
    e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
    e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
    e.setBuffer(p.refs[2].buf, p.refs[2].off, 2);
    e.setBuffer(codes.buf, codes.off, 3);
    e.setBuffer(masked.buf, masked.off, 4);
    const a = PickArgs{ .p0 = p0, .hn = k.i32_(s.k), .penalty = penalty, .temp = if (p.variant == 1) temp else 0, .seed = seed, .frame = frame, .NQ = nq, .size = size };
    e.setBytes(std.mem.asBytes(&a), 5);
    Probe.go(e, .{ s.rows, 1, 1 }, 256);
}

fn pickCount(_: *Probe) usize {
    return 3 * nq * 4;
}

fn ln(x: f64) f64 {
    return @log(x);
}

fn pickCheck(p: *Probe, _: k.g.Gate, out: []const u8) !?[]const u8 {
    const s = p.shape;
    const got = std.mem.bytesAsSlice(i32, out);
    const init = p.get(i32, 1);
    const log = p.get(u16, 0);
    const hist = p.get(i32, 2);
    var lo: [size]f64 = undefined;
    var hi: [size]f64 = undefined;
    var skip: [size]bool = @splat(false);
    for (0..nq) |j| if (got[j] != init[j]) return "the order was changed";
    for (0..nq) |q| {
        const picked = std.mem.indexOfScalar(i32, init[p0 .. p0 + s.rows], @intCast(q)) != null;
        if (!picked) {
            if (got[nq + q] != init[nq + q] or got[2 * nq + q] != init[2 * nq + q]) return "an unpicked codebook was written";
            continue;
        }
        if (got[2 * nq + q] != 0) return "a picked codebook stays masked";
        var seen: [size]bool = @splat(false);
        for (0..s.k) |j| seen[@intCast(hist[j * nq + q])] = true;
        for (0..size) |i| {
            var v = R.bf16(log[q * size + i]);
            if (seen[i]) v = if (v.v < 0) v.mul(R.of(penalty)) else v.div(R.of(penalty));
            if (p.variant == 1) {
                const h = mix64(seed ^ mix64((@as(u64, frame) << 32) ^ (@as(u64, q) << 16) ^ @as(u64, i)));
                const u = (@as(f64, @floatFromInt(h >> 40)) + 0.5) * (1.0 / 16777216.0);
                const g = R.of(u).fp(0.5).up(ln, k.fn_ulps).neg().lip(ln, 1 / (-@log(u) * (1 - 1e-6)), k.fn_ulps);
                v = v.div(R.of(temp)).sub(g);
            }
            lo[i] = v.lo;
            hi[i] = v.hi;
            if (std.math.isNan(v.v)) {
                lo[i] = std.math.nan(f64);
            }
        }
        if (!valid(got[nq + q], &lo, &hi, &skip)) {
            std.debug.print("  codebook {d}: picked {d}\n", .{ q, got[nq + q] });
            return "a code is not the lowest-id largest of its penalized (and noised) logits";
        }
    }
    return null;
}

const pick_def: k.Def = .{ .prep = pickPrep, .launch = pickLaunch, .count = pickCount, .eval = zeroEval, .check = pickCheck };

// --- om_frame_end: variant 0 a frame without end-of-speech (pushed at a slot), 1 a frame with one

const EndArgs = extern struct { NQ: i32, eos: i32, slot: i32, push: i32 };

fn endPrep(p: *Probe, rng: std.Random) !bool {
    const codes = try p.alloc(i32, 0, nq);
    const hist = try p.alloc(i32, 1, hist_cap * nq);
    for (codes) |*x| x.* = @intCast(rng.uintLessThan(u32, eos));
    for (hist) |*x| x.* = @intCast(rng.uintLessThan(u32, eos));
    if (p.variant == 1) codes[0] = eos;
    try p.out(0, true);
    try p.out(1, true);
    return true;
}

fn endLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    p.bind(e, &.{ 0, 1 }, EndArgs{ .NQ = nq, .eos = eos, .slot = k.i32_(p.shape.k), .push = 1 });
    Probe.go(e, .{ 1, 1, 1 }, 32);
}

fn endCount(_: *Probe) usize {
    return (nq + hist_cap * nq) * 4;
}

fn endCheck(p: *Probe, _: k.g.Gate, out: []const u8) !?[]const u8 {
    const want = try p.a.alloc(i32, nq + hist_cap * nq);
    defer p.a.free(want);
    @memcpy(want[0..nq], p.get(i32, 0));
    @memcpy(want[nq..], p.get(i32, 1));
    if (std.mem.indexOfScalar(i32, want[0..nq], eos) != null) @memset(want[0..nq], eos) else @memcpy(want[nq + p.shape.k * nq ..][0..nq], want[0..nq]);
    return k.sameBytes(out, std.mem.sliceAsBytes(want));
}

const end_def: k.Def = .{ .prep = endPrep, .launch = endLaunch, .count = endCount, .eval = zeroEval, .check = endCheck };

fn stated(name: []const u8, rule: []const u8, served: []const c.Shape) c.Spec {
    return .{ .name = name, .accum = .none, .depth = k.noSums, .out = .raw, .in_place = true, .served = served, .nan = .{ .stated = rule }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "picks compare values: no sums" } };
}

const pick_rule = "a NaN logit is never picked; ties (and -Inf) to the lowest id; none left gives 2^31 - 1";

pub const entries = [_]k.Entry{
    .{ .def = &part_def, .spec = stated("metal lfm3 omni om_argmax_part", pick_rule, &.{ .{ .rows = 1, .n = 128000, .k = 1024 }, .{ .rows = 1, .n = 3000, .k = 1024 } }), .file = .rows, .pipe = "om_argmax_part", .control = .{ "if (v > best || (v == best && i < at)) {", "if (v > best || (v == best && i > at)) {" }, .what = "ties to the highest id" },
    .{ .def = &final_def, .spec = stated("metal lfm3 omni om_argmax_final", pick_rule, &.{ .{ .rows = 1, .n = 125, .k = 1 }, .{ .rows = 1, .n = 3, .k = 1 } }), .file = .rows, .pipe = "om_argmax_final", .control = .{ "if (PV[i] > best || (PV[i] == best && PI[i] < at)) {", "if (PV[i] >= best) {" }, .what = "ties to the last part" },
    .{ .def = &pick_def, .spec = stated("metal lfm3 omni om_depth_pick greedy", pick_rule, &.{ .{ .rows = 4, .n = size, .k = hist_cap }, .{ .rows = 4, .n = size, .k = 3 } }), .file = .gemm, .pipe = "om_depth_pick", .variant = 0, .control = .{ "if (v > best || (v == best && i < at)) {", "if (v > best || (v == best && i > at)) {" }, .what = "ties to the highest id" },
    .{ .def = &pick_def, .spec = stated("metal lfm3 omni om_depth_pick sampled", "a NaN logit is never drawn; equal noised values to the lowest id", &.{ .{ .rows = 4, .n = size, .k = hist_cap }, .{ .rows = 4, .n = size, .k = 0 } }), .file = .gemm, .pipe = "om_depth_pick", .variant = 1, .control = .{ "if (seen[i]) v = v < 0.0f ? v * a.penalty : v / a.penalty;", "if (seen[i]) v = v < 0.0f ? v / a.penalty : v * a.penalty;" }, .what = "inverts the repetition penalty" },
    .{ .def = &end_def, .spec = stated("metal lfm3 omni om_frame_end pushed", "integer codes: every word as the rule says", &.{ .{ .rows = 1, .n = nq, .k = 17 }, .{ .rows = 1, .n = nq, .k = 63 } }), .file = .gemm, .pipe = "om_frame_end", .variant = 0, .control = .{ "for (int q = 0; q < a.NQ; ++q) HIST[a.slot * a.NQ + q] = CODES[q];", "for (int q = 1; q < a.NQ; ++q) HIST[a.slot * a.NQ + q] = CODES[q];" }, .what = "pushes all but the first code" },
    .{ .def = &end_def, .spec = stated("metal lfm3 omni om_frame_end end of speech", "integer codes: every word as the rule says", &.{.{ .rows = 1, .n = nq, .k = 5 }}), .file = .gemm, .pipe = "om_frame_end", .variant = 1, .control = .{ "for (int q = 0; q < a.NQ; ++q) any |= CODES[q] == a.eos;", "for (int q = 1; q < a.NQ; ++q) any |= CODES[q] == a.eos;" }, .what = "misses an end-of-speech code in codebook 0" },
};
