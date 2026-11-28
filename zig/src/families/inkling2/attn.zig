//! Inkling 2's attention on the host: the formula in float64, and in fp32 at Transformers' eager rounding points.
const std = @import("std");
const bf16_round = @import("bf16_round");

/// The widest head, the most bias states and the most keys of a row for `attend`'s fp32 model.
pub const most_hd = 128;
pub const most_rel = 32;
pub const most_keys = 8192;

/// `rows` rows from position `pos0` over the cache's positions 0 to the last row's; `window` 0: all keys.
pub const Shape = struct { rows: usize, pos0: usize, heads: usize, kv_heads: usize, hd: usize, drel: usize, rext: usize, window: usize };

/// The bf16 inputs as fp32: q, gate [rows, heads, hd]; k, v [pos0 + rows, kv_heads, hd]; rel; proj [drel, rext]; sink.
pub const Inputs = struct { q: []const f32, k: []const f32, v: []const f32, rel: []const f32, proj: []const f32, sink: []const f32, gate: []const f32 };

pub const Error = error{AttnShape};

/// Refuses empty sizes, ungrouped heads, sizes past the limits above, over 65535 rows and arrays of other lengths.
pub fn shaped(s: Shape, in: Inputs, out_len: usize) Error!void {
    return shapedWithin(s, in, out_len, most_keys);
}

/// `shaped` with `keys` as the most keys of a row.
pub fn shapedWithin(s: Shape, in: Inputs, out_len: usize, keys: usize) Error!void {
    if (s.heads == 0 or s.kv_heads == 0 or s.heads % s.kv_heads != 0) return error.AttnShape;
    if (s.hd == 0 or s.hd > most_hd or s.drel == 0 or s.drel > most_rel or s.rext == 0) return error.AttnShape;
    if (s.rows == 0 or s.rows > 65535) return error.AttnShape;
    const pos = s.pos0 + s.rows;
    if ((if (s.window == 0) pos else s.window) > keys) return error.AttnShape;
    const out = s.rows * s.heads * s.hd;
    const cache = pos * s.kv_heads * s.hd;
    if (out_len != out or in.q.len != out or in.gate.len != out or in.k.len != cache or in.v.len != cache) return error.AttnShape;
    if (in.rel.len != s.rows * s.heads * s.drel or in.proj.len != s.drel * s.rext or in.sink.len != s.rows * s.heads) return error.AttnShape;
}

/// The first key a row at position `t` attends.
pub fn first(s: Shape, t: usize) usize {
    return if (s.window > 0 and t + 1 > s.window) t + 1 - s.window else 0;
}

/// The reference's scale, head_dim ** -0.5, for the arithmetic type T.
pub fn scaleOf(comptime T: type, hd: usize) T {
    return @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
}

/// A warp's xor-tree sum of its 32 lanes: every lane ends with the same bits.
fn tree(lane: *[32]f32) f32 {
    var o: usize = 16;
    while (o > 0) : (o >>= 1) {
        const before = lane.*;
        for (lane, 0..) |*l, i| l.* = before[i] + before[i ^ o];
    }
    return lane[0];
}

fn rnd(rounded: bool, x: f32) f32 {
    return if (rounded) bf16_round.round32(x) else x;
}

/// The formula in float64 (T = f64), or fp32 in the eager model's order of sums, rounded as Transformers if `rounded`.
pub fn attend(comptime T: type, rounded: bool, s: Shape, in: Inputs, out: []T) Error!void {
    try shaped(s, in, out.len);
    for (0..s.rows) |row| for (0..s.heads) |h| head(T, rounded, s, in, row, h, out[(row * s.heads + h) * s.hd ..][0..s.hd]);
}

fn head(comptime T: type, rounded: bool, s: Shape, in: Inputs, row: usize, h: usize, out: []T) void {
    const t = s.pos0 + row;
    const lo = first(s, t);
    const nk = t - lo + 1;
    const kh = h / (s.heads / s.kv_heads);
    const hrow = row * s.heads + h;
    const q = in.q[hrow * s.hd ..][0..s.hd];
    const rel = in.rel[hrow * s.drel ..][0..s.drel];
    var lg: [most_keys]T = undefined;
    for (0..nk) |j| {
        const key = lo + j;
        const k = in.k[(key * s.kv_heads + kh) * s.hd ..][0..s.hd];
        var l: T = scaledDot(T, rounded, q, k, scaleOf(T, s.hd));
        const dist = t - key;
        if (dist < s.rext) l = biased(T, rounded, l, rel, in.proj, s.rext, dist);
        lg[j] = l;
    }
    const sink: T = in.sink[hrow];
    var m: T = sink;
    for (lg[0..nk]) |l| m = @max(m, l); // a NaN logit is passed by, as fmaxf does; its exp below makes the sum NaN
    var z: T = softmaxSum(T, lg[0..nk], m);
    z += @exp(sink - m);
    for (lg[0..nk]) |*e| e.* = if (T == f32) rnd(rounded, e.* / z) else e.* / z;
    for (out, 0..) |*o, d| {
        var acc: T = 0;
        for (0..nk) |j| acc = @mulAdd(T, lg[j], in.v[((lo + j) * s.kv_heads + kh) * s.hd + d], acc);
        const g: T = in.gate[hrow * s.hd + d];
        const y: T = if (T == f32) rnd(rounded, acc) else acc;
        const gated = y * (1 / (1 + @exp(-g)));
        o.* = if (T == f32) rnd(rounded, gated) else gated;
    }
}

/// (q . k) * scale: float64 in a chain, or fp32 as the eager model adds: lane l over l, l + 32, ..., then the tree.
fn scaledDot(comptime T: type, rounded: bool, q: []const f32, k: []const f32, scale: T) T {
    if (T == f64) {
        var sum: f64 = 0;
        for (q, k) |a, b| sum += @as(f64, a) * @as(f64, b);
        return sum * scale;
    }
    var lane: [32]f32 = @splat(0);
    for (&lane, 0..) |*l, i| {
        var j = i;
        while (j < q.len) : (j += 32) l.* = @mulAdd(f32, q[j], k[j], l.*);
    }
    return rnd(rounded, rnd(rounded, tree(&lane)) * scale);
}

/// l plus the bias r[dist] = rel . proj[:, dist]: float64, or fp32 with r a product per lane and the tree.
fn biased(comptime T: type, rounded: bool, l: T, rel: []const f32, proj: []const f32, rext: usize, dist: usize) T {
    if (T == f64) {
        var r: f64 = 0;
        for (rel, 0..) |a, j| r += @as(f64, a) * @as(f64, proj[j * rext + dist]);
        return l + r;
    }
    var lane: [32]f32 = @splat(0);
    for (rel, 0..) |a, j| lane[j] = a * proj[j * rext + dist];
    return rnd(rounded, l + rnd(rounded, tree(&lane)));
}

/// The keys' exp(l - m), left in `lg`, and their sum: float64 in a chain; fp32 as the block adds (threads, warps).
fn softmaxSum(comptime T: type, lg: []T, m: T) T {
    if (T == f64) {
        var sum: f64 = 0;
        for (lg) |*l| {
            l.* = @exp(l.* - m);
            sum += l.*;
        }
        return sum;
    }
    var thread: [128]f32 = @splat(0);
    for (lg, 0..) |*l, j| {
        l.* = @exp(l.* - m);
        thread[j % 128] += l.*;
    }
    var warp: [4]f32 = undefined;
    for (&warp, 0..) |*w, i| {
        var lane: [32]f32 = thread[32 * i ..][0..32].*;
        w.* = tree(&lane);
    }
    return (warp[0] + warp[1]) + (warp[2] + warp[3]);
}

test "attend: one key and no bias leave the softmax a sink and a key; the gate scales the output" {
    const s: Shape = .{ .rows = 1, .pos0 = 0, .heads = 1, .kv_heads = 1, .hd = 2, .drel = 1, .rext = 4, .window = 0 };
    const in: Inputs = .{ .q = &.{ 1, 2 }, .k = &.{ 3, 4 }, .v = &.{ 10, 20 }, .rel = &.{0}, .proj = &.{ 0, 0, 0, 0 }, .sink = &.{0}, .gate = &.{ 0, 0 } };
    var y: [2]f64 = undefined;
    try attend(f64, false, s, in, &y);
    // logit = (1 * 3 + 2 * 4) / sqrt(2); p = e^l / (e^l + e^0); sigmoid(0) = 1/2
    const l: f64 = 11.0 / @sqrt(2.0);
    const p = @exp(l) / (@exp(l) + 1);
    try std.testing.expectApproxEqAbs(p * 10 / 2, y[0], 1e-12);
    try std.testing.expectApproxEqAbs(p * 20 / 2, y[1], 1e-12);
}

test "attend: the bias adds for distances under rext and not at or past it" {
    // two keys (positions 0 and 1) for the row at position 1: distances 1 and 0; rext 1 gives a bias to distance 0 only
    const s: Shape = .{ .rows = 1, .pos0 = 1, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 1, .window = 0 };
    const in: Inputs = .{ .q = &.{0}, .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .rel = &.{1}, .proj = &.{3}, .sink = &.{-100}, .gate = &.{100} };
    var y: [1]f64 = undefined;
    try attend(f64, false, s, in, &y);
    // logits: 0 for position 0 (distance 1: no bias), 3 for position 1 (distance 0); sink and gate do not matter
    const p1 = @exp(@as(f64, 3)) / (@exp(@as(f64, 3)) + 1);
    try std.testing.expectApproxEqAbs((1 - p1) * 1 + p1 * 2, y[0], 1e-12);
}

test "attend: the window drops keys older than its width; a row at the window's edge keeps exactly `window` keys" {
    const s: Shape = .{ .rows = 1, .pos0 = 3, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 8, .window = 2 };
    // keys at positions 0..3, all logits 0: the row at 3 keeps positions 2 and 3, values 30 and 40
    const zeros: [8]f32 = @splat(0);
    const in: Inputs = .{ .q = &.{0}, .k = &.{ 0, 0, 0, 0 }, .v = &.{ 10, 20, 30, 40 }, .rel = &.{0}, .proj = &zeros, .sink = &.{-100}, .gate = &.{100} };
    var y: [1]f64 = undefined;
    try attend(f64, false, s, in, &y);
    try std.testing.expectApproxEqAbs(@as(f64, 35), y[0], 1e-9);
    var wide = s;
    wide.window = 3;
    try attend(f64, false, wide, in, &y);
    try std.testing.expectApproxEqAbs(@as(f64, 30), y[0], 1e-9);
}

test "attend: grouped heads read their group's keys" {
    const s: Shape = .{ .rows = 1, .pos0 = 0, .heads = 4, .kv_heads = 2, .hd = 1, .drel = 1, .rext = 1, .window = 0 };
    const in: Inputs = .{ .q = &.{ 0, 0, 0, 0 }, .k = &.{ 0, 0 }, .v = &.{ 1, 2 }, .rel = &.{ 0, 0, 0, 0 }, .proj = &.{0}, .sink = &.{ -100, -100, -100, -100 }, .gate = &.{ 100, 100, 100, 100 } };
    var y: [4]f64 = undefined;
    try attend(f64, false, s, in, &y);
    try std.testing.expectApproxEqAbs(@as(f64, 1), y[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1), y[1], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2), y[2], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2), y[3], 1e-9);
}

test "attend: a NaN logit or sink makes the head NaN; an infinite value under a weight stays infinite" {
    const nan = std.math.nan(f32);
    const s: Shape = .{ .rows = 1, .pos0 = 1, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 2, .window = 0 };
    const base: Inputs = .{ .q = &.{1}, .k = &.{ 1, 1 }, .v = &.{ 1, 2 }, .rel = &.{0}, .proj = &.{ 0, 0 }, .sink = &.{0}, .gate = &.{0} };
    inline for (.{ f64, f32 }) |T| {
        var y: [1]T = undefined;
        var in = base;
        in.sink = &.{nan};
        try attend(T, false, s, in, &y);
        try std.testing.expect(std.math.isNan(y[0]));
        in = base;
        in.k = &.{ nan, 1 };
        try attend(T, false, s, in, &y);
        try std.testing.expect(std.math.isNan(y[0]));
        in = base;
        in.v = &.{ std.math.inf(f32), 2 };
        try attend(T, false, s, in, &y);
        try std.testing.expect(std.math.isInf(y[0]));
    }
}

test "attend: a +inf sink makes the head NaN" {
    const s: Shape = .{ .rows = 1, .pos0 = 1, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 2, .window = 0 };
    const base: Inputs = .{ .q = &.{1}, .k = &.{ 1, 1 }, .v = &.{ 1, 2 }, .rel = &.{0}, .proj = &.{ 0, 0 }, .sink = &.{0}, .gate = &.{0} };
    inline for (.{ f64, f32 }) |T| {
        var y: [1]T = undefined;
        var in = base;
        in.sink = &.{std.math.inf(f32)}; // m is infinite and exp(inf - inf) is NaN, so the denominator is NaN
        try attend(T, false, s, in, &y);
        try std.testing.expect(std.math.isNan(y[0]));
        in.sink = &.{0}; // the same inputs with a finite sink are finite
        try attend(T, false, s, in, &y);
        try std.testing.expect(std.math.isFinite(y[0]));
    }
}

test "attend: a -inf sink adds nothing to the softmax" {
    const s: Shape = .{ .rows = 1, .pos0 = 1, .heads = 1, .kv_heads = 1, .hd = 1, .drel = 1, .rext = 2, .window = 0 };
    // two equal logits of 1 (no bias): each weight 1/2, the output (1 + 2) / 2, the gate 0 halves it
    const base: Inputs = .{ .q = &.{1}, .k = &.{ 1, 1 }, .v = &.{ 1, 2 }, .rel = &.{0}, .proj = &.{ 0, 0 }, .sink = &.{0}, .gate = &.{0} };
    inline for (.{ f64, f32 }) |T| {
        var y: [1]T = undefined;
        var in = base;
        in.sink = &.{-std.math.inf(f32)}; // exp(-inf - m) is 0: the output of the keys alone
        try attend(T, false, s, in, &y);
        try std.testing.expectApproxEqAbs(@as(T, 0.75), y[0], 1e-6);
        in.sink = &.{0}; // a finite sink of 0 does take weight
        try attend(T, false, s, in, &y);
        try std.testing.expect(y[0] < 0.7 and y[0] > 0.6);
    }
}

const big_zeros: [1 << 20]f32 = @splat(0);

/// Arrays of the lengths `s` needs, so a refusal can only come from the shape's own condition.
fn fitting(s: Shape) Inputs {
    const out = s.rows * s.heads * s.hd;
    const cache = (s.pos0 + s.rows) * s.kv_heads * s.hd;
    return .{ .q = big_zeros[0..out], .k = big_zeros[0..cache], .v = big_zeros[0..cache], .rel = big_zeros[0 .. s.rows * s.heads * s.drel], .proj = big_zeros[0 .. s.drel * s.rext], .sink = big_zeros[0 .. s.rows * s.heads], .gate = big_zeros[0..out] };
}

test "attn: shaped refuses each bad shape on its own condition, with arrays that fit the shape" {
    const ok: Shape = .{ .rows = 1, .pos0 = 0, .heads = 2, .kv_heads = 1, .hd = 2, .drel = 1, .rext = 2, .window = 0 };
    try shaped(ok, fitting(ok), 4);
    const bad = error.AttnShape;
    var s = ok;
    s.heads = 0;
    try std.testing.expectError(bad, shaped(s, fitting(s), 0));
    s = ok;
    s.kv_heads = 0;
    try std.testing.expectError(bad, shaped(s, fitting(s), 4));
    s = ok;
    s.heads = 3; // not in whole groups
    s.kv_heads = 2;
    try std.testing.expectError(bad, shaped(s, fitting(s), 6));
    s = ok;
    s.hd = 0;
    try std.testing.expectError(bad, shaped(s, fitting(s), 0));
    s = ok;
    s.hd = most_hd + 1;
    try std.testing.expectError(bad, shaped(s, fitting(s), 2 * (most_hd + 1)));
    s = ok;
    s.drel = 0;
    try std.testing.expectError(bad, shaped(s, fitting(s), 4));
    s = ok;
    s.drel = most_rel + 1;
    try std.testing.expectError(bad, shaped(s, fitting(s), 4));
    s = ok;
    s.rext = 0;
    try std.testing.expectError(bad, shaped(s, fitting(s), 4));
    s = ok;
    s.rows = 0;
    try std.testing.expectError(bad, shaped(s, fitting(s), 0));
    s = ok;
    s.rows = 65536;
    s.window = 4; // a window keeps the keys few: only the rows are too many
    try std.testing.expectError(bad, shaped(s, fitting(s), 65536 * 4));
    s = ok;
    s.pos0 = most_keys; // a whole cache over the most keys
    try std.testing.expectError(bad, shaped(s, fitting(s), 4));
    s = ok;
    s.window = most_keys + 1;
    try std.testing.expectError(bad, shaped(s, fitting(s), 4));
    // the lengths, each alone
    const fit = fitting(ok);
    try std.testing.expectError(bad, shaped(ok, fit, 3));
    var short = fit;
    short.q = big_zeros[0..3];
    try std.testing.expectError(bad, shaped(ok, short, 4));
    short = fit;
    short.k = big_zeros[0..1];
    try std.testing.expectError(bad, shaped(ok, short, 4));
    short = fit;
    short.v = big_zeros[0..1];
    try std.testing.expectError(bad, shaped(ok, short, 4));
    short = fit;
    short.rel = big_zeros[0..1];
    try std.testing.expectError(bad, shaped(ok, short, 4));
    short = fit;
    short.proj = big_zeros[0..1];
    try std.testing.expectError(bad, shaped(ok, short, 4));
    short = fit;
    short.sink = big_zeros[0..1];
    try std.testing.expectError(bad, shaped(ok, short, 4));
    short = fit;
    short.gate = big_zeros[0..3];
    try std.testing.expectError(bad, shaped(ok, short, 4));
}
