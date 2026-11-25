//! The HIP loop kernels' host model: their named HIP baselines (W32 sums, Zig f32 math) and gate 4's simulation.
const std = @import("std");
const bf = @import("bf16_round");
const core = @import("core");
const K = @import("inkling2_loop_hip");

const threads = core.contract.reference_threads;
const Accum = core.contract.spec.Accum;
const Rows = core.contract.rows.Order;
const H = K.hidden;

/// Two fp32 outputs agree: both NaN, or the same bits.
pub fn same32(x: f32, y: f32) bool {
    return (std.math.isNan(x) and std.math.isNan(y)) or @as(u32, @bitCast(x)) == @as(u32, @bitCast(y));
}

/// The baselines' f32 functions the probe runs on the device: Zig 0.17's std.math.tanh, @exp and 1 / @sqrt.
pub fn zig(op: K.Math, x: f32) f32 {
    return switch (op) {
        .tanh, .device_tanh => std.math.tanh(x),
        .exp, .device_exp => @exp(x),
        .inv_sqrt, .device_rsqrt => 1 / @sqrt(x),
    };
}

/// Lanes of a W32 sum and elements of a chunk: element e goes to lane (e / 8) % 32.
pub const lanes = 32;
pub const chunk = 8;

/// A sum's order: the W32 baseline, or a control's (one serial unfused chain, butterfly reversed, unfused products).
pub const Order = enum { w32, serial, w32_desc, w32_unfused };

/// The longest chain of roundings in a W32 sum of n terms: n / 32 lane steps, then 5 butterfly adds.
pub fn depthW32(n: usize) usize {
    return n / lanes + 5;
}

/// Sum of a[e] * b[e] in `order`; W32: lane chains ascending from +0 by @mulAdd, then the xor butterfly 1, 2, 4, 8, 16.
pub fn sumIn(order: Order, a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len and a.len % (lanes * chunk) == 0);
    if (order == .serial) {
        var s: f32 = 0;
        for (a, b) |x, y| s = s + x * y;
        return s;
    }
    var v: [lanes]f32 = @splat(0);
    for (0..a.len / chunk) |c| for (c * chunk..(c + 1) * chunk) |e| {
        const l = c % lanes;
        v[l] = if (order == .w32_unfused) v[l] + a[e] * b[e] else @mulAdd(f32, a[e], b[e], v[l]);
    };
    const up = [5]usize{ 1, 2, 4, 8, 16 };
    const down = [5]usize{ 16, 8, 4, 2, 1 };
    for (if (order == .w32_desc) down else up) |o| {
        var n: [lanes]f32 = undefined;
        for (&n, 0..) |*x, l| x.* = v[l] + v[l ^ o];
        v = n;
    }
    return v[0];
}

/// HIP-RMS-W32, HIP-RETRIEVE-W32 and HIP-ZETA-W32's sum.
pub fn w32Sum(a: []const f32, b: []const f32) f32 {
    return sumIn(.w32, a, b);
}

/// rho's inverse from a row's sum of squares: the mean, plus eps, then 1 / sqrt, each an fp32 step.
pub fn inverse(squares: f32) f32 {
    const mean = squares / @as(f32, H);
    return 1 / @sqrt(mean + K.eps);
}

/// Where a call's sums run in another order: rho's sum of squares, and the dots (retrieve's logits, zeta).
pub const Orders = struct { rho: Order = .w32, dot: Order = .w32 };

/// One cross-pass message under HIP-RMS-W32 and HIP-RETRIEVE-W32: rho of both banks, logits, softmax and shares.
pub fn message(o: Orders, x0: []const f32, x1: []const f32, q: []const f32, out: []f32) void {
    var b: [2][H]f32 = undefined;
    var logit: [2]f32 = undefined;
    for ([2][]const f32{ x0, x1 }, &b, &logit) |x, *row, *l| {
        const inv = inverse(sumIn(o.rho, x, x));
        for (row, x) |*v, s| v.* = s * inv;
        l.* = sumIn(o.dot, q, row) * K.queryScale();
    }
    const top = @max(@max(-std.math.inf(f32), logit[0]), logit[1]);
    const e0 = @exp(logit[0] - top);
    const e1 = @exp(logit[1] - top);
    const den = (0 + e0) + e1;
    const w0 = e0 / den;
    const w1 = e1 / den;
    for (out, b[0], b[1]) |*m, v0, v1| m.* = (0 + w0 * v0) + w1 * v1;
}

/// A message call's host inputs: bf16 banks [rows][H], both whole query tensors [24][H], `count` layers from `first`.
pub const MessageCall = struct {
    bank0: []const u16,
    bank1: []const u16,
    qm: []const f32,
    kv: []const f32,
    first: usize,
    count: usize,
    rows: usize,
};

/// Core layer j's query for role r (q, k, v, mlp): q and mlp read q_mlp, k and v read kv, at rows 2j and 2j + 1.
pub fn query(c: MessageCall, j: usize, r: usize) []const f32 {
    const row = 2 * (c.first + j) + @intFromBool(r >= 2);
    return (if (r == 0 or r == 3) c.qm else c.kv)[row * H ..][0..H];
}

/// bf16 bits widened to f32.
pub fn widen(dst: []f32, src: []const u16) void {
    for (dst, src) |*d, s| d.* = bf.value(s);
}

const MsgJob = struct {
    c: *const MessageCall,
    lo: usize,
    hi: usize,
    out: []f32,

    pub fn run(j: *MsgJob) void {
        var x: [2][H]f32 = undefined;
        for (j.lo..j.hi) |t| {
            widen(&x[0], j.c.bank0[t * H ..][0..H]);
            widen(&x[1], j.c.bank1[t * H ..][0..H]);
            for (0..j.c.count) |l| for (0..4) |r| {
                message(.{}, &x[0], &x[1], query(j.c.*, l, r), j.out[((l * 4 + r) * j.c.rows + t) * H ..][0..H]);
            };
        }
    }
};

/// Every (core layer, role, row)'s message under the baselines, in the kernel's [message][rows][H] layout.
pub fn messages(c: MessageCall, out: []f32) !void {
    if (out.len != c.count * 4 * c.rows * H or c.qm.len != 24 * H or c.kv.len != 24 * H) return error.BadShape;
    var jobs: [threads.maximum]MsgJob = undefined;
    const n = threads.count(threads.maximum, c.rows);
    for (jobs[0..n], 0..) |*j, i| j.* = .{ .c = &c, .lo = c.rows * i / n, .hi = c.rows * (i + 1) / n, .out = out };
    threads.run(MsgJob, jobs[0..n], .normal);
}

/// One sum of terms as an accumulator gives it (gate 4).
pub fn sum(acc: Accum, order: Rows, terms: []const f64) f32 {
    return @floatCast(core.contract.rows.sumIn(acc, order, terms));
}

/// rho's inverse with the sum of squares in `acc` and `order` (gate 4).
pub fn simInverse(acc: Accum, order: Rows, x: []const f64, terms: []f64) f32 {
    for (terms, x) |*t, v| {
        const a: f32 = @floatCast(v);
        t.* = a * a;
    }
    const mean = sum(acc, order, terms) / @as(f32, H);
    return 1 / @sqrt(mean + K.eps);
}

/// One message's outputs with every sum in `acc` and `order`, retrieve's other steps in f32 (gate 4).
pub fn simMessage(acc: Accum, order: Rows, bank0: []const f64, bank1: []const f64, q: []const f32, out: []f64) void {
    var terms: [H]f64 = undefined;
    var b: [2][H]f32 = undefined;
    var logits: [2]f32 = undefined;
    for ([2][]const f64{ bank0, bank1 }, &b, &logits) |x, *row, *l| {
        const inv = simInverse(acc, order, x, &terms);
        for (row, x) |*v, s| v.* = @as(f32, @floatCast(s)) * inv;
        for (&terms, q, row) |*t, qv, v| t.* = qv * v;
        l.* = sum(acc, order, &terms) * K.queryScale();
    }
    const top = @max(@max(-std.math.inf(f32), logits[0]), logits[1]);
    const e = [2]f64{ @exp(logits[0] - top), @exp(logits[1] - top) };
    const den = sum(acc, order, &e);
    const w = [2]f32{ @as(f32, @floatCast(e[0])) / den, @as(f32, @floatCast(e[1])) / den };
    for (out, 0..) |*o, c| o.* = sum(acc, order, &.{ w[0] * b[0][c], w[1] * b[1][c] });
}

/// The dot order cell's terms: +1, +2^24, +1, -2^24 at elements 0, 8, 16 and 24 (lanes 0-3), every other term 0.
pub fn dotCell(q: []f32) void {
    @memset(q, 0);
    for ([4]usize{ 0, 8, 16, 24 }, [4]f32{ 1, 0x1p24, 1, -0x1p24 }) |e, v| q[e] = v;
}

/// The messages' rho cell bank: 2^12, then 2,303 ones (serial keeps only 2^24, W32 loses lane 0's 71 unit squares).
pub fn rhoCellBank(bank: []u16) void {
    @memset(bank, bf.bits(1));
    bank[0] = bf.bits(4096);
}

test "the query scale and eps the launcher passes are binary32(1 / sqrt(2,304)) and float64 1e-6 narrowed" {
    try std.testing.expectEqual(@as(f32, @floatCast(1 / @sqrt(@as(f64, H)))), K.queryScale());
    try std.testing.expectEqual(@as(f32, @floatCast(@as(f64, 1e-6))), K.eps);
    try std.testing.expectEqual(@as(usize, 2304), H);
}

test "a W32 sum sends element e to lane (e / 8) % 32, each lane ascending, then the butterfly 1, 2, 4, 8, 16" {
    var a: [256]f32 = undefined;
    var ones: [256]f32 = @splat(1);
    for (&a, 0..) |*v, e| v.* = @floatFromInt(e);
    var lane: [lanes]f32 = @splat(0);
    for (a, 0..) |v, e| lane[(e / chunk) % lanes] = @mulAdd(f32, v, 1, lane[(e / chunk) % lanes]);
    var want: f32 = 0;
    for ([5]usize{ 1, 2, 4, 8, 16 }) |o| {
        var n: [lanes]f32 = undefined;
        for (&n, 0..) |*x, l| x.* = lane[l] + lane[l ^ o];
        lane = n;
    }
    want = lane[0];
    try std.testing.expectEqual(want, w32Sum(&a, &ones));
    try std.testing.expectEqual(@as(f32, 32640), w32Sum(&a, &ones));
}

test "the dot cell: W32 gives 1, one serial chain 0, the reversed butterfly 2" {
    var q: [H]f32 = undefined;
    dotCell(&q);
    const one: [H]f32 = @splat(1);
    try std.testing.expectEqual(@as(f32, 1), sumIn(.w32, &q, &one));
    try std.testing.expectEqual(@as(f32, 0), sumIn(.serial, &q, &one));
    try std.testing.expectEqual(@as(f32, 2), sumIn(.w32_desc, &q, &one));
    try std.testing.expectEqual(@as(f32, 1), sumIn(.w32_unfused, &q, &one));
}

test "the rho cell: W32 and serial sums of squares differ, and so does every message output" {
    var bank: [H]u16 = undefined;
    rhoCellBank(&bank);
    var x: [H]f32 = undefined;
    widen(&x, &bank);
    try std.testing.expectEqual(@as(f32, 0x1p24 + 31 * 72), sumIn(.w32, &x, &x));
    try std.testing.expectEqual(@as(f32, 0x1p24), sumIn(.serial, &x, &x));
    try std.testing.expectEqual(sumIn(.w32, &x, &x), sumIn(.w32_desc, &x, &x));
    var prng = std.Random.DefaultPrng.init(0x72686f);
    const r = prng.random();
    var other: [H]f32 = undefined;
    var q: [H]f32 = undefined;
    for (&other, &q) |*o, *v| {
        o.* = bf.value(bf.bits(r.floatNorm(f32) * 3));
        v.* = r.floatNorm(f32) * 2;
    }
    var served: [H]f32 = undefined;
    var serial: [H]f32 = undefined;
    message(.{}, &x, &other, &q, &served);
    message(.{ .rho = .serial }, &x, &other, &q, &serial);
    var differ: usize = 0;
    for (served, serial) |p, s| differ += @intFromBool(!same32(p, s));
    try std.testing.expect(differ > H - 8);
}

test "the dot cell's message: banks of +-1,024 make b +-1, so W32, serial and reversed logits give three outputs" {
    var b0: [H]f32 = @splat(1024);
    var b1: [H]f32 = @splat(-1024);
    var q: [H]f32 = undefined;
    dotCell(&q);
    var out: [3][H]f32 = undefined;
    for ([3]Order{ .w32, .serial, .w32_desc }, &out) |o, *m| message(.{ .dot = o }, &b0, &b1, &q, m);
    for (0..3) |i| for (0..i) |j| try std.testing.expect(!same32(out[i][5], out[j][5]));
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(out[1][5]))); // serial: equal logits, w0 - w1 = +0
}

/// |W32 - exact| over |terms| for one row, and the fp32 bound for its depth (gamma(depth) plus float64's own error).
fn w32Miss(a: []const f32, b: []const f32) [2]f64 {
    var exact: f64 = 0;
    var mag: f64 = 0;
    for (a, b) |x, y| {
        const p = @as(f64, x) * @as(f64, y);
        exact += p;
        mag += @abs(p);
    }
    const allow = core.contract.spec.gamma(@floatFromInt(depthW32(a.len)), 0x1p-24) * mag + core.contract.spec.gamma(@floatFromInt(a.len), 0x1p-53) * mag;
    return .{ @abs(@as(f64, w32Sum(a, b)) - exact), allow };
}

test "W32 sums of 2,304 and 4,608 stay within fp32's bound for depth n / 32 + 5 against float64, random, wide and cancelling rows" {
    var prng = std.Random.DefaultPrng.init(0x773332);
    const r = prng.random();
    var a: [2 * H]f32 = undefined;
    var b: [2 * H]f32 = undefined;
    var worst: f64 = 0;
    for (0..60) |trial| {
        for (&a, &b, 0..) |*x, *y, e| {
            x.* = switch (trial % 3) {
                0 => r.floatNorm(f32),
                1 => std.math.ldexp(r.floatNorm(f32), @as(i32, @intCast(r.uintLessThan(u32, 41))) - 20),
                else => if (e % 2 == 0) 1 + r.float(f32) * 0x1p-10 else -1 - r.float(f32) * 0x1p-10,
            };
            y.* = if (trial % 3 == 2) 1 else r.floatNorm(f32);
        }
        for ([2]usize{ H, 2 * H }) |n| {
            const miss, const allow = w32Miss(a[0..n], b[0..n]);
            try std.testing.expect(miss <= allow);
            worst = @max(worst, miss / allow);
        }
    }
    try std.testing.expect(worst > 0);
}

fn randomBank(r: std.Random, x: []u16) void {
    for (x) |*v| v.* = bf.bits(r.floatNorm(f32) * 3);
}

test "gate 4's message simulation in f32 equals the W32 message where every sum is exact: banks of +-1, three-term queries" {
    var prng = std.Random.DefaultPrng.init(0x4d5347);
    const r = prng.random();
    var b0: [H]u16 = undefined;
    var b1: [H]u16 = undefined;
    for (&b0, &b1) |*x, *y| {
        x.* = bf.bits(if (r.boolean()) 1 else -1);
        y.* = bf.bits(if (r.boolean()) 1 else -1);
    }
    var q: [H]f32 = @splat(0);
    for ([3]f32{ 2, -1, 1 }, 0..) |v, i| q[(131 + i * 977) % H] = v;
    var x0: [H]f32 = undefined;
    var x1: [H]f32 = undefined;
    widen(&x0, &b0);
    widen(&x1, &b1);
    var want: [H]f32 = undefined;
    message(.{}, &x0, &x1, &q, &want);
    var w0: [H]f64 = undefined;
    var w1: [H]f64 = undefined;
    for (&w0, &w1, x0, x1) |*a, *b, u, v| {
        a.* = u;
        b.* = v;
    }
    var sim: [H]f64 = undefined;
    for ([_]Rows{ .sequential, .pairwise, .strided }) |o| {
        simMessage(.f32, o, &w0, &w1, &q, &sim);
        for (want, sim) |x, y| try std.testing.expect(same32(x, @floatCast(y)));
    }
}

test "a message is +0 where both banks hold -0: the shares' sum starts at +0, where m0 + m1 alone would give -0" {
    var b0: [H]u16 = undefined;
    var b1: [H]u16 = undefined;
    for (&b0, &b1, 0..) |*x, *y, i| {
        x.* = bf.bits(@floatFromInt(@as(i32, @intCast(i % 7)) - 3));
        y.* = bf.bits(@floatFromInt(@as(i32, @intCast(i % 5)) - 2));
    }
    b0[11] = 0x8000;
    b1[11] = 0x8000;
    var q: [48 * H]f32 = undefined;
    for (&q, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i % 3)) - 1);
    var out: [4 * H]f32 = undefined;
    try messages(.{ .bank0 = &b0, .bank1 = &b1, .qm = q[0 .. 24 * H], .kv = q[24 * H ..], .first = 0, .count = 1, .rows = 1 }, &out);
    for (0..4) |role| try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(out[role * H + 11])));
    const neg: f32 = -0.0;
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(neg * 0.5 + neg * 0.5)));
}

test "messages read each role's query row: q and mlp from q_mlp, k and v from kv, rows 2j and 2j + 1" {
    var prng = std.Random.DefaultPrng.init(0x726f6c);
    const r = prng.random();
    var b0: [2 * H]u16 = undefined;
    var b1: [2 * H]u16 = undefined;
    randomBank(r, &b0);
    randomBank(r, &b1);
    var q: [48 * H]f32 = undefined;
    for (&q) |*v| v.* = r.floatNorm(f32);
    const c: MessageCall = .{ .bank0 = &b0, .bank1 = &b1, .qm = q[0 .. 24 * H], .kv = q[24 * H ..], .first = 7, .count = 2, .rows = 2 };
    var out: [2 * 4 * 2 * H]f32 = undefined;
    try messages(c, &out);
    var x0: [H]f32 = undefined;
    var x1: [H]f32 = undefined;
    widen(&x0, b0[H..]);
    widen(&x1, b1[H..]);
    var one: [H]f32 = undefined;
    // core layer 8 (j = 1), role v: kv row 17; row 1 of the call
    message(.{}, &x0, &x1, q[24 * H + 17 * H ..][0..H], &one);
    const at = ((1 * 4 + 2) * 2 + 1) * H;
    for (out[at..][0..H], one) |got, want| try std.testing.expect(same32(got, want));
}

test {
    _ = @import("contract_inkling2_loop_ref_gate.zig");
}
