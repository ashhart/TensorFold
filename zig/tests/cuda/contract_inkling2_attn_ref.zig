//! Inkling 2's attention in float64, each device rounding point named: the fixed reference of the attn contract.
const std = @import("std");
const core = @import("core");

const contract = core.contract;
const c = contract.spec;
const sumIn = contract.rows.sumIn;

pub const D = 128;
pub const heads = 24; // a TP=2 rank's query heads
pub const kv = 4; // and its key heads
pub const group = heads / kv;
pub const extent = 1024; // rel_extent: the bias covers distances 0..1023
pub const d_rel = 16;
pub const qk_clip: f64 = 22.627416997969522; // config.json qk_l2_clip_threshold
pub const rel_clip: f64 = 8.0;
pub const floor_pos: f64 = 128000; // log_scaling_n_floor
pub const alpha: f64 = 0.1; // log_scaling_alpha
pub const eps: f64 = 1e-6; // rms_norm_eps
pub const scale: f64 = 0.08838834764831845; // 128^-1/2; the device's fp32 constant is within 2^-24 of it
pub const max_keys = 4096 + extent; // distinct keys a row's reference walks: ring slots, then the biased distances

const u: f64 = 0x1p-24; // fp32 unit roundoff
const u_mma: f64 = 0x1p-23; // a tensor-core add: its adders may truncate
const u_bf: f64 = 0x1p-8; // bf16 unit roundoff (8 significant bits)
const log2e: f64 = 1.4426950408889634;

pub fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// x rounded once to bf16 from float64 (nearest, ties to even); non-finite values pass.
pub fn rn(x: f64) f64 {
    if (!std.math.isFinite(x)) return x;
    var b: u64 = @bitCast(x);
    b += (@as(u64, 1) << 44) - 1 + ((b >> 45) & 1);
    b &= ~((@as(u64, 1) << 45) - 1);
    return @bitCast(b);
}

/// rn(x) as the bits a device buffer holds.
pub fn bits(x: f64) u16 {
    return c.toBf16(@floatCast(rn(x)));
}

/// fmin as the device's fminf: a NaN operand drops out (IEEE minNum).
pub fn minNum(x: f64, y: f64) f64 {
    if (std.math.isNan(x)) return y;
    if (std.math.isNan(y)) return x;
    return @min(x, y);
}

/// Inkling2L2Clip's factor in float64: min(thr / (|x| + 1e-6), 1), with fminf's NaN rule.
pub fn clipFactor(x: []const f64, thr: f64) f64 {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    return minNum(thr / (@sqrt(ss) + 1e-6), 1);
}

/// The log-scaling factor of a global layer's q and rel at position `pos`: 1 + alpha ln(max((pos + 1) / floor, 1)).
pub fn tau(pos: usize) f64 {
    return 1 + alpha * @log(@max(@as(f64, @floatFromInt(pos + 1)) / floor_pos, 1));
}

/// One attention call's host inputs as the device holds them (bf16 bits) and where its rows sit.
pub const Call = struct {
    q: []const u16, // [rows][heads][D]: prep's clipped q (tau-scaled on a global layer)
    k: []const u16, // [streams][kv][ring][D]: a stream's key at position p sits in slot p % ring
    v: []const u16,
    tab: []const u16, // [rows][heads][extent]: the bf16 bias table, rel . rel_logits_proj rounded once
    sink: []const u16, // [rows][heads]
    gate: []const u16, // [rows][heads][D]
    pos: []const usize,
    stream: []const usize,
    ring: usize,
    window: usize, // 0: a global layer, every key from 0
    exact: bool = false, // gate 4's rows: every p is 1 and every sum an integer below 2^24

    /// The fp32 sums' longest chain at a row: 4 mma (k16) a 64-key tile and its rescale, then the folds.
    fn depth(call: *const Call, pos: usize) f64 {
        const global = call.window == 0;
        const tiles: f64 = if (global) 16 else 4; // a block's tiles: a 1,024-key sub-chunk, or a 256-key chunk
        const chunks: f64 = @floatFromInt(if (global) pos / 16384 + 1 else 5);
        const subs: f64 = if (global) 16 else 0;
        return tiles * (4 * 17 + 1) + 2 * (subs + chunks) + 8;
    }

    /// Rescales a key's weight meets (exp2f within 2 ulp, then a product): in-tile, sub-chunk and chunk folds.
    fn rescales(call: *const Call, pos: usize) f64 {
        const global = call.window == 0;
        return (if (global) 16 + 16 + @as(f64, @floatFromInt(pos / 16384 + 1)) else 4 + 5) + 2;
    }

    fn keyRow(call: *const Call, which: []const u16, st: usize, kvh: usize, slot: usize) []const u16 {
        return which[((st * kv + kvh) * call.ring + slot) * D ..][0..D];
    }
};

/// A row's keys for the reference: a slot, its multiplicity and its bias; scores and error terms filled later.
pub const Keys = struct {
    n: usize = 0,
    slot: [max_keys]usize = undefined,
    mult: [max_keys]f64 = undefined,
    bias: [max_keys]f64 = undefined,
    s: [max_keys]f64 = undefined,
    w: [max_keys]f64 = undefined, // exp(s - M), then times mult
    e: [max_keys]f64 = undefined, // the weight's relative error on the device

    /// Keys lo..pos: past the bias extent a slot's keys share one score (global rings wrap), so they group by slot.
    fn gather(ks: *Keys, call: *const Call, r: usize, h: usize) void {
        const pos = call.pos[r];
        const lo = if (call.window > 0) (pos + 1) -| call.window else 0;
        ks.n = 0;
        var near = lo;
        if (call.window == 0 and pos >= extent) {
            const last = pos - extent; // keys 0..last see no bias
            for (0..@min(call.ring, last + 1)) |slot| ks.add(slot, @floatFromInt((last - slot) / call.ring + 1), 0);
            near = last + 1;
        }
        const tab = call.tab[(r * heads + h) * extent ..][0..extent];
        for (near..pos + 1) |p| ks.add(p % call.ring, 1, if (pos - p < extent) bf(tab[pos - p]) else 0);
    }

    fn add(ks: *Keys, slot: usize, m: f64, b: f64) void {
        ks.slot[ks.n] = slot;
        ks.mult[ks.n] = m;
        ks.bias[ks.n] = b;
        ks.n += 1;
    }
};

fn maxNan(x: f64, y: f64) f64 {
    return if (std.math.isNan(x) or std.math.isNan(y)) std.math.nan(f64) else @max(x, y);
}

fn sigmoid(g: f64) f64 {
    return 1 / (1 + @exp(-g));
}

/// Row r, head h: y = sum p v / (sum p + exp(sink - M)), out = y sigmoid(gate), p = exp(s - M); tol bounds it.
pub fn row(call: *const Call, r: usize, h: usize, ks: *Keys, want: []f64, tol: []f64) void {
    ks.gather(call, r, h);
    const st = call.stream[r];
    const kvh = h / group;
    const pos = call.pos[r];
    const qr = call.q[(r * heads + h) * D ..][0..D];
    const z = bf(call.sink[r * heads + h]);
    var most = z;
    for (0..ks.n) |i| {
        const kr = call.keyRow(call.k, st, kvh, ks.slot[i]);
        var dot: f64 = 0;
        var abs: f64 = 0;
        for (qr, kr) |qv, kv_| {
            dot += bf(qv) * bf(kv_); // bf16 x bf16: exact in fp32, so the mma's 128 products are exact
            abs += @abs(bf(qv) * bf(kv_));
        }
        ks.s[i] = dot * scale + ks.bias[i]; // fp32 on the device: dot by 8 chained k16 mma, then scale, bias, log2e
        ks.e[i] = abs * scale * 136 * u_mma + (@abs(dot * scale) + @abs(ks.bias[i])) * 6 * u;
        most = maxNan(most, ks.s[i]);
    }
    const chain = call.depth(pos) * u_mma + call.rescales(pos) * 0x1p-22; // fp32 sums and the folds' exp2f factors
    var l: f64 = @exp(z - most); // the sink: one extra logit with no value (eager_attention_forward)
    for (0..ks.n) |i| {
        ks.w[i] = @exp(ks.s[i] - most) * ks.mult[i];
        l += ks.w[i];
        ks.e[i] += (most - ks.s[i]) * u + 0x1p-22 + u_bf + chain; // x - mu rounded, exp2f, P rounded to bf16 for P.V and l
    }
    const p0 = @exp(z - most);
    const e0 = (@abs(z) * 3 + (most - z)) * u + 0x1p-22 + chain; // the sink's logit times log2e, then exp2f in fp32
    const gate = call.gate[(r * heads + h) * D ..][0..D];
    for (0..D) |d| {
        var num: f64 = 0;
        for (0..ks.n) |i| num += ks.w[i] * bf(call.keyRow(call.v, st, kvh, ks.slot[i])[d]);
        const y = num / l;
        const sg = sigmoid(bf(gate[d])); // fp32 on the device: expf, add, divide, then the product
        want[d] = y * sg;
        if (call.exact) { // q = 0, no bias, no sink: p = exp2(0) = 1 exactly, so only the division, gate and output round
            tol[d] = @abs(want[d]) * (u_bf + 12 * u) + 0x1p-132;
            continue;
        }
        var spread: f64 = p0 * @abs(y) * e0;
        for (0..ks.n) |i| {
            const vd = bf(call.keyRow(call.v, st, kvh, ks.slot[i])[d]);
            spread += (ks.w[i] * ks.e[i] + ks.mult[i] * 0x1p-133) * (@abs(vd) + @abs(y)); // a weight's error moves y by p (v - y)
        }
        tol[d] = sg * spread / l + @abs(want[d]) * (u_bf + 12 * u) + 0x1p-132; // then the output rounds to bf16
    }
}

/// Row r, head h with every sum in `acc` and `order` over the keys one by one (gate 4), output rounded to bf16.
pub fn simulate(call: *const Call, r: usize, h: usize, acc: c.Accum, order: contract.gates.Order, p: []f64, col: []f64, out: []f64) void {
    const st = call.stream[r];
    const kvh = h / group;
    const pos = call.pos[r];
    const lo = if (call.window > 0) (pos + 1) -| call.window else 0;
    const n = pos + 1 - lo;
    const qr = call.q[(r * heads + h) * D ..][0..D];
    const tab = call.tab[(r * heads + h) * extent ..][0..extent];
    const z = bf(call.sink[r * heads + h]);
    var most = z;
    for (p[0..n], lo..) |*x, kp| {
        var dot: f64 = 0;
        for (qr, call.keyRow(call.k, st, kvh, kp % call.ring)) |qv, kv_| dot += bf(qv) * bf(kv_);
        x.* = dot * scale + (if (pos - kp < extent) bf(tab[pos - kp]) else 0);
        most = maxNan(most, x.*);
    }
    for (p[0..n]) |*x| x.* = @exp(x.* - most);
    const l = sumIn(acc, order, p[0..n]) + @exp(z - most);
    const gate = call.gate[(r * heads + h) * D ..][0..D];
    for (out, 0..) |*o, d| {
        for (col[0..n], p[0..n], lo..) |*x, w, kp| x.* = w * bf(call.keyRow(call.v, st, kvh, kp % call.ring)[d]);
        const y = sumIn(acc, order, col[0..n]) / l * sigmoid(bf(gate[d]));
        o.* = bf(c.toBf16(@floatCast(y)));
    }
}

test "rn rounds once to nearest even and keeps bf16 values" {
    try std.testing.expectEqual(@as(f64, 1.0), rn(1.00390625)); // halfway: to even
    try std.testing.expectEqual(@as(f64, 1.0078125), rn(1.005859375)); // halfway: to even, up
    try std.testing.expectEqual(@as(f64, -3.0), rn(-3.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bits(1.001));
    try std.testing.expect(std.math.isNan(minNum(std.math.nan(f64), std.math.nan(f64))));
    try std.testing.expectEqual(@as(f64, 1), minNum(std.math.nan(f64), 1));
    try std.testing.expectEqual(@as(f64, 1), tau(127998));
}
