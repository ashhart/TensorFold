//! Inkling 2's attention in float64, each device rounding point named: the fixed reference of the attn contracts.
const std = @import("std");
const c = @import("contract.zig");
const gates = @import("gates.zig");
const sumIn = @import("rows.zig").sumIn;

pub const D = 128;
pub const heads = 24; // a call's default query heads: CUDA's served Small rank (TP=2); a Call names its own
pub const kv = 4; // and its key heads
pub const group = heads / kv;
pub const extent = 1024; // a call's default rel_extent, and the longest one: the bias covers distances 0..1023
pub const d_rel = 16;
pub const qk_clip: f64 = 22.627416997969522; // config.json qk_l2_clip_threshold
pub const rel_clip: f64 = 8.0;
pub const floor_pos: f64 = 128000; // Small's log_scaling_n_floor (Mini's is 8,192: tauAt)
pub const alpha: f64 = 0.1; // log_scaling_alpha
pub const eps: f64 = 1e-6; // rms_norm_eps
pub const scale: f64 = 0.08838834764831845; // 128^-1/2; the device's fp32 constant is within 2^-24 of it
pub const max_keys = 4096 + extent; // distinct keys a row's reference walks: ring slots, then the biased distances

const u: f64 = 0x1p-24; // fp32 unit roundoff
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

/// The log-scaling factor of a global layer's q and rel at position `pos` with Small's floor.
pub fn tau(pos: usize) f64 {
    return tauAt(pos, floor_pos);
}

/// The log-scaling factor at `pos` for a model's n_floor: 1 + alpha ln(max((pos + 1) / floor, 1)).
pub fn tauAt(pos: usize, floor: f64) f64 {
    return 1 + alpha * @log(@max(@as(f64, @floatFromInt(pos + 1)) / floor, 1));
}

/// A backend's fp32 chains: adds in a q.k dot and in a 64-key tile of a row's P.V sum, and each add's unit roundoff.
pub const Chain = struct {
    dot: f64,
    tile: f64,
    unit: f64,

    /// 8 chained mma (k16) a dot, 4 a tile and its rescale; a tensor-core add may truncate (2^-23).
    pub const cuda: Chain = .{ .dot = 8 * 17, .tile = 4 * 17 + 1, .unit = 0x1p-23 };
    /// HIP's scalar IEEE adds: 128 a dot from 0, a tile's 64 keys after its rescale.
    pub const hip: Chain = .{ .dot = 128, .tile = 64 + 1, .unit = u };
};

/// One attention call's host inputs as the device holds them (bf16 bits) and where its rows sit.
pub const Call = struct {
    q: []const u16, // [rows][heads][D]: prep's clipped q (tau-scaled on a global layer)
    k: []const u16, // [streams][kv][ring][D]: a stream's key at position p sits in slot p % ring
    v: []const u16,
    tab: []const u16, // [rows][heads][call.extent]: the bf16 bias table, rel . rel_logits_proj rounded once
    sink: []const u16, // [rows][heads]
    gate: []const u16, // [rows][heads][D]
    pos: []const usize,
    stream: []const usize,
    ring: usize,
    window: usize, // 0: a global layer, every key from 0
    extent: usize = extent, // the layer's bias extent (config's extent(i)): no bias at distances from it on
    exact: bool = false, // gate 4's rows: every p is 1 and every sum an integer below 2^24
    heads: usize = heads, // the call's query heads, then its key heads (GQA groups of heads / kv)
    kv: usize = kv,
    chain: Chain = .cuda,

    fn groupSize(call: *const Call) usize {
        return call.heads / call.kv;
    }

    /// The fp32 sums' longest chain at a row: a 64-key tile's adds and its rescale (`chain.tile`), then the folds.
    fn depth(call: *const Call, pos: usize) f64 {
        const global = call.window == 0;
        const tiles: f64 = if (global) 16 else 4; // a block's tiles: a 1,024-key sub-chunk, or a 256-key chunk
        const chunks: f64 = @floatFromInt(if (global) pos / 16384 + 1 else 5);
        const subs: f64 = if (global) 16 else 0;
        return tiles * call.chain.tile + 2 * (subs + chunks) + 8;
    }

    /// Rescales a key's weight meets (exp2f within 2 ulp, then a product): in-tile, sub-chunk and chunk folds.
    fn rescales(call: *const Call, pos: usize) f64 {
        const global = call.window == 0;
        return (if (global) 16 + 16 + @as(f64, @floatFromInt(pos / 16384 + 1)) else 4 + 5) + 2;
    }

    fn keyRow(call: *const Call, which: []const u16, st: usize, kvh: usize, slot: usize) []const u16 {
        return which[((st * call.kv + kvh) * call.ring + slot) * D ..][0..D];
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
        const ext = call.extent;
        std.debug.assert(ext <= extent); // max_keys holds a ring's slots and at most 1,024 biased distances
        const lo = if (call.window > 0) (pos + 1) -| call.window else 0;
        ks.n = 0;
        var near = lo;
        if (call.window == 0 and pos >= ext) {
            const last = pos - ext; // keys 0..last see no bias
            for (0..@min(call.ring, last + 1)) |slot| ks.add(slot, @floatFromInt((last - slot) / call.ring + 1), 0);
            near = last + 1;
        }
        const tab = call.tab[(r * call.heads + h) * ext ..][0..ext];
        for (near..pos + 1) |p| ks.add(p % call.ring, 1, if (pos - p < ext) bf(tab[pos - p]) else 0);
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
    const kvh = h / call.groupSize();
    const pos = call.pos[r];
    const qr = call.q[(r * call.heads + h) * D ..][0..D];
    const z = bf(call.sink[r * call.heads + h]);
    var most = z;
    for (0..ks.n) |i| {
        const kr = call.keyRow(call.k, st, kvh, ks.slot[i]);
        var dot: f64 = 0;
        var abs: f64 = 0;
        for (qr, kr) |qv, kv_| {
            dot += bf(qv) * bf(kv_); // bf16 x bf16: exact in fp32, so the mma's 128 products are exact
            abs += @abs(bf(qv) * bf(kv_));
        }
        ks.s[i] = dot * scale + ks.bias[i]; // fp32 on the device: the dot's chain (`chain.dot`), then scale, bias, log2e
        ks.e[i] = abs * scale * call.chain.dot * call.chain.unit + (@abs(dot * scale) + @abs(ks.bias[i])) * 6 * u;
        most = maxNan(most, ks.s[i]);
    }
    const chain = call.depth(pos) * call.chain.unit + call.rescales(pos) * 0x1p-22; // fp32 sums and the folds' exp2f factors
    var l: f64 = @exp(z - most); // the sink: one extra logit with no value (eager_attention_forward)
    for (0..ks.n) |i| {
        ks.w[i] = @exp(ks.s[i] - most) * ks.mult[i];
        l += ks.w[i];
        ks.e[i] += (most - ks.s[i]) * u + 0x1p-22 + u_bf + chain; // x - mu rounded, exp2f, P rounded to bf16 for P.V and l
    }
    const p0 = @exp(z - most);
    const e0 = (@abs(z) * 3 + (most - z)) * u + 0x1p-22 + chain; // the sink's logit times log2e, then exp2f in fp32
    const gate = call.gate[(r * call.heads + h) * D ..][0..D];
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
pub fn simulate(call: *const Call, r: usize, h: usize, acc: c.Accum, order: gates.Order, p: []f64, col: []f64, out: []f64) void {
    const st = call.stream[r];
    const kvh = h / call.groupSize();
    const pos = call.pos[r];
    const lo = if (call.window > 0) (pos + 1) -| call.window else 0;
    const n = pos + 1 - lo;
    const qr = call.q[(r * call.heads + h) * D ..][0..D];
    const tab = call.tab[(r * call.heads + h) * call.extent ..][0..call.extent];
    const z = bf(call.sink[r * call.heads + h]);
    var most = z;
    for (p[0..n], lo..) |*x, kp| {
        var dot: f64 = 0;
        for (qr, call.keyRow(call.k, st, kvh, kp % call.ring)) |qv, kv_| dot += bf(qv) * bf(kv_);
        x.* = dot * scale + (if (pos - kp < call.extent) bf(tab[pos - kp]) else 0);
        most = maxNan(most, x.*);
    }
    for (p[0..n]) |*x| x.* = @exp(x.* - most);
    const l = sumIn(acc, order, p[0..n]) + @exp(z - most);
    const gate = call.gate[(r * call.heads + h) * D ..][0..D];
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

test "a call carries its rank's heads, key heads and chains; tau takes the model's floor" {
    try std.testing.expectEqual(@as(f64, 1), tauAt(8191, 8192));
    try std.testing.expectApproxEqAbs(1 + 0.1 * @log(2.0), tauAt(16383, 8192), 1e-15);
    try std.testing.expectEqual(tau(300000), tauAt(300000, floor_pos));
    const call: Call = .{ .q = &.{}, .k = &.{}, .v = &.{}, .tab = &.{}, .sink = &.{}, .gate = &.{}, .pos = &.{}, .stream = &.{}, .ring = 4096, .window = 0 };
    try std.testing.expectEqual(@as(usize, heads), call.heads);
    try std.testing.expectEqual(@as(usize, kv), call.kv);
    try std.testing.expectEqual(Chain.cuda, call.chain);
    try std.testing.expect(Chain.hip.unit < Chain.cuda.unit and Chain.hip.tile < Chain.cuda.tile);
}

fn pinned(i: usize, m: usize) u16 {
    const x: f32 = @floatFromInt(@as(i64, @intCast((i * 37 + m) % 29)) - 14);
    return c.toBf16(x * 0.0625);
}

test "CUDA's defaults keep the bits the reference had in zig/tests/cuda (#4125)" {
    var q: [heads * D]u16 = undefined;
    var k: [kv * 128 * D]u16 = undefined;
    var v: [kv * 128 * D]u16 = undefined;
    var tab: [heads * extent]u16 = undefined;
    var sink: [heads]u16 = undefined;
    var gate: [heads * D]u16 = undefined;
    for ([_][]u16{ &q, &k, &v, &tab, &gate, &sink }, 1..) |x, m| for (x, 0..) |*e, i| {
        e.* = pinned(i, m);
    };
    var pos = [_]usize{100};
    var st = [_]usize{0};
    const want = [2][4]u64{ .{ 0xbfc6dd62d3109c22, 0xbfbd967537eb01a7, 0x3f64049c6441daa5, 0x3f584dec199b1a8c }, .{ 0xbfc2ef9685604c58, 0xbfbd36245440324a, 0x3f617d0618cb7670, 0x3f579dc3fed2449e } };
    for ([_]usize{ 0, 32 }, want) |window, bits_| {
        const call: Call = .{ .q = &q, .k = &k, .v = &v, .tab = &tab, .sink = &sink, .gate = &gate, .pos = &pos, .stream = &st, .ring = 128, .window = window };
        var ks: Keys = .{};
        var w: [D]f64 = undefined;
        var t: [D]f64 = undefined;
        row(&call, 0, 13, &ks, &w, &t);
        const got = [4]u64{ @bitCast(w[0]), @bitCast(w[77]), @bitCast(t[0]), @bitCast(t[77]) };
        try std.testing.expectEqualSlices(u64, &bits_, &got);
    }
}

test "a call's extent defaults to 1,024; a shorter one reads its own table and no bias past it" {
    const a = std.testing.allocator;
    const hs = 4;
    const kvs = 2;
    const ring = 128;
    const q = try a.alloc(u16, hs * D);
    defer a.free(q);
    const k = try a.alloc(u16, kvs * ring * D);
    defer a.free(k);
    const v = try a.alloc(u16, k.len);
    defer a.free(v);
    const full = try a.alloc(u16, hs * extent); // a bias at every distance to 1,023
    defer a.free(full);
    const sink = try a.alloc(u16, hs);
    defer a.free(sink);
    const gate = try a.alloc(u16, hs * D);
    defer a.free(gate);
    for ([_][]u16{ q, k, v, full, gate, sink }, 1..) |x, m| for (x, 0..) |*e, i| {
        e.* = pinned(i, m);
    };
    const st = [_]usize{0};
    const ks = try a.create(Keys);
    defer a.destroy(ks);
    const p = try a.alloc(f64, 5001);
    defer a.free(p);
    const col = try a.alloc(f64, 5001);
    defer a.free(col);
    // a global row past extent 256 and a sliding row of window 128 at extent 128 (tiny-inkling2-mini's two layers)
    for ([_]usize{ 5000, 700 }, [_]usize{ 0, 128 }, [_]usize{ 256, 128 }) |at, window, ext| {
        const pos = [_]usize{at};
        const short = try a.alloc(u16, hs * ext);
        defer a.free(short);
        const padded = try a.dupe(u16, full);
        defer a.free(padded);
        for (0..hs) |h| {
            @memcpy(short[h * ext ..][0..ext], full[h * extent ..][0..ext]);
            @memset(padded[h * extent + ext ..][0 .. extent - ext], 0);
        }
        const base: Call = .{ .q = q, .k = k, .v = v, .tab = padded, .sink = sink, .gate = gate, .pos = &pos, .stream = &st, .ring = ring, .window = window, .heads = hs, .kv = kvs };
        try std.testing.expectEqual(@as(usize, extent), base.extent);
        var own = base;
        own.tab = short;
        own.extent = ext;
        var rest = base;
        rest.tab = full;
        var w: [3][D]f64 = undefined;
        var t: [3][D]f64 = undefined;
        for ([_]*const Call{ &own, &base, &rest }, &w, &t) |cl, *wx, *tx| row(cl, 0, 3, ks, wx, tx);
        for (w[0], w[1], t[0], t[1]) |x, y, tx, ty| {
            try std.testing.expectApproxEqRel(y, x, 1e-12); // keys past the extent group by slot instead of one by one
            try std.testing.expectApproxEqRel(ty, tx, 1e-9);
        }
        if (window == 0) try std.testing.expect(!std.mem.eql(f64, &w[0], &w[2])); // the bias past 256 moves the row
        var o: [2][D]f64 = undefined;
        for ([_]*const Call{ &own, &base }, &o) |cl, *ox| simulate(cl, 0, 2, .f32, .sequential, p, col, ox);
        for (o[0], o[1]) |x, y| try std.testing.expectEqual(@as(u64, @bitCast(y)), @as(u64, @bitCast(x)));
    }
}

// bits from the base file (5c1fea67f:zig/tests/cuda/contract_inkling2_attn_ref.zig) on these inputs
test "CUDA's defaults keep the base bits past the extent, on a full window, the exact branch, gate 4 and tau" {
    const a = std.testing.allocator;
    const ring = 128;
    const q = try a.alloc(u16, 2 * heads * D);
    defer a.free(q);
    const k = try a.alloc(u16, kv * ring * D);
    defer a.free(k);
    const v = try a.alloc(u16, k.len);
    defer a.free(v);
    const tab = try a.alloc(u16, 2 * heads * extent);
    defer a.free(tab);
    const sink = try a.alloc(u16, 2 * heads);
    defer a.free(sink);
    const gate = try a.alloc(u16, 2 * heads * D);
    defer a.free(gate);
    for ([_][]u16{ q, k, v, tab, gate, sink }, 1..) |x, m| for (x, 0..) |*e, i| {
        e.* = pinned(i, m);
    };
    q[(1 * heads + 13) * D + 3] = c.toBf16(std.math.nan(f32));
    sink[1 * heads + 7] = c.toBf16(std.math.inf(f32));
    const st = [_]usize{ 0, 0 };
    const Case = struct { pos: [2]usize, window: usize, exact: bool, r: usize, h: usize, bits: ?[4]u64 };
    const cases = [_]Case{
        .{ .pos = .{ 5000, 5001 }, .window = 0, .exact = false, .r = 0, .h = 13, .bits = .{ 0xbfc94c2fe7b239f9, 0xbfbbd5eb445673c4, 0x3f65377f773ff541, 0x3f571c5526af1cd6 } },
        .{ .pos = .{ 2000, 2001 }, .window = 1024, .exact = false, .r = 0, .h = 5, .bits = .{ 0x3fc15c3191d15f7c, 0x3fd77f821709cc70, 0x3f5c6a56e75e29fd, 0x3f735aa88f30c395 } },
        .{ .pos = .{ 300, 301 }, .window = 1024, .exact = true, .r = 0, .h = 0, .bits = .{ 0xbfc56a13a38f6bbf, 0xbfc0eaa55736282d, 0x3f456b149c7b1678, 0x3f40eb7056f63eb7 } },
        .{ .pos = .{ 300, 301 }, .window = 0, .exact = false, .r = 1, .h = 13, .bits = null }, // a NaN q: want and tol NaN
        .{ .pos = .{ 300, 301 }, .window = 0, .exact = false, .r = 1, .h = 7, .bits = null }, // a +inf sink: want and tol NaN
    };
    const ks = try a.create(Keys);
    defer a.destroy(ks);
    for (cases) |cs| {
        const call: Call = .{ .q = q, .k = k, .v = v, .tab = tab, .sink = sink, .gate = gate, .pos = &cs.pos, .stream = &st, .ring = ring, .window = cs.window, .exact = cs.exact };
        var w: [D]f64 = undefined;
        var t: [D]f64 = undefined;
        row(&call, cs.r, cs.h, ks, &w, &t);
        if (cs.bits) |b| {
            try std.testing.expectEqualSlices(u64, &b, &.{ @bitCast(w[0]), @bitCast(w[77]), @bitCast(t[0]), @bitCast(t[77]) });
        } else for (w, t) |x, y| try std.testing.expect(std.math.isNan(x) and std.math.isNan(y));
    }
    const p = try a.alloc(f64, 5001);
    defer a.free(p);
    const col = try a.alloc(f64, 5001);
    defer a.free(col);
    const sp = [_]usize{ 5000, 5001 };
    const call: Call = .{ .q = q, .k = k, .v = v, .tab = tab, .sink = sink, .gate = gate, .pos = &sp, .stream = &st, .ring = ring, .window = 0 };
    var o: [D]f64 = undefined;
    simulate(&call, 0, 12, .f32, .sequential, p, col, &o);
    try std.testing.expectEqualSlices(u64, &.{ 0x3fd4c00000000000, 0xbfc2200000000000 }, &.{ @bitCast(o[0]), @bitCast(o[77]) });
    simulate(&call, 0, 12, .bf16, .pairwise, p, col, &o);
    try std.testing.expectEqualSlices(u64, &.{ 0x3fd4e00000000000, 0xbfc2000000000000 }, &.{ @bitCast(o[0]), @bitCast(o[77]) });
    for ([_]usize{ 127999, 128000, 300000, 1048575 }, [_]u64{ 0x3ff0000000000000, 0x3ff00000d1b6e1a9, 0x3ff15ce10ac9824d, 0x3ff35d741b82a5cf }) |pos, b|
        try std.testing.expectEqual(b, @as(u64, @bitCast(tau(pos))));
}

/// Key-by-key float64 attention from the definition, sums in key order; `full` biases distances 0 to 1,023.
fn keyByKey(call: *const Call, full: []const u16, r: usize, h: usize, ext: usize, s: []f64, y: []f64, out: []u16) void {
    const pos = call.pos[r];
    const st = call.stream[r];
    const lo = if (call.window > 0) (pos + 1) -| call.window else 0;
    const kvh = h / (call.heads / call.kv);
    const z = bf(call.sink[r * call.heads + h]);
    var m = z;
    for (lo..pos + 1, 0..) |kp, i| {
        const kr = call.k[((st * call.kv + kvh) * call.ring + kp % call.ring) * D ..][0..D];
        var dot: f64 = 0;
        for (0..D) |d| dot += bf(call.q[(r * call.heads + h) * D + d]) * bf(kr[d]);
        s[i] = dot * scale + (if (pos - kp < ext) bf(full[(r * call.heads + h) * extent + pos - kp]) else 0);
        m = @max(m, s[i]);
    }
    const n = pos + 1 - lo;
    var l: f64 = 0;
    for (s[0..n]) |*e| {
        e.* = @exp(e.* - m);
        l += e.*;
    }
    l += @exp(z - m);
    for (0..D) |d| {
        var num: f64 = 0;
        for (lo..pos + 1, 0..) |kp, i| num += s[i] * bf(call.v[((st * call.kv + kvh) * call.ring + kp % call.ring) * D + d]);
        y[d] = num / l * sigmoid(bf(call.gate[(r * call.heads + h) * D + d]));
        out[d] = c.toBf16(@floatCast(y[d]));
    }
}

test "a call's extent: row and simulate equal a key-by-key float64 attention at 128, 256 and 1,024; one off differs" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed_e7e7);
    const rng = prng.random();
    const Case = struct { heads: usize, kv: usize, ring: usize, window: usize, ext: ?usize, pos: []const usize };
    const cases = [_]Case{
        // tiny-mini's local and global layers (a global ring that wraps), extent 128 on a global ring, CUDA's default
        .{ .heads = 4, .kv = 2, .ring = 128, .window = 128, .ext = 128, .pos = &.{ 126, 127, 128, 129, 300 } },
        .{ .heads = 4, .kv = 2, .ring = 512, .window = 0, .ext = 256, .pos = &.{ 254, 255, 256, 257, 1000 } },
        .{ .heads = 4, .kv = 2, .ring = 256, .window = 0, .ext = 128, .pos = &.{ 127, 128, 129 } },
        .{ .heads = 24, .kv = 4, .ring = 4096, .window = 0, .ext = null, .pos = &.{ 1023, 1024, 1025, 4100 } },
    };
    const ks = try a.create(Keys);
    defer a.destroy(ks);
    const s = try a.alloc(f64, 8192);
    defer a.free(s);
    const p = try a.alloc(f64, 8192);
    defer a.free(p);
    const col = try a.alloc(f64, 8192);
    defer a.free(col);
    var moved: usize = 0;
    for (cases) |cs| {
        const rows = 2;
        const q = try a.alloc(u16, rows * cs.heads * D);
        defer a.free(q);
        const k = try a.alloc(u16, 2 * cs.kv * cs.ring * D);
        defer a.free(k);
        const v = try a.alloc(u16, k.len);
        defer a.free(v);
        const sink = try a.alloc(u16, rows * cs.heads);
        defer a.free(sink);
        const gate = try a.alloc(u16, rows * cs.heads * D);
        defer a.free(gate);
        const full = try a.alloc(u16, rows * cs.heads * extent);
        defer a.free(full);
        for (q) |*e| e.* = bits(rng.floatNorm(f64) * 0.6);
        for (k) |*e| e.* = bits(rng.floatNorm(f64) * 0.6);
        for (v) |*e| e.* = bits(rng.float(f64) * 2 - 1);
        for (sink) |*e| e.* = bits(rng.floatNorm(f64));
        for (gate) |*e| e.* = bits(rng.floatNorm(f64));
        for (full) |*e| e.* = bits(1 + rng.float(f64) * 2); // never 0: a missing or extra bias moves the row
        const ext = cs.ext orelse extent;
        const tab = try a.alloc(u16, rows * cs.heads * ext); // the reference's own layout: [rows][heads][ext]
        defer a.free(tab);
        for (0..rows * cs.heads) |rh| @memcpy(tab[rh * ext ..][0..ext], full[rh * extent ..][0..ext]);
        for (cs.pos) |at| {
            const pos = [rows]usize{ at, at -| 1 };
            const st = [rows]usize{ 0, 1 };
            var call: Call = .{ .q = q, .k = k, .v = v, .tab = tab, .sink = sink, .gate = gate, .pos = &pos, .stream = &st, .ring = cs.ring, .window = cs.window, .heads = cs.heads, .kv = cs.kv, .chain = .hip };
            if (cs.ext) |e| call.extent = e;
            try std.testing.expectEqual(ext, call.extent);
            for (0..rows) |r| for ([_]usize{ 0, cs.heads - 1 }) |h| {
                var want: [D]f64 = undefined;
                var tol: [D]f64 = undefined;
                row(&call, r, h, ks, &want, &tol);
                var y: [D]f64 = undefined;
                var b16: [D]u16 = undefined;
                keyByKey(&call, full, r, h, ext, s, &y, &b16);
                // keys past the extent group by slot, so row's sums differ from key order only by rounding
                for (want, y) |w, x| try std.testing.expectApproxEqAbs(x, w, 1e-12);
                var sim: [D]f64 = undefined;
                simulate(&call, r, h, .none, .sequential, p, col, &sim);
                for (sim, b16) |o, w| try std.testing.expectEqual(w, c.toBf16(@floatCast(o)));
                // one off: ext - 1 drops the bias at distance ext - 1, ext + 1 adds one at distance ext
                const lo = if (cs.window > 0) (pos[r] + 1) -| cs.window else 0;
                for ([_]usize{ ext - 1, ext + 1 }) |off| {
                    if (off > extent or pos[r] - lo < @min(off, ext)) continue; // no key at the moved distance
                    keyByKey(&call, full, r, h, off, s, &y, &b16);
                    var most: f64 = 0;
                    for (want, y) |w, x| most = @max(most, @abs(w - x));
                    try std.testing.expect(most > 1e-9);
                    moved += 1;
                }
            };
        }
    }
    try std.testing.expect(moved >= 40);
}

fn splitmix(x: *u64) u64 {
    x.* +%= 0x9e3779b97f4a7c15;
    var z = x.*;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

fn fnv(h: *u64, xs: []const f64) void {
    for (xs) |x| {
        const b: u64 = @bitCast(x);
        for (0..8) |i| {
            h.* ^= (b >> @intCast(i * 8)) & 0xff;
            h.* *%= 0x100000001b3;
        }
    }
}

// the pinned hash of these calls' outputs: CUDA's default extent must keep it
test "CUDA's default extent and an explicit 1,024 keep the base bits on global rows past 1,024, windows and rings" {
    const a = std.testing.allocator;
    const Case = struct { heads: usize, kv: usize, ring: usize, window: usize, pos: [2]usize, chain: Chain };
    const cases = [_]Case{
        .{ .heads = 24, .kv = 4, .ring = 4096, .window = 0, .pos = .{ 1023, 1025 }, .chain = .cuda },
        .{ .heads = 24, .kv = 4, .ring = 4096, .window = 0, .pos = .{ 4097, 5000 }, .chain = .cuda },
        .{ .heads = 24, .kv = 8, .ring = 2048, .window = 1024, .pos = .{ 1024, 3000 }, .chain = .hip },
        .{ .heads = 4, .kv = 2, .ring = 128, .window = 0, .pos = .{ 1024, 2048 }, .chain = .hip },
        .{ .heads = 24, .kv = 4, .ring = 128, .window = 32, .pos = .{ 100, 1025 }, .chain = .cuda },
        .{ .heads = 8, .kv = 2, .ring = 1024, .window = 0, .pos = .{ 31, 1024 }, .chain = .hip },
    };
    const ks = try a.create(Keys);
    defer a.destroy(ks);
    const p = try a.alloc(f64, 8192);
    defer a.free(p);
    const col = try a.alloc(f64, 8192);
    defer a.free(col);
    var seed: u64 = 1024;
    var h: u64 = 0xcbf29ce484222325;
    for (cases) |cs| {
        const q = try a.alloc(u16, 2 * cs.heads * D);
        defer a.free(q);
        const k = try a.alloc(u16, 2 * cs.kv * cs.ring * D);
        defer a.free(k);
        const v = try a.alloc(u16, k.len);
        defer a.free(v);
        const tab = try a.alloc(u16, 2 * cs.heads * extent);
        defer a.free(tab);
        const sink = try a.alloc(u16, 2 * cs.heads);
        defer a.free(sink);
        const gate = try a.alloc(u16, 2 * cs.heads * D);
        defer a.free(gate);
        for ([_][]u16{ q, k, v, tab, sink, gate }, [_]f64{ 1, 1, 1, 2, 2, 3 }) |xs, span| for (xs) |*e| {
            const f: f64 = @floatFromInt(splitmix(&seed) >> 11);
            e.* = bits((f * 0x1p-52 - 1) * span);
        };
        const st = [2]usize{ 0, 1 };
        const call: Call = .{ .q = q, .k = k, .v = v, .tab = tab, .sink = sink, .gate = gate, .pos = &cs.pos, .stream = &st, .ring = cs.ring, .window = cs.window, .heads = cs.heads, .kv = cs.kv, .chain = cs.chain };
        var calls = [2]Call{ call, call };
        calls[1].extent = 1024;
        var hc = [2]u64{ h, h };
        for (&calls, &hc) |*cl, *hh| for (0..2) |r| for ([_]usize{ 0, cs.heads - 1 }) |hd| {
            var want: [D]f64 = undefined;
            var tol: [D]f64 = undefined;
            row(cl, r, hd, ks, &want, &tol);
            fnv(hh, &want);
            fnv(hh, &tol);
            var out: [D]f64 = undefined;
            simulate(cl, r, hd, .f32, .sequential, p, col, &out);
            fnv(hh, &out);
        };
        try std.testing.expectEqual(hc[0], hc[1]);
        h = hc[0];
    }
    try std.testing.expectEqual(@as(u64, 0xf5b6b6e0a1ef3f34), h);
}
