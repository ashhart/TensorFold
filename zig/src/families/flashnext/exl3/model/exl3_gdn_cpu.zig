//! CPU oracle for one Gated DeltaNet layer of Qwen3.8-Flash-Next (qwen4_exp_text), everything between the two EXL3

const std = @import("std");

pub const Dims = struct {
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    taps: usize,

    /// conv channels: q (nk*dk), k (nk*dk), v (nv*dv)
    pub fn channels(d: Dims) usize {
        return 2 * d.nk * d.dk + d.nv * d.dv;
    }
    pub fn zWidth(d: Dims) usize {
        return d.nv * d.dv;
    }
    /// conv state rows kept per layer
    pub fn keep(d: Dims) usize {
        return d.taps - 1;
    }
    pub fn recElems(d: Dims) usize {
        return d.nv * d.dv * d.dk;
    }
    pub fn convElems(d: Dims) usize {
        return d.keep() * d.channels();
    }
};

/// Flash Next: 16 key heads, 48 value heads, 128 x 128, kernel 4.
pub const qwen: Dims = .{ .nk = 16, .nv = 48, .dk = 128, .dv = 128, .taps = 4 };

/// Per-sequence, per-layer state.
pub const State = struct {
    conv: []f64,
    rec: []f64,

    pub fn init(gpa: std.mem.Allocator, d: Dims) !State {
        const conv = try gpa.alloc(f64, d.convElems());
        errdefer gpa.free(conv);
        const rec = try gpa.alloc(f64, d.recElems());
        @memset(conv, 0);
        @memset(rec, 0);
        return .{ .conv = conv, .rec = rec };
    }
    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        gpa.free(s.conv);
        gpa.free(s.rec);
    }
    pub fn reset(s: *State) void {
        @memset(s.conv, 0);
        @memset(s.rec, 0);
    }
};

/// The layer's non-EXL3 parameters, as f64.
pub const Params = struct {
    conv_w: []const f64, // [channels][taps]
    a_log: []const f64, // [nv]
    dt_bias: []const f64, // [nv]
    norm_w: []const f64, // [dv]
    eps: f64 = 1e-6,
};

pub fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

pub fn silu(x: f64) f64 {
    return x / (1.0 + @exp(-x));
}

/// torch.nn.functional.softplus with threshold 20.
pub fn softplus(x: f64) f64 {
    return if (x > 20.0) x else std.math.log1p(@exp(x));
}

/// y[j] = sum_i w[j * k + i] * x[i], for a dense [rows][k] weight (in_proj_a / in_proj_b).
pub fn rowsDot(w: []const f64, x: []const f64, y: []f64) void {
    const k = x.len;
    for (y, 0..) |*o, j| {
        var s: f64 = 0;
        for (x, w[j * k ..][0..k]) |xi, wi| s += xi * wi;
        o.* = s;
    }
}

/// One token of the causal depthwise conv: out = silu(sum_tap w[c][tap] * x[t - (taps-1) + tap]), x before the
pub fn convStep(d: Dims, p: Params, conv: []f64, x: []const f64, out: []f64) void {
    const c = d.channels();
    const kp = d.keep();
    for (0..c) |ch| {
        var acc: f64 = 0;
        for (0..d.taps) |tap| {
            const v = if (tap < kp) conv[tap * c + ch] else x[ch];
            acc += p.conv_w[ch * d.taps + tap] * v;
        }
        out[ch] = silu(acc);
    }
    for (0..kp) |r| {
        for (0..c) |ch| conv[r * c + ch] = if (r + 1 < kp) conv[(r + 1) * c + ch] else x[ch];
    }
}

/// Decay exp(g) and beta for each value head from the raw a and b projections.
pub fn gates(d: Dims, p: Params, a: []const f64, b: []const f64, decay: []f64, beta: []f64) void {
    for (0..d.nv) |h| {
        const g = -@exp(p.a_log[h]) * softplus(a[h] + p.dt_bias[h]);
        decay[h] = @exp(g);
        beta[h] = sigmoid(b[h]);
    }
}

/// One token of the delta rule on all value heads. `act` is the conv output [q | k | v]; y is [nv][dv].
pub fn deltaStep(d: Dims, rec: []f64, act: []const f64, decay: []const f64, beta: []const f64, y: []f64) void {
    const group = d.nv / d.nk;
    const kbase = d.nk * d.dk;
    const vbase = 2 * kbase;
    for (0..d.nv) |hv| {
        const hk = hv / group;
        const q = act[hk * d.dk ..][0..d.dk];
        const k = act[kbase + hk * d.dk ..][0..d.dk];
        var qq: f64 = 0;
        var kk: f64 = 0;
        for (q, k) |a, b| {
            qq += a * a;
            kk += b * b;
        }
        const iq = 1.0 / @sqrt(qq + 1e-6) / @sqrt(@as(f64, @floatFromInt(d.dk)));
        const ik = 1.0 / @sqrt(kk + 1e-6);
        for (0..d.dv) |v| {
            const row = rec[(hv * d.dv + v) * d.dk ..][0..d.dk];
            var kv: f64 = 0;
            for (row, k) |*s, ki| {
                s.* *= decay[hv];
                kv += s.* * ki * ik;
            }
            const delta = (act[vbase + hv * d.dv + v] - kv) * beta[hv];
            var o: f64 = 0;
            for (row, k, q) |*s, ki, qi| {
                s.* += ki * ik * delta;
                o += s.* * qi * iq;
            }
            y[hv * d.dv + v] = o;
        }
    }
}

/// Gated RMS norm per value head: w * (y * rsqrt(mean(y^2) + eps)) * sigmoid(z).
pub fn gatedNorm(d: Dims, p: Params, y: []const f64, z: []const f64, out: []f64) void {
    for (0..d.nv) |hv| {
        const row = y[hv * d.dv ..][0..d.dv];
        var ss: f64 = 0;
        for (row) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(d.dv)) + p.eps);
        for (row, 0..) |v, i| out[hv * d.dv + i] = p.norm_w[i] * (v * inv) * sigmoid(z[hv * d.dv + i]);
    }
}

/// `n` tokens, in order, from the projection outputs: qkv [n][channels], z [n][nv*dv],.
pub fn tokens(gpa: std.mem.Allocator, d: Dims, p: Params, st: State, n: usize, qkv: []const f64, z: []const f64, a: []const f64, b: []const f64, out: []f64) !void {
    const c = d.channels();
    const zw = d.zWidth();
    const act = try gpa.alloc(f64, c);
    defer gpa.free(act);
    const y = try gpa.alloc(f64, zw);
    defer gpa.free(y);
    const decay = try gpa.alloc(f64, d.nv);
    defer gpa.free(decay);
    const beta = try gpa.alloc(f64, d.nv);
    defer gpa.free(beta);
    for (0..n) |t| {
        convStep(d, p, st.conv, qkv[t * c ..][0..c], act);
        gates(d, p, a[t * d.nv ..][0..d.nv], b[t * d.nv ..][0..d.nv], decay, beta);
        deltaStep(d, st.rec, act, decay, beta, y);
        gatedNorm(d, p, y, z[t * zw ..][0..zw], out[t * zw ..][0..zw]);
    }
}

/// Byte sizes of the device state table: conv rows are fp16 (they are raw EXL3 projection outputs), the recurrent
pub const Table = struct {
    dims: Dims,
    layers: usize,
    slots: usize,

    pub fn convBytesPer(t: Table) usize {
        return t.dims.convElems() * 2;
    }
    pub fn recBytesPer(t: Table) usize {
        return t.dims.recElems() * 4;
    }
    pub fn convBytes(t: Table) usize {
        return t.convBytesPer() * t.layers * t.slots;
    }
    pub fn recBytes(t: Table) usize {
        return t.recBytesPer() * t.layers * t.slots;
    }
    pub fn convOffset(t: Table, slot: usize, layer: usize) usize {
        return (slot * t.layers + layer) * t.convBytesPer();
    }
    pub fn recOffset(t: Table, slot: usize, layer: usize) usize {
        return (slot * t.layers + layer) * t.recBytesPer();
    }
};

// ---- tests (small dims; the arithmetic does not depend on the Flash Next sizes) ----

const small: Dims = .{ .nk = 2, .nv = 4, .dk = 8, .dv = 8, .taps = 4 };

fn fill(rng: *std.Random.DefaultPrng, xs: []f64, scale: f64) void {
    for (xs) |*x| x.* = (rng.random().float(f64) - 0.5) * 2.0 * scale;
}

const TestParams = struct {
    conv_w: []f64,
    a_log: []f64,
    dt_bias: []f64,
    norm_w: []f64,

    fn init(gpa: std.mem.Allocator, d: Dims, rng: *std.Random.DefaultPrng) !TestParams {
        const t: TestParams = .{
            .conv_w = try gpa.alloc(f64, d.channels() * d.taps),
            .a_log = try gpa.alloc(f64, d.nv),
            .dt_bias = try gpa.alloc(f64, d.nv),
            .norm_w = try gpa.alloc(f64, d.dv),
        };
        fill(rng, t.conv_w, 0.7);
        fill(rng, t.a_log, 1.0);
        fill(rng, t.dt_bias, 1.0);
        for (t.norm_w) |*w| w.* = 0.5 + rng.random().float(f64);
        return t;
    }
    fn deinit(t: TestParams, gpa: std.mem.Allocator) void {
        gpa.free(t.conv_w);
        gpa.free(t.a_log);
        gpa.free(t.dt_bias);
        gpa.free(t.norm_w);
    }
    fn view(t: TestParams) Params {
        return .{ .conv_w = t.conv_w, .a_log = t.a_log, .dt_bias = t.dt_bias, .norm_w = t.norm_w };
    }
};

test "conv is a causal depthwise filter over the stored rows" {
    const gpa = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(1);
    const tp = try TestParams.init(gpa, small, &rng);
    defer tp.deinit(gpa);
    const c = small.channels();
    const n = 7;
    const xs = try gpa.alloc(f64, n * c);
    defer gpa.free(xs);
    fill(&rng, xs, 1.0);
    var st = try State.init(gpa, small);
    defer st.deinit(gpa);
    const out = try gpa.alloc(f64, c);
    defer gpa.free(out);
    for (0..n) |t| {
        convStep(small, tp.view(), st.conv, xs[t * c ..][0..c], out);
        for (0..c) |ch| {
            var acc: f64 = 0;
            for (0..small.taps) |tap| {
                // input index t - (taps - 1) + tap; before the sequence it is zero
                const at: isize = @as(isize, @intCast(t)) - @as(isize, @intCast(small.taps - 1)) + @as(isize, @intCast(tap));
                if (at >= 0) acc += tp.conv_w[ch * small.taps + tap] * xs[@as(usize, @intCast(at)) * c + ch];
            }
            try std.testing.expectApproxEqAbs(silu(acc), out[ch], 1e-12);
        }
    }
    // the state ends as the last three raw rows
    for (0..3) |r| for (0..c) |ch| try std.testing.expectEqual(xs[(n - 3 + r) * c + ch], st.conv[r * c + ch]);
}

test "one scalar head follows S = g S + k (v - g S k) beta, o = S q" {
    const d: Dims = .{ .nk = 1, .nv = 1, .dk = 1, .dv = 1, .taps = 4 };
    var rec = [_]f64{0.5};
    const act = [_]f64{ 2.0, 3.0, 0.75 }; // q, k, v
    const decay = [_]f64{0.9};
    const beta = [_]f64{0.4};
    var y = [_]f64{0};
    deltaStep(d, &rec, &act, &decay, &beta, &y);
    const ik = 1.0 / @sqrt(9.0 + 1e-6);
    const iq = 1.0 / @sqrt(4.0 + 1e-6);
    const k = 3.0 * ik;
    const s0 = 0.5 * 0.9;
    const delta = (0.75 - s0 * k) * 0.4;
    const s1 = s0 + k * delta;
    try std.testing.expectApproxEqAbs(s1, rec[0], 1e-14);
    try std.testing.expectApproxEqAbs(s1 * 2.0 * iq, y[0], 1e-14);
}

test "with beta 1 and no decay the state returns v for the key just written" {
    const d: Dims = .{ .nk = 1, .nv = 2, .dk = 4, .dv = 3, .taps = 4 };
    var rng = std.Random.DefaultPrng.init(2);
    var rec: [2 * 3 * 4]f64 = undefined;
    fill(&rng, &rec, 1.0);
    var act: [2 * 4 + 2 * 3]f64 = undefined;
    fill(&rng, &act, 1.0);
    const decay = [_]f64{ 1, 1 };
    const beta = [_]f64{ 1, 1 };
    var y: [6]f64 = undefined;
    deltaStep(d, &rec, &act, &decay, &beta, &y);
    var kk: f64 = 0;
    for (act[4..8]) |k| kk += k * k;
    const ik = 1.0 / @sqrt(kk + 1e-6);
    for (0..2) |hv| for (0..3) |v| {
        var read: f64 = 0;
        for (0..4) |i| read += rec[(hv * 3 + v) * 4 + i] * act[4 + i] * ik;
        // S k = v * (|k|^2 / (|k|^2 + 1e-6)), the l2norm eps leaves a tiny shortfall
        try std.testing.expectApproxEqAbs(act[8 + hv * 3 + v], read, 1e-5);
    };
}

test "value heads share their key head in groups (repeat_interleave)" {
    const d: Dims = .{ .nk = 2, .nv = 4, .dk = 4, .dv = 2, .taps = 4 };
    var rng = std.Random.DefaultPrng.init(3);
    var act: [2 * 2 * 4 + 4 * 2]f64 = undefined;
    fill(&rng, &act, 1.0);
    const decay = [_]f64{ 0.8, 0.8, 0.8, 0.8 };
    const beta = [_]f64{ 0.5, 0.5, 0.5, 0.5 };
    var rec: [4 * 2 * 4]f64 = @splat(0);
    var y: [8]f64 = undefined;
    deltaStep(d, &rec, &act, &decay, &beta, &y);
    // zero state: kv = 0, so S = k (v beta)^T and o = (k . q) v beta, with q, k of key head hv / 2
    for (0..4) |hv| {
        const hk = hv / 2;
        const q = act[hk * 4 ..][0..4];
        const k = act[8 + hk * 4 ..][0..4];
        var qq: f64 = 0;
        var kk: f64 = 0;
        var qk: f64 = 0;
        for (q, k) |a, b| {
            qq += a * a;
            kk += b * b;
            qk += a * b;
        }
        const iq = 1.0 / @sqrt(qq + 1e-6) / @sqrt(4.0);
        const ik = 1.0 / @sqrt(kk + 1e-6);
        for (0..2) |v| try std.testing.expectApproxEqAbs(qk * iq * ik * act[16 + hv * 2 + v] * 0.5, y[hv * 2 + v], 1e-12);
    }
}

test "gated norm is w * rms-normalised y * sigmoid(z)" {
    const d: Dims = .{ .nk = 1, .nv = 1, .dk = 1, .dv = 4, .taps = 4 };
    const p: Params = .{ .conv_w = &.{}, .a_log = &.{}, .dt_bias = &.{}, .norm_w = &.{ 1, 2, 3, 4 } };
    const y = [_]f64{ 1, -2, 3, -4 };
    const z = [_]f64{ 0, 1, -1, 2 };
    var out: [4]f64 = undefined;
    gatedNorm(d, p, &y, &z, &out);
    const inv = 1.0 / @sqrt((1.0 + 4 + 9 + 16) / 4.0 + 1e-6);
    for (0..4) |i| try std.testing.expectApproxEqAbs(p.norm_w[i] * y[i] * inv * sigmoid(z[i]), out[i], 1e-14);
}

test "processing tokens in two calls equals one call, state included" {
    const gpa = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(4);
    const tp = try TestParams.init(gpa, small, &rng);
    defer tp.deinit(gpa);
    const n = 9;
    const c = small.channels();
    const zw = small.zWidth();
    const qkv = try gpa.alloc(f64, n * c);
    defer gpa.free(qkv);
    const z = try gpa.alloc(f64, n * zw);
    defer gpa.free(z);
    const a = try gpa.alloc(f64, n * small.nv);
    defer gpa.free(a);
    const b = try gpa.alloc(f64, n * small.nv);
    defer gpa.free(b);
    fill(&rng, qkv, 1.0);
    fill(&rng, z, 1.0);
    fill(&rng, a, 1.0);
    fill(&rng, b, 1.0);
    var whole = try State.init(gpa, small);
    defer whole.deinit(gpa);
    var split = try State.init(gpa, small);
    defer split.deinit(gpa);
    const o1 = try gpa.alloc(f64, n * zw);
    defer gpa.free(o1);
    const o2 = try gpa.alloc(f64, n * zw);
    defer gpa.free(o2);
    try tokens(gpa, small, tp.view(), whole, n, qkv, z, a, b, o1);
    const cut = 5;
    try tokens(gpa, small, tp.view(), split, cut, qkv, z, a, b, o2);
    try tokens(gpa, small, tp.view(), split, n - cut, qkv[cut * c ..], z[cut * zw ..], a[cut * small.nv ..], b[cut * small.nv ..], o2[cut * zw ..]);
    try std.testing.expectEqualSlices(f64, o1, o2);
    try std.testing.expectEqualSlices(f64, whole.rec, split.rec);
    try std.testing.expectEqualSlices(f64, whole.conv, split.conv);
}

test "state table addresses a (slot, layer) pair and Flash Next sizes" {
    const t: Table = .{ .dims = qwen, .layers = 36, .slots = 2 };
    try std.testing.expectEqual(@as(usize, 10240), qwen.channels());
    try std.testing.expectEqual(@as(usize, 3 * 10240 * 2), t.convBytesPer());
    try std.testing.expectEqual(@as(usize, 48 * 128 * 128 * 4), t.recBytesPer());
    try std.testing.expectEqual(t.recBytesPer() * 37, t.recOffset(1, 1));
    try std.testing.expectEqual(t.convBytesPer() * 36 * 2, t.convBytes());
}
