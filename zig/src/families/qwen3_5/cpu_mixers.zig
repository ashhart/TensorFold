//! The Qwen3.5-2B mixers on the CPU: the gated delta rule and the attention, fp32 sums, one bf16 rounding an output.
const std = @import("std");
const c = @import("config.zig");
const k = @import("cpu_kernels.zig");
pub const ops = @import("cpu_ops.zig");
const f32of = k.f32of;
const bf16of = k.bf16of;
const bf16of64 = ops.bf16of64;
const V8 = @Vector(8, f32);

/// The lanes' pairwise reduction, `v[i] = v[i + w] + v[i]` for w = 1, 2, 4, ...: the fixed order of every sum here.
fn reduce(v: []f32) f32 {
    var width: usize = 1;
    while (width < v.len) : (width *= 2) {
        var i: usize = 0;
        while (i + width < v.len) : (i += 2 * width) v[i] = v[i + width] + v[i];
    }
    return v[0];
}

/// The inputs a delta layer's conv reads for a segment: its three saved rows, then the chunk's qkv rows.
pub const ConvInput = struct {
    saved: []const u16,
    qkv: []const u16,

    fn at(in: ConvInput, row: usize, col: usize) f32 {
        return f32of(if (row < c.conv_taps - 1) in.saved[row * c.conv_dim + col] else in.qkv[(row - (c.conv_taps - 1)) * c.conv_dim + col]);
    }
};

/// The delta rule's inputs for row `w`: the causal conv and SiLU, the L2-normed q and k, v, the decay and beta.
pub fn gdnPre(in: ConvInput, w: usize, conv_w: []const u16, a: []const u16, b: []const u16, a_log: []const f32, dt_bias: []const u16, q: []u16, kk: []u16, v: []u16, g: []f32, beta: []u16, co: []u16) void {
    var vals: [c.linear_dim]f64 = undefined;
    for (0..3 * c.linear_heads) |head| {
        const col0 = head * c.linear_dim;
        for (&vals, 0..) |*val, j| {
            const col = col0 + j;
            var acc: f32 = 0;
            for (0..c.conv_taps) |tap| acc = @mulAdd(f32, f32of(conv_w[col * c.conv_taps + tap]), in.at(w + tap, col), acc);
            val.* = ops.silu(acc);
        }
        if (head < 2 * c.linear_heads) {
            var ss: f64 = 0;
            for (vals) |x| ss += x * x;
            const inv = 1.0 / @sqrt(ss / @as(f64, c.linear_dim) + 1e-6 / @as(f64, c.linear_dim));
            const is_q = head < c.linear_heads;
            const scale: f64 = if (is_q) 1.0 / @as(f64, c.linear_dim) else 1.0 / @sqrt(@as(f64, c.linear_dim));
            const out = if (is_q) q[head * c.linear_dim ..][0..c.linear_dim] else kk[(head - c.linear_heads) * c.linear_dim ..][0..c.linear_dim];
            for (out, vals) |*o, x| o.* = bf16of64(scale * (x * inv));
        } else {
            const hv = head - 2 * c.linear_heads;
            for (v[hv * c.linear_dim ..][0..c.linear_dim], vals) |*o, x| o.* = bf16of64(x);
            const s = @as(f64, f32of(a[hv])) + @as(f64, f32of(dt_bias[hv]));
            const softplus = @max(s, 0.0) + std.math.log1p(@exp(-@abs(s)));
            g[hv] = @floatCast(@exp(-@exp(@as(f64, a_log[hv])) * softplus));
            beta[hv] = bf16of64(ops.sigmoid(f32of(b[hv])));
        }
    }
    for (0..c.conv_taps - 1) |r| {
        const row = w + 1 + r;
        for (0..c.conv_dim) |col| co[r * c.conv_dim + col] = if (row < c.conv_taps - 1) in.saved[row * c.conv_dim + col] else in.qkv[(row - (c.conv_taps - 1)) * c.conv_dim + col];
    }
}

/// The recurrence's dot over 128: 32 lanes of four consecutive fused terms, then the pairwise reduction.
fn dot128(state: []const f32, x: *const [c.linear_dim]f32) f32 {
    const V32 = @Vector(32, f32);
    var lanes: V32 = @splat(0);
    inline for (0..4) |i| {
        var s: [32]f32 = undefined;
        var xv: [32]f32 = undefined;
        inline for (0..32) |t| {
            s[t] = state[4 * t + i];
            xv[t] = x[4 * t + i];
        }
        lanes = @mulAdd(V32, s, xv, lanes);
    }
    var v: [32]f32 = lanes;
    return reduce(&v);
}

/// The gated delta rule for one head over `rows` tokens on the fp32 state [dv][dk]; row states kept when recording.
pub fn gdnChain(state: []f32, q: []const u16, kk: []const u16, v: []const u16, g: []const f32, beta: []const u16, rows: usize, head: usize, y: []u16, saved: []f32, record: bool) void {
    const hd = c.linear_dim;
    var qc: [hd]f32 = undefined;
    var kc: [hd]f32 = undefined;
    for (0..rows) |node| {
        const stride = c.linear_heads * hd;
        for (0..hd) |i| {
            qc[i] = f32of(q[node * stride + head * hd + i]);
            kc[i] = f32of(kk[node * stride + head * hd + i]);
        }
        const gc = g[node * c.linear_heads + head];
        const bc = f32of(beta[node * c.linear_heads + head]);
        for (0..hd) |dv| {
            const row = state[dv * hd ..][0..hd];
            const gv: V8 = @splat(gc);
            inline for (0..hd / 8) |i| {
                const s: V8 = row[8 * i ..][0..8].*;
                row[8 * i ..][0..8].* = s * gv;
            }
            const kv_mem = dot128(row, &kc);
            const delta = (f32of(v[node * stride + head * hd + dv]) - kv_mem) * bc;
            const dv8: V8 = @splat(delta);
            inline for (0..hd / 8) |i| {
                const s: V8 = row[8 * i ..][0..8].*;
                const kv: V8 = kc[8 * i ..][0..8].*;
                row[8 * i ..][0..8].* = @mulAdd(V8, kv, dv8, s);
            }
            y[node * stride + head * hd + dv] = bf16of(dot128(row, &qc));
        }
        if (record) @memcpy(saved[node * c.linear_heads * hd * hd + head * hd * hd ..][0 .. hd * hd], state) else if (node + 1 == rows) @memcpy(saved[head * hd * hd ..][0 .. hd * hd], state);
    }
}

/// The delta layer's output for one row: each head's RMS norm of y, times silu(z), rounded to bf16 through fp32.
pub fn gdnPost(y: []const u16, z: []const u16, norm_w: []const u16, out: []u16) void {
    for (0..c.linear_heads) |hv| {
        var ss: f64 = 0;
        for (y[hv * c.linear_dim ..][0..c.linear_dim]) |yb| ss += @as(f64, f32of(yb)) * f32of(yb);
        const inv = 1.0 / @sqrt(ss / @as(f64, c.linear_dim) + c.eps);
        for (0..c.linear_dim) |d| {
            const x = @as(f64, f32of(norm_w[d])) * (@as(f64, f32of(y[hv * c.linear_dim + d])) * inv);
            out[hv * c.linear_dim + d] = bf16of64(ops.silu(f32of(z[hv * c.linear_dim + d])) * x);
        }
    }
}

/// The q and k norms over one head of 256: RMS with weight, rounded to bf16 through fp32.
pub fn headNorm(x: []const u16, weight: []const u16, out: []u16) void {
    var ss: f64 = 0;
    for (x[0..c.head_dim]) |xb| ss += @as(f64, f32of(xb)) * f32of(xb);
    const inv = 1.0 / @sqrt(ss / @as(f64, c.head_dim) + 1e-6);
    for (out[0..c.head_dim], x[0..c.head_dim], weight[0..c.head_dim]) |*o, xb, wb| o.* = bf16of64(@as(f64, f32of(wb)) * (@as(f64, f32of(xb)) * inv));
}

pub const Partial = struct { m: f32 = -std.math.inf(f32), l: f32 = 0, o: [c.head_dim]f32 = @splat(0) };

fn dot256(q: *const [c.head_dim]f32, key: []const u16) f32 {
    var lanes: [32]f32 = undefined;
    for (&lanes, 0..) |*lane, t| {
        var d: f32 = 0;
        for (0..8) |i| d = @mulAdd(f32, q[8 * t + i], f32of(key[8 * t + i]), d);
        lane.* = d;
    }
    return reduce(&lanes);
}

/// e^x for the softmax's fp32 weights: through float64, rounded once.
fn expf(x: f32) f32 {
    return @floatCast(@exp(@as(f64, x)));
}

/// Partials merged by the running max, each weighted by exp(m - max).
fn merge(parts: []const Partial) Partial {
    var mx = -std.math.inf(f32);
    for (parts) |p| mx = @max(mx, p.m);
    var out: Partial = .{ .m = mx };
    for (parts) |p| {
        const e: f32 = if (p.l > 0) expf(p.m - mx) else 0;
        out.l = @mulAdd(f32, p.l, e, out.l);
        for (&out.o, p.o) |*acc, o| acc.* = @mulAdd(f32, o, e, acc.*);
    }
    return out;
}

/// Keys per attention chunk, and the chunk partials a row's attention needs for `capacity` positions.
pub const chunk_keys = 128;
pub fn chunksFor(capacity: usize) usize {
    return (capacity + chunk_keys - 1) / chunk_keys;
}

/// One query row's attention over keys 0..last: chunks of 128 keys, four interleaved splits, merged alike.
pub fn attention(q: []const u16, keys: []const u16, values: []const u16, capacity: usize, kv_head: usize, last: usize, chunks: []Partial, out: []u16) void {
    var qf: [c.head_dim]f32 = undefined;
    for (&qf, q[0..c.head_dim]) |*f, b| f.* = f32of(b);
    const head_keys = keys[kv_head * capacity * c.head_dim ..];
    const head_values = values[kv_head * capacity * c.head_dim ..];
    const nch = chunksFor(last + 1);
    for (0..nch) |ch| {
        const k0 = ch * chunk_keys;
        const k1 = @min(k0 + chunk_keys, last + 1);
        var splits: [4]Partial = .{ .{}, .{}, .{}, .{} };
        for (&splits, 0..) |*sp, s| {
            var base = k0 + s;
            while (base < k1) : (base += 16) {
                var sc: [4]f32 = undefined;
                var bm = -std.math.inf(f32);
                for (0..4) |j| {
                    const pos = base + j * 4;
                    if (pos >= k1) break;
                    sc[j] = dot256(&qf, head_keys[pos * c.head_dim ..][0..c.head_dim]) * (1.0 / 16.0);
                    bm = @max(bm, sc[j]);
                }
                const mn = @max(sp.m, bm);
                const a = expf(sp.m - mn);
                sp.l *= a;
                for (&sp.o) |*o| o.* *= a;
                for (0..4) |j| {
                    const pos = base + j * 4;
                    if (pos >= k1) break;
                    const b = expf(sc[j] - mn);
                    sp.l += b;
                    const vr = head_values[pos * c.head_dim ..][0..c.head_dim];
                    for (&sp.o, vr) |*o, val| o.* = @mulAdd(f32, b, f32of(val), o.*);
                }
                sp.m = mn;
            }
        }
        chunks[ch] = merge(&splits);
    }
    const all = merge(chunks[0..nch]);
    for (out[0..c.head_dim], all.o) |*o, acc| o.* = bf16of64(@as(f64, acc) / @as(f64, all.l));
}

test "the recurrence's dot: 32 lanes of four contracted terms, reduced pairwise, against float64" {
    var state: [c.linear_dim]f32 = undefined;
    var x: [c.linear_dim]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    var want: f64 = 0;
    for (&state, &x) |*s, *v| {
        s.* = rnd.floatNorm(f32);
        v.* = f32of(bf16of(rnd.floatNorm(f32)));
        want += @as(f64, s.*) * @as(f64, v.*);
    }
    const got = dot128(&state, &x);
    try std.testing.expect(@abs(@as(f64, got) - want) <= 1e-4 * (1 + @abs(want)));
    var lanes: [32]f32 = undefined;
    for (&lanes, 0..) |*lane, t| {
        var acc: f32 = 0;
        for (0..4) |i| acc = @mulAdd(f32, state[4 * t + i], x[4 * t + i], acc);
        lane.* = acc;
    }
    try std.testing.expectEqual(reduce(&lanes), got);
}

test "attention over one key is that key's value and agrees with a float64 softmax across chunk and split boundaries" {
    const a = std.testing.allocator;
    const cap = 300;
    const keys = try a.alloc(u16, 2 * cap * c.head_dim);
    defer a.free(keys);
    const values = try a.alloc(u16, 2 * cap * c.head_dim);
    defer a.free(values);
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    for (keys) |*kb| kb.* = bf16of(rnd.floatNorm(f32) * 0.1);
    for (values) |*vb| vb.* = bf16of(rnd.floatNorm(f32));
    var q: [c.head_dim]u16 = undefined;
    for (&q) |*v| v.* = bf16of(rnd.floatNorm(f32) * 0.1);
    var out: [c.head_dim]u16 = undefined;
    var chunks: [3]Partial = undefined;
    attention(&q, keys, values, cap, 1, 0, &chunks, &out);
    try std.testing.expectEqualSlices(u16, values[cap * c.head_dim ..][0..c.head_dim], &out);
    const last = 259;
    attention(&q, keys, values, cap, 0, last, &chunks, &out);
    var qf: [c.head_dim]f64 = undefined;
    for (&qf, q) |*f, b| f.* = f32of(b);
    var scores: [last + 1]f64 = undefined;
    var mx = -std.math.inf(f64);
    for (&scores, 0..) |*s, pos| {
        var d: f64 = 0;
        for (0..c.head_dim) |i| d += qf[i] * f32of(keys[pos * c.head_dim + i]);
        s.* = d / 16.0;
        mx = @max(mx, s.*);
    }
    var denom: f64 = 0;
    var acc: [c.head_dim]f64 = @splat(0);
    for (scores, 0..) |s, pos| {
        const w = @exp(s - mx);
        denom += w;
        for (&acc, 0..) |*o, i| o.* += w * f32of(values[pos * c.head_dim + i]);
    }
    for (out, acc) |o, want| try std.testing.expect(@abs(f32of(o) - want / denom) <= 1e-2 + 1e-2 * @abs(want / denom));
}

test "the head norm and the delta output norm agree with float64 RMS norms to one bf16 rounding" {
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    var x: [c.head_dim]u16 = undefined;
    var w: [c.head_dim]u16 = undefined;
    for (&x) |*v| v.* = bf16of(rnd.floatNorm(f32) * 3);
    for (&w) |*v| v.* = bf16of(rnd.float(f32) + 0.5);
    var out: [c.head_dim]u16 = undefined;
    headNorm(&x, &w, &out);
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, f32of(v)) * f32of(v);
    const inv = 1.0 / @sqrt(ss / @as(f64, c.head_dim) + 1e-6);
    for (out, x, w) |o, xv, wv| {
        const want = @as(f64, f32of(wv)) * f32of(xv) * inv;
        try std.testing.expect(@abs(@as(f64, f32of(o)) - want) <= 0x1p-8 * @abs(want) + 0x1p-133);
    }
    var y: [c.hidden]u16 = undefined;
    var z: [c.hidden]u16 = undefined;
    var nw: [c.linear_dim]u16 = undefined;
    for (&y) |*v| v.* = bf16of(rnd.floatNorm(f32));
    for (&z) |*v| v.* = bf16of(rnd.floatNorm(f32));
    for (&nw) |*v| v.* = bf16of(rnd.float(f32) + 0.5);
    var post: [c.hidden]u16 = undefined;
    gdnPost(&y, &z, &nw, &post);
    for (0..c.linear_heads) |hv| {
        var hs: f64 = 0;
        for (y[hv * c.linear_dim ..][0..c.linear_dim]) |v| hs += @as(f64, f32of(v)) * f32of(v);
        const hinv = 1.0 / @sqrt(hs / @as(f64, c.linear_dim) + c.eps);
        for (0..c.linear_dim) |d| {
            const at = hv * c.linear_dim + d;
            const want = ops.silu(f32of(z[at])) * (@as(f64, f32of(nw[d])) * f32of(y[at]) * hinv);
            try std.testing.expect(@abs(@as(f64, f32of(post[at])) - want) <= 0x1p-8 * @abs(want) + 0x1p-133);
        }
    }
}
