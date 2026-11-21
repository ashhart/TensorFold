//! Float64 formulas of Inkling 2's attention (eager_attention_forward, relative logits) over the whole stream, no ring.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;

pub const dim = 128;
pub const d_rel = 16;

pub const Config = struct {
    rows: usize,
    pos0: usize,
    heads: usize,
    kv_heads: usize,
    window: usize,
    extent: usize,
    scale: f32,
};

/// [rows][heads]: q/out x128, rel x16, sink x1, gate x128; k/v [position][kv_heads][128]; proj [16][extent].
pub const Data = struct {
    q: []const u16,
    rel: []const u16,
    sink: []const u16,
    gate: []const u16,
    k: []const u16,
    v: []const u16,
    proj: []const u16,
};

/// `want`; per output `att` (before the gate), `err` (fp32 bound on it), `sig` (gate sigmoid), all [rows][heads][128].
pub const Result = struct { want: []f64, att: []f64, err: []f64, sig: []f64 };

fn val(w: u16) f64 {
    return c.fromBf16(w);
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// Position x is visible to the query at p.
fn visible(x: usize, p: usize, window: usize) bool {
    return x <= p and (window == 0 or x + window > p);
}

fn firstVisible(p: usize, window: usize) usize {
    var x: usize = 0;
    while (!visible(x, p, window)) x += 1;
    return x;
}

/// u = fp32 unit roundoff; a weight's relative error is 4u + u|x|, so output moves by vmax * expm1(2e) plus roundings.
const eps32 = 0x1p-24;

pub fn attention(a: std.mem.Allocator, cfg: Config, d: Data, out: Result) !void {
    const group = cfg.heads / cfg.kv_heads;
    const total = cfg.pos0 + cfg.rows;
    const lo_min = firstVisible(cfg.pos0, cfg.window);
    const span = total - lo_min;
    const ks = try a.alloc(f64, span * dim);
    defer a.free(ks);
    const vs = try a.alloc(f64, span * dim);
    defer a.free(vs);
    const w = try a.alloc(f64, span);
    defer a.free(w);
    const pj = try a.alloc(f64, d_rel * cfg.extent);
    defer a.free(pj);
    for (pj, d.proj) |*o, word| o.* = val(word);
    const scale: f64 = cfg.scale;
    const inf = std.math.inf(f64);
    const nan = std.math.nan(f64);
    var vmax: [dim]f64 = undefined;
    for (0..cfg.kv_heads) |kvh| {
        for (0..span) |i| for (0..dim) |j| {
            ks[i * dim + j] = val(d.k[((lo_min + i) * cfg.kv_heads + kvh) * dim + j]);
            vs[i * dim + j] = val(d.v[((lo_min + i) * cfg.kv_heads + kvh) * dim + j]);
        };
        @memset(&vmax, 0);
        for (0..span) |i| for (0..dim) |j| {
            vmax[j] = @max(vmax[j], @abs(vs[i * dim + j]));
        };
        for (0..cfg.rows) |r| {
            const p = cfg.pos0 + r;
            const lo = firstVisible(p, cfg.window);
            const keys = p + 1 - lo;
            for (kvh * group..(kvh + 1) * group) |h| {
                const base = (r * cfg.heads + h) * dim;
                var q: [dim]f64 = undefined;
                for (&q, 0..) |*o, j| o.* = val(d.q[base + j]);
                var rel: [d_rel]f64 = undefined;
                for (&rel, 0..) |*o, i| o.* = val(d.rel[(r * cfg.heads + h) * d_rel + i]);
                const sink = val(d.sink[r * cfg.heads + h]);

                var bad = std.math.isNan(sink);
                var top: f64 = if (bad) -inf else sink;
                var low: f64 = sink;
                var worst_mag: f64 = 0;
                var worst_abs: f64 = 0;
                for (0..keys) |i| {
                    const x = lo + i;
                    const kk = ks[(x - lo_min) * dim ..][0..dim];
                    var dot: f64 = 0;
                    var mag: f64 = 0;
                    for (0..dim) |j| {
                        dot += q[j] * kk[j];
                        mag += @abs(q[j] * kk[j]);
                    }
                    var s = dot * scale;
                    const dist = p - x;
                    if (dist < cfg.extent) {
                        var b: f64 = 0;
                        for (0..d_rel) |t| b += rel[t] * pj[t * cfg.extent + dist];
                        s += bf(b);
                    }
                    w[i] = s;
                    if (std.math.isNan(s)) {
                        bad = true;
                    } else {
                        top = @max(top, s);
                        low = @min(low, s);
                        worst_mag = @max(worst_mag, mag * scale);
                        worst_abs = @max(worst_abs, @abs(s));
                    }
                }
                if (bad or top == inf or top == -inf) {
                    for (0..dim) |j| {
                        out.want[base + j] = nan;
                        out.att[base + j] = nan;
                        out.err[base + j] = 0;
                        out.sig[base + j] = 0;
                    }
                    continue;
                }
                var z = @exp(sink - top);
                for (w[0..keys]) |*o| {
                    o.* = @exp(o.* - top);
                    z += o.*;
                }
                var acc: [dim]f64 = @splat(0);
                for (0..keys) |i| {
                    const vv = vs[(lo + i - lo_min) * dim ..][0..dim];
                    for (0..dim) |j| acc[j] += w[i] * vv[j];
                }
                const e_s = c.gamma(131, eps32) * worst_mag + eps32 * worst_abs;
                const range = top - low;
                const blocks: f64 = @floatFromInt((@min(keys, 2048) + 127) / 128);
                const chain = 64 + blocks * (40 + range);
                const share = std.math.expm1(2 * e_s) + 2 * (4 * eps32 + eps32 * range) + 2 * c.gamma(chain, eps32) + 4 * eps32;
                for (0..dim) |j| {
                    const att = acc[j] / z;
                    const gv = val(d.gate[base + j]);
                    const sig = 1.0 / (1.0 + @exp(-gv));
                    out.att[base + j] = att;
                    out.err[base + j] = vmax[j] * share;
                    out.sig[base + j] = sig;
                    out.want[base + j] = att * sig;
                }
            }
        }
    }
}

fn run1(rows: usize, pos0: usize, window: usize, q: []const u16, rel: []const u16, sink: []const u16, gate: []const u16, k: []const u16, v: []const u16, proj: []const u16, want: []f64) !void {
    const a = std.testing.allocator;
    const n = want.len;
    const att = try a.alloc(f64, n);
    defer a.free(att);
    const err = try a.alloc(f64, n);
    defer a.free(err);
    const sig = try a.alloc(f64, n);
    defer a.free(sig);
    try attention(a, .{ .rows = rows, .pos0 = pos0, .heads = 1, .kv_heads = 1, .window = window, .extent = 1, .scale = 1 }, .{ .q = q, .rel = rel, .sink = sink, .gate = gate, .k = k, .v = v, .proj = proj }, .{ .want = want, .att = att, .err = err, .sig = sig });
}

test "attention: one key and a zero sink split the mass, the gate halves it again" {
    const one = c.toBf16(1);
    const q: [dim]u16 = @splat(0);
    const k: [dim]u16 = @splat(0);
    const v: [dim]u16 = @splat(one);
    const rel: [d_rel]u16 = @splat(0);
    const sink = [_]u16{0};
    const gate: [dim]u16 = @splat(0);
    const proj: [d_rel]u16 = @splat(0);
    var want: [dim]f64 = undefined;
    try run1(1, 0, 0, &q, &rel, &sink, &gate, &k, &v, &proj, &want);
    try std.testing.expectApproxEqAbs(0.25, want[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.25, want[dim - 1], 1e-12);
}

test "attention: a window of one sees only the row's own key" {
    const q: [2 * dim]u16 = @splat(0);
    const k: [2 * dim]u16 = @splat(0);
    var v: [2 * dim]u16 = @splat(0);
    const rel: [2 * d_rel]u16 = @splat(0);
    const sink = [_]u16{ c.toBf16(-100), c.toBf16(-100) };
    const gate: [2 * dim]u16 = @splat(c.toBf16(100));
    const proj: [d_rel]u16 = @splat(0);
    for (0..dim) |j| {
        v[j] = c.toBf16(5);
        v[dim + j] = c.toBf16(7);
    }
    var want: [2 * dim]f64 = undefined;
    try run1(2, 0, 1, &q, &rel, &sink, &gate, &k, &v, &proj, &want);
    try std.testing.expectApproxEqAbs(5.0, want[0], 1e-9);
    try std.testing.expectApproxEqAbs(7.0, want[dim], 1e-9);
    // a global window averages the two keys (equal scores) for the second row
    try run1(2, 0, 0, &q, &rel, &sink, &gate, &k, &v, &proj, &want);
    try std.testing.expectApproxEqAbs(6.0, want[dim], 1e-9);
}

test "attention: a NaN or +inf sink gives NaN, a -inf sink weighs 0" {
    const q: [dim]u16 = @splat(0);
    const k: [dim]u16 = @splat(0);
    const v: [dim]u16 = @splat(0);
    const rel: [d_rel]u16 = @splat(0);
    const gate: [dim]u16 = @splat(0);
    const proj: [d_rel]u16 = @splat(0);
    var want: [dim]f64 = undefined;
    try run1(1, 0, 0, &q, &rel, &[_]u16{0xff80}, &gate, &k, &v, &proj, &want);
    // the key's logit is 0, finite: the sink at -inf weighs 0
    try std.testing.expect(std.math.isFinite(want[0]));
    try run1(1, 0, 0, &q, &rel, &[_]u16{0x7fc0}, &gate, &k, &v, &proj, &want);
    try std.testing.expect(std.math.isNan(want[0]));
    try run1(1, 0, 0, &q, &rel, &[_]u16{0x7f80}, &gate, &k, &v, &proj, &want);
    try std.testing.expect(std.math.isNan(want[0]));
}
