//! DeltaNet launches: gate, serial and chunked recurrence, and the gated norm.

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const tri = util.tri;
const invalid = util.invalid;

/// The chunked DeltaNet's chunk, in rows.
pub const gdn_chunk = 64;

/// Byte sizes of the chunked DeltaNet's scratch, rows padded to whole chunks.
pub const GdnScratch = struct { qk: usize, gc: usize, wu: usize, h: usize, total: usize };

pub fn gdnScratch(length: usize, key_heads: usize, value_heads: usize) GdnScratch {
    const chunks = cdiv(length, gdn_chunk);
    const rows = chunks * gdn_chunk;
    const qk = std.mem.alignForward(usize, rows * key_heads * 128 * 2, 256);
    const gc = std.mem.alignForward(usize, length * value_heads * 4, 256);
    const wu = std.mem.alignForward(usize, rows * value_heads * 128 * 2, 256);
    const h = chunks * value_heads * 128 * 128 * 2;
    return .{ .qk = qk, .gc = gc, .wu = wu, .h = h, .total = 3 * qk + gc + 4 * wu + h };
}

pub fn tf_gdn_gate(l: *const Launcher, av: C, bv: C, kind: c_int, a_log: CF, dt_bias: CF, gate: F, beta: F, count: c_int, heads: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(av));
    try a.add(ad(bv));
    try a.add(ad(a_log));
    try a.add(ad(dt_bias));
    try a.add(ad(gate));
    try a.add(ad(beta));
    try a.add(count);
    try a.add(heads);
    try l.go(l.gdn_gate[tri(kind)], dim(cdiv(count, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_gated_delta(l: *const Launcher, q: CF, k: CF, v: CF, gate: CF, beta: CF, state: F, y: F, batch: c_int, length: c_int, key_heads: c_int, value_heads: c_int, dk: c_int, dv: c_int, s: S, states: F) Error!void {
    if ((dk != 16 and dk != 128) or dv < 1 or key_heads < 1 or @rem(value_heads, key_heads) != 0 or batch < 1 or length < 1) {
        return invalid("gated delta");
    }
    var a: Args = .{};
    try a.add(ad(q));
    try a.add(ad(k));
    try a.add(ad(v));
    try a.add(ad(gate));
    try a.add(ad(beta));
    try a.add(ad(state));
    try a.add(ad(y));
    try a.add(length);
    try a.add(key_heads);
    try a.add(value_heads);
    try a.add(dv);
    try a.add(ad(states));
    const wide: usize = if (dk == 128) 0 else 1;
    // Eight rows share one k/q tile; a short dv stays on the one-row wave, which is the same scan.
    if (@rem(dv, 8) == 0) {
        try l.go(l.gated_delta_tile[wide], dim(@divExact(dv, 8), value_heads, batch), dim(32, 8, 1), 0, s, &a);
    } else {
        try l.go(l.gated_delta_wave[wide], dim(dv, value_heads, batch), dim(32, 1, 1), 0, s, &a);
    }
}

/// The chunked DeltaNet prefill of `length` rows (dk = dv = 128, batch 1), `scratch` of `gdnScratch` bytes.
pub fn gdnChunked(l: *const Launcher, q: CF, k: CF, v: CF, gate: CF, beta: CF, state: F, y: F, length: usize, key_heads: usize, value_heads: usize, scratch: u64, s: S) Error!void {
    if (length < 1 or key_heads < 1 or @rem(value_heads, key_heads) != 0) return invalid("gdn chunked");
    const sc = gdnScratch(length, key_heads, value_heads);
    const qh = scratch;
    const kh = qh + sc.qk;
    const kt = kh + sc.qk;
    const vh = kt + sc.qk;
    const gc = vh + sc.wu;
    const w = gc + sc.gc;
    const u = w + sc.wu;
    const vt = u + sc.wu;
    const hb = vt + sc.wu;
    const chunks = cdiv(length, gdn_chunk);
    const len: c_int = @intCast(length);
    const hk: c_int = @intCast(key_heads);
    const hv: c_int = @intCast(value_heads);
    var a: Args = .{};
    try a.add(ad(q));
    try a.add(ad(k));
    try a.add(ad(v));
    try a.add(qh);
    try a.add(kh);
    try a.add(vh);
    try a.add(len);
    try a.add(hk);
    try a.add(hv);
    try l.go(l.gdn_chunked[0], dim(cdiv(length * (key_heads + value_heads) * 32, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
    var t: Args = .{};
    try t.add(kh);
    try t.add(kt);
    try t.add(len);
    try t.add(hk);
    try l.go(l.gdn_chunked[1], dim(chunks, key_heads, 1), dim(256, 1, 1), 0, s, &t);
    var b: Args = .{};
    try b.add(kh);
    try b.add(vh);
    try b.add(ad(beta));
    try b.add(ad(gate));
    try b.add(gc);
    try b.add(w);
    try b.add(u);
    try b.add(len);
    try b.add(hk);
    try b.add(hv);
    try l.go(l.gdn_chunked[2], dim(chunks, value_heads, 1), dim(256, 1, 1), 0, s, &b);
    var c: Args = .{};
    try c.add(kt);
    try c.add(w);
    try c.add(u);
    try c.add(vt);
    try c.add(gc);
    try c.add(ad(state));
    try c.add(hb);
    try c.add(len);
    try c.add(hk);
    try c.add(hv);
    try l.go(l.gdn_chunked[3], dim(8, value_heads, 1), dim(256, 1, 1), 0, s, &c);
    var d: Args = .{};
    try d.add(qh);
    try d.add(kh);
    try d.add(vt);
    try d.add(hb);
    try d.add(gc);
    try d.add(ad(y));
    try d.add(len);
    try d.add(hk);
    try d.add(hv);
    try l.go(l.gdn_chunked[4], dim(chunks, value_heads, 1), dim(256, 1, 1), 0, s, &d);
}

pub fn tf_gdn_gate_prefill(l: *const Launcher, av: C, bv: C, kind: c_int, a_log: CF, dt_bias: CF, gate: F, beta: F, count: c_int, heads: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(av));
    try a.add(ad(bv));
    try a.add(kind);
    try a.add(ad(a_log));
    try a.add(ad(dt_bias));
    try a.add(ad(gate));
    try a.add(ad(beta));
    try a.add(count);
    try a.add(heads);
    try l.flat(l.op.gdn_gate_prefill, @as(i64, count) * heads, s, &a);
}

pub fn tf_gnorm_silu(l: *const Launcher, y: CF, z: C, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(y));
    try a.add(ad(z));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(n);
    try l.flat(l.op.gnorm_silu, n, s, &a);
}
