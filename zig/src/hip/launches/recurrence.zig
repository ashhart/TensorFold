//! DeltaNet launches: gate, recurrence and the gated norm.

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

pub fn tf_gnorm_silu(
    l: *const Launcher,
    y: CF,
    z: C,
    out: P,
    kind: c_int,
    n: c_longlong,
    sigmoid: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    try a.add(ad(y));
    try a.add(ad(z));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(n);
    try a.add(sigmoid);
    try l.flat(l.op.gnorm_silu, n, s, &a);
}
