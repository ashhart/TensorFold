//! Causal conv launches: decode, window rows, prefill and the split with its norms and gates.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const C = util.C;
const F = util.F;
const CF = util.CF;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;
const PlanArgs = @import("plan.zig").PlanArgs;

/// decode.hip's ConvArgs: the linear attention's conv launch.
pub const ConvArgs = extern struct {
    x: u64,
    kind: c_int,
    weight: u64,
    state: u64,
    states: u64 = 0,
    qn: u64,
    kn: u64,
    v: u64,
    channels: c_int,
    kernel: c_int,
    rows: c_int,
    kw: c_int,
    vw: c_int,
    qw: u64 = 0,
    kw_w: u64 = 0,
    eps: f32 = 0,
    norm: c_int = 0,
    ga: u64 = 0,
    gb: u64 = 0,
    a_log: u64 = 0,
    dt_bias: u64 = 0,
    gate: u64 = 0,
    beta: u64 = 0,
    gcount: c_int = 0,
    heads: c_int = 0,
    /// A round's plan: a grid row a slot, its rows and state from the plan (no plan: this launch's own).
    plan: PlanArgs = .{},
    layer: c_int = 0,
    slots: c_int = 1,
};

pub fn tf_conv_decode(l: *const Launcher, x: CF, weight: CF, state: F, y: F, batch: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(weight));
    try a.add(ad(state));
    try a.add(ad(y));
    try a.add(channels);
    try a.add(kernel);
    try l.go(l.conv_decode, dim(cdiv(channels, 128), batch, 1), dim(128, 1, 1), 0, s, &a);
}

pub fn tf_conv_rows(l: *const Launcher, x: CF, weight: CF, state: F, y: F, states: F, rows: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(weight));
    try a.add(ad(state));
    try a.add(ad(y));
    try a.add(ad(states));
    try a.add(channels);
    try a.add(kernel);
    try a.add(rows);
    try l.go(l.conv_rows, dim(cdiv(channels, 128), 1, 1), dim(128, 1, 1), 0, s, &a);
}

/// A window's rows through the linear attention's conv, split into q, k and v, normed and gated when `c` asks.
pub fn tf_conv_split(l: *const Launcher, c: ConvArgs, s: S) Error!void {
    if (c.kernel < 1 or c.kernel > 8 or (c.norm != 0 and (c.norm != 128 or @rem(c.kw, 128) != 0))) return invalid("conv split");
    var a: Args = .{};
    try a.add(c);
    const gate_blocks = if (c.ga != 0) cdiv(c.gcount, 128) else 0;
    try l.go(l.dec.conv_split, dim(cdiv(c.channels, 128) + gate_blocks, c.slots, 1), dim(128, 1, 1), 0, s, &a);
}

pub fn tf_conv_prefill(l: *const Launcher, x: C, kind: c_int, weight: CF, state: CF, out: F, new_state: F, len: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(kind);
    try a.add(ad(weight));
    try a.add(ad(state));
    try a.add(ad(out));
    try a.add(ad(new_state));
    try a.add(len);
    try a.add(channels);
    try a.add(kernel);
    // a thread a channel over 16 rows (conv_prefill_kernel's kConvRows), at most 8 taps
    if (kernel > 8) return invalid("conv_prefill");
    try l.go(l.op.conv_prefill, dim(cdiv(channels, 256), @max(1, cdiv(len, 16)), 1), dim(256, 1, 1), 0, s, &a);
}
