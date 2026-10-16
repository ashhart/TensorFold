//! Nemotron's Mamba-2 launches: the conv, dt, the fp32 state recurrence and the group norm (recurrence/mamba.hip).

const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
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
const invalid = util.invalid;

/// The conv's taps (conv_kernel), the rows a conv thread slides over, the SSM state width and a scan block's rows.
pub const taps = 4;
pub const conv_rows = 16;
pub const state = 128;
pub const scan_rows = 32;

pub const Fns = struct {
    conv: Function,
    dt: Function,
    scan: Function,
    gnorm: Function,

    pub fn load(m: Module) Error!Fns {
        return .{
            .conv = try m.function("tf_mamba_conv"),
            .dt = try m.function("tf_mamba_dt"),
            .scan = try m.function("tf_mamba_scan"),
            .gnorm = try m.function("tf_mamba_gnorm"),
        };
    }
};

/// xc = round(silu(conv(state_in rows, P's x columns))), state_out the last three raw inputs (state_in itself allowed).
pub fn tf_mamba_conv(
    l: *const Launcher,
    p: C,
    kind: c_int,
    state_in: C,
    xc: P,
    cw: CF,
    cb: CF,
    state_out: P,
    rows: c_int,
    proj: c_int,
    xoff: c_int,
    cd: c_int,
    s: S,
) Error!void {
    if (rows < 1 or cd < 1 or proj < xoff + cd) return invalid("mamba conv");
    var a: Args = .{};
    try a.add(ad(p));
    try a.add(kind);
    try a.add(ad(state_in));
    try a.add(ad(xc));
    try a.add(ad(cw));
    try a.add(ad(cb));
    try a.add(ad(state_out));
    try a.add(rows);
    try a.add(proj);
    try a.add(xoff);
    try a.add(cd);
    try l.go(l.mamba.conv, dim(cdiv(cd, 256), cdiv(rows, conv_rows), 1), dim(256, 1, 1), 0, s, &a);
}

/// dt[row][head] = clamp(softplus(P[row][dtoff + head] + dt_bias[head]), lo, hi).
pub fn tf_mamba_dt(
    l: *const Launcher,
    p: C,
    kind: c_int,
    dt_bias: CF,
    dt: F,
    rows: c_int,
    proj: c_int,
    dtoff: c_int,
    heads: c_int,
    lo: f32,
    hi: f32,
    s: S,
) Error!void {
    if (rows < 1 or heads < 1 or proj < dtoff + heads) return invalid("mamba dt");
    var a: Args = .{};
    try a.add(ad(p));
    try a.add(kind);
    try a.add(ad(dt_bias));
    try a.add(ad(dt));
    try a.add(rows);
    try a.add(proj);
    try a.add(dtoff);
    try a.add(heads);
    try a.add(lo);
    try a.add(hi);
    try l.flat(l.mamba.dt, @as(i64, rows) * heads, s, &a);
}

/// The recurrence over `rows` in order, state [heads][dh][128] fp32 in place; y = round(silu(z)) round(x D + s C).
pub fn tf_mamba_scan(
    l: *const Launcher,
    p: C,
    xc: C,
    kind: c_int,
    dt: CF,
    st: F,
    a_log: CF,
    dsk: CF,
    y: P,
    rows: c_int,
    proj: c_int,
    xd: c_int,
    cd: c_int,
    heads: c_int,
    dh: c_int,
    groups: c_int,
    s: S,
) Error!void {
    if (rows < 1 or heads < 1 or groups < 1 or @rem(
        heads,
        groups,
    ) != 0 or dh < 1 or @rem(dh, scan_rows) != 0) return invalid("mamba scan");
    if (xd != heads * dh or cd != xd + 2 * groups * state or proj < xd + cd + heads) return invalid("mamba scan");
    var a: Args = .{};
    try a.add(ad(p));
    try a.add(ad(xc));
    try a.add(kind);
    try a.add(ad(dt));
    try a.add(ad(st));
    try a.add(ad(a_log));
    try a.add(ad(dsk));
    try a.add(ad(y));
    try a.add(rows);
    try a.add(proj);
    try a.add(xd);
    try a.add(cd);
    try a.add(heads);
    try a.add(dh);
    try a.add(groups);
    try l.go(l.mamba.scan, dim(heads, @divExact(dh, scan_rows), 1), dim(scan_rows * 4, 1, 1), 0, s, &a);
}

/// out = round(w * round(x / rms(x))) per group of gs (at most 1024) of each row's xd columns.
pub fn tf_mamba_gnorm(
    l: *const Launcher,
    x: C,
    kind: c_int,
    w: CF,
    out: P,
    rows: c_int,
    xd: c_int,
    gs: c_int,
    eps: f32,
    s: S,
) Error!void {
    if (rows < 1 or gs < 1 or gs > 1024 or @rem(xd, gs) != 0) return invalid("mamba gnorm");
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(kind);
    try a.add(ad(w));
    try a.add(ad(out));
    try a.add(xd);
    try a.add(gs);
    try a.add(eps);
    try l.go(l.mamba.gnorm, dim(rows, @divExact(xd, gs), 1), dim(128, 1, 1), 0, s, &a);
}
