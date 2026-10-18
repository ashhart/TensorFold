//! Nemotron's Mamba-2 mixer between its in_proj output and its out_proj input, one fp32 recurrence at every length.

const std = @import("std");
const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const launches = @import("../launches/mamba.zig");
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const int = t.int;

/// The layer's dimensions on this rank (config.zig's mamba fields); the kernels serve one state width and tap count.
pub const Shape = struct {
    heads: usize,
    dh: usize,
    groups: usize,
    state: usize = launches.state,
    conv_kernel: usize = launches.taps,

    fn check(s: Shape) Error!void {
        if (s.state != launches.state or s.conv_kernel != launches.taps) return error.BadShape;
    }

    pub fn xd(s: Shape) usize {
        return s.heads * s.dh;
    }

    /// The conv's channels: x, then each group's B, then each group's C.
    pub fn cd(s: Shape) usize {
        return s.xd() + 2 * s.groups * launches.state;
    }

    /// An in_proj row: z (xd), the conv's input (cd), then dt (heads).
    pub fn proj(s: Shape) usize {
        return s.xd() + s.cd() + s.heads;
    }
};

/// The dt clamp (config's time_step_limit) and the group norm's eps.
pub const Limits = struct { dt_min: f32 = 0, dt_max: f32 = std.math.inf(f32), eps: f32 };

/// A stream's state in a layer: conv [3][cd] of the kind (the last three raw inputs), ssm [heads][dh][128] fp32.
pub const State = struct { conv: u64, ssm: u64 };

/// The layer's fp32 vectors: conv_w [cd][4] channel-major, conv_b [cd], dt_bias, a_log (raw), d [heads], gnorm [xd].
pub const Weights = struct { conv_w: u64, conv_b: u64, dt_bias: u64, a_log: u64, d: u64, gnorm: u64 };

/// The mixer's scratch for `rows`: xc [rows][cd] of the kind, dt [rows][heads] fp32, y [rows][xd] of the kind.
pub const Scratch = struct { xc: u64, dt: u64, y: u64 };

/// xc = round(silu(conv([state_in; P's x columns]))), state_out the last three raw inputs; state_out may be state_in.
pub fn mambaConv(
    o: Ops,
    src: Tensor,
    state_in: u64,
    xc: u64,
    w: Weights,
    state_out: u64,
    rows: usize,
    s: Shape,
) Error!void {
    try s.check();
    if (src.kind == .f32) return error.BadShape;
    try o.l.tf_mamba_conv(
        p(src.ptr),
        @backingInt(src.kind),
        p(state_in),
        p(xc),
        f(w.conv_w),
        f(w.conv_b),
        p(state_out),
        int(rows),
        int(s.proj()),
        int(s.xd()),
        int(s.cd()),
        o.stream,
    );
}

/// dt = clamp(softplus(P's dt columns + dt_bias), dt_min, dt_max), fp32 [rows][heads].
pub fn mambaDt(o: Ops, src: Tensor, dt: u64, w: Weights, rows: usize, s: Shape, lim: Limits) Error!void {
    try s.check();
    if (src.kind == .f32) return error.BadShape;
    try o.l.tf_mamba_dt(
        p(src.ptr),
        @backingInt(src.kind),
        f(w.dt_bias),
        f(dt),
        int(rows),
        int(s.proj()),
        int(s.xd() + s.cd()),
        int(s.heads),
        lim.dt_min,
        lim.dt_max,
        o.stream,
    );
}

/// The recurrence over xc's rows with dt, ssm in place: y = round(silu(z)) round(x D + s C) in the kind.
pub fn mambaScan(
    o: Ops,
    src: Tensor,
    xc: u64,
    dt: u64,
    ssm: u64,
    w: Weights,
    y: u64,
    rows: usize,
    s: Shape,
) Error!void {
    try s.check();
    if (src.kind == .f32) return error.BadShape;
    try o.l.tf_mamba_scan(
        p(src.ptr),
        p(xc),
        @backingInt(src.kind),
        f(dt),
        f(ssm),
        f(w.a_log),
        f(w.d),
        p(y),
        int(rows),
        int(s.proj()),
        int(s.xd()),
        int(s.cd()),
        int(s.heads),
        int(s.dh),
        int(s.groups),
        o.stream,
    );
}

/// out = round(norm * round(rmsnorm(y))) per group of xd / groups, the out_proj input.
pub fn mambaGroupNorm(o: Ops, y: Tensor, out: u64, w: Weights, rows: usize, s: Shape, lim: Limits) Error!void {
    try s.check();
    if (y.kind == .f32 or s.xd() % s.groups != 0) return error.BadShape;
    try o.l.tf_mamba_gnorm(
        p(y.ptr),
        @backingInt(y.kind),
        f(w.gnorm),
        p(out),
        int(rows),
        int(s.xd()),
        int(s.xd() / s.groups),
        lim.eps,
        o.stream,
    );
}

/// Where a window keeps each row's state: slot 0's conv tail and SSM state of this layer, and the bytes between slots.
pub const Snap = struct { conv: u64, ssm: u64, stride: usize };

/// `mambaMixer`'s bits in three launches; slot r + 1 of `snap` takes the state after row r, but the last row's.
pub fn mambaWindow(
    o: Ops,
    src: Tensor,
    st: State,
    sc: Scratch,
    out: u64,
    w: Weights,
    rows: usize,
    s: Shape,
    lim: Limits,
    snap: ?Snap,
) Error!void {
    try s.check();
    if (rows == 0) return;
    if (src.kind == .f32 or s.xd() % s.groups != 0) return error.BadShape;
    const sn: Snap = snap orelse .{ .conv = 0, .ssm = 0, .stride = 0 };
    try o.l.tf_mamba_window(.{
        .p = src.ptr,
        .kind = @backingInt(src.kind),
        .conv = st.conv,
        .ssm = st.ssm,
        .xc = sc.xc,
        .dt = sc.dt,
        .y = sc.y,
        .cw = w.conv_w,
        .cb = w.conv_b,
        .dt_bias = w.dt_bias,
        .a_log = w.a_log,
        .dsk = w.d,
        .snap_conv = sn.conv,
        .snap_ssm = sn.ssm,
        .stride = @intCast(sn.stride),
        .rows = int(rows),
        .proj = int(s.proj()),
        .xd = int(s.xd()),
        .cd = int(s.cd()),
        .heads = int(s.heads),
        .dh = int(s.dh),
        .groups = int(s.groups),
        .lo = lim.dt_min,
        .hi = lim.dt_max,
    }, .{ .gw = w.gnorm, .out = out, .gs = int(s.xd() / s.groups), .eps = lim.eps }, o.stream);
}

/// The whole mixer over `rows` of in_proj output `src` into `out`, both states advanced in place.
pub fn mambaMixer(
    o: Ops,
    src: Tensor,
    st: State,
    sc: Scratch,
    out: u64,
    w: Weights,
    rows: usize,
    s: Shape,
    lim: Limits,
) Error!void {
    if (rows == 0) return;
    try mambaConv(o, src, st.conv, sc.xc, w, st.conv, rows, s);
    try mambaDt(o, src, sc.dt, w, rows, s, lim);
    try mambaScan(o, src, sc.xc, sc.dt, st.ssm, w, sc.y, rows, s);
    try mambaGroupNorm(o, .{ .ptr = sc.y, .kind = src.kind }, out, w, rows, s, lim);
}
