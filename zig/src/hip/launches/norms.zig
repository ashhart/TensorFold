//! Norm launches and the decode tails that fuse a norm.

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

pub fn tf_rms(l: *const Launcher, x: C, weight: CF, y: P, kind: c_int, rows: c_int, width: c_int, eps: f32, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    if (width <= 1024) {
        // a wave a row, rms_kernel's sums
        try a.add(kind);
        try a.add(ad(weight));
        try a.add(ad(y));
        try a.add(rows);
        try a.add(width);
        try a.add(eps);
        return l.go(l.op.rms_rows, dim(cdiv(rows, 8), 1, 1), dim(256, 1, 1), 0, s, &a);
    }
    try a.add(ad(weight));
    try a.add(ad(y));
    try a.add(width);
    try a.add(eps);
    try l.go(l.rms[tri(kind)], dim(rows, 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// Two sets of fp32 rows of `width` (at most 1024), each with its own weight, normed in one launch.
pub fn tf_rms2(l: *const Launcher, x0: CF, w0: CF, y0: F, x1: CF, w1: CF, y1: F, rows: c_int, width: c_int, eps: f32, s: S) Error!void {
    if (width > 1024) return invalid("rms2");
    var a: Args = .{};
    try a.add(ad(x0));
    try a.add(ad(w0));
    try a.add(ad(y0));
    try a.add(ad(x1));
    try a.add(ad(w1));
    try a.add(ad(y1));
    try a.add(rows);
    try a.add(width);
    try a.add(eps);
    try l.go(l.dec.rms2, dim(cdiv(rows, 8), 2, 1), dim(256, 1, 1), 0, s, &a);
}

/// out = rms(y) * weight * round(silu(z)) rounded to the kind, rows of `width` (at most 1024).
pub fn tf_gnorm_out(l: *const Launcher, y: CF, weight: CF, z: C, out: P, kind: c_int, rows: c_int, width: c_int, eps: f32, s: S) Error!void {
    if (width > 1024) return invalid("gnorm out");
    var a: Args = .{};
    try a.add(ad(y));
    try a.add(ad(weight));
    try a.add(ad(z));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(rows);
    try a.add(width);
    try a.add(eps);
    try l.go(l.dec.gnorm_out, dim(cdiv(rows, 8), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// x = round(x + t), t being y or the weighted sum of the row's `slots`; normed = rms(x) * weight, a block a row.
pub fn tf_tail(l: *const Launcher, x: P, y: C, wts: CF, weight: CF, normed: P, kind: c_int, rows: c_int, slots: c_int, width: c_int, eps: f32, s: S) Error!void {
    if (width > 8192 or rows < 1) return invalid("tail");
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(y));
    try a.add(ad(wts));
    try a.add(ad(weight));
    try a.add(ad(normed));
    try a.add(kind);
    try a.add(slots);
    try a.add(width);
    try a.add(eps);
    try l.go(l.dec.tail, dim(rows, 1, 1), dim(512, 1, 1), 0, s, &a);
}
