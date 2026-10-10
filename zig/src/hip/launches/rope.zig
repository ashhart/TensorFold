//! RoPE launches: decode, prefill and the window's fused q / k norm and rotation.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

pub fn tf_qk_rope(l: *const Launcher, src: C, kind: c_int, s_row: c_longlong, s_head: c_int, weight: CF, eps: f32, rows: c_int, heads: c_int, width: c_int, rotary: c_int, theta: f32, pos: CI, wide: F, cache: P, total: c_int, s: S) Error!void {
    if (width > 512) return invalid("qk_rope");
    var a: Args = .{};
    try a.add(ad(src));
    try a.add(kind);
    try a.add(s_row);
    try a.add(s_head);
    try a.add(ad(weight));
    try a.add(eps);
    try a.add(rows);
    try a.add(heads);
    try a.add(width);
    try a.add(rotary);
    try a.add(theta);
    try a.add(ad(pos));
    try a.add(ad(wide));
    try a.add(ad(cache));
    try a.add(total);
    // a wave a (row, head), 8 a block
    try l.go(l.op.qk_rope, dim(cdiv(@as(i64, rows) * heads, 8), 1, 1), dim(256, 1, 1), 0, s, &a);
}

pub fn tf_rope_decode(l: *const Launcher, x: CF, y: F, rows: c_int, width: c_int, rotary: c_int, pos: c_int, theta: f32, s: S, pos_dev: CI, per: c_int) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(y));
    try a.add(width);
    try a.add(rotary);
    try a.add(pos);
    try a.add(theta);
    try a.add(ad(pos_dev));
    try a.add(per);
    try l.go(l.rope_decode, dim(rows, 1, 1), dim(32, 1, 1), 0, s, &a);
}

pub fn tf_rope_prefill(l: *const Launcher, x: C, kind: c_int, out: P, out_kind: c_int, s_head: c_longlong, s_row: c_longlong, len: c_int, heads: c_int, d: c_int, rotary: c_int, pos0: c_int, theta: f32, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(kind);
    try a.add(ad(out));
    try a.add(out_kind);
    try a.add(s_head);
    try a.add(s_row);
    try a.add(len);
    try a.add(heads);
    try a.add(d);
    try a.add(rotary);
    try a.add(pos0);
    try a.add(theta);
    try l.flat(l.op.rope_prefill, @as(i64, len) * heads * d, s, &a);
}
