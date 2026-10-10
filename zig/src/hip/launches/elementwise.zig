//! Element-wise launches: embedding rows, casts, sums, column copies and the draft projection.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const CF = util.CF;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;

pub fn tf_embed_rows(l: *const Launcher, words: C, scale: C, bias: C, scale_kind: c_int, ids: CI, n: c_int, bits: c_int, group: c_int, k: c_int, out: P, out_kind: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(words));
    try a.add(ad(scale));
    try a.add(ad(bias));
    try a.add(scale_kind);
    try a.add(ad(ids));
    try a.add(n);
    try a.add(bits);
    try a.add(group);
    try a.add(k);
    try a.add(ad(out));
    try a.add(out_kind);
    try l.flat(l.op.embed_rows, @as(i64, n) * k, s, &a);
}

pub fn tf_embed_dense(l: *const Launcher, table: CF, ids: CI, n: c_int, k: c_int, out: P, out_kind: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(table));
    try a.add(ad(ids));
    try a.add(n);
    try a.add(k);
    try a.add(ad(out));
    try a.add(out_kind);
    try l.flat(l.op.embed_dense, @as(i64, n) * k, s, &a);
}

pub fn tf_cast(l: *const Launcher, src: C, skind: c_int, dst: P, dkind: c_int, n: c_longlong, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(src));
    try a.add(skind);
    try a.add(ad(dst));
    try a.add(dkind);
    try a.add(n);
    try l.flat(l.op.cast, n, s, &a);
}

pub fn tf_silu_mul(l: *const Launcher, gate: C, up: C, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(gate));
    try a.add(ad(up));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(n);
    try l.flat(l.op.silu_mul, n, s, &a);
}

pub fn tf_add(l: *const Launcher, x: C, y: C, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(y));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(n);
    try l.flat(l.op.add, n, s, &a);
}

pub fn tf_copy_cols(l: *const Launcher, src: C, stride: c_longlong, offset: c_int, dst: P, kind: c_int, rows: c_int, cols: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(src));
    try a.add(stride);
    try a.add(offset);
    try a.add(ad(dst));
    try a.add(kind);
    try a.add(rows);
    try a.add(cols);
    try l.flat(l.op.copy_cols, @as(i64, rows) * cols, s, &a);
}

pub fn tf_dense_rows(l: *const Launcher, x: C, kind: c_int, w: CF, out: P, rows: c_int, n: c_int, k: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(kind);
    try a.add(ad(w));
    try a.add(ad(out));
    try a.add(rows);
    try a.add(n);
    try a.add(k);
    try l.go(l.op.dense_rows, dim(n, rows, 1), dim(32, 1, 1), 0, s, &a);
}
