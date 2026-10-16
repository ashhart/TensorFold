//! Argmax and top-k of logits rows: one 1024-thread block a row.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const C = util.C;
const P = util.P;
const I = util.I;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;

/// out[r] = torch.argmax of row r of (rows, n) logits of `kind` (1 fp16, 2 bf16).
pub fn tf_argmax_rows(l: *const Launcher, logits: C, kind: c_int, rows: c_int, n: c_int, out: I, s: S) Error!void {
    if (rows == 0) return;
    var a: Args = .{};
    try a.add(ad(logits));
    try a.add(kind);
    try a.add(n);
    try a.add(ad(out));
    try l.go(l.op.argmax, dim(rows, 1, 1), dim(1024, 1, 1), 0, s, &a);
}

/// Row r's ks[r] largest by (value desc, id asc) into ids and 16-bit values, `stride` apart; ks[r] = 0 skips the row.
pub fn tf_topk_rows(l: *const Launcher, logits: C, rows: c_int, n: c_int, ks: CI, stride: c_int, ids: I, values: P, s: S) Error!void {
    if (rows == 0) return;
    var a: Args = .{};
    try a.add(ad(logits));
    try a.add(n);
    try a.add(ad(ks));
    try a.add(stride);
    try a.add(ad(ids));
    try a.add(ad(values));
    try l.go(l.op.topk, dim(rows, 1, 1), dim(1024, 1, 1), 0, s, &a);
}
