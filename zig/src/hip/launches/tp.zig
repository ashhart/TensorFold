//! A tensor-parallel rank's launches (the fp32 sum into a residual, the local expert ids) and MTP's token probability.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const C = util.C;
const P = util.P;
const I = util.I;
const CI = util.CI;
const F = util.F;
const CF = util.CF;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;

fn blocks(n: i64, threads: i64) c_int {
    return @intCast(@divFloor(n + threads - 1, threads));
}

/// out = round(x + y) of `n` values, x and out of `kind` (1 fp16, 2 bf16), y fp32.
pub fn tf_add_wide(l: *const Launcher, x: C, y: CF, out: P, kind: c_int, n: c_longlong, s: S) Error!void {
    if (n == 0) return;
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(y));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(n);
    try l.go(l.op.add_wide, dim(blocks(n, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// out[i] = remap[pick[i]], or `skip` where it is negative.
pub fn tf_moe_localize(l: *const Launcher, pick: CI, remap: CI, out: I, n: c_int, skip: c_int, s: S) Error!void {
    if (n == 0) return;
    var a: Args = .{};
    try a.add(ad(pick));
    try a.add(ad(remap));
    try a.add(ad(out));
    try a.add(n);
    try a.add(skip);
    try l.go(l.op.moe_localize, dim(blocks(n, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// Zeroes the pair count of every plan item of the `skip` expert.
pub fn tf_moe_foreign_items(l: *const Launcher, items: I, count: c_int, skip: c_int, s: S) Error!void {
    if (count == 0) return;
    var a: Args = .{};
    try a.add(ad(items));
    try a.add(count);
    try a.add(skip);
    try l.go(l.op.moe_foreign_items, dim(blocks(count, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// Zeroes the rows of `y` (pairs, d) fp32 whose pick is `skip`.
pub fn tf_moe_zero_foreign(l: *const Launcher, y: F, picks: CI, skip: c_int, pairs: c_longlong, d: c_int, s: S) Error!void {
    if (pairs == 0) return;
    var a: Args = .{};
    try a.add(ad(y));
    try a.add(ad(picks));
    try a.add(skip);
    try a.add(pairs);
    try a.add(d);
    try l.go(l.op.moe_zero_foreign, dim(blocks(pairs * d, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// out[r] = softmax(row r)[id[r]] of (rows, n) logits of `kind`, or of the row's largest value when `id` is null.
pub fn tf_token_prob(l: *const Launcher, logits: C, kind: c_int, rows: c_int, n: c_int, id: CI, out: F, s: S) Error!void {
    if (rows == 0) return;
    var a: Args = .{};
    try a.add(ad(logits));
    try a.add(kind);
    try a.add(n);
    try a.add(ad(id));
    try a.add(ad(out));
    try l.go(l.op.token_prob, dim(rows, 1, 1), dim(256, 1, 1), 0, s, &a);
}
