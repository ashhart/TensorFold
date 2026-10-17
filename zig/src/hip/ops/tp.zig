//! The launches a tensor-parallel rank adds: the fp32 sum into a residual and the local expert ids.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const i = t.i;
const int = t.int;

/// out = round(x + y): an activation `x` plus an fp32 sum `y`, rounded once to x's kind (a tp residual).
pub fn addWide(o: Ops, x: Tensor, y: u64, out: Tensor, n: usize) Error!void {
    if (x.kind == .f32 or x.kind != out.kind) return error.BadShape;
    try o.l.tf_add_wide(p(x.ptr), f(y), p(out.ptr), @backingInt(out.kind), @intCast(n), o.stream);
}

/// out[i] = remap[pick[i]], or `skip` where it is negative: a tp rank's local expert ids.
pub fn moeLocalize(o: Ops, pick: u64, remap: u64, out: u64, n: usize, skip: usize) Error!void {
    try o.l.tf_moe_localize(i(pick), i(remap), i(out), int(n), int(skip), o.stream);
}

/// Zeroes the pair count of every plan item of the `skip` expert.
pub fn moeForeignItems(o: Ops, items: u64, count: usize, skip: usize) Error!void {
    try o.l.tf_moe_foreign_items(i(items), int(count), int(skip), o.stream);
}

/// Zeroes the rows of `y` (pairs, d) fp32 whose pick is `skip`.
pub fn moeZeroForeign(o: Ops, y: u64, picks: u64, skip: usize, pairs: usize, d: usize) Error!void {
    try o.l.tf_moe_zero_foreign(f(y), i(picks), int(skip), @intCast(pairs), int(d), o.stream);
}
