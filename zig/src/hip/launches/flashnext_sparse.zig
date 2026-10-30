//! Flash Next's sparse attention: the parts tile and merge.

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

/// Attention of each row's heads over its listed keys (or its first count[r] keys) in `parts` parts, then merged.
pub fn sparseAttention(
    l: *const Launcher,
    q: CF,
    k: C,
    v: C,
    cache_kind: c_int,
    ids: CI,
    id_stride: i64,
    count: CI,
    sparse: CI,
    parts_o: F,
    parts_ms: F,
    out: F,
    rows: c_int,
    heads: c_int,
    kv_heads: c_int,
    dims: c_int,
    cap: i64,
    scale: f32,
    parts: c_int,
    s: S,
) Error!void {
    if (dims > 256 or @rem(heads, kv_heads) != 0 or @divExact(heads, kv_heads) > 12)
        return util.invalid("sparseAttention");
    var a: Args = .{};
    inline for (.{
        ad(q),
        ad(k),
        ad(v),
        cache_kind,
        ad(ids),
        id_stride,
        ad(count),
        ad(sparse),
        ad(parts_o),
        ad(parts_ms),
        heads,
        kv_heads,
        dims,
        cap,
        scale,
        parts,
    }) |x| try a.add(x);
    const wide: usize = if (@divExact(heads, kv_heads) <= 6) 0 else 1;
    try l.go(l.fnx.sparse_parts[wide], dim(kv_heads, rows, parts), dim(256, 1, 1), 0, s, &a);
    var b: Args = .{};
    inline for (.{ ad(parts_o), ad(parts_ms), ad(out), heads, dims, parts }) |x| try b.add(x);
    try l.go(l.fnx.sparse_merge, dim(heads, rows, 1), dim(256, 1, 1), 0, s, &b);
}
