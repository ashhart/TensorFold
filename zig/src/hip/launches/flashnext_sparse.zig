//! Flash Next's sparse attention: the parts tile and merge, and the matrix-core prompt and decode tiles.

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
const caps = @import("../caps.zig");
const kernels = @import("../kernels.zig");

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

/// The prompt attention's geometry: Qwen3.8 Flash Next's 24 heads of 256 over 2 kv heads, bf16 caches.
pub const prompt_heads = 24;
pub const prompt_kv_heads = 2;
pub const prompt_dims = 256;

/// Whether this build's kernels run the prompt attention on matrix cores (gfx11 WMMA, bf16 activations).
pub fn promptOnMatrix() bool {
    const c = caps.Caps.of(kernels.arch) orelse return false;
    return c.matrix == .wmma11 and c.act == .bf16;
}

/// A prompt's sparse attention in one matrix-core pass; `exact_q`: q bf16, else hi + lo; `p_lo` off: p in bf16 once.
pub fn sparsePrompt(
    l: *const Launcher,
    q: CF,
    k: C,
    v: C,
    ids: CI,
    id_stride: i64,
    count: CI,
    sparse: CI,
    qg: C,
    out: P,
    rows: c_int,
    cap: i64,
    scale: f32,
    exact_q: bool,
    p_lo: bool,
    s: S,
) Error!void {
    if (!promptOnMatrix() or (!p_lo and !exact_q)) return util.invalid("sparsePrompt");
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ad(q), ad(k), ad(v), ad(ids), id_stride, ad(count), ad(sparse), ad(qg), ad(out), cap, scale }) |x|
        try a.add(x);
    const f = l.fnx.sparse_prompt[if (!p_lo) 2 else if (exact_q) 0 else 1];
    try l.go(f, dim(rows, prompt_kv_heads, 1), dim(128, 1, 1), 0, s, &a);
}

/// A few rows' attention: sparsePrompt's tile over `parts` parts of each row's keys, merged in order and gated.
pub fn sparseDecode(
    l: *const Launcher,
    q: CF,
    k: C,
    v: C,
    ids: CI,
    id_stride: i64,
    count: CI,
    sparse: CI,
    parts_o: F,
    parts_ms: F,
    qg: C,
    out: P,
    kind: c_int,
    rows: c_int,
    cap: i64,
    scale: f32,
    parts: c_int,
    exact_q: bool,
    s: S,
) Error!void {
    if (!promptOnMatrix() or parts < 1) return util.invalid("sparseDecode");
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ad(q), ad(k), ad(v), ad(ids), id_stride, ad(count), ad(sparse), ad(parts_o), ad(parts_ms), cap, scale }) |x|
        try a.add(x);
    try l.go(l.fnx.sparse_tile_parts[if (exact_q) 0 else 1], dim(rows, prompt_kv_heads, parts), dim(128, 1, 1), 0, s, &a);
    var b: Args = .{};
    inline for (.{ ad(parts_o), ad(parts_ms), ad(qg), kind, ad(out), @as(c_int, prompt_heads), @as(c_int, prompt_dims), parts }) |x|
        try b.add(x);
    try l.go(l.fnx.sparse_merge_gate, dim(prompt_heads, rows, 1), dim(256, 1, 1), 0, s, &b);
}
