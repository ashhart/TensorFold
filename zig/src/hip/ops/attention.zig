//! Attention launches: the gate, RoPE, cache writes and the causal kernels over a cache.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const Kind = t.Kind;
const p = t.p;
const f = t.f;
const int = t.int;

/// The gated attention's o input from att fp32, (heads, len, d) or (len, heads, d) with `rows_major`, and the gate.
pub fn attnGate(o: Ops, att: u64, qg: Tensor, out: Tensor, len: usize, heads: usize, d: usize, rows_major: bool) Error!void {
    try o.l.tf_attn_gate(f(att), p(qg.ptr), p(out.ptr), @backingInt(out.kind), int(len), int(heads), int(d), @intFromBool(rows_major), o.stream);
}

/// Prefill RoPE of x (len, heads, d), rounded to x's kind, stored as `out` kind at out[h * s_head + r * s_row + j].
pub fn ropePrefill(o: Ops, x: Tensor, out: Tensor, s_head: usize, s_row: usize, len: usize, heads: usize, d: usize, rotary: usize, pos0: usize, theta: f32) Error!void {
    try o.l.tf_rope_prefill(p(x.ptr), @backingInt(x.kind), p(out.ptr), @backingInt(out.kind), @intCast(s_head), @intCast(s_row), int(len), int(heads), int(d), int(rotary), int(pos0), theta, o.stream);
}

/// A window's q or k heads RMS-normed with `weight`, rotated at `pos`, rounded; into fp32 `wide` and/or cache slots.
pub fn qkRope(o: Ops, src: Tensor, s_row: usize, s_head: usize, weight: u64, eps: f32, rows: usize, heads: usize, width: usize, rotary: usize, theta: f32, pos: u64, wide: ?u64, cache: ?u64, total: usize) Error!void {
    if (width > 512 or src.kind == .f32) return error.BadShape;
    try o.l.tf_qk_rope(p(src.ptr), @backingInt(src.kind), @intCast(s_row), int(s_head), f(weight), eps, int(rows), int(heads), int(width), int(rotary), theta, @ptrFromInt(pos), if (wide) |w| f(w) else null, if (cache) |c| p(c) else null, int(total), o.stream);
}

/// The decode RoPE over fp32 rows; `pos_dev` (int32, one a group of `per` rows) or the host `pos`.
pub fn ropeDecode(o: Ops, x: u64, y: u64, rows: usize, width: usize, rotary: usize, pos: usize, theta: f32, pos_dev: ?u64, per: usize) Error!void {
    try o.l.tf_rope_decode(f(x), f(y), int(rows), int(width), int(rotary), int(pos), theta, o.stream, if (pos_dev) |a| @ptrFromInt(a) else null, int(per));
}

pub fn kvWrite(o: Ops, src: Tensor, cache: u64, len: usize, kv_heads: usize, d: usize, total: usize, pos0: usize) Error!void {
    try o.l.tf_kv_write(p(src.ptr), p(cache), @backingInt(src.kind), int(len), int(kv_heads), int(d), int(total), int(pos0), o.stream);
}

/// kv_write with the first slot read from the device (one int32), for graphs replayed at new positions.
pub fn kvWriteAt(o: Ops, src: Tensor, cache: u64, len: usize, kv_heads: usize, d: usize, total: usize, pos: u64) Error!void {
    try o.l.tf_kv_write_at(p(src.ptr), p(cache), @backingInt(src.kind), int(len), int(kv_heads), int(d), int(total), @ptrFromInt(pos), o.stream);
}

/// The cache's layout: `kv_heads` heads of `total` slots of `d` values (batch 1).
pub const Cache = struct { k: u64, v: u64, kind: Kind, kv_heads: usize, total: usize, d: usize };

/// Prefill attention on the prefill tile (any query count): q (heads, qlen, d) fp32 over the first `span` slots.
pub fn causalPrefill(o: Ops, q: u64, c: Cache, out: u64, qlen: usize, span: usize, heads: usize, scale: f32, q_pos0: usize) Error!void {
    if (c.d > 256 or heads % c.kv_heads != 0) return error.BadShape;
    const sh: c_longlong = @intCast(c.total * c.d);
    const ss: c_longlong = @intCast(c.d);
    try o.l.tf_causal(f(q), p(c.k), p(c.v), f(out), 1, int(qlen), int(span), int(heads), int(c.kv_heads), int(c.d), scale, int(q_pos0), sh * @as(c_longlong, @intCast(c.kv_heads)), sh, ss, sh * @as(c_longlong, @intCast(c.kv_heads)), sh, ss, c.kind.cache(), null, null, null, o.stream, null);
}

/// One attention layer of a stream's paged cache: key and value pools (`count` 64-position pages a head), page table.
pub const Paged = struct { k: u64, v: u64, table: u64, kind: Kind, kv_heads: usize, d: usize, count: usize };

fn zigLaunches(o: Ops) Error!*const @import("../launches.zig").Launcher {
    return o.l;
}

/// 16-bit `src` values (h * s_head + r * s_row + j elements) of `len` positions from `pos0` into a pool by the table.
pub fn pageWrite(o: Ops, src: u64, pool: u64, table: u64, len: usize, kv_heads: usize, d: usize, s_head: usize, s_row: usize, pos0: usize, count: usize) Error!void {
    try (try zigLaunches(o)).pagesWrite(src, pool, table, len, kv_heads, d, s_head, s_row, pos0, count, o.stream);
}

/// Prefill attention over a stream's pages: the 64-row tile reads pages; other shapes read a flat copy.
pub fn causalPaged(o: Ops, q: u64, c: Paged, out: u64, qlen: usize, span: usize, heads: usize, scale: f32, q_pos0: usize) Error!void {
    if (c.d > 256 or heads % c.kv_heads != 0 or c.kind == .f32) return error.BadShape;
    const z = try zigLaunches(o);
    if (c.d % 64 == 0 and z.wide) return z.pagedCausal(q, c.k, c.v, c.table, out, qlen, span, heads, c.kv_heads, c.d, scale, q_pos0, c.count, c.kind.cache(), o.stream);
    const bytes = c.kv_heads * span * c.d * 2;
    const flat: Cache = .{ .k = try o.arena.take(bytes), .v = try o.arena.take(bytes), .kind = c.kind, .kv_heads = c.kv_heads, .total = span, .d = c.d };
    try z.pagesGather(c.k, c.table, flat.k, span, c.kv_heads, c.d, c.count, o.stream);
    try z.pagesGather(c.v, c.table, flat.v, span, c.kv_heads, c.d, c.count, o.stream);
    try causalPrefill(o, q, flat, out, qlen, span, heads, scale, q_pos0);
}

/// causal_at: `rows` queries (rows, heads, 1, d) fp32 each at its device position over one shared cache.
pub fn causalAt(o: Ops, q: u64, c: Cache, out: u64, rows: usize, heads: usize, scale: f32, pos: u64) Error!void {
    if (c.d > 256 or heads % c.kv_heads != 0) return error.BadShape;
    const span = c.total;
    const scores = try o.arena.of(f32, rows * heads * span);
    const stats = try o.arena.of(f32, rows * heads * 2);
    const partials = try o.arena.of(f32, rows * heads * ((span + 127) / 128) * c.d);
    const sh: c_longlong = @intCast(c.total * c.d);
    const ss: c_longlong = @intCast(c.d);
    try o.l.tf_causal(f(q), p(c.k), p(c.v), f(out), int(rows), 1, int(span), int(heads), int(c.kv_heads), int(c.d), scale, 0, 0, sh, ss, 0, sh, ss, c.kind.cache(), f(scores), f(stats), f(partials), o.stream, @ptrFromInt(pos));
}
