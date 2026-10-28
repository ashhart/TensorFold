//! The DFlash2 drafter's ops between projections; its q, k heads go through `qkRope` into fp32 rows these read.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const launches = @import("../launches/draft.zig");
const Error = t.Error;
const Tensor = t.Tensor;

pub const head = launches.head;
pub const most_rows = launches.most_rows;
pub const Segment = launches.Segment;
pub const Partials = launches.Partials;
pub const Attention = launches.Attention;

fn half(x: Tensor) Error!void {
    if (x.kind.size() != 2) return error.BadShape;
}

/// out (n, width) 16-bit = a tap ring's rows (start + i) % ring_rows, each `width` 16-bit values (taps * hidden).
pub fn gatherTaps(o: Ops, ring: u64, ring_rows: usize, start: u64, n: usize, width: usize, out: u64) Error!void {
    if (width % 2 != 0) return error.BadShape;
    try launches.gather(o.l, ring, ring_rows, start, null, n, width / 2, out, o.stream);
}

/// gatherTaps of ring rows (start + rows[i]) % ring_rows: a kept path's rows in order, at most `most_rows`.
pub fn gatherRows(o: Ops, ring: u64, ring_rows: usize, start: u64, rows: []const u32, width: usize, out: u64) Error!void {
    if (width % 2 != 0 or rows.len > most_rows) return error.BadShape;
    try launches.gather(o.l, ring, ring_rows, start, rows, rows.len, width / 2, out, o.stream);
}

/// keys (fp32, from qkRope) and values of `rows` rows of `width` into a layer's rings at (pos0 + i) % ring_rows.
pub fn ringPut(o: Ops, keys: u64, values: Tensor, k_ring: u64, v_ring: u64, rows: usize, width: usize, pos0: u64, ring_rows: usize) Error!void {
    try half(values);
    try launches.ringPut(o.l, keys, values.ptr, @backingInt(values.kind), k_ring, v_ring, rows, width, pos0, ring_rows, o.stream);
}

/// out = the dynamic conv's `branch` of x with the coefficients `dyn` and fp32 `base` kernel, plus `res` when given.
pub fn dynConv(o: Ops, x: Tensor, dyn: Tensor, base: u64, res: ?Tensor, out: Tensor, rows: usize, hidden: usize, group: usize, branch: usize, block_rows: usize) Error!void {
    try half(x);
    if (dyn.kind != x.kind or out.kind != x.kind) return error.BadShape;
    if (res) |r| if (r.kind != x.kind) return error.BadShape;
    const r = if (res) |v| v.ptr else 0;
    try launches.dynConv(o.l, x.ptr, dyn.ptr, base, r, out.ptr, @backingInt(x.kind), rows, hidden, group, branch, block_rows, o.stream);
}

/// out = every stream's block rows but the first (its pending token), packed: streams * (block_rows - 1) rows.
pub fn maskRows(o: Ops, x: Tensor, out: Tensor, streams: usize, hidden: usize, block_rows: usize) Error!void {
    try half(x);
    if (out.kind != x.kind) return error.BadShape;
    try launches.maskRows(o.l, x.ptr, out.ptr, streams, hidden, block_rows, o.stream);
}

/// out (rows, heads, 128) = q (fp32) over each row's ring window and block (kb fp32, vb); `mask` bits: block rows seen.
pub fn attention(o: Ops, q: u64, kb: u64, vb: Tensor, mask: ?u64, segments: u64, at: Attention, p: Partials, out: Tensor) Error!void {
    try half(vb);
    if (out.kind != vb.kind or !at.check(mask != null)) return error.BadShape;
    const kind = @backingInt(vb.kind);
    try launches.attention(o.l, q, kb, vb.ptr, mask orelse 0, segments, kind, at, p, out.ptr, o.stream);
}
