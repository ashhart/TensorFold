//! The draw inputs read from logits rows: argmax and the top ids.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const i = t.i;
const int = t.int;

/// torch.argmax of each of `rows` logits rows (width `n`, fp16 or bf16) into device i32 `out`.
pub fn argmaxRows(o: Ops, logits: Tensor, rows: usize, n: usize, out: u64) Error!void {
    if (logits.kind == .f32) return error.BadShape;
    // a wide row in parts over the card, then their best: the same pick, as the order is total
    if (o.fused() and n >= 2 * part_values) {
        const parts = (n + part_values - 1) / part_values;
        const v = try o.arena.of(f32, rows * parts);
        const ix = try o.arena.of(i32, rows * parts);
        return o.l.tf_argmax_parts(
            p(logits.ptr),
            @backingInt(logits.kind),
            int(rows),
            int(n),
            int(part_values),
            int(parts),
            p(v),
            i(ix),
            i(out),
            o.stream,
        );
    }
    try o.l.tf_argmax_rows(p(logits.ptr), @backingInt(logits.kind), int(rows), int(n), i(out), o.stream);
}

/// Values a part of a wide row's argmax takes.
const part_values = 1024;

/// Values a part of a wide row's top-k takes at least, and the most parts a row takes (the join reads parts * stride).
const part_topk = 4096;
const max_topk_parts = 64;
/// Candidates tf_topk_join holds a row (draw.hpp kJoinMost).
const join_most = 4096;

/// softmax(row r)[ids[r]] of `rows` logits rows (fp16 or bf16) into fp32 `out`; of the row's largest when `ids` is 0.
pub fn tokenProb(o: Ops, logits: Tensor, rows: usize, n: usize, ids: u64, out: u64) Error!void {
    if (logits.kind == .f32) return error.BadShape;
    try o.l.tf_token_prob(p(logits.ptr), @backingInt(logits.kind), int(rows), int(n), if (ids == 0) null else @ptrFromInt(ids), f(out), o.stream);
}

/// Each row's `ks[r]` largest by (value desc, id asc): ids (i32) and the values' 16-bit patterns, `stride` apart.
pub fn topkRows(o: Ops, logits: Tensor, rows: usize, n: usize, ks: u64, stride: usize, ids: u64, values: u64) Error!void {
    if (logits.kind == .f32) return error.BadShape;
    // a wide row in parts over the card, then each row's from the parts' own: the same ids, values and order
    const parts = @min(max_topk_parts, (n + part_topk - 1) / part_topk);
    if (o.fused() and n >= 2 * part_topk and parts * stride <= join_most) {
        const per = (n + parts - 1) / parts;
        const cand_ids = try o.arena.of(i32, rows * parts * stride);
        const cand_vals = try o.arena.of(u16, rows * parts * stride);
        return o.l.tf_topk_parts(p(logits.ptr), int(rows), int(n), i(ks), int(stride), int(per), int(parts), i(cand_ids), p(cand_vals), i(ids), p(values), o.stream);
    }
    try o.l.tf_topk_rows(p(logits.ptr), int(rows), int(n), i(ks), int(stride), i(ids), p(values), o.stream);
}
