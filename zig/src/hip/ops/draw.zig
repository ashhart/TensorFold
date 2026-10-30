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

/// softmax(row r)[ids[r]] of `rows` logits rows (fp16 or bf16) into fp32 `out`; of the row's largest when `ids` is 0.
pub fn tokenProb(o: Ops, logits: Tensor, rows: usize, n: usize, ids: u64, out: u64) Error!void {
    if (logits.kind == .f32) return error.BadShape;
    try o.l.tf_token_prob(p(logits.ptr), @backingInt(logits.kind), int(rows), int(n), if (ids == 0) null else @ptrFromInt(ids), f(out), o.stream);
}

/// Each row's `ks[r]` largest by (value desc, id asc): ids (i32) and the values' 16-bit patterns, `stride` apart.
pub fn topkRows(o: Ops, logits: Tensor, rows: usize, n: usize, ks: u64, stride: usize, ids: u64, values: u64) Error!void {
    if (logits.kind == .f32) return error.BadShape;
    try o.l.tf_topk_rows(p(logits.ptr), int(rows), int(n), i(ks), int(stride), i(ids), p(values), o.stream);
}
