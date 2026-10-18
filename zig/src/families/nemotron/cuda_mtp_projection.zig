//! One output head per span, with a full-vocabulary stride only when heads differ in the batch.
pub const Vocab = enum { subset, full };
pub fn project(o: anytype, spans: anytype, heads: anytype, widths: [2]usize, row: u64, xs: u64, logits: u64, d: u64) !usize {
    if (spans.len == 0) return error.EmptyHeadSpans;
    const v = spans[0].vocab;
    for (spans[1..spans.len]) |s| {
        if (s.vocab != v) break;
    } else {
        const index = @backingInt(v);
        try o.dense(row, xs, heads[index], logits, spans.len);
        return widths[index];
    }
    for (spans, 0..) |s, k| {
        try o.dense(row + k * d * 2, xs + k * d / 64 * 4, heads[@backingInt(s.vocab)], logits + k * widths[1] * 2, 1);
    }
    return widths[1];
}
