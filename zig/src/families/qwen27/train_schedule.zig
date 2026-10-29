//! The native row program stays serial; only independent head rows and submissions are scheduled here.
const mode_api = @import("living").train_mode;
const to = @import("train_ops.zig");

pub fn commitEvery(learning_rows: bool) usize {
    return if (learning_rows) 0 else 4;
}
pub fn head(mode: mode_api.Mode, row: usize, start: usize) bool {
    return mode != .keys and row + 1 >= start;
}
pub fn q4Threads(rows: usize, width: usize) usize {
    return ((rows + 3) / 4) * width;
}
pub fn loss(t: anytype, o: anytype) !void {
    for (t.answers[0..t.nseq]) |a| {
        try o.loss(to.plus(t.logits, a.first * t.g.vocab * 2), to.plus(t.targets, a.first * 4), to.plus(t.weights, a.first * 4), to.plus(t.dlogits, a.first * t.g.vocab * 4), to.plus(t.stats, a.first * 8), a.rows, t.g.vocab);
    }
}
pub fn headBack(t: anytype, o: anytype) !void {
    // Zero-loss rows have +0 input gradients; every answer retains its original fixed-order affine product.
    try o.zero(t.dy, t.rows * t.g.hidden);
    for (t.answers[0..t.nseq]) |a| {
        try o.q4Back(to.plus(t.dlogits, a.first * t.g.vocab * 4), t.head, to.plus(t.dy, a.first * t.g.hidden * 4), a.rows, false);
    }
}
