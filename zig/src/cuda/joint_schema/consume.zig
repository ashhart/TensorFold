//! A readout piece on the device: LN_hidden of its rows, M's rows appended, the span sums of its rows, g at the end.
const Stream = @import("cuda").Stream;
const DeviceBuffer = @import("cuda").DeviceBuffer;
const Head = @import("head.zig").Head;
const State = @import("state.zig").State;

/// Prompt rows first..first+count of the stream's final-norm rows (BF16 at `rows`), in order; work goes on `s`.
pub fn consume(h: *Head, st: *State, rows: u64, first: usize, count: usize, s: Stream) !void {
    if (first != st.consumed or count == 0 or first + count > st.rows) return error.ReadoutPieceOutOfOrder;
    if (count > h.bounds.piece) return error.ReadoutPieceTooLong;
    const H = h.cfg.hidden_size;
    const W = h.cfg.width;
    const normed = h.at(h.layout.piece);
    try h.norm(s, rows, h.w.w.hidden_norm, normed, count, H);
    try h.blas.project(s, normed, h.w.w.memory, st.memory(h) + first * W * 2, count, W, H);
    try h.fns.spanSums(s, normed, first, count, H, st.spans(h), st.questions + st.options, st.sums(h));
    if (first + count == st.rows) {
        const g: DeviceBuffer = .{ .d = h.d, .ptr = st.g(h), .len = H * 2, .borrowed = true };
        try g.copyFrom(0, normed + (count - 1) * H * 2, H * 2, s.handle);
    }
    st.consumed = first + count;
}
