//! A readout piece on the device: LN_hidden of its rows, M's rows appended, the span sums of its rows, g at the end.
const Stream = @import("../stream.zig").Stream;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Head = @import("head.zig").Head;
const State = @import("state.zig").State;

/// Prompt rows first..first+count of the stream's final-norm rows (BF16 at `rows`), in order; work goes on `s`.
pub fn consume(h: *Head, st: *State, rows: u64, first: usize, count: usize, s: Stream) !void {
    if (first != st.consumed or count == 0 or first + count > st.rows) return error.ReadoutPieceOutOfOrder;
    // rocBLAS runs on the stream its handle is bound to: the kernels must go on that one too
    if (s.handle != h.blas.stream.handle) return error.ReadoutStreamNotBlas;
    const H = h.cfg.hidden_size;
    const W = h.cfg.width;
    if (count > h.bounds.piece) return error.ReadoutPieceTooLong;
    const normed = h.at(h.layout.piece);
    try h.norm(s, rows, h.w.w.hidden_norm, normed, count, H);
    try h.blas.project(normed, h.w.w.memory, st.memory(h) + first * W * 2, count, W, H);
    try h.fns.spanSums(s, normed, 0, first, count, H, st.spans(h), st.questions + st.options, st.sums(h));
    if (first + count == st.rows) {
        const g: DeviceBuffer = .{ .r = h.r, .ptr = @ptrFromInt(st.g(h)), .len = H * 2 };
        try g.copyFromAsync(0, normed + (count - 1) * H * 2, H * 2, s);
    }
    st.consumed = first + count;
}
