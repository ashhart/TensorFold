//! Exact output-column projections: gather BF16 heads, compute full dot products, then gather FP32 outputs.
const wt = @import("cuda_prompt_wt.zig");
const ops = @import("cuda_batch_ops.zig");
const Layer = @import("cuda_prompt_load.zig").Layer;

const dims = wt.dims;
const out_sk = wt.out_sk;

/// Write full FP32 output rows into part without adding partial products across ranks.
pub fn out(self: anytype, x: u64, xs: u64, lw: *const Layer) !usize {
    const s = self.s;
    const n = self.rows;
    const cut = lw.dense_cut orelse return self.lane(x, xs, lw.out, s.dummy, s.part, n, out_sk);
    const native = lw.out.native orelse return error.UnexpectedTensor;
    const c = self.m.collective orelse return error.TensorParallelUnavailable;
    if (n == 0 or n > s.cap or !lw.heads.eql(.half) or cut.width() != dims / 2 or
        cut.from != c.rank * (dims / 2) or native.n != dims / 2 or native.k != wt.out_k or
        native.sk != @import("cuda").qmmf.splitK(dims, wt.out_k)) return error.UnexpectedTensor;
    const width = lw.heads.valueDim();
    try c.gather(self.stream, x, s.rank_gather, n * width);
    for (0..2) |rank| try ops.copyCols(self.k, self.stream, s.rank_gather + rank * n * width * 2, width, s.rank_input + rank * width * 2, 2 * width, width, n);
    try self.k.native_lane.matmulF32(self.stream, s.rank_input, 2 * width, n, native, s.rank_output, s.native_part);
    try c.gatherF32(self.stream, s.rank_output, s.rank_result, n * cut.width());
    // copyCols moves 16-bit words, so two words preserve every FP32 value's bits.
    for (0..2) |rank| try ops.copyCols(self.k, self.stream, s.rank_result + rank * n * cut.width() * 4, cut.width() * 2, s.part + rank * cut.width() * 4, dims * 2, cut.width() * 2, n);
    return 1;
}

test {
    _ = @import("cuda_dense_split.zig");
    if (@import("builtin").os.tag != .windows) _ = @import("cuda_split_lane_test.zig");
}
