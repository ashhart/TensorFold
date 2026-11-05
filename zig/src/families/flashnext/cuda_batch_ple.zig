//! The shared batched pass's per-layer embedding block: the n-gram keys and values, the gate and the convolution.
const wt = @import("cuda_prompt_wt.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const eps = wt.eps;
const ple_tail_n = wt.ple_tail_n;

pub fn pleBlock(self: anytype, lw: anytype, ids: []const u32) !void {
    const s = self.s;
    const n = self.rows;
    // the rows' n-gram embeddings (host gathers from the mapped table) are already uploaded by `stage`
    _ = ids;
    _ = try self.lane(s.ple_emb, s.ple_xs, lw.key, s.ple_keys, s.dummy, n, 1);
    const vsk = try self.lane(s.ple_emb, s.ple_xs, lw.val, s.dummy, s.ple_part, n, 4);
    try self.sumSlices(s.ple_part, s.ple_vals, true, n * dims, vsk);
    try self.tri.pleGate(s.ple_keys, s.ple_vals, s.h, lw.nk.ptr, lw.nq.ptr, s.ple_gated, s.ple_pss, eps, n, dims, streams);
    if (self.lanes.len == 0) {
        try self.tri.pleConv(s.ple_gated, s.ple_pss, lw.nc.ptr, self.seq.tail.ptr, lw.ple_cw.ptr, s.h, s.h, s.ple_nrow, eps, n, dims, streams, 4, 3);
        try self.shiftWindow(self.seq.tail.ptr, s.ple_nrow, wide, n, ple_tail_n, wide);
        return;
    }
    // shared rounds: each lane's rows convolved over its own stream's tail, the launch its lone pass makes
    var row0: usize = 0;
    for (self.lanes) |l| {
        const g = row0 * wide * 2;
        try self.tri.pleConv(s.ple_gated + g, s.ple_pss + row0 * streams * 4, lw.nc.ptr, l.seq.tail.ptr, lw.ple_cw.ptr, s.h + g, s.h + g, s.ple_nrow + g, eps, l.rows, dims, streams, 4, 3);
        try self.shiftWindow(l.seq.tail.ptr, s.ple_nrow + g, wide, l.rows, ple_tail_n, wide);
        row0 += l.rows;
    }
}
