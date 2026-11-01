//! A shared expert keeps full dot products: gather BF16 intermediate rows, then write disjoint output columns.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const ops = @import("cuda_batch_ops.zig");
const mixed = @import("cuda_mixed_experts.zig");

pub fn run(self: anytype, native: *const mixed.Owned, n: usize) !void {
    if (!native.split_shared and native.rank != 0) return;
    const s = self.s;
    const dims = wt.dims;
    const glue = self.tri.own orelse return error.NativeGlueRequired;
    const width = if (native.split_shared) wt.moe_w / 2 else wt.moe_w;
    try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part);
    try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part);
    try glue.sharedSwiGLU(self.stream, s.shared_gate, s.shared_up, s.shared_act, n, width);
    var input = s.shared_act;
    if (native.split_shared) {
        const c = self.m.collective orelse return error.TensorParallelUnavailable;
        try c.gather(self.stream, input, s.rank_gather, n * width);
        for (0..2) |rank| try ops.copyCols(self.k, self.stream, s.rank_gather + rank * n * width * 2, width, s.rank_input + rank * width * 2, wt.moe_w, width, n);
        input = s.rank_input;
    }
    const down = native.shared[2].native.?;
    if (!native.split_shared and down.n == dims and cuda.qmmf_shared.eligible(self.k.native_lane, n, down)) {
        const target = s.y + (wt.slots - 1) * dims * 4;
        try self.k.native_lane.matmulF32Strided(self.stream, input, wt.moe_w, n, down, target, wt.slots * dims);
    } else {
        try self.k.native_lane.matmulF32(self.stream, input, wt.moe_w, n, down, s.shared_out, s.native_part);
        if (native.split_shared) {
            // Every routed slot and the peer's output columns stay at the zeros set before the expert launches.
            try ops.copyCols(self.k, self.stream, s.shared_out, dims, s.y + (wt.slots - 1) * dims * 4 + native.rank * dims * 2, wt.slots * dims * 2, dims, n);
        } else try glue.sharedScatter(self.stream, s.shared_out, s.y, n, dims, wt.slots);
    }
}
