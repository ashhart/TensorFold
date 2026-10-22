//! Part's cap-stride zero slice: a one-rank native out projection's writeback adds it once, not sk - 1 cleared slices.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");

const dims = wt.dims;

/// A lane write into part that reaches the zero slice: a legacy tile's slices, or a native slice past cap rows.
pub fn spends(native: bool, sk: usize, m: usize, n: usize, cap: usize) bool {
    return sk > 1 and (!native or m * n > cap * dims);
}

/// Pass.lane into s.part: spends the zero slice as it must; `zero` and native: one slice, no clears (1), else null.
pub fn lane(k: *const parts.Kernels, stream: cuda.Stream, s: *parts.Scratch, zero: bool, x: u64, w: wt.Q4, part: u64, m: usize, sk: usize) !?usize {
    if (part != s.part) return null;
    if (spends(w.native != null, sk, m, w.n, s.cap)) s.part_zero = false;
    const native = w.native orelse return null;
    if (!zero or sk == 1) return null;
    try k.native_lane.matmulF32(stream, x, w.k, m, native, part, s.native_part);
    return 1;
}

/// A layer's out projection may skip its clears: the zero slice intact, nothing kept, one rank, a native projection.
pub fn usable(intact: bool, mix_branch: u64, collective: bool, native_out: bool) bool {
    return intact and mix_branch == 0 and !collective and native_out;
}

pub const Slices = struct { stride: usize, count: usize };

/// The layer writeback's slices of part: the projection's and the zero slice at cap rows, else its own at n rows.
pub fn slices(zero: bool, cap: usize, n: usize, out_slices: usize) Slices {
    return if (zero) .{ .stride = cap * dims, .count = 2 } else .{ .stride = n * dims, .count = out_slices };
}

/// One rank's down kernel and sharedScatter write every MoE slot; another rank needs y cleared for the rank sum.
pub fn clearsY(world: usize, rank: usize) bool {
    return world != 1 or rank != 0;
}

test {
    _ = @import("cuda_zero_slice_test.zig");
}
