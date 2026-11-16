//! A prompt pass's routed-expert exchange at --tp 2: each rank sums its share of rows' slots, then gathers the sums.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const moe = @import("cuda_batch_moe.zig");

const dims = wt.dims;
const slots = wt.slots;
const share = cuda.rank_scatter.share;

/// Rows past any decode round or verify window: `world` and `n` are what both ranks' passes agree on.
pub fn rowSplit(world: usize, n: usize) bool {
    return world == 2 and n > mtp.round_rows;
}

/// Rows of each rank's reduce-scatter block: rank 0's share, which rank 1's pads to (row n when n is odd).
pub fn block(n: usize) usize {
    return share(n, 2, 0)[1];
}

/// Rows of y the clear spans: a split pass's two blocks, else the pass's own.
pub fn clearedRows(world: usize, n: usize) usize {
    return if (rowSplit(world, n)) 2 * block(n) else n;
}

/// y's slots, a rank's values and the other's +0, to their fp32 sum in moe_f32 on both ranks: the pending_f32 form.
pub fn run(pass: anytype, c: anytype, n: usize) !void {
    const s = pass.s;
    const h = block(n);
    const own = share(n, 2, c.rank);
    try c.sumBlock(pass.stream, s.y, h * slots * dims);
    try moe.sumSlots(pass, own[0], own[1] - own[0]);
    try c.gatherF32(pass.stream, s.moe_f32 + own[0] * dims * 4, s.moe_f32, h * dims);
    pass.pending_f32 = true;
}

test {
    if (@import("builtin").os.tag != .windows) _ = @import("cuda_moe_exchange_test.zig"); // over a Unix socket pair
}
