//! The prompt scratch's buffers sized by use (Scratch.bytes / alloc): head_local, native_part, hc_part and part.
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");

const hc_sk = wt.hc_sk;
const out_sk = wt.out_sk;
const streams = wt.streams;
const att_rows = @import("cuda_batch_state.zig").att_rows;

/// head_local's rows on one GPU: the draft head's lanes (<= 16) times its ids; a split head needs a pass's rows.
pub const head_rows_one: usize = 32;

/// head_local's rows, the lane's K-slice reduce, a row's expert slots and the widest shared expert (cuda_widths.zig).
pub const Sizes = struct { head_rows: usize = head_rows_one, lane_reduce: bool = true, slots: usize = wt.slots, shared: usize = wt.moe_w };

/// hc_part's rows of (low + 4) floats: hc_sk slices below 64 rows (legacy kernel), <= 8 below 256 (lane), one beyond.
pub fn hcPartRows(cap: usize) usize {
    return @max(cap, @max(hc_sk * @min(cap, 63), @as(usize, 8) * @min(cap, 255)));
}

/// part's rows of dims: an out projection's two cap-stride slices or a legacy lane tile's out_sk, or the head's lane.
pub fn partRows(cap: usize) usize {
    return @max(@max(2 * cap, out_sk * @min(cap, 63)), headRows(cap));
}

/// The head's fc_h lane: streams * rows fp32 plus the zero slice after it, for a pass of at most piece_rows rows.
fn headRows(cap: usize) usize {
    return 2 * streams * @min(cap, mtp.piece_rows);
}

/// The block scores' row stride for a cache of `cap` tokens: its blocks rounded up to 16 (the captured _scores').
pub fn scoreStride(cap: usize) usize {
    return ((cap + wt.index_ratio - 1) / wt.index_ratio + 15) / 16 * 16;
}

/// Bytes of block scores a pass over a cache of `cap` tokens needs for chunks of up to `rows` rows.
pub fn scoreBytes(cap: usize, rows: usize) usize {
    return @min(rows, att_rows) * scoreStride(cap) * 4;
}

test {
    _ = @import("cuda_scratch_sizes_test.zig");
}
