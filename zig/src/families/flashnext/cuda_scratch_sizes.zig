//! The prompt scratch's buffers sized by use (Scratch.bytes / alloc): head_local, native_part, hc_part and part.
const wt = @import("cuda_prompt_wt.zig");

const hc_sk = wt.hc_sk;
const out_sk = wt.out_sk;
const att_rows = @import("cuda_batch_state.zig").att_rows;

/// head_local's rows on one GPU: the draft head's lanes (<= 16) times its ids; a split head needs a pass's rows.
pub const head_rows_one: usize = 32;

/// part's rows: an EXL3 decoder's lane zero-fills every slice it is asked for, so it keeps out_sk whole slices.
pub fn partRowsFor(z: Sizes, cap: usize) usize {
    return if (z.format == .exl3 or z.exl3_lanes) out_sk * cap else partRows(cap);
}

/// EXL3 omits classic expert planning/shared-expert work and classic rank/head staging handled by its own kernels.
pub const ScratchFormat = enum { classic, exl3 };

/// head_local's rows, whether the lane reduces through native_part, the format, whether the decoder's lanes are EXL3's.
pub const Sizes = struct { head_rows: usize = head_rows_one, lane_reduce: bool = true, format: ScratchFormat = .classic, exl3_lanes: bool = false };

/// hc_part's rows of (low + 4) floats: hc_sk slices below 64 rows (legacy kernel), <= 8 below 256 (lane), one beyond.
pub fn hcPartRows(cap: usize) usize {
    return @max(cap, @max(hc_sk * @min(cap, 63), @as(usize, 8) * @min(cap, 255)));
}

/// part's rows of dims: an out projection's slice plus the zero slice, or a legacy lane tile's out_sk below 64 rows.
pub fn partRows(cap: usize) usize {
    return @max(2 * cap, out_sk * @min(cap, 63));
}

/// The block scores' row stride for a cache of `cap` tokens: its blocks rounded up to 16 (the captured _scores').
pub fn scoreStride(cap: usize) usize {
    return ((cap + wt.index_ratio - 1) / wt.index_ratio + 15) / 16 * 16;
}

/// Bytes of block scores a pass over a cache of `cap` tokens needs for chunks of up to `rows` rows.
pub fn scoreBytes(cap: usize, rows: usize) usize {
    return @min(rows, att_rows) * scoreStride(cap) * 4;
}
