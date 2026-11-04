//! The prompt scratch's buffers sized by use (Scratch.bytes / alloc): head_local, native_part, hc_part and part.
const wt = @import("cuda_prompt_wt.zig");

const hc_sk = wt.hc_sk;
const out_sk = wt.out_sk;

/// head_local's rows on one GPU: the draft head's lanes (<= 16) times its ids; a split head needs a pass's rows.
pub const head_rows_one: usize = 32;

/// part's rows for `format`: an EXL3 lane zero-fills every slice it is asked for, so it keeps out_sk whole slices.
pub fn partRowsFor(format: ScratchFormat, cap: usize) usize {
    return if (format == .exl3) out_sk * cap else partRows(cap);
}

/// An EXL3 checkpoint's scratch leaves out the classic MoE buffers its own experts never touch.
pub const ScratchFormat = enum { classic, exl3 };

/// head_local's rows, whether the lane reduces K slices through native_part (never with clusters), and the format.
pub const Sizes = struct { head_rows: usize = head_rows_one, lane_reduce: bool = true, format: ScratchFormat = .classic };

/// hc_part's rows of (low + 4) floats: hc_sk slices below 64 rows (legacy kernel), <= 8 below 256 (lane), one beyond.
pub fn hcPartRows(cap: usize) usize {
    return @max(cap, @max(hc_sk * @min(cap, 63), @as(usize, 8) * @min(cap, 255)));
}

/// part's rows of dims: an out projection's slice plus the zero slice, or a legacy lane tile's out_sk below 64 rows.
pub fn partRows(cap: usize) usize {
    return @max(2 * cap, out_sk * @min(cap, 63));
}
