//! Shape-only TFQ3 pipeline selection, chunk geometry, and sign addressing.
const std = @import("std");

pub fn useSplitK(n: u32, k: u32) bool {
    const n_bc = k >> 7;
    return n <= 34816 and n_bc >= 8 and n_bc % 8 == 0;
}

pub fn capRows(k: u32, max_rows: u32, max_xrot_elems: usize) u32 {
    return @intCast(@min(@as(usize, max_rows), max_xrot_elems / k));
}

pub fn byteOffsets(done: u32, ldx: u32, ldo: u32) [2]usize {
    return .{ @as(usize, done) * ldx * 2, @as(usize, done) * ldo * 2 };
}

/// Qwen27 uses absolute standalone rows; only an explicit fused split resets its sign row.
pub fn signRow(r0: u32, sr_split: u32) u32 {
    return if (sr_split != 0) r0 % sr_split else r0;
}

comptime {
    std.debug.assert(useSplitK(256, 1024));
    std.debug.assert(!useSplitK(256, 512));
}
