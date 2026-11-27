//! NVFP4 prompt staging geometry: 64 regular pairs or 128 paired-chunk pairs an item.

const std = @import("std");

fn stagedSmem(parts: usize, wn: usize, rows: usize, sg: usize) u32 {
    const stage: u32 = @intCast((rows * 4 * sg + wn * sg * parts * 36) * 16);
    const stages: u32 = if (stage * 4 <= 49152) 4 else if (stage * 3 <= 49152) 3 else 2;
    return stages * stage;
}

pub fn promptSmem(parts: usize, wn: usize) u32 {
    return stagedSmem(parts, wn, 64, 2);
}

/// The opt-in R3 prompt entries: two 32-input stages, three CTAs an SM.
pub fn promptResidentSmem(parts: usize, wn: usize) u32 {
    const stage: u32 = @intCast((64 * 4 + wn * parts * 36) * 16);
    return 2 * stage;
}

/// The paired entries' WM=2 staging: 128 rows, two x two input stages.
pub fn pairPromptSmem(parts: usize, wn: usize) u32 {
    return stagedSmem(parts, wn, 128, 2);
}

test "nvfp4 paired and resident prompt staging bytes" {
    try std.testing.expectEqual(@as(u32, 2 * 17408), promptSmem(1, 8));
    try std.testing.expectEqual(@as(u32, 2 * 17408), promptSmem(2, 4));
    try std.testing.expectEqual(@as(u32, 2 * 8704), promptResidentSmem(1, 8));
    try std.testing.expectEqual(@as(u32, 2 * 8704), promptResidentSmem(2, 4));
    try std.testing.expectEqual(@as(u32, 2 * 25600), pairPromptSmem(2, 4));
    try std.testing.expectEqual(@as(u32, 2 * 25600), pairPromptSmem(1, 8));
}
