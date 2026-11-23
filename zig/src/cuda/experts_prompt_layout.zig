//! NVFP4 prompt staging geometry: 64 regular pairs or 128 paired-chunk pairs an item.

const std = @import("std");

fn stagedSmem(parts: usize, wn: usize, rows: usize) u32 {
    const stage: u32 = @intCast((rows * 8 + wn * 2 * parts * 36) * 16);
    const stages: u32 = if (stage * 4 <= 49152) 4 else if (stage * 3 <= 49152) 3 else 2;
    return stages * stage;
}

pub fn promptSmem(parts: usize, wn: usize) u32 {
    return stagedSmem(parts, wn, 64);
}

/// Prompt4<M, 4, WM, WN>'s staging: 64*WM rows and WN x 2 weight blocks.
pub fn promptSmemWM(parts: usize, wm: usize, wn: usize) u32 {
    return if (wm == 1) promptSmem(parts, wn) else pairPromptSmem(parts, wn);
}

/// The paired entries' WM=2 staging: 128 rows, two x two input stages.
pub fn pairPromptSmem(parts: usize, wn: usize) u32 {
    return stagedSmem(parts, wn, 128);
}

test "nvfp4 paired prompt staging bytes" {
    try std.testing.expectEqual(@as(u32, 2 * 17408), promptSmem(1, 8));
    try std.testing.expectEqual(@as(u32, 2 * 17408), promptSmem(2, 4));
    try std.testing.expectEqual(@as(u32, 3 * 12800), promptSmem(1, 4));
    try std.testing.expectEqual(@as(u32, 2 * 25600), pairPromptSmem(2, 4));
    try std.testing.expectEqual(@as(u32, 2 * 25600), pairPromptSmem(1, 8));
}
