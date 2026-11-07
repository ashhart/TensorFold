//! TF_GEMMA4_CHUNK_SOLO: whole grid chunks a solo prompt pass takes, on lanes.solo_chunk's rule.
const std = @import("std");
const solo_chunk = @import("lanes").solo_chunk;

/// The grid chunk (a pass beside a decode), and the widest solo pass: the rings and the prompt scratch grow with it.
pub const limits: solo_chunk.Limits = .{ .busy = grid, .most = 4096, .grid = grid };
const grid = @import("cuda_forward.zig").prompt_rows;

pub fn rows() !usize {
    return solo_chunk.rows("TF_GEMMA4_CHUNK_SOLO", limits);
}

test "TF_GEMMA4_CHUNK_SOLO takes whole grid chunks up to 4096" {
    try std.testing.expectEqual(@as(usize, 512), try solo_chunk.parse(null, limits));
    try std.testing.expectEqual(@as(usize, 2048), try solo_chunk.parse("2048", limits));
    try std.testing.expectError(error.BadChunkSolo, solo_chunk.parse("1000", limits));
    try std.testing.expectError(error.BadChunkSolo, solo_chunk.parse("8192", limits));
}
