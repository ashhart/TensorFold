//! The chunks a prompt pass runs in, and the gap between the snapshots it keeps.

const std = @import("std");
const pages = @import("../forward/pages.zig");

/// A snapshot this close to the prompt's end saves too little to keep next to the one before it.
pub const min_gap = 256;

/// Cuts sit on page edges, multiples of the recurrence's chunk, so a resumed span chunks as a fresh prompt does.
pub const step = pages.tokens;

/// Rows a prompt pass advances between two rounds when no other stream decodes; steps end on multiples of it.
pub const chunk = 1024;

/// Rows from which a prompt pass synchronizes and asks its cancel after every layer (fewer: it runs through).
pub const poll_rows = @import("hip_serve").prefix.poll_rows;

/// Single ranks poll layer edges from poll_rows; TP ranks stop together between chunks.
pub const pollsLayers = @import("hip_serve").prefix.pollsLayers;

/// Where the step from `at` ends: the next multiple of `rows`, or `limit` when that comes first.
pub fn chunkEnd(at: usize, limit: usize, rows: usize) usize {
    return @min(limit, (at / rows + 1) * rows);
}

test "a prompt step ends on the next multiple of its rows, or the prompt's cut" {
    try std.testing.expectEqual(@as(usize, 1024), chunkEnd(0, 5000, chunk));
    try std.testing.expectEqual(@as(usize, 512), chunkEnd(0, 5000, 512));
    try std.testing.expectEqual(@as(usize, 1024), chunkEnd(512, 5000, 512));
    try std.testing.expectEqual(@as(usize, 1024), chunkEnd(700, 5000, chunk));
    try std.testing.expectEqual(@as(usize, 900), chunkEnd(700, 900, 512));
}

test "every prompt chunk a server pass runs stops at a layer edge when cancelled" {
    // a 1024-row chunk of a 27B on gfx1151 takes about 5 s: the cancel must not wait for all of it
    try std.testing.expect(pollsLayers(1, chunk));
    try std.testing.expect(pollsLayers(1, 512)); // the policy's default fill_rows, while other streams decode
    try std.testing.expect(pollsLayers(1, 2048));
    try std.testing.expect(!pollsLayers(1, 20)); // a short prompt runs through: a sync a layer would only slow it
    try std.testing.expect(!pollsLayers(2, chunk)); // tensor parallelism: the ranks stop together, between chunks
}
