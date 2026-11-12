//! KV pages of the dense engine: 512 rows of every layer's keys and values, the host pool's layout, a request's pages.
const std = @import("std");
const kv = @import("core").kv_pool;
const Geometry = @import("recipe.zig").Geometry;

/// Rows a page holds: one decode attention chunk (nemotron_attention.cu's PAGE).
pub const page_rows = 512;
/// The widest window a stream verifies: a request's pages hold that many rows past its prompt and reply.
pub const window = 16;

/// Pages that hold positions [0, positions).
pub fn pagesOf(positions: usize) usize {
    return (positions + page_rows - 1) / page_rows;
}

/// Pages a request holds at most: its prompt, its reply and the verify window.
pub fn pagesFor(prompt: usize, max_new: usize) usize {
    return pagesOf(prompt + max_new + window);
}

/// Table entries a stream takes at a `context`-token window, its verify window included.
pub fn perSlot(context: usize) usize {
    return pagesOf(context + window);
}

/// bf16 elements a page holds: [layer][K, V][512][kv width].
pub fn pageElems(g: Geometry) usize {
    return g.layers * 2 * page_rows * g.kvWidth();
}

pub fn pageBytes(g: Geometry) u64 {
    return pageElems(g) * 2;
}

/// Bytes of one stream at a whole `context`-token window: its pages and its table row.
pub fn streamBytes(g: Geometry, context: usize) u64 {
    return perSlot(context) * (pageBytes(g) + 4);
}

/// What the host's pool plans: a token's keys and values over every layer, a stream's table row fixed.
pub fn layout(g: Geometry, context: usize) kv.Layout {
    const row: u64 = 2 * g.kvWidth() * 2;
    return kv.Layout.ofRows(.{ .kind = .full, .row = row, .layers = @intCast(g.layers) }, page_rows, perSlot(context) * 4, 0);
}

const dots: Geometry = .{ .hidden = 1536, .layers = 28, .heads = 12, .kv_heads = 2, .head_dim = 128, .vocab = 151936, .eps = 1e-6, .context = 131072, .acc = 2048, .wide = 17920, .mid = 8960, .theta = 1e6 };

test "dots' page: 512 rows of 28 layers' keys and values at 2 x 128 dims, 14 MiB" {
    try std.testing.expectEqual(@as(u64, 28 * 2 * 512 * 256 * 2), pageBytes(dots));
    try std.testing.expectEqual(@as(u64, 14680064), pageBytes(dots));
    const l = layout(dots, 32768);
    try std.testing.expectEqual(pageBytes(dots), l.pageBytes());
    try std.testing.expectEqual(@as(u64, 65 * 4), l.fixed_bytes);
    try std.testing.expectEqual(@as(usize, 65), l.pages(32768 + window));
    try std.testing.expectEqual(@as(u64, 65 * (14680064 + 4)), streamBytes(dots, 32768));
}

test "a request holds its prompt, its reply and the verify window in whole pages" {
    try std.testing.expectEqual(@as(usize, 1), pagesFor(1, 1));
    try std.testing.expectEqual(@as(usize, 1), pagesFor(400, 96));
    try std.testing.expectEqual(@as(usize, 2), pagesFor(400, 97));
    try std.testing.expectEqual(@as(usize, 65), pagesFor(32768, 0));
    try std.testing.expectEqual(@as(usize, 65), perSlot(32768));
    try std.testing.expectEqual(@as(usize, 1), perSlot(496));
    try std.testing.expectEqual(@as(usize, 2), perSlot(497));
}
