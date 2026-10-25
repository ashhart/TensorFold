//! Dory's full-attention caches in pages: one arena of the pool's pages and a table of page ids a slot.
const std = @import("std");
const kv = @import("core").kv_pool;
const config = @import("config.zig");
const memory = @import("memory.zig");

/// Slots a round holds: its rows over the widest window (cuda_model's round_rows / window_rows).
pub const max_slots = 8;
/// Positions a page holds: window_rows.cu's decode chunk, so a chunk and a prompt tile stay inside one page.
pub const tokens = 256;

/// The arena of `pool` pages, every slot's leases among them, and the slots' page tables.
pub const Paging = struct {
    pool: usize,
    per_slot: usize, // table entries a slot
    page_elems: usize, // bf16 elements a page, over every full cache's K and V
    arena: u64,
    table: u64,
    ids: []u32, // the table on the host, [max_slots * per_slot]
    caps: [max_slots]usize = @splat(0), // positions the slot's pages cover

    pub fn pageBytes(p: Paging) usize {
        return p.page_elems * 2;
    }
};

/// Pages that hold `positions`.
pub fn pagesOf(positions: usize) usize {
    return (positions + tokens - 1) / tokens;
}

/// bf16 elements of a page across `full` caches: each cache's K block, then its V block, of `tokens` rows of `kv_dim`.
pub fn pageElems(c: config.Config, full: usize) usize {
    return 2 * full * tokens * c.kvDim();
}

/// What the host's pool plans: a slot costs its rings and table row, a token its full caches' rows.
pub fn layout(c: config.Config, full: usize, sliding: usize, context: usize, ring: usize) kv.Layout {
    return .{
        .fixed_bytes = sliding * ring * memory.tokenBytes(c) + pagesOf(context) * 4,
        .token_bytes = full * memory.tokenBytes(c),
        .page_tokens = tokens,
        .snap_bytes = 0, // kept states are copies of varying size: no slots
    };
}

/// Pages a request holds: its prompt, its reply and one last verify window (positions written, from 0).
pub fn need(prompt: usize, max_new: usize, window: usize) usize {
    return pagesOf(prompt + max_new + window);
}

test "a page holds 256 positions of every full cache: 9, 14 and 24 MiB at 1, 2 and 4 loops" {
    const text = try config.pinnedText(std.testing.allocator);
    defer std.testing.allocator.free(text);
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, text, &why);
    for ([_]usize{ 1, 2, 4 }, [_]usize{ 9, 14, 24 }) |loops, full| {
        const l = layout(try c.withLoops(loops), full, 3 * full, 32768, 3072);
        try std.testing.expectEqual(@as(u64, full << 20), l.pageBytes());
    }
    try std.testing.expectEqual(@as(usize, 129), pagesOf(32768 + 1));
    try std.testing.expectEqual(@as(usize, 1), need(1, 1, 16));
}
