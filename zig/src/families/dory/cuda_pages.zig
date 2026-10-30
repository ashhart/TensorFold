//! Dory's full-attention caches in pages: an arena (each cache's K, then V, over every page) and a table of ids a slot.
const std = @import("std");
const kv = @import("core").kv_pool;
const config = @import("config.zig");
const memory = @import("memory.zig");
const fwd = @import("cuda_model.zig");

/// Slots a round of `round` rows holds over windows of `window` rows.
pub fn slotsOf(round: usize, window: usize) usize {
    return round / window;
}

/// Slots a round holds: its rows over the widest window.
pub const max_slots = slotsOf(fwd.round_rows, fwd.window_rows);
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
    wins: [max_slots]?u32 = @splat(null), // w when the slot's pages are ids from w * per_slot on (contiguous slot w)

    pub fn pageBytes(p: Paging) usize {
        return p.page_elems * 2;
    }
};

/// w when `ids` are consecutive from w * per (a contiguous slot of the arena's caches), else null.
pub fn windowOf(ids: []const u32, per: usize) ?u32 {
    if (ids.len == 0 or ids[0] % per != 0) return null;
    for (ids, ids[0]..) |id, want| if (id != want) return null;
    return @intCast(ids[0] / per);
}

/// Table entries a slot of `context` positions takes: whole pages only, else error.BadAttach.
pub fn perSlot(context: usize) !usize {
    const n = pagesOf(context);
    if (n == 0 or n * tokens != context) return error.BadAttach;
    return n;
}

/// Pages that hold `positions`.
pub fn pagesOf(positions: usize) usize {
    return (positions + tokens - 1) / tokens;
}

/// bf16 elements a page holds across `full` caches: `tokens` rows of `kv_dim` in each cache's K and V.
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

test "a window: consecutive ids from a multiple of the slot's pages; a gap, a step back, an offset or none is not" {
    try std.testing.expectEqual(@as(?u32, 2), windowOf(&.{ 8, 9, 10 }, 4));
    try std.testing.expectEqual(@as(?u32, 0), windowOf(&.{0}, 4));
    try std.testing.expectEqual(@as(?u32, null), windowOf(&.{ 7, 8 }, 4));
    try std.testing.expectEqual(@as(?u32, null), windowOf(&.{ 8, 10 }, 4));
    try std.testing.expectEqual(@as(?u32, null), windowOf(&.{ 9, 8 }, 4));
    try std.testing.expectEqual(@as(?u32, null), windowOf(&.{}, 4));
}

test "a round's slots are its rows over the widest window, not a number kept beside them" {
    try std.testing.expectEqual(@as(usize, 8), slotsOf(128, 16));
    try std.testing.expectEqual(@as(usize, 4), slotsOf(128, 32));
    try std.testing.expectEqual(@as(usize, 16), slotsOf(256, 16));
    try std.testing.expectEqual(fwd.round_rows / fwd.window_rows, max_slots);
    try std.testing.expect(max_slots * fwd.window_rows <= fwd.round_rows);
}

test "a slot's table is whole pages: a context off the page size refuses to attach" {
    try std.testing.expectEqual(@as(usize, 132), try perSlot(33792));
    try std.testing.expectEqual(@as(usize, 4), try perSlot(1024));
    try std.testing.expectError(error.BadAttach, perSlot(33793));
    try std.testing.expectError(error.BadAttach, perSlot(1000));
    try std.testing.expectError(error.BadAttach, perSlot(255));
    try std.testing.expectError(error.BadAttach, perSlot(0));
    for (1..200) |k| try std.testing.expectEqual(k, try perSlot(k * tokens));
}
