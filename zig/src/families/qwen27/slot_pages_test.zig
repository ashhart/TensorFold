//! slot_pages over a host pool: promised on open, grown as rows come, every page and promise back as a stream goes.
const std = @import("std");
const kvp = @import("core").metal_kv_pages;
const kv_pool = @import("core").kv_pool;
const sp = @import("slot_pages.zig");
const t = std.testing;
const page = kvp.page_tokens;
const pass = 512; // one widest pass past a prompt, as qwen27_slots.promiseRows adds

/// `lanes` lanes of `capacity` rows each at most over `pages` pages.
fn pool(lanes: usize, capacity: u32, pages: usize) !*kvp.Pages {
    const l: kv_pool.Layout = .{ .fixed_bytes = 0, .token_bytes = 64, .page_tokens = page, .snap_bytes = 0 };
    return kvp.Pages.init(t.allocator, null, l, lanes, pages, capacity, 0);
}

/// Nothing held, nothing promised, every page free.
fn empty(p: *kvp.Pages) !void {
    const c = p.pool.census();
    try t.expect(p.idle());
    try t.expectEqual(c.total, c.free);
    try t.expectEqual(@as(usize, 0), c.promised);
}

test "two streams promised their prompts open on a pool short of two windows, each grows as its rows come" {
    const window: u32 = 64 * page;
    const p = try pool(2, window, 96); // one and a half windows
    defer p.deinit();
    try sp.promise(p, 0, 20 * page + pass);
    try sp.promise(p, 1, 30 * page + pass);
    try sp.grow(p, 0, 20 * page);
    try sp.grow(p, 1, 30 * page);
    try sp.grow(p, 0, 40 * page); // a reply past its promise takes its pages as it writes them
    try t.expectEqual(@as(usize, 40), p.ids(0).len);
    for (0..2) |lane| try t.expectEqualSlices(u32, p.ids(lane), p.tables.entries(lane, p.ids(lane).len));
    for (0..2) |lane| sp.leave(p, lane, null);
    try empty(p);
}

test "two streams grow in turn to their whole windows, and leaving gives every page back" {
    const capacity: u32 = 262144 + 16; // --context 262144 and a 16-row window
    const p = try pool(2, capacity, 2 * ((capacity + page - 1) / page));
    defer p.deinit();
    try sp.promise(p, 0, 1000 + pass);
    try sp.promise(p, 1, 30_000 + pass);
    var rows = [2]u64{ 1000, 30_000 };
    for (0..2) |lane| try sp.grow(p, lane, rows[lane]);
    while (rows[0] < capacity or rows[1] < capacity) for (0..2) |lane| {
        rows[lane] = @min(rows[lane] + page, capacity);
        try sp.grow(p, lane, rows[lane]);
        try t.expectEqualSlices(u32, p.ids(lane), p.tables.entries(lane, p.ids(lane).len));
    };
    try t.expectEqual(@as(usize, 0), p.pool.census().free); // both windows whole: the pool's every page
    for (0..2) |lane| sp.leave(p, lane, null);
    try empty(p);
}

test "a promise or a growth the pool cannot meet fails with LaneMemory before a page is taken or an entry written" {
    const window: u32 = 32 * page;
    const p = try pool(2, window, 40);
    defer p.deinit();
    try sp.promise(p, 0, 30 * page);
    try sp.grow(p, 0, 30 * page);
    try t.expectError(error.LaneMemory, sp.promise(p, 1, 20 * page)); // 10 pages left
    try t.expectEqual(@as(usize, 0), p.leases[1].promised);
    try sp.promise(p, 1, 4 * page);
    try sp.grow(p, 1, 4 * page);
    const before = p.pool.census();
    const next = p.tables.entries(1, 5)[4];
    try t.expectError(error.LaneMemory, sp.grow(p, 1, 16 * page));
    try t.expectEqual(@as(usize, 4), p.ids(1).len);
    try t.expectEqual(next, p.tables.entries(1, 5)[4]); // its next entry as it was
    try t.expectEqual(before.free, p.pool.census().free);
    try t.expectEqual(before.promised, p.pool.census().promised);
    for (0..2) |lane| sp.leave(p, lane, null);
    try empty(p);
}

test "a stream that leaves mid-prompt, cancelled or failed, gives every page and its promise back" {
    const p = try pool(2, 16 * page, 32);
    defer p.deinit();
    try sp.promise(p, 1, 3 * page);
    try sp.grow(p, 1, 3 * page);
    try sp.promise(p, 0, 12 * page + pass);
    try sp.grow(p, 0, 5 * page); // five of its twelve prompt pages written
    sp.leave(p, 0, null);
    try t.expectEqual(@as(usize, 0), p.ids(0).len);
    try t.expectEqual(@as(usize, 0), p.leases[0].promised);
    try t.expectEqual(@as(usize, 3), p.ids(1).len); // the other stream keeps its own
    sp.leave(p, 1, null);
    try empty(p);
}

test "a reset mid-prompt gives the slot's pages back before the new prompt takes its own; a kept holder's page stays" {
    const p = try pool(2, 4 * page, 8);
    defer p.deinit();
    try sp.promise(p, 0, 3 * page);
    try sp.grow(p, 0, 3 * page);
    var old: [3]u32 = undefined;
    @memcpy(&old, p.ids(0));
    p.pool.retain(old[1]); // a kept state's hold on one of them
    sp.startOver(p, 0);
    try t.expectEqual(@as(usize, 0), p.ids(0).len);
    try t.expectEqual(@as(u32, 0), p.pool.holders(old[0]));
    try t.expectEqual(@as(u32, 1), p.pool.holders(old[1]));
    try t.expectEqual(@as(u32, 0), p.pool.holders(old[2]));
    try sp.promise(p, 0, 2 * page); // the new prompt
    try sp.grow(p, 0, 2 * page);
    for (p.ids(0)) |id| try t.expect(id != old[1]);
    try t.expectEqual(@as(usize, 2), p.ids(0).len);
    sp.leave(p, 0, null);
    p.pool.release(old[1]);
    try empty(p);
}

test "a stream moving to slot 0 leaves its old slot empty and unpromised; slot 0 holds its rows" {
    const p = try pool(2, 16 * page, 24);
    defer p.deinit();
    try sp.promise(p, 0, 6 * page);
    try sp.grow(p, 0, 6 * page);
    try sp.promise(p, 1, 8 * page + pass);
    try sp.grow(p, 1, 8 * page);
    sp.leave(p, 0, null); // slot 0's reply ended
    try sp.grow(p, 0, 8 * page); // qwen27_slots.promote: the copy covers slot 0 first
    sp.leave(p, 1, null);
    try sp.promise(p, 0, 8 * page + pass);
    try t.expectEqual(@as(usize, 0), p.ids(1).len);
    try t.expectEqual(@as(usize, 0), p.leases[1].promised);
    try t.expectEqual(@as(usize, 8), p.ids(0).len);
    try t.expectEqual(@as(usize, 2), p.leases[0].promised);
    sp.leave(p, 0, null);
    try empty(p);
}

test "a reply that finds no page mid-run fails its growth; its round's streams leave and no page or promise stays" {
    const p = try pool(2, 32 * page, 36);
    defer p.deinit();
    try sp.promise(p, 0, 20 * page + pass);
    try sp.grow(p, 0, 20 * page);
    try sp.promise(p, 1, 12 * page + pass);
    try sp.grow(p, 1, 12 * page);
    try sp.grow(p, 0, 22 * page);
    try t.expectError(error.LaneMemory, sp.grow(p, 1, 16 * page)); // Runner.cover in the round fails the verify
    for (0..2) |lane| sp.leave(p, lane, null); // lane_host_end.failRound ends both; each unbind releases its slot
    try empty(p);
}

test "a standing slot 0 keeps the pages of its committed rows, nothing past them and no promise" {
    const p = try pool(2, 16 * page, 16);
    defer p.deinit();
    try sp.promise(p, 0, 6 * page + pass);
    try sp.grow(p, 0, 8 * page);
    sp.leave(p, 0, 6 * page + 9); // its rows stay for the next prompt's resume
    try t.expectEqual(@as(usize, 7), p.ids(0).len);
    try t.expectEqual(@as(usize, 0), p.leases[0].promised);
    try t.expectEqual(@as(usize, 0), p.pool.census().promised);
    try sp.promise(p, 0, 7 * page + pass); // the resume's promise counts the pages it holds
    try t.expectEqual(@as(usize, 2), p.leases[0].promised);
    sp.leave(p, 0, null);
    try empty(p);
}
