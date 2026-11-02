//! A retained-payload cap beside live slots: shared KV pages, bounded snapshots and indexed token payloads.
const std = @import("std");
const Store = @import("store.zig").Store;
const none = @import("store.zig").none;
const max_world = @import("pool.zig").max_world;

/// The most-loaded rank's KV plus index and logical bounded payload; unequal ranks supply Bounded.logical_bytes.
pub fn bytes(s: *const Store) u64 {
    const p = s.pool;
    var pages: [max_world]u64 = @splat(0);
    for (p.refs, p.slot_refs, 0..) |refs, slots, pg| {
        if (refs > slots) pages[pg % p.world] += 1;
    }
    var total: u64 = s.bounded_logical_total;
    var all: u64 = 0;
    var worst: u64 = 0;
    for (pages[0..p.world]) |n| {
        all += n;
        worst = @max(worst, n);
    }
    for (p.layout.families) |f| total += f.pageBytes(p.page) * (if (f.split and p.world > 1) worst else all);
    for (s.index.entries.items) |e| if (e.live and e.ram) {
        total += @sizeOf(@TypeOf(e)) + 4 * @as(u64, e.tail.len + e.extra.pages.len);
    };
    for (s.index.nodes.items) |n| if (n.sub_ram > 0) {
        total += @sizeOf(@TypeOf(n));
        if (n.tokens) |ids| total += 4 * @as(u64, ids.len);
    };
    return total;
}

/// Evict LRU entries, including the newest if it alone exceeds the cap; live slots keep their own page references.
pub fn trim(s: *Store, cap: u64) !void {
    try s.settleAll();
    while (bytes(s) > cap and s.oldest != none) {
        _ = try s.evict(s.oldest);
        try s.settleAll();
    }
}
