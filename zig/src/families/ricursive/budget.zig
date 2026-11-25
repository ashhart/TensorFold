//! What a Ripple load on Metal may hold: llama's ledger and limit, and the longest context Ripple's caches fit.
const std = @import("std");
const cfg = @import("config.zig");
const st = @import("state.zig");
const llama = @import("../llama/budget.zig");

pub const Ledger = llama.Ledger;
pub const caches = llama.caches;
pub const admitStream = llama.admitStream;

/// A stream's ledger bytes from admission to release: its caches, and its drafter ring if it drafts.
pub fn charge(c: *const cfg.Config, capacity: usize, drafts: bool, ring: u64) u64 {
    return st.cacheBytes(c, capacity) + (if (drafts) ring else 0);
}

/// The longest context whose stream (its caches and logits row) fits in `bytes`, at most `most`.
pub fn longest(c: *const cfg.Config, bytes: u64, most: usize) usize {
    if (st.cacheBytes(c, 2) > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (st.cacheBytes(c, mid + 1) <= bytes) lo = mid else hi = mid - 1;
    }
    return lo;
}

/// The longest context whose first stream fits in `bytes` with its drafter ring (`ring`; 0 without a drafter).
pub fn longestDrafted(c: *const cfg.Config, bytes: u64, ring: u64, most: usize) usize {
    return if (ring > bytes) 0 else longest(c, bytes - ring, most);
}

test "ricursive budget: a drafter's first ring bounds the context, C + D fits the window and C + D - 1 does not" {
    const gpa = std.testing.allocator;
    const c = try cfg.parse(gpa, @embedFile("fixtures/config.json"));
    const dflash2 = @import("dflash2.zig");
    const dcfg = try @import("dflash2_config.zig").parse(gpa, @embedFile("fixtures/dflash2_config.json"), &c);
    const ring: u64 = dflash2.Drafter.laneBytes(&dcfg);
    const n: usize = 262144;
    const caches_n = st.cacheBytes(&c, n + 1); // a stream of the whole window: its caches and its logits row
    // automatic sizing takes the longest that fits; --context n is refused past it: both read this bound
    try std.testing.expectEqual(n, longestDrafted(&c, caches_n + ring, ring, n));
    try std.testing.expect(longestDrafted(&c, caches_n + ring - 1, ring, n) < n);
    // without a drafter the bound is the caches' alone, as before
    try std.testing.expectEqual(n, longestDrafted(&c, caches_n, 0, n));
    try std.testing.expect(longestDrafted(&c, caches_n - 1, 0, n) < n);
    try std.testing.expectEqual(longest(&c, caches_n + ring - 1, n), longestDrafted(&c, caches_n + ring - 1, 0, n));
    try std.testing.expectEqual(@as(usize, 0), longestDrafted(&c, ring - 1, ring, n));
}

test "ricursive budget: a 262,144-token stream's caches fit 17 GiB, not 16" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    const gib: u64 = 1 << 30;
    try std.testing.expect(longest(&c, 17 * gib, 1 << 20) >= 262144);
    try std.testing.expect(longest(&c, 16 * gib, 1 << 20) < 262144);
    try std.testing.expectEqual(@as(usize, 0), longest(&c, 1 << 20, 1 << 20));
}
