//! What a Ripple load on Metal may hold: llama's ledger and limit, and the longest context Ripple's caches fit.
const std = @import("std");
const cfg = @import("config.zig");
const st = @import("state.zig");
const llama = @import("../llama/budget.zig");

pub const Ledger = llama.Ledger;
pub const caches = llama.caches;
pub const admitStream = llama.admitStream;

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

test "ricursive budget: a 262,144-token stream's caches fit 17 GiB, not 16" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    const gib: u64 = 1 << 30;
    try std.testing.expect(longest(&c, 17 * gib, 1 << 20) >= 262144);
    try std.testing.expect(longest(&c, 16 * gib, 1 << 20) < 262144);
    try std.testing.expectEqual(@as(usize, 0), longest(&c, 1 << 20, 1 << 20));
}
