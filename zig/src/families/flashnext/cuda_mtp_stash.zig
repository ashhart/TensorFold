//! Rounds the draft head skips while a stream runs plain, absorbed in one pass when it drafts again.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const Draft = mtp.Draft;

const wide = wt.wide;

/// Keep `next.len` rows of `rows` for later; false when full (taken first, so the caller's absorb follows in order).
pub fn stash(o: anytype, d: *Draft, next: []const u32, rows: u64) !bool {
    if (!d.canStash() or next.len == 0 or next.len > Draft.stash_rows) return false;
    if (d.stashed + next.len > Draft.stash_rows) {
        try flush(o, d);
        return false;
    }
    d.verify_cache.invalidate();
    const at = d.streams.ptr + (mtp.max_rows + d.stashed) * wide * 2;
    const n = next.len * wide * 2;
    const st = o.stream();
    try st.d.check(st.d.api.cuMemcpyDtoDAsync_v2(at, rows, n, st.handle), "cuMemcpyDtoDAsync");
    @memcpy(d.stash_next[d.stashed..][0..next.len], next);
    d.stashed += next.len;
    return true;
}

/// The kept rounds through the head in one pass (the cache the rounds would have built, up to bits).
pub fn flush(o: anytype, d: *Draft) !void {
    const n = d.stashed;
    if (n == 0) return;
    d.stashed = 0;
    try o.absorb(d, d.stash_next[0..n], d.streams.ptr + mtp.max_rows * wide * 2, false);
}

test "rounds are kept only where streams has rows past the widest window's" {
    var d: Draft = undefined;
    d.rows_cap = mtp.max_rows;
    try std.testing.expect(!d.canStash());
    d.rows_cap = mtp.max_rows + Draft.stash_rows - 1;
    try std.testing.expect(!d.canStash());
    d.rows_cap = mtp.max_rows + Draft.stash_rows;
    try std.testing.expect(d.canStash());
    try std.testing.expect(Draft.stash_rows <= mtp.piece_rows); // one head pass takes them all
}
