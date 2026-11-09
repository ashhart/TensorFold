//! #1594 step 2b on the real lane host, continued: which copy a stream stands on, what it lets go, what its reply says.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const fixture = @import("lane_remote_test.zig");
const Peer = fixture.Peer;
const Exp = fixture.Exp;
const Kept = fixture.Kept;
const Slot = fixture.Slot;
const runWith = fixture.runWith;
const runKeep = fixture.runKeep;
const unborrowed = fixture.unborrowed;
const six = fixture.six;
const eight = fixture.eight;
const id = fixture.id;
const t = std.testing;

var none: u8 = 0;

/// A prompt cache over Kept's copies, as lane_remote_test.zig's tests make one.
fn cache() pc.Store {
    return pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
}

/// Where the state a lookup finds stands, and whether it is remote.
const Found = struct { at: u32, remote: bool };

fn found(store: *pc.Store, prompt: []const u32) ?Found {
    const e = store.find(prompt, &.{}, &.{}) orelse return null;
    return .{ .at = e.at, .remote = e.remote };
}

test "handoff: a repeat whose lookup finds its exact cut resumes below it on a remote state, and the stream stays remote" {
    var store = cache();
    defer store.deinit();
    store.keep_remote = true;
    try t.expect(store.keep(&six, 3, null, &.{}, &.{})); // built on another machine's prefix, before the cut
    store.keep_remote = false;
    try t.expect(store.keep(&six, 5, null, &.{}, &.{})); // the engine's own at the cut, which the lookup finds first
    var peer: Peer = .{ .at = 0 };
    var e: Exp = .{};
    const seen = try runWith(&peer, &.{ .prompt = &six, .max_tokens = 1, .handoff = id }, &e, &store);
    try t.expect(seen.restoring and seen.remote); // the remote state's provenance, not the exact cut's
    try t.expectEqual(@as(u32, 3), seen.at);
    try t.expectEqual(@as(?usize, 5), e.tokens);
    try t.expectEqual(@as(?u32, 3), seen.remote_prefix); // so its reply is outside the exactness contract
    try t.expectEqual(@as(?Found, .{ .at = 5, .remote = false }), found(&store, &six)); // and served the pass's keep
    try t.expectEqual(@as(u64, 2), store.counts.kept);
    try unborrowed(&store);
}

test "remote prefix kept: a refused pull leaves the shorter kept state it beat, let go before the pass and never refused" {
    var store = cache();
    defer store.deinit();
    try t.expect(store.keep(&six, 2, null, &.{}, &.{})); // shorter than the pull: the lookup borrows it, the pull wins
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    const seen = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true, .refuse = true });
    try t.expect(seen.restoring and seen.remote and seen.at == 4); // the pull's copy, which the first chunk refused
    try t.expectEqual(@as(u32, 0), seen.borrowed); // the state it beat was let go before the pass began
    try t.expectEqual(@as(?u32, 0), seen.remote_prefix); // refused: the whole prompt was prefilled here
    try t.expectEqual(@as(u64, 0), store.counts.failed); // the refusal is the pull's, never the kept state's
    try t.expectEqual(@as(?Found, .{ .at = 2, .remote = false }), found(&store, six[0..3])); // still kept, still exact
    try t.expectEqual(@as(?Found, .{ .at = 5, .remote = false }), found(&store, &six)); // the pass ran from token 0
    try t.expectEqual(@as(u32, 1), peer.drops.load(.acquire));
    try unborrowed(&store);
}

test "remote prefix kept: a kept remote state that slot 0's own state reaches is let go before the pass runs" {
    var store = cache();
    defer store.deinit();
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{}));
    store.keep_remote = false;
    const held = store.find(&six, &.{}, &.{}).?;
    held.borrowers += 1; // another job's borrow of the same state
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    var slot: Slot = .{ .at = 4 };
    const peek: remote.ResumePeek = .{ .ctx = &slot, .at = Slot.peek };
    const level = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true, .peek = peek });
    try t.expect(!level.restoring and !level.remote); // slot 0's own exact state goes on
    try t.expectEqual(@as(u32, 1), level.borrowed); // the lookup's borrow is not held while the pass runs, the other's is
    try t.expectEqual(@as(u32, 0), peer.fills);
    try t.expectEqual(@as(u32, 1), held.borrowers); // let go once: a lookup still named after it would take the other's
    store.release(held);
    try unborrowed(&store);
}

test "remote prefix: a reply on a fresh pull reports all of its remote prefix pulled, none of it a kept remote copy" {
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    const seen = try runWith(&peer, &.{ .prompt = &six, .max_tokens = 2 }, null, null);
    try t.expect(seen.restoring and seen.remote and seen.at == 4);
    try t.expectEqual(@as(?u32, 4), seen.remote_prefix);
    try t.expectEqual(@as(?u32, 4), seen.pulled_prefix);
    try t.expectEqual(@as(u32, 0), seen.remote_prefix.? - seen.pulled_prefix.?); // kept_remote_cache_tokens: none
}

test "remote prefix kept (TF_PREFILL_KEEP): a reply on a kept remote state reports none of it pulled, all of it kept" {
    var store = cache();
    defer store.deinit();
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    const first = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true });
    try t.expectEqual(@as(?u32, 4), first.remote_prefix); // pulled over the link for this reply; its mark 5 kept remote
    try t.expectEqual(@as(?u32, 4), first.pulled_prefix);
    const next = try runKeep(&peer, &.{ .prompt = &eight, .max_tokens = 2, .history_len = 7 }, null, &store, .{ .keep = true });
    try t.expect(next.restoring and next.remote and next.at == 5);
    try t.expectEqual(@as(u32, 1), peer.fills); // nothing pulled for this reply
    try t.expectEqual(@as(?u32, 5), next.remote_prefix);
    try t.expectEqual(@as(?u32, 0), next.pulled_prefix);
    try t.expectEqual(@as(u32, 5), next.remote_prefix.? - next.pulled_prefix.?); // kept_remote_cache_tokens: all five
    try unborrowed(&store);
}
