//! TF_PREFILL_KEEP on the lane host: a pass's remote states are kept, resumed, reported remote and released.
const std = @import("std");
const api = @import("engine_api.zig");
const platform = @import("sys");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const t = std.testing;
const fixture = @import("lane_remote_test.zig");
const Peer = fixture.Peer;
const Exp = fixture.Exp;
const Kept = fixture.Kept;
const runKeep = fixture.runKeep;
const unborrowed = fixture.unborrowed;
const six = fixture.six;
const eight = fixture.eight;
const id = fixture.id;
const Slot = fixture.Slot;

test "remote prefix kept (TF_PREFILL_KEEP): its pass keeps remote states; the next turn resumes one, counted cached, reported remote, and keeps remote states" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    // a chat's history ends at 5: the store marks it, past the 4 pulled tokens
    const first = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true });
    try t.expect(first.restoring and first.remote and first.hooked);
    try t.expectEqual(@as(u32, 4), first.at);
    try t.expectEqual(@as(u32, 1), peer.fills);
    try t.expectEqual(@as(?u32, 0), first.prefilled); // a fresh pull is not the cache's
    try t.expectEqual(@as(u32, 1), peer.drops.load(.acquire)); // its storage went once, at the first mark
    try t.expectEqual(@as(u64, 1), store.counts.kept);
    try t.expect(store.entries.items[0].remote and store.entries.items[0].at == 5); // kept past the prefix, marked remote
    // the chat's next turn: its history ends at 7, past the kept state it resumes
    const next = try runKeep(&peer, &.{ .prompt = &eight, .max_tokens = 2, .history_len = 7 }, null, &store, .{ .keep = true });
    try t.expect(next.restoring and next.remote); // a kept remote state: restored here, still remote
    try t.expectEqual(@as(u32, 5), next.at);
    try t.expectEqual(@as(u32, 1), peer.fills); // the cache's state beat a new fill
    try t.expectEqual(@as(?u32, 5), next.prefilled); // kept here, so counted as cached
    try t.expectEqual(@as(?u32, 5), next.remote_prefix); // and still reported as a remote prefix, never as exact
    try t.expectEqual(@as(u32, 1), peer.drops.load(.acquire));
    try t.expectEqual(@as(u64, 2), store.counts.kept);
    const later = store.find(&eight, &.{}, &.{}).?;
    try t.expect(later.at == 7 and later.remote); // nothing pulled this turn, yet what it keeps stands on a remote state
}

test "kept remote states: an exact keep hides the remote copy, and a remote keep rides an exact one" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{}));
    store.keep_remote = false;
    try t.expect(store.find(&six, &.{}, &.{}).?.remote);
    try t.expect(store.keep(&six, 4, null, &.{}, &.{}));
    try t.expect(!store.find(&six, &.{}, &.{}).?.remote); // the engine's own copy is the one a lookup finds
    try t.expectEqual(@as(u64, 2), store.counts.kept);
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{})); // the exact copy serves a remote keep: nothing new
    store.keep_remote = false;
    try t.expectEqual(@as(u64, 2), store.counts.kept);
}

test "remote prefix kept: a warm pass resuming a kept remote state keeps its end remote and off disk, and the next turn is remote" {
    var buf: [64]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-remote-warm-{d}", .{platform.c.getpid()});
    defer rmTree(root);
    var none: u8 = 0;
    const family: pc.Snapshots = .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop, .rows = &.{ .parts = Kept.parts, .gather = Kept.gather, .blank = Kept.blank, .scatter = Kept.scatter } } };
    const disk = try pc.rows.Rows.open(t.allocator, family, .{ .root = root, .identity = 2, .cap = 1 << 30, .floor = 0, .daily = 1 << 30 });
    defer disk.deinit();
    var store = pc.Store.init(t.allocator, family, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.disk = disk;
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{})); // a pulled turn's history state
    store.keep_remote = false;
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    // the server's warm pass after a reply: background, no tokens, its history the whole prompt
    const warm = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 0, .background = true, .history_len = 6 }, null, &store, .{ .keep = true, .warm = true });
    try t.expect(warm.restoring and warm.remote and warm.at == 4);
    const end = store.find(&eight, &.{}, &.{}).?;
    try t.expect(end.at == 6 and end.remote); // the end it kept stands on the remote state: remote too
    try t.expectEqual(@as(usize, 0), disk.states.items.len); // nothing written, though the host's stop writes what waits
    const next = try runKeep(&peer, &.{ .prompt = &eight, .max_tokens = 2, .history_len = 7 }, null, &store, .{ .keep = true, .warm = true });
    try t.expect(next.restoring and next.remote and next.at == 6);
    try t.expectEqual(@as(?u32, 6), next.remote_prefix); // reported as a remote prefix, never as exact
    try t.expectEqual(@as(u32, 0), peer.fills);
}

test "remote prefix kept: a pull past a shorter kept state releases the lookup's borrow, and keeps only marks past the pull" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    try t.expect(store.keep(&six, 2, null, &.{}, &.{})); // shorter than the pull: the lookup borrows it, the pull beats it
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    // a history ending where the pull does: the pass never stands there, so no mark is left to keep
    const at_pull = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 4 }, null, &store, .{ .keep = true });
    try t.expect(at_pull.restoring and at_pull.remote and at_pull.hooked and at_pull.marks == 0);
    try t.expectEqual(@as(u64, 1), store.counts.kept);
    const beyond = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true });
    try t.expect(beyond.restoring and beyond.remote and beyond.at == 4 and beyond.marks == 1);
    try t.expectEqual(@as(u32, 2), peer.fills);
    for (store.entries.items) |e| try t.expectEqual(@as(u32, 0), e.borrowers); // else the shorter state is never evicted
    try t.expect(store.find(&six, &.{}, &.{}).?.remote); // the pass kept 5, marked remote
}

test "remote prefix kept: slot 0's own state as far as a kept remote state wins over it, and the pass keeps exact states" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{}));
    store.keep_remote = false;
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    var slot: Slot = .{ .at = 3 };
    const peek: remote.ResumePeek = .{ .ctx = &slot, .at = Slot.peek };
    const short = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 4 }, null, &store, .{ .keep = true, .peek = peek });
    try t.expect(short.restoring and short.remote and short.at == 4); // slot 0 short of it: the remote state is restored
    slot.at = 4;
    const level = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true, .peek = peek });
    try t.expect(!level.restoring and !level.remote and level.marks == 1); // as far: slot 0's own exact state goes on
    try t.expectEqual(@as(u32, 0), peer.fills);
    for (store.entries.items) |e| try t.expectEqual(@as(u32, 0), e.borrowers); // the remote state was let go
    try t.expect(!store.find(&six, &.{}, &.{}).?.remote); // and the mark past it is kept exact
}
