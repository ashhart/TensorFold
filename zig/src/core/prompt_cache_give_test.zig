//! A stream's new memory taken back from kept states: least valuable first, never the state that stream restores.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const Store = pc.Store;

/// Memory as a backend sees it: `total` less what the kept states hold, against what a new slot needs.
const Left = struct {
    store: *const Store,
    total: u64,
    need: u64,
    pub fn fits(l: Left) bool {
        return l.total -| l.store.held >= l.need;
    }
};

const c1 = [_]u32{ 1, 1, 1, 1, 9 };
const c1_next = [_]u32{ 1, 1, 1, 1, 9, 5, 5, 9 };
const c2 = [_]u32{ 2, 2, 2, 2, 2, 2, 9 };
const c3 = [_]u32{ 3, 3, 3, 3, 3, 3, 3, 3, 9 };
const c3_next = [_]u32{ 3, 3, 3, 3, 3, 3, 3, 3, 9, 4, 9 };

/// Four kept states, oldest first: c2 at 6, c3 at 8, c1 at 4 (resumed, then moved past) and c1 at 7.
fn conversations(a: std.mem.Allocator, f: *Fake, s: *Store) !void {
    _ = f.pass(s, &c1, try s.begin(a, &c1, 4, &.{}, &.{}, null, &.{}));
    _ = f.pass(s, &c2, try s.begin(a, &c2, 6, &.{}, &.{}, null, &.{}));
    _ = f.pass(s, &c3, try s.begin(a, &c3, 8, &.{}, &.{}, null, &.{}));
    _ = f.pass(s, &c1_next, try s.begin(a, &c1_next, 7, &.{}, &.{}, null, &.{}));
}

fn ats(s: *const Store, out: []u32) []u32 {
    for (s.entries.items, 0..) |e, i| out[i] = e.at;
    std.mem.sort(u32, out[0..s.entries.items.len], {}, std.sort.asc(u32));
    return out[0..s.entries.items.len];
}

test "a new slot's memory comes from kept states: the superseded one, then the oldest, never the one the stream restores" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    try conversations(a, &f, &s);
    var buf: [8]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 4, 6, 7, 8 }, ats(&s, &buf));
    try std.testing.expectEqual(@as(u64, 104 + 106 + 108 + 107), s.held);
    const l = try s.lookup(a, &c3_next, 9, &.{}, &.{}, &.{}); // c3's next turn restores its state at 8
    defer s.release(l.entry.?);
    const keep = l.entry.?.saved;
    try std.testing.expect(s.giveBack(keep, &c3_next, Left{ .store = &s, .total = 500, .need = 285 }));
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, ats(&s, &buf)); // c1 at 4 (moved past), then c2 (oldest)
    try std.testing.expectEqual(@as(u64, 2), s.counts.evicted);
    try std.testing.expectEqual(@as(usize, 2), f.live);
    try std.testing.expect(s.giveBack(keep, &c3_next, Left{ .store = &s, .total = 500, .need = 285 })); // fits: nothing goes
    try std.testing.expectEqual(@as(u64, 2), s.counts.evicted);
}

test "kept states that cannot free enough all go but the restored one, and the slot is refused" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    try conversations(a, &f, &s);
    const l = try s.lookup(a, &c3_next, 9, &.{}, &.{}, &.{});
    try std.testing.expect(!s.giveBack(l.entry.?.saved, &c3_next, Left{ .store = &s, .total = 500, .need = 400 }));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expect(s.entries.items[0].saved == l.entry.?.saved); // its restore still finds its state
    try std.testing.expectEqual(@as(u64, 108), s.held);
    s.release(l.entry.?); // the refused pass no longer borrows a restore state
    try std.testing.expect(s.giveBack(null, &c3_next, Left{ .store = &s, .total = 500, .need = 400 })); // nothing to keep: it goes too
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 0), f.live);
}

test "a store of `most` slots keeps no more states than that: a keep past them frees the least valuable first" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.most = 2;
    try conversations(a, &f, &s);
    var buf: [8]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len); // the budget holds four, the slots two
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, ats(&s, &buf));
    f.at = 10; // the pass stands at row 10
    try std.testing.expect(s.keep(&c3_next, 10, null, &.{}, &.{})); // a keep no lookup made room for evicts first
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 2), f.live);
    _ = try s.lookup(a, &c2, 6, &.{}, &.{}, &.{}); // the pass will keep one state the store lacks: room made before it runs
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    s.most = 0;
    try std.testing.expect(!s.keep(&c2, 6, null, &.{}, &.{})); // no slot at all: refused, not saved
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
}

test "reusing spare storage still respects the kept-state count" {
    const gpa = std.testing.allocator;
    var f: Fake = .{ .gpa = gpa, .spare_bytes = 1000 };
    var s = Store.init(gpa, f.pooled(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.most = 1;
    f.at = 4;
    try std.testing.expect(s.keep(&c1, 4, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    f.at = 6;
    try std.testing.expect(s.keep(&c2, 6, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    try std.testing.expectEqual(@as(u32, 6), s.entries.items[0].at);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
}
