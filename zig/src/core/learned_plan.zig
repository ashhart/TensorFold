//! Choose one LRU victim set only when it can satisfy both disk floors and the learned-store cap.
const std = @import("std");
pub const Candidate = struct { key: u64, local: u64, peer: u64, cap: u64, used: u64 };
pub const Need = struct { local: u64, peer: u64, cap: u64 };
fn fits(have: Need, need: Need) bool {
    return have.local >= need.local and have.peer >= need.peer and have.cap >= need.cap;
}
pub fn choose(a: std.mem.Allocator, candidates: []const Candidate, need: Need) !?[]u64 {
    var total: Need = .{ .local = 0, .peer = 0, .cap = 0 };
    for (candidates) |c| {
        total.local +|= c.local;
        total.peer +|= c.peer;
        total.cap +|= c.cap;
    }
    if (!fits(total, need)) return null;
    const ordered = try a.dupe(Candidate, candidates);
    defer a.free(ordered);
    std.mem.sort(Candidate, ordered, {}, struct {
        fn less(_: void, x: Candidate, y: Candidate) bool {
            return x.used < y.used or (x.used == y.used and x.key < y.key);
        }
    }.less);
    var victims: std.ArrayList(u64) = .empty;
    errdefer victims.deinit(a);
    var have: Need = .{ .local = 0, .peer = 0, .cap = 0 };
    for (ordered) |c| {
        if (fits(have, need)) break;
        try victims.append(a, c.key);
        have.local +|= c.local;
        have.peer +|= c.peer;
        have.cap +|= c.cap;
    }
    return try victims.toOwnedSlice(a);
}
test "both ranks use one victim set and an impossible shortfall deletes nothing" {
    const a = std.testing.allocator;
    const items = [_]Candidate{ .{ .key = 1, .local = 10, .peer = 5, .cap = 10, .used = 1 }, .{ .key = 2, .local = 10, .peer = 25, .cap = 10, .used = 2 }, .{ .key = 3, .local = 50, .peer = 0, .cap = 50, .used = 3 } };
    const selected = (try choose(a, &items, .{ .local = 20, .peer = 30, .cap = 0 })).?;
    defer a.free(selected);
    try std.testing.expectEqualSlices(u64, &.{ 1, 2 }, selected);
    try std.testing.expect((try choose(a, &items, .{ .local = 1, .peer = 31, .cap = 0 })) == null);
    const empty = (try choose(a, &items, .{ .local = 0, .peer = 0, .cap = 0 })).?;
    defer a.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);
}
