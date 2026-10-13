//! Host-only Context storage ownership and initialization provenance regressions.
const std = @import("std");
const p = @import("draft_pool.zig");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

test "strict flag is default off and requires exact cache" {
    try expect(!try p.enabled(null, false));
    try expect(!try p.enabled("0", false));
    try expect(try p.enabled("1", true));
    try std.testing.expectError(error.DraftReuseRequiresExactCache, p.enabled("1", false));
    try std.testing.expectError(error.InvalidDraftReuseFlag, p.enabled("yes", true));
}

test "fresh small drafted requests include cross-key and capacity changes" {
    try expect(p.eligible(true, true, 14, 286, true, false));
    try expect(p.eligible(true, true, 31, 303, true, false));
    try expect(p.eligible(true, true, 1, 512, true, false));
    try expect(!p.eligible(false, true, 14, 286, true, false));
    try expect(!p.eligible(true, false, 14, 286, true, false));
    try expect(!p.eligible(true, true, 31, 513, true, false));
    try expect(!p.eligible(true, true, 0, 286, true, false));
    try expect(!p.eligible(true, true, 31, 31, true, false));
    try expect(!p.eligible(true, true, 31, 303, false, false));
    try expect(!p.eligible(true, true, 31, 303, true, true));
}

test "detached storage transfers once and total ownership is bounded" {
    var owners = [_]u8{ 1, 2, 3 };
    var pool = p.Pool(u8).init(true, 2, 80);
    try expect(pool.retain(&owners[0], 1));
    try expect(!pool.retain(&owners[1], 1));
    try expect(!pool.hasRoom(1));
    try expect(pool.take().? == &owners[0]);
    try expect(pool.take() == null);
    try equal(@as(u64, 1), pool.stats.reuse_hits);
    try expect(pool.retain(&owners[0], 0));
    try expect(pool.retain(&owners[1], 0));
    try expect(!pool.retain(&owners[2], 0));
    try equal(@as(usize, 160), pool.stats.retained_bytes);
    try expect(pool.evict().? == &owners[1]);
    try expect(pool.evict().? == &owners[0]);
    try equal(@as(usize, 0), pool.stats.retained_bytes);
    try equal(@as(u64, 2), pool.stats.evictions);
}

test "retained bytes and backend configured ownership cap are enforced" {
    var owners = [_]u8{ 1, 2, 3 };
    var pool = p.Pool(u8).init(true, 32, p.max_bytes / 2);
    try equal(@as(usize, 16), pool.limit);
    try expect(pool.retain(&owners[0], 0));
    try expect(pool.retain(&owners[1], 0));
    try expect(!pool.retain(&owners[2], 0));
    try equal(p.max_bytes, pool.stats.retained_bytes);
    var disabled = p.Pool(u8).init(false, 16, 1);
    try expect(!disabled.retain(&owners[0], 0));
    var zero = p.Pool(u8).init(true, 0, 1);
    try expect(!zero.hasRoom(0));
}

test "failed fence quarantines without reuse or free until successful fence" {
    var owners = [_]u8{ 1, 2 };
    var pool = p.Pool(u8).init(true, 2, 80);
    pool.park(&owners[0], 1);
    try expect(pool.stats.failed);
    try expect(!pool.hasRoom(1));
    try expect(pool.take() == null);
    try expect(pool.reap(false) == null);
    try equal(@as(usize, 1), pool.stats.quarantined_slots);
    try equal(@as(usize, 80), pool.stats.quarantined_bytes);
    try expect(pool.reap(true).? == &owners[0]);
    try expect(pool.reap(true) == null);
    try equal(@as(usize, 0), pool.stats.quarantined_bytes);
    try expect(!pool.retain(&owners[1], 0));
    try equal(@as(u64, 1), pool.stats.fence_failures);
}

test "initialization proof binds independent request target wrapper and arena" {
    var owners = [_]u8{ 1, 2, 3, 4 };
    const proof: p.Proof = .{ .stream = &owners[0], .seq = &owners[1], .wrapper = &owners[2], .arena = 4096, .tokens = 31, .capacity = 303 };
    var proofs: p.Proofs = .{};
    try proofs.mark(proof);
    try std.testing.expectError(error.DuplicateDraftInitialization, proofs.mark(proof));
    try expect(proofs.remove(&owners[3]) == null);
    const taken = proofs.remove(&owners[0]).?;
    try expect(taken.seq == @as(*const anyopaque, &owners[1]) and taken.wrapper == @as(*const anyopaque, &owners[2]));
    try equal(@as(u64, 4096), taken.arena);
    try expect(proofs.remove(&owners[0]) == null);
    var invalid = proof;
    invalid.capacity = 513;
    try std.testing.expectError(error.InvalidDraftInitialization, proofs.mark(invalid));
    invalid = proof;
    invalid.arena = 0;
    try std.testing.expectError(error.InvalidDraftInitialization, proofs.mark(invalid));
}

test "retention requires every independent clean completion gate" {
    try expect(p.completed(true, true, true, true, true, false));
    try expect(!p.completed(false, true, true, true, true, false));
    try expect(!p.completed(true, false, true, true, true, false));
    try expect(!p.completed(true, true, false, true, true, false));
    try expect(!p.completed(true, true, true, false, true, false));
    try expect(!p.completed(true, true, true, true, false, false));
    try expect(!p.completed(true, true, true, true, true, true));
}
