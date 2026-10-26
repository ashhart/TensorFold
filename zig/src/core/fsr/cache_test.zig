const std = @import("std");
const Cache = @import("cache.zig").Cache;
const absent = @import("cache.zig").absent;
const a = std.testing.allocator;

fn fill(c: *Cache, keys: []const usize) !void {
    try c.protect(keys);
    errdefer c.release();
    for (keys) |k| if (try c.reserve(k, false)) |t| try c.publish(t);
}

test "FSR CLOCK protects the whole incoming union before evicting early misses" {
    var c = try Cache.init(a, 8, 3);
    defer c.deinit();
    try fill(&c, &.{ 0, 1, 2 });
    c.release();
    const stable = (try c.resident(2)).?;
    try fill(&c, &.{ 3, 2, 3 });
    try std.testing.expectEqual(stable, (try c.resident(2)).?);
    try std.testing.expect(try c.resident(3) != null);
    try std.testing.expectEqual(@as(usize, 3), c.slots.len);
    c.release();
    try std.testing.expectError(error.ActiveUnionExceedsPool, c.protect(&.{ 0, 1, 2, 3 }));
    try std.testing.expectError(error.InvalidExpert, c.protect(&.{ 0, 8 }));
    for (c.demand) |x| try std.testing.expect(!x);
}

test "FSR reads publish all components atomically and cannot replace leased consumers" {
    var c = try Cache.init(a, 4, 1);
    defer c.deinit();
    try c.protect(&.{0});
    const t = (try c.reserve(0, false)).?;
    try std.testing.expectEqual(absent, c.map[0]);
    try std.testing.expectError(error.NoEvictableSlot, c.reserve(1, true));
    try c.abort(t);
    try std.testing.expectEqual(absent, c.map[0]);
    const retry = (try c.reserve(0, false)).?;
    try std.testing.expectError(error.StaleRead, c.publish(t));
    try c.publish(retry);
    try std.testing.expectError(error.NoEvictableSlot, c.reserve(1, true));
    try std.testing.expectError(error.DemandStillLeased, c.protect(&.{1}));
    c.release();
    try c.protect(&.{1});
    const next = (try c.reserve(1, false)).?;
    try std.testing.expectEqual(absent, c.map[0]);
    try c.abort(next);
    c.release();
}

test "FSR correct and wrong forecasts use safe capacity and demand joins its own read" {
    var c = try Cache.init(a, 6, 3);
    defer c.deinit();
    try fill(&c, &.{0});
    const hit = (try c.reserve(1, true)).?;
    const wrong = (try c.reserve(2, true)).?;
    try std.testing.expect(try c.reserve(1, true) == null);
    try c.publish(wrong);
    c.release();
    try c.protect(&.{1});
    try std.testing.expect(try c.reserve(1, false) == null);
    try std.testing.expectEqual(hit.generation, c.pending(1).?.generation);
    try c.publish(hit);
    try std.testing.expect(try c.resident(1) != null);
    c.release();
    try fill(&c, &.{ 3, 4, 5 });
    for ([_]usize{ 3, 4, 5 }) |key| try std.testing.expect(c.map[key] >= 0);
    c.release();
}

test "FSR publishes into the borrowed GPU map and never frees its owner" {
    var map: [8]i32 = undefined;
    var c = try Cache.initMap(std.testing.allocator, &map, 2);
    try c.protect(&.{5});
    const ticket = (try c.reserve(5, false)).?;
    try std.testing.expectEqual(absent, map[5]);
    try c.publish(ticket);
    try std.testing.expectEqual(@as(i32, @intCast(ticket.slot)), map[5]);
    c.release();
    c.deinit();
    map[0] = 17;
    try std.testing.expectEqual(@as(i32, 17), map[0]);
}

test "FSR forecast cannot spend demand second chances or promote unused predictions" {
    var c = try Cache.init(a, 8, 3);
    defer c.deinit();
    try fill(&c, &.{ 0, 1, 2 });
    c.release();
    const result = c.reserve(3, true);
    if (result) |ticket| {
        if (ticket) |t| try c.abort(t);
        return error.ForecastEvictedRecentDemand;
    } else |err| try std.testing.expectEqual(error.NoEvictableSlot, err);
    for (c.slots) |s| try std.testing.expect(s.referenced);
    try fill(&c, &.{3});
    c.release();
    const predicted = (try c.reserve(4, true)).?;
    try c.publish(predicted);
    try std.testing.expect(!c.slots[predicted.slot].referenced);
    try std.testing.expect(try c.peek(4) != null);
    try std.testing.expect(!c.slots[predicted.slot].referenced);
    _ = try c.resident(4);
    try std.testing.expect(c.slots[predicted.slot].referenced);
}
