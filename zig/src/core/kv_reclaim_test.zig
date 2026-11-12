//! The reclaim chain (#2444 C7): stores asked in join order, the walk stopping once enough pages are free.
const std = @import("std");
const kv = @import("kv_pool.zig");
const most = @import("kv_reclaim.zig").most;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualSlices = std.testing.expectEqualSlices;

/// A store holding pages of `pool`; asked, it frees up to `step` of them and logs its name and the want it saw.
const Store = struct {
    pool: *kv.Pool,
    name: u8,
    step: usize,
    log: *Log,
    kept: std.ArrayList(u32) = .empty,

    fn fill(s: *Store, n: usize) !void {
        const ids = try s.kept.addManyAsSlice(std.testing.allocator, n);
        try expect(s.pool.hold(ids));
    }

    fn call(ctx: *anyopaque, want: usize) bool {
        const s: *Store = @ptrCast(@alignCast(ctx));
        s.log.add(s.name, want);
        for (0..@min(s.step, s.kept.items.len)) |_| s.pool.release(s.kept.pop().?);
        return s.pool.available() >= want;
    }

    fn hook(s: *Store) kv.Reclaim {
        return .{ .ctx = s, .call = call };
    }

    fn deinit(s: *Store) void {
        for (s.kept.items) |id| s.pool.release(id);
        s.kept.deinit(std.testing.allocator);
    }
};

const Log = struct {
    names: [16]u8 = undefined,
    wants: [16]usize = undefined,
    n: usize = 0,
    fn add(l: *Log, name: u8, want: usize) void {
        l.names[l.n] = name;
        l.wants[l.n] = want;
        l.n += 1;
    }
};

test "a promise asks the stores in the order they joined, each with the pages still wanted" {
    var p = try kv.Pool.init(std.testing.allocator, 6, 1);
    defer p.deinit();
    var log: Log = .{};
    var a: Store = .{ .pool = &p, .name = 'a', .step = 1, .log = &log };
    var b: Store = .{ .pool = &p, .name = 'b', .step = 1, .log = &log };
    var c: Store = .{ .pool = &p, .name = 'c', .step = 2, .log = &log };
    defer for ([_]*Store{ &a, &b, &c }) |s| s.deinit();
    try a.fill(2);
    try b.fill(2);
    try c.fill(2);
    try p.joinReclaim(c.hook());
    try p.joinReclaim(a.hook());
    try p.joinReclaim(b.hook());
    var l: kv.Lease = .{};
    defer p.finish(&l);
    try p.promise(&l, 4); // c frees 2, a 1, b 1: all three asked, in join order
    try expectEqualSlices(u8, "cab", log.names[0..log.n]);
    try expectEqualSlices(usize, &.{ 4, 4, 4 }, log.wants[0..log.n]);
}

test "the walk stops once enough pages are free: a later store is never asked" {
    var p = try kv.Pool.init(std.testing.allocator, 4, 1);
    defer p.deinit();
    var log: Log = .{};
    var a: Store = .{ .pool = &p, .name = 'a', .step = 2, .log = &log };
    var b: Store = .{ .pool = &p, .name = 'b', .step = 2, .log = &log };
    defer for ([_]*Store{ &a, &b }) |s| s.deinit();
    try a.fill(2);
    try b.fill(2);
    try p.joinReclaim(a.hook());
    try p.joinReclaim(b.hook());
    var ids: [2]u32 = undefined;
    try expect(p.hold(&ids));
    defer for (ids) |id| p.release(id);
    try expectEqualSlices(u8, "a", log.names[0..log.n]);
    try expectEqual(@as(usize, 2), b.kept.items.len); // b kept its pages
}

test "several stores on one pool: leave keeps the others' order, a store joins once, the chain holds `most`" {
    var p = try kv.Pool.init(std.testing.allocator, 8, 1);
    defer p.deinit();
    var log: Log = .{};
    var s: [most]Store = undefined;
    for (&s, 0..) |*x, i| x.* = .{ .pool = &p, .name = @intCast('a' + i), .step = 1, .log = &log };
    defer for (&s) |*x| x.deinit();
    for (&s) |*x| try x.fill(2);
    for (&s) |*x| try p.joinReclaim(x.hook());
    try std.testing.expectError(error.Joined, p.joinReclaim(s[1].hook()));
    var extra: u8 = 0;
    try std.testing.expectError(error.ChainFull, p.joinReclaim(.{ .ctx = &extra, .call = Store.call }));
    p.leaveReclaim(&s[1]);
    p.leaveReclaim(&extra); // never joined: nothing changes
    var l: kv.Lease = .{};
    defer p.finish(&l);
    try p.promise(&l, 3);
    try expectEqualSlices(u8, "acd", log.names[0..log.n]);
    try p.joinReclaim(s[1].hook()); // it joins again, last
    try p.promise(&l, 3);
    try expectEqualSlices(u8, "acdacd", log.names[0..log.n]);
}

test "one store is asked as the single hook was: once with the want, and a promise that still does not fit fails" {
    var p = try kv.Pool.init(std.testing.allocator, 4, 1);
    defer p.deinit();
    var log: Log = .{};
    var a: Store = .{ .pool = &p, .name = 'a', .step = 1, .log = &log };
    defer a.deinit();
    try a.fill(3);
    try p.joinReclaim(a.hook());
    var l: kv.Lease = .{};
    defer p.finish(&l);
    try p.promise(&l, 1); // one page free: nothing asked
    try expectEqual(@as(usize, 0), log.n);
    try std.testing.expectError(error.LaneMemory, p.promise(&l, 3)); // asked once for 3, frees 1, still short
    try expectEqualSlices(u8, "a", log.names[0..log.n]);
    try expectEqualSlices(usize, &.{3}, log.wants[0..log.n]);
    try p.promise(&l, 1);
    try expectEqual(@as(usize, 1), log.n); // it fit without the store
}
