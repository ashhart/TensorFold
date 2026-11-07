//! Owner classes (DeepSeek's residue book): one class is today's pool id for id, and a lease's ids share its class.
const std = @import("std");
const kv = @import("kv_pool.zig");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;
const t = std.testing.allocator;

/// 40 scripted steps on 24 pages, traced: the ids each leaves in a table or hold, each census, then the free stack.
fn script(a: std.mem.Allocator, out: *std.ArrayList(u32)) !void {
    var p = try kv.Pool.init(a, 24, 4096);
    defer p.deinit();
    var ls: [4]kv.Lease = @splat(.{});
    defer for (&ls) |*l| p.finish(l);
    var held: std.ArrayList(u32) = .empty;
    defer held.deinit(a);
    var prng = std.Random.DefaultPrng.init(0x1431);
    const r = prng.random();
    for (0..40) |_| {
        const l = &ls[r.uintLessThan(usize, ls.len)];
        switch (r.uintLessThan(u8, 6)) {
            0, 1 => {
                const n = 1 + r.uintLessThan(usize, 5);
                p.promise(l, n) catch {};
                p.cover(l, l.ids.items.len + @min(n, l.promised), null) catch {};
            },
            2 => p.trim(l, l.ids.items.len / 2),
            3 => p.finish(l),
            4 => {
                var two: [2]u32 = undefined;
                if (p.hold(&two)) try held.appendSlice(a, &two);
            },
            else => if (held.pop()) |id| p.release(id),
        }
        for (&ls) |*x| try out.appendSlice(a, x.ids.items);
        try out.appendSlice(a, held.items);
        const c = p.census();
        try out.appendSlice(a, &.{ @intCast(c.free), @intCast(c.promised), @intCast(c.kept), @intCast(c.leased) });
    }
    for (held.items) |id| p.release(id);
    try out.appendSlice(a, p.free[0..p.nfree]);
}

test "owner classes: one class is today's pool, id for id and in the same order (the trace main records)" {
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(t);
    try script(t, &out); // every family's pool (Pool.init): Nemotron, Qwen3.5, Dory, Gemma 4, Metal, the prompt store
    try equal(@as(usize, 738), out.items.len);
    try std.testing.expectEqualSlices(u32, &.{ 0, 23, 0, 0, 1, 0, 1, 2, 3, 20, 0, 0, 4, 0, 1, 22, 2, 0, 2, 1, 23, 2, 0, 1 }, out.items[0..24]);
    try equal(@as(u64, 0x6a51e5e6ca5c61f6), std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(out.items)));
    var one = try kv.Pool.initClasses(t, 24, 4096, 1);
    defer one.deinit();
    var base = try kv.Pool.init(t, 24, 4096);
    defer base.deinit();
    try std.testing.expectEqualSlices(u32, base.free, one.free);
}

test "owner classes: each class's stack holds its residue, lowest id on top, and the pool refuses a bad split" {
    var p = try kv.Pool.initClasses(t, 12, 64, 3);
    defer p.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 9, 6, 3, 0, 10, 7, 4, 1, 11, 8, 5, 2 }, p.free);
    try equal(@as(usize, 4), p.classCap());
    for (0..3) |c| try equal(@as(usize, 4), p.availableIn(@intCast(c)));
    try std.testing.expectError(error.BadClasses, kv.Pool.initClasses(t, 13, 64, 3)); // not a multiple of the classes
    try std.testing.expectError(error.BadClasses, kv.Pool.initClasses(t, 12, 64, 0));
    try std.testing.expectError(error.BadClasses, kv.Pool.initClasses(t, 34, 64, kv.max_classes + 1));
    var most = try kv.Pool.initClasses(t, 32, 64, kv.max_classes);
    most.deinit();
}

test "owner classes: a lease takes every id from its owner's class, through cover, trim, finish and reuse" {
    for ([_]u32{ 2, 3 }) |w| {
        var p = try kv.Pool.initClasses(t, 6 * w, 64, w);
        defer p.deinit();
        var ls: [3]kv.Lease = @splat(.{});
        for (0..4) |round| {
            for (&ls, 0..) |*l, i| {
                l.owner = @intCast((i + round) % w);
                try p.promise(l, 2);
                try p.cover(l, 2, null);
            }
            for (&ls) |*l| {
                p.trim(l, 1);
                try p.cover(l, 2, null);
                for (l.ids.items) |id| try equal(l.owner, id % w);
            }
            for (&ls) |*l| p.finish(l);
            try equal(@as(u32, 0), ls[0].owner); // finish resets the lease, owner included
            for (0..w) |c| try equal(@as(usize, 6), p.classFree(@intCast(c)));
        }
        try expect(p.idle());
    }
}

test "owner classes: a full class refuses a promise while another class still has room" {
    var p = try kv.Pool.initClasses(t, 8, 64, 2);
    defer p.deinit();
    var a: kv.Lease = .{ .owner = 1 };
    var b: kv.Lease = .{ .owner = 1 };
    defer p.finish(&a);
    defer p.finish(&b);
    try p.promise(&a, 3);
    try std.testing.expectError(error.LaneMemory, p.promise(&b, 2)); // class 1 has 1 left, the pool 5
    try equal(@as(usize, 5), p.available());
    try equal(@as(usize, 3), p.classPromised(1));
    b.owner = 0;
    try p.promise(&b, 4);
    try equal(@as(usize, 0), p.availableIn(0));
    try p.cover(&a, 3, null);
    try equal(@as(usize, 1), p.classFree(1));
}

test "owner classes: the owner rule is the class with the most available pages, the lowest on ties" {
    var p = try kv.Pool.initClasses(t, 12, 64, 3);
    defer p.deinit();
    try equal(@as(u32, 0), p.roomiest()); // all tie
    var a: kv.Lease = .{ .owner = 0 };
    defer p.finish(&a);
    try p.promise(&a, 1);
    try equal(@as(u32, 1), p.roomiest()); // 1 and 2 tie
    var b: kv.Lease = .{ .owner = 1 };
    defer p.finish(&b);
    try p.promise(&b, 2);
    try equal(@as(u32, 2), p.roomiest());
    var c: kv.Lease = .{ .owner = 2 };
    defer p.finish(&c);
    try p.promise(&c, 3);
    try equal(@as(u32, 0), p.roomiest()); // 3, 2 and 1 available
    try p.cover(&a, 1, null);
    p.unpromise(&b);
    try equal(@as(u32, 1), p.roomiest()); // 3 against 4: promised and taken count alike
}

test "owner classes: a run spans classes, so runs are refused above one and cover takes the class's ids" {
    var p = try kv.Pool.initClasses(t, 16, 64, 2);
    defer p.deinit();
    var l: kv.Lease = .{ .owner = 1 };
    defer p.finish(&l);
    try p.promise(&l, 6);
    try equal(@as(?u32, null), p.claimRun(&l, 2));
    try p.coverRun(&l, 4, 4, null); // scattered, as cover
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5, 7 }, l.ids.items);
}

test "owner classes: planClasses is plan at one class, floors the pages to the classes and refuses a short class" {
    const l: kv.Layout = .{ .fixed_bytes = 1 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 };
    var two = l;
    two.classes = 2;
    const page = l.pageBytes();
    for ([_]u64{ 0, 1, 7, 513 }) |extra| { // the room of 1,029 pages at 128k x 4 (257 a window), and around it
        const b: kv.Budget = .{ .room = 1029 * page + (4 << 20) + extra * page, .engine = 0, .lanes = 4, .window = 257 * 512, .engine_paged = true };
        const one = try kv.plan(l, b);
        try equal(one, try kv.planClasses(l, b));
        const split = try kv.planClasses(two, b);
        try equal(one.pages - one.pages % 2, split.pages);
        try equal(one.lanes, split.lanes);
    }
    const b: kv.Budget = .{ .room = 1029 * page + (4 << 20), .engine = 0, .lanes = 4, .window = 257 * 512, .engine_paged = true };
    try equal(@as(usize, 1028), (try kv.planClasses(two, b)).pages); // 514 a class, two windows each
    const wide: kv.Budget = .{ .room = 2057 * page + (8 << 20), .engine = 0, .lanes = 8, .window = 2049 * 512, .engine_paged = true };
    try equal(@as(usize, 2057), (try kv.plan(l, wide)).pages);
    try std.testing.expectError(error.ClassTooSmall, kv.planClasses(two, wide)); // 1,028 a class, 2,049 a window
    try equal(@as(usize, 2057), (try kv.planClasses(l, wide)).pages);
}

/// Each class's counts are a scan's: its free ids on the stack, its leases' promises; and every lease id its owner's.
fn scanned(p: *const kv.Pool, ls: []const kv.Lease) !void {
    for (0..p.classes) |c| {
        var free: usize = 0;
        for (p.refs, 0..) |r, id| free += @intFromBool(r == 0 and id % p.classes == c);
        try equal(free, p.classFree(@intCast(c)));
        for (p.free[c * p.classCap() ..][0..free]) |id| try equal(c, id % p.classes);
        var promised: usize = 0;
        for (ls) |l| promised += if (l.owner == c) l.promised else 0;
        try equal(promised, p.classPromised(@intCast(c)));
    }
    for (ls) |l| for (l.ids.items) |id| try equal(l.owner, id % p.classes);
}

test "owner classes: seeded walks keep each class's counts a scan's, and fitsAny answers as a promise would" {
    for ([_]u32{ 1, 2, 3 }) |w| for (0..24) |seed| {
        var prng = std.Random.DefaultPrng.init(0x1431_0000 + seed);
        const r = prng.random();
        var p = try kv.Pool.initClasses(t, w * (4 + seed % 9), 64, w);
        defer p.deinit();
        var ls: [5]kv.Lease = @splat(.{});
        defer for (&ls) |*l| p.finish(l);
        for (0..300) |_| {
            const l = &ls[r.uintLessThan(usize, ls.len)];
            switch (r.uintLessThan(u8, 5)) {
                0, 1 => if (l.ids.items.len == 0 and l.promised == 0) {
                    l.owner = p.roomiest();
                    const n = 1 + r.uintLessThan(usize, p.classCap());
                    if (p.fitsAny(n)) try p.promise(l, n) else try std.testing.expectError(error.LaneMemory, p.promise(l, n));
                },
                2 => try p.cover(l, l.ids.items.len + r.uintLessThan(usize, l.promised + 1), null),
                3 => p.trim(l, r.uintLessThan(usize, l.ids.items.len + 1)),
                else => p.finish(l),
            }
            try scanned(&p, &ls);
            for (0..p.total() + 2) |need| {
                var x: kv.Lease = .{ .owner = p.roomiest() };
                const fit = p.fitsAny(need);
                if (p.promise(&x, need)) |_| try expect(fit) else |_| try expect(!fit);
                p.unpromise(&x);
            }
        }
    };
}
