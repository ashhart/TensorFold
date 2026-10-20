//! prefix.Index under allocation failure: an insert that fails changes nothing, and the same insert then succeeds.

const std = @import("std");
const prefix = @import("prefix.zig");
const testing = std.testing;
const T = prefix.Index(u64);
const Digest = prefix.Digest;
const page = 16;

/// What a failed insert must leave as it was; spare token blocks may stay in the reuse pool.
const Shape = struct { keys: u32, digests: u32, ram: u32, held: u32, sub_all: u32, sub_ram: u32 };

fn shape(x: *T, tag: u32) Shape {
    var s: Shape = .{ .keys = x.by_key.count(), .digests = x.by_digest.count(), .ram = 0, .held = 0, .sub_all = 0, .sub_ram = 0 };
    var e = x.by_key.valueIterator();
    while (e.next()) |id| s.ram += @intFromBool(x.get(id.*).ram);
    var n = x.by_digest.valueIterator();
    while (n.next()) |id| s.held += @intFromBool(x.node(id.*).tokens != null);
    if (x.by_digest.get(prefix.root(tag))) |r| {
        s.sub_all = x.node(r).sub_all;
        s.sub_ram = x.node(r).sub_ram;
    }
    return s;
}

/// Every slot is live in its map or on its free list, and a node keeps its page's tokens exactly while sub_ram > 0.
fn consistent(x: *T) !void {
    try testing.expectEqual(x.entries.items.len - x.free_entries.items.len, x.by_key.count());
    try testing.expectEqual(x.nodes.items.len - x.free_nodes.items.len, x.by_digest.count());
    var n = x.by_digest.valueIterator();
    while (n.next()) |id| {
        const nd = x.node(id.*);
        if (nd.depth > 0) try testing.expectEqual(nd.sub_ram > 0, nd.tokens != null);
    }
}

fn seq(gpa: std.mem.Allocator, n: usize, salt: i32) ![]i32 {
    const v = try gpa.alloc(i32, n);
    for (v, 0..) |*t, i| t.* = @as(i32, @intCast(i % 997)) * 3 + salt;
    return v;
}

fn add(x: *T, tag: u32, ids: []const i32, ram: bool, used: u64) !u32 {
    var d: [64]Digest = undefined;
    prefix.chain(tag, ids, page, d[0 .. ids.len / page]);
    return (try x.insert(tag, ids, d[0 .. ids.len / page], ram, used)).id;
}

/// The (k + 1)-th allocation from now fails, and every one after it until `heal`.
fn failAfter(fa: *testing.FailingAllocator, k: usize) void {
    fa.fail_index = fa.alloc_index + k;
}

fn heal(fa: *testing.FailingAllocator) void {
    fa.fail_index = std.math.maxInt(usize);
}

test "a RAM insert failing at any allocation leaves the index as it was, and the same insert then succeeds" {
    const chat = try seq(testing.allocator, 160, 0);
    defer testing.allocator.free(chat);
    var d: [16]Digest = undefined;
    prefix.chain(7, chat[0..150], page, d[0..9]);
    var k: usize = 0;
    while (k < 256) : (k += 1) {
        var fa = testing.FailingAllocator.init(testing.allocator, .{});
        var x = T.init(fa.allocator(), page);
        defer x.deinit();
        // what the insert meets: entries sharing its first pages, a disk entry, another tag, and free slots to reuse
        _ = try add(&x, 7, chat[0..40], true, 1);
        _ = try add(&x, 7, chat[0..72], false, 2);
        _ = try add(&x, 8, chat[0..33], true, 3);
        x.remove(try add(&x, 9, chat[0..48], true, 4));
        try consistent(&x);
        const before = shape(&x, 7);
        failAfter(&fa, k);
        const got = x.insert(7, chat[0..150], d[0..9], true, 9) catch |err| {
            try testing.expectEqual(error.OutOfMemory, err);
            try consistent(&x);
            try testing.expectEqual(before, shape(&x, 7));
            heal(&fa);
            const again = try x.insert(7, chat[0..150], d[0..9], true, 9);
            try testing.expect(again.new);
            try consistent(&x);
            var out: [150]i32 = undefined;
            try x.idsOf(again.id, &out);
            try testing.expectEqualSlices(i32, chat[0..150], &out);
            var d10: [10]Digest = undefined;
            prefix.chain(7, chat[0..160], page, &d10);
            try testing.expectEqual(@as(?u32, again.id), x.find(7, chat[0..160], &d10, .ram));
            x.remove(again.id);
            try consistent(&x);
            try testing.expectEqual(before, shape(&x, 7));
            continue;
        };
        // no allocation left to fail: every failure point before this one was injected and rolled back
        try testing.expect(got.new and k > 0);
        return;
    }
    return error.TestUnexpectedResult;
}

test "a disk insert into an empty index failing at any allocation leaves it empty, and the same insert then succeeds" {
    const chat = try seq(testing.allocator, 90, 0);
    defer testing.allocator.free(chat);
    var k: usize = 0;
    while (k < 256) : (k += 1) {
        var fa = testing.FailingAllocator.init(testing.allocator, .{});
        var x = T.init(fa.allocator(), page);
        defer x.deinit();
        failAfter(&fa, k);
        const got = x.insertFrom(7, prefix.none, chat, 5) catch |err| {
            try testing.expectEqual(error.OutOfMemory, err);
            try consistent(&x);
            try testing.expectEqual(@as(u32, 0), x.by_digest.count()); // the tag's root and pages made before the failure
            try testing.expectEqual(@as(u32, 0), x.by_key.count());
            heal(&fa);
            const again = try x.insertFrom(7, prefix.none, chat, 5);
            try testing.expect(again.new);
            try testing.expectEqual(@as(u32, 1 + 5), x.by_digest.count());
            try consistent(&x);
            continue;
        };
        try testing.expect(got.new and k > 0);
        return;
    }
    return error.TestUnexpectedResult;
}
