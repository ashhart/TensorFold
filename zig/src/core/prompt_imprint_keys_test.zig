//! Imprint's key index against the linear scans it replaced: random adds, touches, removes, evictions and reloads.
const std = @import("std");
const platform = @import("sys");
const imp = @import("prompt_imprint.zig");
const Imprint = imp.Imprint;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const donate = @import("prompt_imprint_open_test.zig").donate;
const Allocator = std.mem.Allocator;
const expectEqual = std.testing.expectEqual;

const cap: u64 = 1 << 40;
const values = 48; // tokens { v, 7, 9 } for v below this: keys come back, so adds meet held, removed and donated ones

fn scratch(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-imprint-keys-{s}-{d}", .{ tag, platform.c.getpid() });
}

fn tokensOf(v: u32) [3]u32 {
    return .{ v, 7, 9 };
}

const Rec = struct { key: u64, at: u32, bytes: u64, used: u64 = 0 };

fn linear(list: anytype, key: u64) ?usize {
    for (list, 0..) |x, i| if (x.key == key) return i;
    return null;
}

/// The old Imprint, by linear scans: the index file's records, this writer's, and whether another wrote since.
const Model = struct {
    file: std.ArrayList(Rec) = .empty,
    mine: std.ArrayList(Rec) = .empty,
    clock: u64 = 0,
    behind: bool = false,

    fn deinit(o: *Model, a: Allocator) void {
        o.file.deinit(a);
        o.mine.deinit(a);
    }
    /// The index read again: each key's first record, `used` kept for those held, the others stamped in order.
    fn reload(o: *Model, a: Allocator) !void {
        var fresh: std.ArrayList(Rec) = .empty;
        errdefer fresh.deinit(a);
        for (o.file.items) |x| if (linear(fresh.items, x.key) == null) {
            var y = x;
            if (linear(o.mine.items, x.key)) |i| y.used = o.mine.items[i].used else {
                o.clock += 1;
                y.used = o.clock;
            }
            try fresh.append(a, y);
        };
        o.mine.deinit(a);
        o.mine = fresh;
        o.behind = false;
    }
    fn add(o: *Model, a: Allocator, x: Rec) !void {
        if (linear(o.mine.items, x.key) != null) return;
        if (o.behind) try o.reload(a);
        if (linear(o.mine.items, x.key) != null) return;
        o.clock += 1;
        var y = x;
        y.used = o.clock;
        try o.mine.append(a, y);
        try o.file.append(a, y);
    }
    fn touch(o: *Model, key: u64) void {
        const i = linear(o.mine.items, key) orelse return;
        o.clock += 1;
        o.mine.items[i].used = o.clock;
    }
    /// The index rewritten from this writer's records less `key`: another's records since its last read are lost.
    fn remove(o: *Model, a: Allocator, key: u64) !void {
        const i = linear(o.mine.items, key) orelse return;
        _ = o.mine.orderedRemove(i);
        o.file.clearRetainingCapacity();
        try o.file.appendSlice(a, o.mine.items);
        o.behind = false;
    }
    fn victim(o: *const Model) ?u64 {
        var out: ?Rec = null;
        for (o.mine.items) |x| if (out == null or x.used < out.?.used) {
            out = x;
        };
        return if (out) |x| x.key else null;
    }
    fn reopen(o: *Model, a: Allocator) !void {
        o.mine.clearRetainingCapacity();
        o.clock = 0;
        try o.reload(a);
    }
};

/// The same records in the same order, the same clock and victim, and each key found exactly where a scan finds it.
fn expectSame(o: *const Model, m: *const Imprint) !void {
    try expectEqual(o.mine.items.len, m.metas.items.len);
    for (o.mine.items, m.metas.items) |x, y| {
        try expectEqual(x.key, y.key);
        try expectEqual(x.at, y.at);
        try expectEqual(x.bytes, y.bytes);
        try expectEqual(x.used, y.used);
    }
    try expectEqual(o.clock, m.clock);
    try expectEqual(o.victim(), m.victim());
    try expectScans(m);
}

/// `has` agrees with a linear scan of the records for every key the test uses, and for keys it never adds.
fn expectScans(m: *const Imprint) !void {
    for (0..values + 8) |v| {
        const key = Imprint.keyOf(&tokensOf(@intCast(v)));
        try expectEqual(linear(m.metas.items, key) != null, m.has(key));
    }
}

test "the key index keeps the records, order, clock and victims of the linear scans over random operations" {
    const a = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "random");
    defer rmTree(root);
    var o: Model = .{};
    defer o.deinit(a);
    var m = try Imprint.open(a, root, 1, cap);
    defer m.deinit();
    var prng: std.Random.DefaultPrng = .init(0x6b65_7973);
    const r = prng.random();
    var donor: u64 = 100;
    const Op = enum { add, reload, touch, remove, evict, donate, reopen };
    var ran: std.EnumArray(Op, usize) = .initFill(0);
    for (0..700) |_| {
        const t = tokensOf(r.uintLessThan(u32, values));
        const x: Rec = .{ .key = Imprint.keyOf(&t), .at = r.intRangeAtMost(u32, 1, 3), .bytes = r.int(u16) };
        const op: Op = switch (r.uintLessThan(u32, 20)) {
            0...6 => if (o.behind and linear(o.mine.items, x.key) == null) .reload else .add,
            7...9 => .touch,
            10, 11 => .remove,
            12, 13 => .evict,
            14...18 => .donate,
            else => .reopen,
        };
        switch (op) {
            .add, .reload => {
                try o.add(a, x);
                try m.add(x.key, x.at, &t, &.{}, x.bytes);
            },
            .touch => {
                o.touch(x.key);
                m.touch(x.key);
            },
            .remove => {
                try o.remove(a, x.key);
                try m.remove(x.key);
            },
            .evict => if (o.victim()) |key| {
                try expectEqual(key, m.victim().?);
                try o.remove(a, key);
                try m.remove(key);
            },
            .donate => {
                donor += 1;
                try donate(a, root, 1, donor, &t, x.at, &.{}, x.bytes);
                try o.file.append(a, x);
                o.behind = true;
            },
            .reopen => {
                const again = try Imprint.open(a, root, 1, cap);
                m.deinit();
                m = again;
                try o.reopen(a);
            },
        }
        ran.getPtr(op).* += 1;
        try expectSame(&o, &m);
    }
    for (ran.values) |n| try std.testing.expect(n >= 15);
    try std.testing.expect(o.mine.items.len >= 10);
}

test "an add that runs out of memory during its reload leaves records and keys in step; its retry matches" {
    const a = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "oom");
    defer rmTree(root);
    var o: Model = .{};
    defer o.deinit(a);
    var failing: std.testing.FailingAllocator = .init(a, .{});
    var m = try Imprint.open(failing.allocator(), root, 1, cap);
    defer m.deinit();
    for (0..4) |v| {
        const t = tokensOf(@intCast(v));
        const x: Rec = .{ .key = Imprint.keyOf(&t), .at = 2, .bytes = v };
        try o.add(a, x);
        try m.add(x.key, x.at, &t, &.{}, x.bytes);
    }
    for ([_]u32{ 1, 9, 2, 10 }, 100..) |v, donor| { // two keys held, two new: the reload drops one copy of each held
        const t = tokensOf(v);
        try donate(a, root, 1, donor, &t, 3, &.{}, 50 + v);
        try o.file.append(a, .{ .key = Imprint.keyOf(&t), .at = 3, .bytes = 50 + v });
    }
    o.behind = true;
    const t = tokensOf(20);
    const x: Rec = .{ .key = Imprint.keyOf(&t), .at = 1, .bytes = 7 };
    var failed: usize = 0;
    while (true) : (failed += 1) {
        failing.fail_index = failing.alloc_index + failed;
        m.add(x.key, x.at, &t, &.{}, x.bytes) catch |err| {
            try expectEqual(error.OutOfMemory, err);
            try expectScans(&m);
            continue;
        };
        break;
    }
    failing.fail_index = std.math.maxInt(usize);
    try std.testing.expect(failed > 3);
    try o.add(a, x);
    try expectSame(&o, &m);
}
