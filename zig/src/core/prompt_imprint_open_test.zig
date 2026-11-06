//! Imprint.open's results at load: duplicate records, unindexed payloads, records without files, failed allocations.
const std = @import("std");
const platform = @import("sys");
const imp = @import("prompt_imprint.zig");
const Imprint = imp.Imprint;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Allocator = std.mem.Allocator;
const expectEqual = std.testing.expectEqual;

const cap: u64 = 1 << 40;
const suffixes = [_][]const u8{ ".bin", ".r0.bin", ".r1.bin" };

fn scratch(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-imprint-open-{s}-{d}", .{ tag, platform.c.getpid() });
}

fn identityDir(buf: []u8, root: []const u8, identity: u64) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}", .{ root, identity }, 0);
}

/// An empty file `name` in `dir`.
fn put(dir: []const u8, name: []const u8) !void {
    var buf: [600]u8 = undefined;
    const fd = platform.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    _ = platform.c.close(fd);
}

fn present(dir: []const u8, name: []const u8) bool {
    var buf: [600]u8 = undefined;
    const fd = platform.c.open(std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0) catch return false, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return false;
    _ = platform.c.close(fd);
    return true;
}

const Names = struct {
    list: std.ArrayList([]u8) = .empty,

    fn add(n: *Names, a: Allocator, comptime fmt: []const u8, args: anytype) !void {
        const s = try std.fmt.allocPrint(a, fmt, args);
        errdefer a.free(s);
        try n.list.append(a, s);
    }
    fn deinit(n: *Names, a: Allocator) void {
        for (n.list.items) |s| a.free(s);
        n.list.deinit(a);
    }
    fn lessThan(_: void, x: []u8, y: []u8) bool {
        return std.mem.order(u8, x, y) == .lt;
    }
    /// Sorted, each name once.
    fn settle(n: *Names, a: Allocator) void {
        std.mem.sort([]u8, n.list.items, {}, lessThan);
        var out: usize = 0;
        for (n.list.items) |s| {
            if (out > 0 and std.mem.eql(u8, n.list.items[out - 1], s)) a.free(s) else {
                n.list.items[out] = s;
                out += 1;
            }
        }
        n.list.shrinkRetainingCapacity(out);
    }
};

/// The regular files in `dir` are exactly `want`.
fn expectFiles(a: Allocator, dir: [:0]const u8, want: *Names) !void {
    var got: Names = .{};
    defer got.deinit(a);
    const d = platform.c.opendir(dir) orelse return error.TestDir;
    defer _ = platform.c.closedir(d);
    while (platform.c.readdir(d)) |ent| if (ent.type == platform.c.DT.REG) try got.add(a, "{s}", .{std.mem.sliceTo(&ent.name, 0)});
    got.settle(a);
    want.settle(a);
    try expectEqual(want.list.items.len, got.list.items.len);
    for (want.list.items, got.list.items) |w, g| try std.testing.expectEqualStrings(w, g);
}

/// Identity `from`'s whole index appended to identity `into`'s, as a second writer's records would be.
fn appendIndex(a: Allocator, root: []const u8, into: u64, from: u64) !void {
    var buf: [600]u8 = undefined;
    const src = platform.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}/index", .{ root, from }, 0), .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (src < 0) return error.TestFile;
    defer _ = platform.c.close(src);
    const end = platform.c.lseek(src, 0, platform.c.SEEK.END);
    if (end <= 0) return error.TestFile;
    const bytes = try a.alloc(u8, @intCast(end));
    defer a.free(bytes);
    if (!imp.readAt(src, bytes, 0)) return error.TestFile;
    const dst = platform.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}/index", .{ root, into }, 0), .{ .ACCMODE = .WRONLY, .APPEND = true }, @as(platform.c.mode_t, 0));
    if (dst < 0) return error.TestFile;
    defer _ = platform.c.close(dst);
    try imp.writeAll(dst, bytes);
}

/// One record learned alone under identity `from`, then appended to identity `into`'s index.
pub fn donate(a: Allocator, root: []const u8, into: u64, from: u64, tokens: []const u32, at: u32, starts: []const u32, bytes: u64) !void {
    {
        var m = try Imprint.open(a, root, from, cap);
        defer m.deinit();
        try m.add(Imprint.keyOf(tokens), at, tokens, starts, bytes);
    }
    try appendIndex(a, root, into, from);
}

test "a duplicate record keeps its first copy, in index order, and the index is read whole" {
    const a = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "dupes");
    defer rmTree(root);
    const t = [_][4]u32{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 }, .{ 9, 10, 11, 12 }, .{ 13, 14, 15, 16 } };
    {
        var m = try Imprint.open(a, root, 1, cap);
        defer m.deinit();
        try m.add(Imprint.keyOf(&t[0]), 3, &t[0], &.{1}, 10);
        try m.add(Imprint.keyOf(&t[1]), 2, &t[1], &.{}, 20);
        try m.add(Imprint.keyOf(&t[2]), 1, &t[2], &.{}, 30);
    }
    try donate(a, root, 1, 2, &t[0], 1, &.{}, 99); // the first key again, at another position with other bytes
    try donate(a, root, 1, 3, &t[3], 2, &.{1}, 40);
    try donate(a, root, 1, 4, &t[1], 3, &.{ 1, 2 }, 77);
    var m = try Imprint.open(a, root, 1, cap);
    defer m.deinit();
    const Want = struct { t: usize, at: u32, bytes: u64, starts: []const u32 };
    const want = [_]Want{ .{ .t = 0, .at = 3, .bytes = 10, .starts = &.{1} }, .{ .t = 1, .at = 2, .bytes = 20, .starts = &.{} }, .{ .t = 2, .at = 1, .bytes = 30, .starts = &.{} }, .{ .t = 3, .at = 2, .bytes = 40, .starts = &.{1} } };
    try expectEqual(want.len, m.metas.items.len);
    for (want, m.metas.items, 1..) |w, x, used| {
        try expectEqual(Imprint.keyOf(&t[w.t]), x.key);
        try expectEqual(w.at, x.at);
        try expectEqual(w.bytes, x.bytes);
        try std.testing.expectEqualSlices(u32, &t[w.t], x.tokens);
        try std.testing.expectEqualSlices(u32, w.starts, x.starts);
        try expectEqual(@as(u64, used), x.used);
    }
    try expectEqual(@as(u64, 4), m.clock);
    try expectEqual(@as(u64, 100), m.total());
    try expectEqual(try m.indexScratchBytes(), m.index_end); // duplicates are read past, not cut off as torn
    const end = m.index_end;
    try m.add(Imprint.keyOf(&t[0]), 2, &t[0], &.{}, 5);
    try expectEqual(end, try m.indexScratchBytes());
}

test "reopen keeps all three suffixes of an indexed key and other files, and removes an unindexed key's" {
    const a = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "suffixes");
    defer rmTree(root);
    const tokens = [_]u32{ 3, 1, 4, 1, 5 };
    const kept = Imprint.keyOf(&tokens);
    const lost: u64 = 0x00c0ffee12345678;
    var db: [256]u8 = undefined;
    const dir = try identityDir(&db, root, 1);
    {
        var m = try Imprint.open(a, root, 1, cap);
        defer m.deinit();
        try m.add(kept, 4, &tokens, &.{}, 64);
    }
    var nb: [64]u8 = undefined;
    for (suffixes) |s| {
        try put(dir, try std.fmt.bufPrint(&nb, "{x:0>16}{s}", .{ kept, s }));
        try put(dir, try std.fmt.bufPrint(&nb, "{x:0>16}{s}", .{ lost, s }));
    }
    try put(dir, "notes.txt");
    var m = try Imprint.open(a, root, 1, cap);
    defer m.deinit();
    try expectEqual(@as(usize, 1), m.metas.items.len);
    try expectEqual(kept, m.metas.items[0].key);
    var want: Names = .{};
    defer want.deinit(a);
    for (suffixes) |s| try want.add(a, "{x:0>16}{s}", .{ kept, s });
    for ([_][]const u8{ "index", "used", "notes.txt" }) |s| try want.add(a, "{s}", .{s});
    try expectFiles(a, dir, &want);
}

test "records whose files are missing all load, and reopening creates and removes nothing" {
    const a = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "missing");
    defer rmTree(root);
    const t = [_][3]u32{ .{ 1, 2, 3 }, .{ 4, 5, 6 }, .{ 7, 8, 9 } };
    var db: [256]u8 = undefined;
    const dir = try identityDir(&db, root, 1);
    {
        var m = try Imprint.open(a, root, 1, cap);
        defer m.deinit();
        for (&t, 1..) |*x, i| try m.add(Imprint.keyOf(x), 2, x, &.{}, i);
    }
    var nb: [64]u8 = undefined;
    try put(dir, try std.fmt.bufPrint(&nb, "{x:0>16}.r1.bin", .{Imprint.keyOf(&t[1])}));
    var m = try Imprint.open(a, root, 1, cap);
    defer m.deinit();
    try expectEqual(t.len, m.metas.items.len);
    for (&t, m.metas.items, 1..) |*x, y, i| {
        try expectEqual(Imprint.keyOf(x), y.key);
        try expectEqual(@as(u64, i), y.bytes);
        try expectEqual(@as(u64, i), y.used);
    }
    var want: Names = .{};
    defer want.deinit(a);
    try want.add(a, "{x:0>16}.r1.bin", .{Imprint.keyOf(&t[1])});
    for ([_][]const u8{ "index", "used" }) |s| try want.add(a, "{s}", .{s});
    try expectFiles(a, dir, &want);
}

test "an open that runs out of memory removes no unindexed file" {
    const a = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "oom");
    defer rmTree(root);
    var db: [256]u8 = undefined;
    const dir = try identityDir(&db, root, 1);
    {
        var m = try Imprint.open(a, root, 1, cap);
        defer m.deinit();
        for (0..3) |i| {
            const x = [_]u32{ @intCast(i), 2 };
            try m.add(Imprint.keyOf(&x), 1, &x, &.{}, 8);
        }
    }
    const lost = [_][]const u8{ "00000000000000a1.bin", "00000000000000a2.r0.bin", "00000000000000a3.r1.bin" };
    var fail: usize = 0;
    while (true) : (fail += 1) {
        for (lost) |name| try put(dir, name);
        var failing: std.testing.FailingAllocator = .init(a, .{ .fail_index = fail });
        var m = Imprint.open(failing.allocator(), root, 1, cap) catch |err| {
            try expectEqual(error.OutOfMemory, err);
            for (lost) |name| try std.testing.expect(present(dir, name));
            continue;
        };
        m.deinit();
        for (lost) |name| try std.testing.expect(!present(dir, name));
        break;
    }
    try std.testing.expect(fail > 3);
}

/// A file name's state key under the old rule: 16 lowercase hex digits and one of the three payload suffixes.
fn oldKey(name: []const u8) ?u64 {
    if (name.len < 16) return null;
    for (name[0..16]) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return null;
    for (suffixes) |s| if (std.mem.eql(u8, name[16..], s)) return std.fmt.parseInt(u64, name[0..16], 16) catch null;
    return null;
}

test "reopen keeps the records and files the old linear scans kept, over a random index and directory" {
    const a = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "oracle");
    defer rmTree(root);
    var db: [256]u8 = undefined;
    const dir = try identityDir(&db, root, 1);
    var prng: std.Random.DefaultPrng = .init(0x1e47_0b5e);
    const r = prng.random();
    const Rec = struct { key: u64, at: u32, bytes: u64 };
    var log: std.ArrayList(Rec) = .empty; // every record in the index, in its order
    defer log.deinit(a);
    for (0..5) |w| {
        var m = try Imprint.open(a, root, 1 + w, cap);
        defer m.deinit();
        for (0..if (w == 0) 120 else 60) |_| {
            const x = [_]u32{ r.uintLessThan(u32, 200), 7, 9 };
            const rec: Rec = .{ .key = Imprint.keyOf(&x), .at = r.intRangeAtMost(u32, 1, 3), .bytes = r.int(u16) };
            if (m.has(rec.key)) continue;
            try m.add(rec.key, rec.at, &x, &.{}, rec.bytes);
            try log.append(a, rec);
        }
    }
    for (2..6) |w| try appendIndex(a, root, 1, w);
    var kept: std.ArrayList(Rec) = .empty; // the old duplicate check: a linear scan of the records so far
    defer kept.deinit(a);
    for (log.items) |x| {
        const seen = for (kept.items) |y| {
            if (y.key == x.key) break true;
        } else false;
        if (!seen) try kept.append(a, x);
    }
    try std.testing.expect(kept.items.len < log.items.len);
    var made: Names = .{};
    defer made.deinit(a);
    const forms = [_][]const u8{ ".bin", ".r0.bin", ".r1.bin", ".r2.bin", ".bin.x", "", ".BIN" };
    for (0..400) |_| {
        const key = switch (r.uintLessThan(u32, 4)) {
            0, 1 => log.items[r.uintLessThan(usize, log.items.len)].key,
            2 => r.int(u64),
            else => Imprint.keyOf(&[_]u32{ 200 + r.uintLessThan(u32, 50), 7, 9 }),
        };
        var nb: [64]u8 = undefined;
        const name = try switch (r.uintLessThan(u32, 10)) {
            0 => std.fmt.bufPrint(&nb, "{X:0>16}.bin", .{key}),
            1 => std.fmt.bufPrint(&nb, "{x:0>15}.bin", .{key >> 4}),
            else => std.fmt.bufPrint(&nb, "{x:0>16}{s}", .{ key, forms[r.uintLessThan(usize, forms.len)] }),
        };
        const clash = for (made.list.items) |s| { // a case-insensitive volume would make these one file
            if (std.ascii.eqlIgnoreCase(s, name)) break true;
        } else false;
        if (clash) continue;
        try made.add(a, "{s}", .{name});
        try put(dir, name);
    }
    var want: Names = .{}; // the old sweep: a state name survives when a linear scan of the records finds its key
    defer want.deinit(a);
    var counts: [3]usize = @splat(0); // names that aren't states', indexed states', unindexed states'
    for (made.list.items) |name| {
        const indexed = if (oldKey(name)) |key| for (kept.items) |y| {
            if (y.key == key) break true;
        } else false else null;
        counts[if (indexed) |i| @as(usize, if (i) 1 else 2) else 0] += 1;
        if (indexed orelse true) try want.add(a, "{s}", .{name});
    }
    for (counts) |n| try std.testing.expect(n >= 20);
    for ([_][]const u8{ "index", "used" }) |s| try want.add(a, "{s}", .{s});
    var m = try Imprint.open(a, root, 1, cap);
    defer m.deinit();
    try expectEqual(kept.items.len, m.metas.items.len);
    for (kept.items, m.metas.items, 1..) |x, y, used| {
        try expectEqual(x.key, y.key);
        try expectEqual(x.at, y.at);
        try expectEqual(x.bytes, y.bytes);
        try expectEqual(@as(u64, used), y.used);
    }
    try expectFiles(a, dir, &want);
}
