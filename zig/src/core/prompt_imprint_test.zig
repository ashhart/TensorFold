//! Learned prompt states on disk: reopen, eviction order, removal and the sweep of other identities.
const std = @import("std");
const imp = @import("prompt_imprint.zig");
const Imprint = imp.Imprint;

/// `root`, its identity directories and their files removed (a test's scratch tree, two levels deep).
pub fn rmTree(root: []const u8) void {
    var buf: [512]u8 = undefined;
    const r = std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0) catch return;
    const d = std.c.opendir(r) orelse return;
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name[0] == '.') continue;
        var sub: [512]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&sub, "{s}/{s}", .{ root, name }, 0) catch continue;
        if (std.c.unlink(p) == 0) continue;
        if (std.c.opendir(p)) |s| {
            while (std.c.readdir(s)) |f| {
                const fname = std.mem.sliceTo(&f.name, 0);
                if (fname[0] == '.') continue;
                var fb: [600]u8 = undefined;
                _ = std.c.unlink(std.fmt.bufPrintSentinel(&fb, "{s}/{s}", .{ p, fname }, 0) catch continue);
            }
            _ = std.c.closedir(s);
        }
        _ = std.c.rmdir(p);
    }
    _ = std.c.closedir(d);
    _ = std.c.rmdir(r);
}

fn scratch(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-imprint-{s}-{d}", .{ tag, std.c.getpid() });
}

/// A file of `n` bytes at `dir`/`name`.
fn fill(dir: []const u8, name: []const u8, n: usize) !void {
    var buf: [512]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = std.c.close(fd);
    var zeros: [256]u8 = @splat(0);
    var left = n;
    while (left > 0) : (left -= @min(left, zeros.len)) try imp.writeAll(fd, zeros[0..@min(left, zeros.len)]);
}

fn exists(path: []const u8) bool {
    var buf: [512]u8 = undefined;
    const d = std.c.opendir(std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0) catch return false) orelse return false;
    _ = std.c.closedir(d);
    return true;
}

test "learned states survive a reopen; the longest usable prefix wins" {
    const gpa = std.testing.allocator;
    var tmp_buf: [256]u8 = undefined;
    const root = try scratch(&tmp_buf, "reopen");
    defer rmTree(root);
    {
        var m = try Imprint.open(gpa, root, 7, 1 << 30);
        defer m.deinit();
        try m.add(Imprint.keyOf(&.{ 1, 2, 3 }), 2, &.{ 1, 2, 3 }, &.{ 1, 2, 5 }, 10);
        try m.add(Imprint.keyOf(&.{ 1, 2, 3, 4, 5 }), 4, &.{ 1, 2, 3, 4, 5 }, &.{ 1, 4 }, 20);
        try std.testing.expect(m.has(Imprint.keyOf(&.{ 1, 2, 3 })));
    }
    var m = try Imprint.open(gpa, root, 7, 1 << 30);
    defer m.deinit();
    try std.testing.expectEqual(@as(usize, 2), m.metas.items.len);
    try std.testing.expectEqual(@as(u64, 30), m.total());
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    try std.testing.expectEqual(@as(u32, 4), m.best(&prompt, &.{ 1, 4, 6 }, true, 0).?.at);
    try std.testing.expectEqual(@as(u32, 2), m.best(&prompt, &.{ 1, 2, 6 }, true, 0).?.at); // other starts below 4: only 2 fits
    try std.testing.expect(m.best(&prompt, &.{ 1, 4, 6 }, true, 4) == null); // memory already holds 4
    try std.testing.expect(m.best(&.{ 1, 2, 9, 9 }, &.{ 1, 2 }, true, 0) == null); // token 2 differs: 3 was read
    var other = try Imprint.open(gpa, root, 8, 1 << 30); // another identity sees nothing
    defer other.deinit();
    try std.testing.expectEqual(@as(usize, 0), other.metas.items.len);
}

test "the least recently used state goes first, and a removed one stays gone after a reopen" {
    const gpa = std.testing.allocator;
    var tmp_buf: [256]u8 = undefined;
    const root = try scratch(&tmp_buf, "lru");
    defer rmTree(root);
    const a = Imprint.keyOf(&.{ 1, 2 });
    const b = Imprint.keyOf(&.{ 3, 4 });
    const c = Imprint.keyOf(&.{ 5, 6 });
    {
        var m = try Imprint.open(gpa, root, 1, 100);
        defer m.deinit();
        try m.add(a, 1, &.{ 1, 2 }, &.{}, 30);
        try m.add(b, 1, &.{ 3, 4 }, &.{}, 30);
        try m.add(c, 1, &.{ 5, 6 }, &.{}, 30);
        m.touch(a);
        try std.testing.expectEqual(b, m.victim().?);
        try std.testing.expect(!m.fits(20)); // 90 held of 100
        try m.remove(b);
        try std.testing.expect(m.fits(20));
        try std.testing.expectEqual(c, m.victim().?);
    }
    var m = try Imprint.open(gpa, root, 1, 100);
    defer m.deinit();
    try std.testing.expect(m.has(a) and m.has(c) and !m.has(b));
    try std.testing.expectEqual(@as(u64, 60), m.total());
}

test "other identities are swept, least recently opened first, until the cap fits" {
    const gpa = std.testing.allocator;
    var tmp_buf: [256]u8 = undefined;
    const root = try scratch(&tmp_buf, "sweep");
    defer rmTree(root);
    var dirs: [2][300]u8 = undefined;
    var names: [2][]const u8 = undefined;
    for (0..2) |i| {
        var m = try Imprint.open(gpa, root, i + 1, 1 << 30);
        defer m.deinit();
        try fill(m.dir, "state.bin", 100); // with its 8-byte stamp: 108 bytes an identity
        names[i] = try std.fmt.bufPrint(&dirs[i], "{s}", .{m.dir});
    }
    var m = try Imprint.open(gpa, root, 3, 150); // Opening never deletes either rank before coordinated admission.
    defer m.deinit();
    try std.testing.expect(exists(names[0]));
    try std.testing.expect(m.fits(0));
    try std.testing.expect(!exists(names[0]));
    try std.testing.expect(exists(names[1]));
    try std.testing.expectEqual(@as(u64, 108), m.others);
    try std.testing.expect(m.fits(42) and !m.fits(43 + 108)); // a larger state sweeps the other identity too
    try std.testing.expect(!exists(names[1]));
    try std.testing.expectEqual(@as(u64, 0), m.others);
}
