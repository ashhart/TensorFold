const std = @import("std");
const Imprint = @import("prompt_imprint.zig").Imprint;
const life = @import("learned_fault_test.zig");

fn scratch(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-learn-part-{s}-{d}", .{ tag, std.c.getpid() });
}
fn put(dir: []const u8, name: []const u8, count: usize) !void {
    var buf: [1200]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = std.c.close(fd);
    if (std.c.ftruncate(fd, @intCast(count)) != 0) return error.TestFile;
}
fn exists(dir: []const u8, name: []const u8) bool {
    var buf: [1200]u8 = undefined;
    const fd = std.c.open(std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0) catch return false, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return false;
    _ = std.c.close(fd);
    return true;
}

test "startup drops only regular owned temporary names before counting every identity" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "names");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    const owned = [_][]const u8{ "index.part", "00000000000000af.bin.part", "00000000000000af.r0.bin.part", "00000000000000af.r1.bin.part", "pending.part", "used.part" };
    const foreign = [_][]const u8{ "notes.part", "00000000000000af.bin.part.bak", "00000000000000af.r2.bin.part", "00000000000000BF.bin.part", "0000000000000af.bin.part", "000000000000000af.bin.part", "state.bin" };
    for (1..3) |id| {
        var im = try Imprint.open(a, root, id, 1 << 20);
        defer im.deinit();
        for (owned) |name| try put(im.dir, name, 4096);
        for (foreign) |name| try put(im.dir, name, 1);
    }
    try put(root, "clock.part", 8);
    try put(root, "notes.part", 1);
    var im = try Imprint.open(a, root, 2, 1 << 20);
    defer im.deinit();
    for (1..3) |id| {
        var path: [1200]u8 = undefined;
        const dir = try std.fmt.bufPrint(&path, "{s}/{x:0>16}", .{ root, id });
        for (owned) |name| try std.testing.expect(!exists(dir, name));
        for (foreign) |name| try std.testing.expect(exists(dir, name));
    }
    try std.testing.expect(!exists(root, "clock.part"));
    try std.testing.expect(exists(root, "notes.part"));
    try std.testing.expectEqual(@as(u64, 15), im.others);
}

test "orphan temporary pressure cannot evict an older valid learned state" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "cap");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const second = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };
    const later = [_]u32{ 21, 22, 23, 24, 25, 26, 27, 28 };
    for (1..3) |id| {
        var im = try Imprint.open(a, root, id, 1 << 20);
        defer im.deinit();
        try life.learnOne(&im, if (id == 1) &first else &second);
    }
    var old: [1200]u8 = undefined;
    const old_dir = try std.fmt.bufPrint(&old, "{s}/0000000000000001", .{root});
    try put(old_dir, "index.part", 4096);
    try put(old_dir, "00000000000000af.bin.part", 4096);
    {
        var im = try Imprint.open(a, root, 2, 900);
        defer im.deinit();
        try life.learnOne(&im, &later);
        try std.testing.expect(im.has(Imprint.keyOf(second[0..6])));
    }
    {
        var im = try Imprint.open(a, root, 1, 900);
        defer im.deinit();
        try std.testing.expect(im.has(Imprint.keyOf(first[0..6])));
        try life.recallOne(&im, &first);
    }
    var im = try Imprint.open(a, root, 2, 900);
    defer im.deinit();
    try life.recallOne(&im, &second);
    try life.recallOne(&im, &later);
}

test "owned-name links and directories stay untouched and foreign directories are not traversed" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "types");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var im = try Imprint.open(a, root, 1, 1 << 20);
    defer im.deinit();
    try put(root, "target", 7);
    var target_buf: [1200]u8 = undefined;
    const target = try std.fmt.bufPrintSentinel(&target_buf, "{s}/target", .{root}, 0);
    var link_buf: [1200]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&link_buf, "{s}/00000000000000af.bin.part", .{im.dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlink(target, link));
    defer _ = std.c.unlink(link);
    var dir_buf: [1200]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "{s}/index.part", .{im.dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(dir, 0o700));
    defer _ = std.c.rmdir(dir);
    var foreign_buf: [1200]u8 = undefined;
    const foreign = try std.fmt.bufPrintSentinel(&foreign_buf, "{s}/foreign", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(foreign, 0o700));
    defer _ = std.c.rmdir(foreign);
    try put(foreign, "index.part", 3);
    var reopened = try Imprint.open(a, root, 1, 1 << 20);
    defer reopened.deinit();
    var actual: [1200]u8 = undefined;
    const n = std.c.readlink(link, &actual, actual.len);
    try std.testing.expectEqual(@as(isize, @intCast(target.len)), n);
    try std.testing.expectEqualSlices(u8, target, actual[0..@intCast(n)]);
    const directory = std.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(directory >= 0);
    _ = std.c.close(directory);
    try std.testing.expect(exists(root, "target"));
    try std.testing.expect(exists(foreign, "index.part"));
    var marker_buf: [1400]u8 = undefined;
    _ = std.c.unlink(try std.fmt.bufPrintSentinel(&marker_buf, "{s}/index.part", .{foreign}, 0));
}

test "an identity-directory symlink refuses open and preserves foreign payloads" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "identity-link");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    {
        var seeded = try Imprint.open(a, root, 0, 1 << 20);
        defer seeded.deinit();
    }
    var target_buf: [1200]u8 = undefined;
    const target = try std.fmt.bufPrintSentinel(&target_buf, "{s}/foreign", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(target, 0o700));
    try put(target, "00000000000000af.bin", 7);
    var link_buf: [1200]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&link_buf, "{s}/0000000000000001", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlink(target, link));
    defer _ = std.c.unlink(link);
    try std.testing.expectError(error.ImprintRead, Imprint.open(a, root, 1, 1 << 20));
    try std.testing.expect(exists(target, "00000000000000af.bin"));
    try std.testing.expect(!exists(target, "used"));
}

test "an explicitly configured root alias still cleans only its real identity children" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "root-target");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    {
        var seeded = try Imprint.open(a, root, 1, 1 << 20);
        defer seeded.deinit();
        try put(seeded.dir, "index.part", 4096);
    }
    var alias_buf: [128]u8 = undefined;
    const alias = try std.fmt.bufPrintSentinel(&alias_buf, "/tmp/tf-learn-root-alias-{d}", .{std.c.getpid()}, 0);
    var target_buf: [128]u8 = undefined;
    const target = try std.fmt.bufPrintSentinel(&target_buf, "{s}", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlink(target, alias));
    defer _ = std.c.unlink(alias);
    var reopened = try Imprint.open(a, alias, 1, 1 << 20);
    defer reopened.deinit();
    try std.testing.expect(!exists(reopened.dir, "index.part"));
}
