const std = @import("std");
const platform = @import("sys");
const Imprint = @import("prompt_imprint.zig").Imprint;
const life = @import("learned_fault_test.zig");
const builtin = @import("builtin");
const dirs = @import("lanes").learned_dirs;
const rowdir = @import("rows_dir.zig");

const SweepFailure = enum { none, unlink, sync };
threadlocal var sweep_failure: SweepFailure = .none;
var open_streams: std.atomic.Value(isize) = .init(0);
threadlocal var capture_stage = false;
threadlocal var stage_name: [256]u8 = undefined;
threadlocal var stage_len: usize = 0;

extern "c" fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) ?*anyopaque;

fn systemCall(comptime T: type, name: [*:0]const u8) T {
    const next: *anyopaque = @ptrFromInt(std.math.maxInt(usize));
    return @ptrCast(@alignCast(dlsym(next, name) orelse unreachable));
}

fn opened(d: ?*platform.c.DIR) ?*platform.c.DIR {
    if (d != null) _ = open_streams.fetchAdd(1, .monotonic);
    return d;
}

pub export fn opendir(path: [*:0]const u8) callconv(.c) ?*platform.c.DIR {
    if (comptime builtin.os.tag == .windows) return opened(platform.c.opendir(path));
    return opened(systemCall(*const fn ([*:0]const u8) callconv(.c) ?*platform.c.DIR, "opendir")(path));
}

pub export fn fdopendir(fd: c_int) callconv(.c) ?*platform.c.DIR {
    if (comptime builtin.os.tag == .windows) return opened(platform.c.fdopendir(fd));
    return opened(systemCall(*const fn (c_int) callconv(.c) ?*platform.c.DIR, "fdopendir")(fd));
}

pub export fn closedir(d: *platform.c.DIR) callconv(.c) c_int {
    const rc = if (comptime builtin.os.tag == .windows)
        platform.c.closedir(d)
    else
        systemCall(*const fn (*platform.c.DIR) callconv(.c) c_int, "closedir")(d);
    if (rc == 0) _ = open_streams.fetchSub(1, .monotonic);
    return rc;
}

pub export fn unlinkat(fd: c_int, path: [*:0]const u8, flags: c_int) callconv(.c) c_int {
    const actual = if (sweep_failure == .unlink) -1 else fd;
    if (comptime builtin.os.tag == .windows) return platform.c.unlinkat(actual, path, @intCast(flags));
    return systemCall(*const fn (c_int, [*:0]const u8, c_int) callconv(.c) c_int, "unlinkat")(actual, path, flags);
}

pub export fn fsync(fd: c_int) callconv(.c) c_int {
    const actual = if (sweep_failure == .sync) -1 else fd;
    if (comptime builtin.os.tag == .windows) return platform.c.fsync(actual);
    return systemCall(*const fn (c_int) callconv(.c) c_int, "fsync")(actual);
}

fn recordStage(path: [*:0]const u8) void {
    if (!capture_stage) return;
    const base = std.fs.path.basename(std.mem.span(path));
    if (base.len > stage_name.len) return;
    @memcpy(stage_name[0..base.len], base);
    stage_len = base.len;
}

pub export fn renameat(old_fd: c_int, old: [*:0]const u8, new_fd: c_int, new: [*:0]const u8) callconv(.c) c_int {
    recordStage(old);
    if (comptime builtin.os.tag == .windows) return platform.c.rename(old, new);
    return systemCall(*const fn (c_int, [*:0]const u8, c_int, [*:0]const u8) callconv(.c) c_int, "renameat")(old_fd, old, new_fd, new);
}

pub export fn rename(old: [*:0]const u8, new: [*:0]const u8) callconv(.c) c_int {
    recordStage(old);
    if (comptime builtin.os.tag == .windows) return platform.c.rename(old, new);
    return systemCall(*const fn ([*:0]const u8, [*:0]const u8) callconv(.c) c_int, "rename")(old, new);
}

fn stagedWord(fd: c_int, name: [*:0]const u8, word: u64, copy: []u8) ![]const u8 {
    stage_len = 0;
    capture_stage = true;
    rowdir.writeWord(fd, name, word);
    capture_stage = false;
    if (stage_len == 0 or stage_len > copy.len) return error.TestStageName;
    @memcpy(copy[0..stage_len], stage_name[0..stage_len]);
    return copy[0..stage_len];
}

fn staleStage(copy: []u8, staged: []const u8) ![]const u8 {
    const pid_at = if (std.mem.indexOf(u8, staged, ".tmp.")) |at| at + ".tmp.".len else (std.mem.indexOfScalar(u8, staged, '.') orelse return error.TestStageName) + 1;
    const end = std.mem.indexOfScalarPos(u8, staged, pid_at, '.') orelse return error.TestStageName;
    return std.fmt.bufPrint(copy, "{s}999999999{s}", .{ staged[0..pid_at], staged[end..] });
}

fn scratch(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-learn-part-{s}-{d}", .{ tag, platform.c.getpid() });
}
fn put(dir: []const u8, name: []const u8, count: usize) !void {
    var buf: [1200]u8 = undefined;
    const fd = platform.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = platform.c.close(fd);
    if (platform.c.ftruncate(fd, @intCast(count)) != 0) return error.TestFile;
}
fn exists(dir: []const u8, name: []const u8) bool {
    var buf: [1200]u8 = undefined;
    const fd = platform.c.open(std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0) catch return false, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return false;
    _ = platform.c.close(fd);
    return true;
}

test "startup drops only regular owned temporary names before counting every identity" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "names");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    const owned = [_][]const u8{ "index.part", "index.tmp.123.0", "pending.part", "pending.tmp.123.1", "used.part", "used.tmp.123.2", "00000000000000af.bin.part", "00000000000000af.bin.tmp.123.3", "00000000000000af.r0.bin.part", "00000000000000af.r0.bin.tmp.123.4", "00000000000000af.r1.bin.part", "00000000000000af.r1.bin.tmp.123.5" };
    const foreign = [_][]const u8{ "notes.part", "notes.tmp.123.0", "index.tmp.123.", "00000000000000af.bin.part.bak", "00000000000000af.bin.tmp.123.0.bak", "00000000000000af.r2.bin.part", "00000000000000BF.bin.part", "0000000000000af.bin.part", "000000000000000af.bin.part", "state.bin" };
    for (1..3) |id| {
        var im = try Imprint.open(a, root, id, 1 << 20);
        defer im.deinit();
        for (owned) |name| try put(im.dir, name, 4096);
        for (foreign) |name| try put(im.dir, name, 1);
    }
    try put(root, "clock.part", 8);
    try put(root, "clock.tmp.123.0", 8);
    try put(root, "notes.part", 1);
    try put(root, "notes.tmp.123.0", 1);
    var im = try Imprint.open(a, root, 2, 1 << 20);
    defer im.deinit();
    for (1..3) |id| {
        var path: [1200]u8 = undefined;
        const dir = try std.fmt.bufPrint(&path, "{s}/{x:0>16}", .{ root, id });
        for (owned) |name| try std.testing.expect(!exists(dir, name));
        for (foreign) |name| try std.testing.expect(exists(dir, name));
    }
    try std.testing.expect(!exists(root, "clock.part"));
    try std.testing.expect(!exists(root, "clock.tmp.123.0"));
    try std.testing.expect(exists(root, "notes.part"));
    try std.testing.expect(exists(root, "notes.tmp.123.0"));
    try std.testing.expectEqual(@as(u64, 18), im.others);
}

test "imprint rewrite preserves a planted legacy temporary link and its target" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "rewrite-link");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var im = try Imprint.open(a, root, 1, 1 << 20);
    defer im.deinit();
    const first = Imprint.keyOf(&.{ 1, 2 });
    const second = Imprint.keyOf(&.{ 3, 4 });
    try im.add(first, 1, &.{ 1, 2 }, &.{}, 10);
    try im.add(second, 1, &.{ 3, 4 }, &.{}, 20);
    try put(root, "target", 7);
    var target_buf: [1200]u8 = undefined;
    const target = try std.fmt.bufPrintSentinel(&target_buf, "{s}/target", .{root}, 0);
    var link_buf: [1200]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&link_buf, "{s}/index.part", .{im.dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.symlink(target, link));
    try im.remove(first);
    var actual: [1200]u8 = undefined;
    try std.testing.expectEqual(@as(isize, @intCast(target.len)), platform.c.readlink(link, &actual, actual.len));
    try std.testing.expectEqual(@as(u64, 7), try @import("lanes").learned_dirs.fileBytes(target));
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
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // symlinks need privilege on Windows
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
    try std.testing.expectEqual(@as(c_int, 0), platform.c.symlink(target, link));
    defer _ = platform.c.unlink(link);
    var dir_buf: [1200]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "{s}/index.part", .{im.dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(dir, 0o700));
    defer _ = platform.c.rmdir(dir);
    var foreign_buf: [1200]u8 = undefined;
    const foreign = try std.fmt.bufPrintSentinel(&foreign_buf, "{s}/foreign", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(foreign, 0o700));
    defer _ = platform.c.rmdir(foreign);
    try put(foreign, "index.part", 3);
    var reopened = try Imprint.open(a, root, 1, 1 << 20);
    defer reopened.deinit();
    var actual: [1200]u8 = undefined;
    const n = platform.c.readlink(link, &actual, actual.len);
    try std.testing.expectEqual(@as(isize, @intCast(target.len)), n);
    try std.testing.expectEqualSlices(u8, target, actual[0..@intCast(n)]);
    const directory = platform.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
    try std.testing.expect(directory >= 0);
    _ = platform.c.close(directory);
    try std.testing.expect(exists(root, "target"));
    try std.testing.expect(exists(foreign, "index.part"));
    var marker_buf: [1400]u8 = undefined;
    _ = platform.c.unlink(try std.fmt.bufPrintSentinel(&marker_buf, "{s}/index.part", .{foreign}, 0));
}

test "an identity-directory symlink refuses open and preserves foreign payloads" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // symlinks need privilege on Windows
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
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(target, 0o700));
    try put(target, "00000000000000af.bin", 7);
    var link_buf: [1200]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&link_buf, "{s}/0000000000000001", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.symlink(target, link));
    defer _ = platform.c.unlink(link);
    try std.testing.expectError(error.ImprintRead, Imprint.open(a, root, 1, 1 << 20));
    try std.testing.expect(exists(target, "00000000000000af.bin"));
    try std.testing.expect(!exists(target, "used"));
}

test "an explicitly configured root alias still cleans only its real identity children" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // symlinks need privilege on Windows
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
    const alias = try std.fmt.bufPrintSentinel(&alias_buf, "/tmp/tf-learn-root-alias-{d}", .{platform.c.getpid()}, 0);
    var target_buf: [128]u8 = undefined;
    const target = try std.fmt.bufPrintSentinel(&target_buf, "{s}", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.symlink(target, alias));
    defer _ = platform.c.unlink(alias);
    var reopened = try Imprint.open(a, alias, 1, 1 << 20);
    defer reopened.deinit();
    try std.testing.expect(!exists(reopened.dir, "index.part"));
}

test "cross-identity eviction refuses a linked identity and preserves its target" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "eviction-link");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var root_buf: [1200]u8 = undefined;
    const root_path = try std.fmt.bufPrintSentinel(&root_buf, "{s}", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(root_path, 0o700));
    var foreign_buf: [1200]u8 = undefined;
    const foreign = try std.fmt.bufPrintSentinel(&foreign_buf, "{s}/foreign", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(foreign, 0o700));
    try put(foreign, "00000000000000af.bin", 7);
    var link_buf: [1200]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&link_buf, "{s}/0000000000000001", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.symlink(foreign, link));
    try std.testing.expectError(error.LearnedDirectoryRead, dirs.removeOther(root, "-", 1));
    try std.testing.expectError(error.LearnedDirectoryRead, dirs.used(root, 1));
    try std.testing.expect(exists(foreign, "00000000000000af.bin"));
}

test "index rewrite ignores a planted legacy staging link and preserves its target" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "index-stage-link");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var im = try Imprint.open(a, root, 1, 1 << 20);
    defer im.deinit();
    const tokens = [_]u32{ 1, 2, 3 };
    const key = Imprint.keyOf(&tokens);
    try im.add(key, 3, &tokens, &.{}, 7);
    try put(root, "victim", 6);
    var victim_buf: [1200]u8 = undefined;
    const victim = try std.fmt.bufPrintSentinel(&victim_buf, "{s}/victim", .{root}, 0);
    var link_buf: [1200]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&link_buf, "{s}/index.part", .{im.dir}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.symlink(victim, link));
    try im.remove(key);
    const fd = platform.c.open(victim, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = platform.c.close(fd);
    var contents: [6]u8 = undefined;
    try std.testing.expectEqual(@as(isize, 6), platform.c.read(fd, &contents, contents.len));
}

test "unindexed payload sweeps close their directory after repeated successes" {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "sweep-success");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var im = try Imprint.open(a, root, 1, 1 << 20);
    defer im.deinit();
    const metas = [_]@import("prompt_imprint.zig").Meta{};
    for (0..2) |_| {
        try put(im.dir, "00000000000000a1.bin", 1);
        const before = open_streams.load(.monotonic);
        try dirs.removeUnindexed(a, im.dir, &metas);
        try std.testing.expectEqual(before, open_streams.load(.monotonic));
        try std.testing.expect(!exists(im.dir, "00000000000000a1.bin"));
    }
}

fn failedSweepBalances(comptime failure: SweepFailure, tag: []const u8) !void {
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, tag);
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var im = try Imprint.open(a, root, 1, 1 << 20);
    defer im.deinit();
    try put(im.dir, "00000000000000a1.bin", 1);
    const metas = [_]@import("prompt_imprint.zig").Meta{};
    sweep_failure = failure;
    defer sweep_failure = .none;
    const before = open_streams.load(.monotonic);
    try std.testing.expectError(error.LearnedUnlink, dirs.removeUnindexed(a, im.dir, &metas));
    try std.testing.expectEqual(before, open_streams.load(.monotonic));
    try std.testing.expectEqual(failure == .sync, !exists(im.dir, "00000000000000a1.bin"));
}

test "unindexed payload sweeps close their directory after an unlink failure" {
    try failedSweepBalances(.unlink, "sweep-unlink");
}

test "unindexed payload sweeps close their directory after a directory-sync failure" {
    try failedSweepBalances(.sync, "sweep-sync");
}

test "row clock crash cleanup removes stale root and recency stages but keeps live writers" {
    // POSIX liveness needs no file timestamps; Windows' age fallback needs an aged fixture.
    if (platform.windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "row-clock");
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var im = try Imprint.open(a, root, 1, 1 << 20);
    defer im.deinit();
    var root_buf: [160]u8 = undefined;
    const root_path = try std.fmt.bufPrintSentinel(&root_buf, "{s}", .{root}, 0);
    const root_fd = try rowdir.openRoot(root_path);
    defer rowdir.close(root_fd);
    const identity_fd = try rowdir.openPath(im.dir);
    defer rowdir.close(identity_fd);
    var clock_buf: [128]u8 = undefined;
    var used_buf: [128]u8 = undefined;
    const clock_stage = try stagedWord(root_fd, "clock", 17, &clock_buf);
    const used_stage = try stagedWord(identity_fd, "used", 17, &used_buf);
    var clock_word: u64 = 0;
    var used_word: u64 = 0;
    try std.testing.expect(rowdir.read(root_fd, "clock", std.mem.asBytes(&clock_word)));
    try std.testing.expect(rowdir.read(identity_fd, "used", std.mem.asBytes(&used_word)));
    try std.testing.expectEqual(@as(u64, 17), clock_word);
    try std.testing.expectEqual(@as(u64, 17), used_word);
    var stale_clock_buf: [128]u8 = undefined;
    var stale_used_buf: [128]u8 = undefined;
    const stale_clock = try staleStage(&stale_clock_buf, clock_stage);
    const stale_used = try staleStage(&stale_used_buf, used_stage);
    try put(root, stale_clock, 8);
    try put(im.dir, stale_used, 8);
    try put(root, clock_stage, 8);
    try put(im.dir, used_stage, 8);
    try dirs.cleanupParts(root);
    try std.testing.expect(!exists(root, stale_clock));
    try std.testing.expect(!exists(im.dir, stale_used));
    try std.testing.expect(exists(root, clock_stage));
    try std.testing.expect(exists(im.dir, used_stage));
    try std.testing.expect(exists(root, "clock"));
    try std.testing.expect(exists(im.dir, "used"));
}
