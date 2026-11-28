//! Standalone platform shim regressions for environment setters and Windows filesystem safety.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const env = platform.env;
fn value(name: [:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    return std.mem.span(raw);
}

test "the platform environment shim sets a variable, keeps it on a no-overwrite write and clears it" {
    const name = "TF_PLATFORM_ENV_SHIM_TEST";
    defer _ = env.unsetenv(name);
    try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, "1", 1));
    try std.testing.expectEqualStrings("1", value(name).?);
    // a write that may not overwrite leaves the value alone
    try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, "2", 0));
    try std.testing.expectEqualStrings("1", value(name).?);
    try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, "2", 1));
    try std.testing.expectEqualStrings("2", value(name).?);
    try std.testing.expectEqual(@as(c_int, 0), env.unsetenv(name));
    // Windows clears a variable to the empty string, elsewhere it is gone
    const left = value(name);
    try std.testing.expect(left == null or left.?.len == 0);
}

test "Windows exclusive staging advances past a symlink collision" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "target", .data = "leave the target unchanged" });
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    const destination = try std.fmt.allocPrintSentinel(a, "{s}/published", .{root}, 0);
    defer a.free(destination);
    const target = try std.fmt.allocPrintSentinel(a, "{s}/target", .{root}, 0);
    defer a.free(target);
    const first = try std.fmt.allocPrintSentinel(a, "{s}.tmp.{d}.0", .{ destination, platform.c.getpid() }, 0);
    defer a.free(first);
    const second = try std.fmt.allocPrintSentinel(a, "{s}.tmp.{d}.1", .{ destination, platform.c.getpid() }, 0);
    defer a.free(second);
    if (platform.c.symlink(target.ptr, first.ptr) != 0) return error.SkipZigTest;
    defer _ = platform.c.unlink(first.ptr);
    const staged = try platform.atomic_file.stage(a, destination, &.{"published through the next candidate"});
    var staged_owned = true;
    defer {
        if (staged_owned) {
            a.free(staged);
            _ = platform.c.unlink(staged.ptr);
        }
    }
    try t.expectEqualStrings(second, staged);
    staged_owned = false;
    try platform.atomic_file.commit(a, staged, destination);
    const published = try tmp.dir.readFileAlloc(t.io, "published", a, .limited(128));
    defer a.free(published);
    try t.expectEqualStrings("published through the next candidate", published);
    const untouched = try tmp.dir.readFileAlloc(t.io, "target", a, .limited(128));
    defer a.free(untouched);
    try t.expectEqualStrings("leave the target unchanged", untouched);
}

test "Windows NOFOLLOW still refuses a non-exclusive dangling staging link" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    const target = try std.fmt.allocPrintSentinel(a, "{s}/absent", .{root}, 0);
    defer a.free(target);
    const link = try std.fmt.allocPrintSentinel(a, "{s}/state.part", .{root}, 0);
    defer a.free(link);
    if (platform.c.symlink(target.ptr, link.ptr) != 0) return error.SkipZigTest;
    defer _ = platform.c.unlink(link.ptr);
    const fd = platform.c.open(link.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0o600));
    if (fd >= 0) {
        _ = platform.c.close(fd);
        return error.UnexpectedFollow;
    }
    try t.expectEqual(platform.c.E.LOOP, platform.c.errno(fd));
    const target_fd = platform.c.open(target.ptr, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (target_fd >= 0) {
        _ = platform.c.close(target_fd);
        return error.UnexpectedTarget;
    }
}

test "Windows directory locks refuse a symlinked lock file" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "target", .data = "not a lock" });
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    const target = try std.fmt.allocPrintSentinel(a, "{s}/target", .{root}, 0);
    defer a.free(target);
    const link = try std.fmt.allocPrintSentinel(a, "{s}/.tensorfold-lock", .{root}, 0);
    defer a.free(link);
    if (platform.c.symlink(target.ptr, link.ptr) != 0) return error.SkipZigTest;
    defer _ = platform.c.unlink(link.ptr);
    try t.expectError(error.Unlockable, platform.folder_lock.hold(a, root, true));
}

test "Windows relative rename and unlink preserve POSIX open-handle semantics" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "old", .data = "replacement" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "new", .data = "held destination" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "victim", .data = "held unlink" });
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    const dir = platform.c.open(root.ptr, .{ .DIRECTORY = true }, @as(platform.c.mode_t, 0));
    try t.expect(dir >= 0);
    defer _ = platform.c.close(dir);
    const destination = platform.c.openat(dir, "new", .{}, @as(platform.c.mode_t, 0));
    try t.expect(destination >= 0);
    defer _ = platform.c.close(destination);
    try t.expectEqual(@as(c_int, 0), platform.c.renameat(dir, "old", dir, "new"));
    const victim = platform.c.openat(dir, "victim", .{}, @as(platform.c.mode_t, 0));
    try t.expect(victim >= 0);
    defer _ = platform.c.close(victim);
    try t.expectEqual(@as(c_int, 0), platform.c.unlinkat(dir, "victim", 0));
    const absent = platform.c.openat(dir, "victim", .{}, @as(platform.c.mode_t, 0));
    if (absent >= 0) {
        _ = platform.c.close(absent);
        return error.UnlinkedNameRemains;
    }
    try t.expectEqual(platform.c.E.NOENT, platform.c.errno(absent));
}

test "Windows exclusive creates stage and publish absolutely and relative to a directory" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    const dir = platform.c.open(root.ptr, .{ .DIRECTORY = true }, @as(platform.c.mode_t, 0));
    try t.expect(dir >= 0);
    defer _ = platform.c.close(dir);
    defer _ = platform.c.unlinkat(dir, "leaf", 0);
    const flags: platform.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true };
    const relative = platform.c.openat(dir, "leaf", flags, @as(platform.c.mode_t, 0o600));
    try t.expect(relative >= 0);
    try t.expectEqual(@as(isize, 5), platform.c.write(relative, "fresh", 5));
    try t.expectEqual(@as(c_int, 0), platform.c.close(relative));
    const relative_collision = platform.c.openat(dir, "leaf", flags, @as(platform.c.mode_t, 0o600));
    if (relative_collision >= 0) {
        _ = platform.c.close(relative_collision);
        return error.UnexpectedRelativeCreate;
    }
    try t.expectEqual(platform.c.E.EXIST, platform.c.errno(relative_collision));
    try t.expectEqual(@as(c_int, 0), platform.c.renameat(dir, "leaf", dir, "published"));
    const published_relative = try tmp.dir.readFileAlloc(t.io, "published", a, .limited(16));
    defer a.free(published_relative);
    try t.expectEqualStrings("fresh", published_relative);

    const absolute = try std.fmt.allocPrintSentinel(a, "{s}/absolute", .{root}, 0);
    defer a.free(absolute);
    defer _ = platform.c.unlink(absolute.ptr);
    const created = platform.c.open(absolute.ptr, flags, @as(platform.c.mode_t, 0o600));
    try t.expect(created >= 0);
    try t.expectEqual(@as(isize, 5), platform.c.write(created, "whole", 5));
    try t.expectEqual(@as(c_int, 0), platform.c.close(created));
    const absolute_collision = platform.c.open(absolute.ptr, flags, @as(platform.c.mode_t, 0o600));
    if (absolute_collision >= 0) {
        _ = platform.c.close(absolute_collision);
        return error.UnexpectedAbsoluteCreate;
    }
    try t.expectEqual(platform.c.E.EXIST, platform.c.errno(absolute_collision));
    const staged = try platform.atomic_file.stagePrivate(a, absolute, &.{"staged and published"});
    try platform.atomic_file.commit(a, staged, absolute);
    const absolute_bytes = try tmp.dir.readFileAlloc(t.io, "absolute", a, .limited(16));
    defer a.free(absolute_bytes);
    try t.expectEqualStrings("staged and published", absolute_bytes);
}

test "Windows dot reopen scans independently of its parent handle" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "entry", .data = "contents" });
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    const dir = platform.c.open(root.ptr, .{ .DIRECTORY = true }, @as(platform.c.mode_t, 0));
    try t.expect(dir >= 0);
    defer _ = platform.c.close(dir);
    const copy = platform.c.openat(dir, ".", .{ .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    try t.expect(copy >= 0);
    {
        const stream = platform.c.fdopendir(copy) orelse {
            _ = platform.c.close(copy);
            return error.CloneScanFailed;
        };
        defer _ = platform.c.closedir(stream);
        var found = false;
        while (platform.c.readdir(stream)) |entry| {
            if (std.mem.eql(u8, std.mem.sliceTo(&entry.name, 0), "entry")) found = true;
        }
        try t.expect(found);
    }
    const file = platform.c.openat(dir, "entry", .{ .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    try t.expect(file >= 0);
    defer _ = platform.c.close(file);
    var bytes: [8]u8 = undefined;
    try t.expectEqual(@as(isize, 8), platform.c.read(file, &bytes, bytes.len));
    try t.expectEqualStrings("contents", &bytes);
}
