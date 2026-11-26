//! Bounded-name learned directories: both ranks plan the same identity deletions before either writes.
const std = @import("std");
const platform = @import("sys");
pub fn rootOf(dir: []const u8) []const u8 {
    return std.fs.path.dirname(dir) orelse ".";
}
fn isCurrent(keep: []const u8, id: u64) bool {
    var name: [16]u8 = undefined;
    const formatted = std.fmt.bufPrint(&name, "{x:0>16}", .{id}) catch unreachable;
    return std.mem.eql(u8, std.fs.path.basename(keep), formatted);
}
pub fn next(root: []const u8, keep: []const u8, after: ?u64) !?u64 {
    const root_fd = openRoot(root);
    if (root_fd < 0) return error.LearnedDirectoryRead;
    defer _ = platform.c.close(root_fd);
    return nextAt(root_fd, keep, after);
}

pub fn nextAt(root_fd: c_int, keep: []const u8, after: ?u64) !?u64 {
    const copy = try cloneDir(root_fd);
    const d = platform.c.fdopendir(copy) orelse {
        _ = platform.c.close(copy);
        return error.LearnedDirectoryRead;
    };
    defer _ = platform.c.closedir(d);
    var out: ?u64 = null;
    while (platform.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name.len != 16) continue;
        const id = std.fmt.parseInt(u64, name, 16) catch continue;
        if (id == std.math.maxInt(u64)) return error.LearnedDirectoryType;
        if (ent.type != platform.c.DT.DIR) return error.LearnedDirectoryType;
        if (isCurrent(keep, id) or (after != null and id <= after.?)) continue;
        if (out == null or id < out.?) out = id;
    }
    return out;
}
pub fn unlink(file: [:0]const u8) !void {
    const rc = platform.c.unlink(file);
    if (rc != 0 and platform.c.errno(rc) != .NOENT) return error.LearnedUnlink;
    if (rc == 0) platform.atomic_file.syncParent(file) catch return error.LearnedUnlink;
}

fn stateKey(name: []const u8) ?u64 {
    if (name.len < 16) return null;
    for (name[0..16]) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return null;
    const suffix = name[16..];
    if (!std.mem.eql(u8, suffix, ".bin") and !std.mem.eql(u8, suffix, ".r0.bin") and !std.mem.eql(u8, suffix, ".r1.bin")) return null;
    return std.fmt.parseInt(u64, name[0..16], 16) catch null;
}

fn stagedAs(name: []const u8, base: []const u8) bool {
    const staged = platform.atomic_file.stagedBy(name) orelse return false;
    return std.mem.eql(u8, staged.file, base);
}
fn partName(name: []const u8) bool {
    if (std.mem.eql(u8, name, "index.part") or std.mem.eql(u8, name, "pending.part") or std.mem.eql(u8, name, "used.part")) return true;
    if (std.mem.endsWith(u8, name, ".part") and stateKey(name[0 .. name.len - 5]) != null) return true;
    if (stagedAs(name, "index") or stagedAs(name, "pending") or stagedAs(name, "used")) return true;
    const staged = platform.atomic_file.stagedBy(name) orelse return false;
    return stateKey(staged.file) != null;
}
fn staleStage(fd: c_int, name: [*:0]const u8) bool {
    const staged = platform.atomic_file.stagedBy(std.mem.span(name)) orelse return false;
    if (platform.windows) {
        const file = platform.c.openat(fd, name, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
        if (file < 0) return false;
        defer _ = platform.c.close(file);
        var stat: platform.c.Stat = undefined;
        var now: platform.c.timespec = undefined;
        if (platform.c.fstat(file, &stat) != 0 or platform.c.clock_gettime(platform.c.CLOCK.REALTIME, &now) != 0) return false;
        const modified = @as(i128, stat.mtime().sec) * std.time.ns_per_s + @as(i128, stat.mtime().nsec);
        const current = @as(i128, now.sec) * std.time.ns_per_s + @as(i128, now.nsec);
        return modified < current - 86_400 * std.time.ns_per_s;
    }
    if (platform.c.kill(staged.pid, @fromBackingInt(@intCast(0))) == 0) return false;
    return platform.c.errno(-1) == .SRCH;
}

fn identityName(name: []const u8) bool {
    if (name.len != 16) return false;
    for (name) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return false;
    return true;
}

fn unlinkAt(fd: c_int, name: [*:0]const u8) !void {
    const rc = platform.c.unlinkat(fd, name, 0);
    if (rc != 0 and platform.c.errno(rc) != .NOENT) return error.LearnedUnlink;
}
fn syncFd(fd: c_int) !void {
    while (platform.c.fsync(fd) != 0) {
        if (platform.c.errno(-1) == .INTR) continue;
        return error.LearnedUnlink;
    }
}

fn openRoot(root: []const u8) c_int {
    var buf: [1100]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0) catch return -1;
    return platform.c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
}

fn cloneDir(fd: c_int) !c_int {
    // Reopen "." for an independent readdir offset; descriptor duplication would share it.
    const copy = platform.c.openat(fd, ".", .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    return if (copy < 0) error.LearnedDirectoryRead else copy;
}

fn openIdentity(root_fd: c_int, id: u64) c_int {
    var leaf_buf: [17]u8 = undefined;
    const leaf = std.fmt.bufPrintSentinel(&leaf_buf, "{x:0>16}", .{id}, 0) catch unreachable;
    return platform.c.openat(root_fd, leaf, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
}

/// Only the configured root may be an alias; cleanup never follows an identity or file link.
pub fn cleanupParts(root: []const u8) !void {
    var buf: [1100]u8 = undefined;
    const fd = platform.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0), .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.LearnedDirectoryRead;
    return cleanupPartsFd(fd);
}

/// The caller keeps `root_fd`; cleanup iterates an independent descriptor pinned to the same directory.
pub fn cleanupPartsAt(root_fd: c_int) !void {
    return cleanupPartsFd(try cloneDir(root_fd));
}

fn cleanupPartsFd(fd: c_int) !void {
    const d = platform.c.fdopendir(fd) orelse {
        _ = platform.c.close(fd);
        return error.LearnedDirectoryRead;
    };
    defer _ = platform.c.closedir(d);
    var root_changed = false;
    while (platform.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (ent.type == platform.c.DT.REG) {
            if (std.mem.eql(u8, name, "clock.part") or (stagedAs(name, "clock") and staleStage(fd, @ptrCast(&ent.name)))) {
                try unlinkAt(fd, @ptrCast(&ent.name));
                root_changed = true;
                continue;
            }
        }
        if (!identityName(name)) continue;
        if (ent.type != platform.c.DT.DIR) return error.LearnedDirectoryType;
        const sub_fd = platform.c.openat(fd, @ptrCast(&ent.name), .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
        if (sub_fd < 0) return error.LearnedDirectoryRead;
        const sub = platform.c.fdopendir(sub_fd) orelse {
            _ = platform.c.close(sub_fd);
            return error.LearnedDirectoryRead;
        };
        {
            defer _ = platform.c.closedir(sub);
            var changed = false;
            while (platform.c.readdir(sub)) |file| {
                const file_name = std.mem.sliceTo(&file.name, 0);
                if (file.type == platform.c.DT.REG and partName(file_name)) {
                    if (stagedAs(file_name, "used") and !staleStage(sub_fd, @ptrCast(&file.name))) continue;
                    try unlinkAt(sub_fd, @ptrCast(&file.name));
                    changed = true;
                }
            }
            if (changed) try syncFd(sub_fd);
        }
    }
    if (root_changed) try syncFd(fd);
}

/// Payloads have no token metadata to recover a lost index entry; only exact owned, unreferenced names go.
pub fn removeUnindexed(gpa: std.mem.Allocator, dir: [:0]const u8, metas: anytype) !void {
    var indexed: std.AutoHashMapUnmanaged(u64, void) = .empty; // whole before any unlink
    defer indexed.deinit(gpa);
    try indexed.ensureTotalCapacity(gpa, std.math.cast(u32, metas.len) orelse return error.OutOfMemory);
    for (metas) |x| indexed.putAssumeCapacity(x.key, {});
    const fd = platform.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.LearnedDirectoryRead;
    const d = platform.c.fdopendir(fd) orelse {
        _ = platform.c.close(fd);
        return error.LearnedDirectoryRead;
    };
    defer _ = platform.c.closedir(d);
    var changed = false;
    while (platform.c.readdir(d)) |ent| {
        if (ent.type != platform.c.DT.REG) continue;
        const name = std.mem.sliceTo(&ent.name, 0);
        const key = stateKey(name) orelse continue;
        if (indexed.contains(key)) continue;
        try unlinkAt(fd, @ptrCast(&ent.name));
        changed = true;
    }
    if (changed) try syncFd(fd);
}
pub fn bytes(dir: [:0]const u8) !u64 {
    const dir_fd = platform.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    if (dir_fd < 0) return if (platform.c.errno(dir_fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = platform.c.close(dir_fd);
    return bytesAt(dir_fd);
}

fn bytesAt(dir_fd: c_int) !u64 {
    const copy = try cloneDir(dir_fd);
    const d = platform.c.fdopendir(copy) orelse {
        _ = platform.c.close(copy);
        return error.LearnedDirectoryRead;
    };
    defer _ = platform.c.closedir(d);
    var n: u64 = 0;
    while (platform.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        if (ent.type != platform.c.DT.REG) return error.LearnedDirectoryType;
        const fd = platform.c.openat(dir_fd, @ptrCast(&ent.name), .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
        if (fd < 0) return error.LearnedDirectoryRead;
        const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        _ = platform.c.close(fd);
        if (end < 0) return error.LearnedDirectoryRead;
        n = try std.math.add(u64, n, @intCast(end));
    }
    return n;
}
pub fn fileBytes(file: [:0]const u8) !u64 {
    const fd = platform.c.open(file, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return if (platform.c.errno(fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = platform.c.close(fd);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < 0) return error.LearnedDirectoryRead;
    return @intCast(end);
}
pub fn otherBytes(root: []const u8, keep: []const u8, id: u64) !u64 {
    const root_fd = openRoot(root);
    if (root_fd < 0) return if (platform.c.errno(root_fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = platform.c.close(root_fd);
    return otherBytesAt(root_fd, keep, id);
}

pub fn otherBytesAt(root_fd: c_int, keep: []const u8, id: u64) !u64 {
    if (isCurrent(keep, id)) return 0;
    const dir_fd = openIdentity(root_fd, id);
    if (dir_fd < 0) return if (platform.c.errno(dir_fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = platform.c.close(dir_fd);
    return bytesAt(dir_fd);
}

pub fn totalOthers(root: []const u8, keep: []const u8) !u64 {
    const root_fd = openRoot(root);
    if (root_fd < 0) return error.LearnedDirectoryRead;
    defer _ = platform.c.close(root_fd);
    return totalOthersAt(root_fd, keep);
}

pub fn totalOthersAt(root_fd: c_int, keep: []const u8) !u64 {
    var cursor: ?u64 = null;
    var n: u64 = 0;
    while (try nextAt(root_fd, keep, cursor)) |id| {
        n = try std.math.add(u64, n, try otherBytesAt(root_fd, keep, id));
        cursor = id;
    }
    return n;
}

pub fn total(root: []const u8) !u64 {
    return totalOthers(root, "-");
}
pub fn used(root: []const u8, id: u64) !u64 {
    const root_fd = openRoot(root);
    if (root_fd < 0) return if (platform.c.errno(root_fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = platform.c.close(root_fd);
    return usedAt(root_fd, id);
}

pub fn usedAt(root_fd: c_int, id: u64) !u64 {
    const dir_fd = openIdentity(root_fd, id);
    if (dir_fd < 0) return if (platform.c.errno(dir_fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = platform.c.close(dir_fd);
    const fd = platform.c.openat(dir_fd, "used", .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return if (platform.c.errno(fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = platform.c.close(fd);
    var n: u64 = 0;
    const data = std.mem.asBytes(&n);
    var done: usize = 0;
    while (done < data.len) {
        const count = platform.c.read(fd, data.ptr + done, data.len - done);
        if (count < 0) {
            if (platform.c.errno(count) == .INTR) continue;
            return error.LearnedDirectoryRead;
        }
        if (count == 0) return 0; // A torn recency hint is oldest; it cannot make a learned payload valid.
        done += @intCast(count);
    }
    return n;
}
pub fn removeOther(root: []const u8, keep: []const u8, id: u64) !void {
    const root_fd = openRoot(root);
    if (root_fd < 0) return if (platform.c.errno(root_fd) == .NOENT) {} else error.LearnedDirectoryRead;
    defer _ = platform.c.close(root_fd);
    return removeOtherAt(root_fd, keep, id);
}

pub fn removeOtherAt(root_fd: c_int, keep: []const u8, id: u64) !void {
    if (isCurrent(keep, id)) return;
    const dir_fd = openIdentity(root_fd, id);
    if (dir_fd < 0) return if (platform.c.errno(dir_fd) == .NOENT) {} else error.LearnedDirectoryRead;
    var leaf_buf: [17]u8 = undefined;
    const leaf = std.fmt.bufPrintSentinel(&leaf_buf, "{x:0>16}", .{id}, 0) catch unreachable;
    const d = platform.c.fdopendir(dir_fd) orelse {
        _ = platform.c.close(dir_fd);
        return error.LearnedDirectoryRead;
    };
    {
        defer _ = platform.c.closedir(d);
        var changed = false;
        while (platform.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            if (ent.type != platform.c.DT.REG) return error.LearnedDirectoryType;
            try unlinkAt(dir_fd, @ptrCast(&ent.name));
            changed = true;
        }
        if (changed) try syncFd(dir_fd);
    }
    const rc = platform.c.unlinkat(root_fd, leaf, platform.c.AT.REMOVEDIR);
    if (rc != 0 and platform.c.errno(rc) != .NOENT) return error.LearnedUnlink;
    if (rc == 0) try syncFd(root_fd);
}
test "acknowledged deletion refuses a directory and accepts an absent file" {
    var buf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&buf, "/tmp/tf-unlink-{d}", .{platform.c.getpid()}, 0);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(dir, 0o700));
    defer _ = platform.c.rmdir(dir);
    try std.testing.expectError(error.LearnedUnlink, unlink(dir));
    var missing: [160]u8 = undefined;
    try unlink(try std.fmt.bufPrintSentinel(&missing, "{s}/absent", .{dir}, 0));
}
