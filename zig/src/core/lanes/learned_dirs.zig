//! Bounded-name learned directories: both ranks plan the same identity deletions before either writes.
const std = @import("std");
pub fn rootOf(dir: []const u8) []const u8 {
    return std.fs.path.dirname(dir) orelse ".";
}
fn path(buf: []u8, root: []const u8, id: u64) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}", .{ root, id }, 0);
}
fn isCurrent(keep: []const u8, id: u64) bool {
    var name: [16]u8 = undefined;
    const formatted = std.fmt.bufPrint(&name, "{x:0>16}", .{id}) catch unreachable;
    return std.mem.eql(u8, std.fs.path.basename(keep), formatted);
}
pub fn next(root: []const u8, keep: []const u8, after: ?u64) !?u64 {
    var buf: [1100]u8 = undefined;
    const d = std.c.opendir(try std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0)) orelse return error.LearnedDirectoryRead;
    defer _ = std.c.closedir(d);
    var out: ?u64 = null;
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name.len != 16) continue;
        const id = std.fmt.parseInt(u64, name, 16) catch continue;
        if (id == std.math.maxInt(u64)) return error.LearnedDirectoryType;
        if (ent.type != std.c.DT.DIR) return error.LearnedDirectoryType;
        if (isCurrent(keep, id) or (after != null and id <= after.?)) continue;
        if (out == null or id < out.?) out = id;
    }
    return out;
}
pub fn unlink(file: [:0]const u8) !void {
    const rc = std.c.unlink(file);
    if (rc != 0 and std.c.errno(rc) != .NOENT) return error.LearnedUnlink;
}

fn stateKey(name: []const u8) ?u64 {
    if (name.len < 16) return null;
    for (name[0..16]) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return null;
    const suffix = name[16..];
    if (!std.mem.eql(u8, suffix, ".bin") and !std.mem.eql(u8, suffix, ".r0.bin") and !std.mem.eql(u8, suffix, ".r1.bin")) return null;
    return std.fmt.parseInt(u64, name[0..16], 16) catch null;
}

fn partName(name: []const u8) bool {
    if (std.mem.eql(u8, name, "index.part") or std.mem.eql(u8, name, "pending.part") or std.mem.eql(u8, name, "used.part")) return true;
    return std.mem.endsWith(u8, name, ".part") and stateKey(name[0 .. name.len - 5]) != null;
}

fn identityName(name: []const u8) bool {
    if (name.len != 16) return false;
    for (name) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return false;
    return true;
}

fn unlinkAt(fd: c_int, name: [*:0]const u8) !void {
    const rc = std.c.unlinkat(fd, name, 0);
    if (rc != 0 and std.c.errno(rc) != .NOENT) return error.LearnedUnlink;
}

/// Only the configured root may be an alias; cleanup never follows an identity or file link.
pub fn cleanupParts(root: []const u8) !void {
    var buf: [1100]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0), .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.LearnedDirectoryRead;
    const d = std.c.fdopendir(fd) orelse {
        _ = std.c.close(fd);
        return error.LearnedDirectoryRead;
    };
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (ent.type == std.c.DT.REG and std.mem.eql(u8, name, "clock.part")) {
            try unlinkAt(fd, @ptrCast(&ent.name));
            continue;
        }
        if (!identityName(name)) continue;
        if (ent.type != std.c.DT.DIR) return error.LearnedDirectoryType;
        const sub_fd = std.c.openat(fd, @ptrCast(&ent.name), .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
        if (sub_fd < 0) return error.LearnedDirectoryRead;
        const sub = std.c.fdopendir(sub_fd) orelse {
            _ = std.c.close(sub_fd);
            return error.LearnedDirectoryRead;
        };
        {
            defer _ = std.c.closedir(sub);
            while (std.c.readdir(sub)) |file| {
                if (file.type == std.c.DT.REG and partName(std.mem.sliceTo(&file.name, 0))) try unlinkAt(sub_fd, @ptrCast(&file.name));
            }
        }
    }
}

/// Payloads have no token metadata to recover a lost index entry; only exact owned, unreferenced names go.
pub fn removeUnindexed(dir: [:0]const u8, metas: anytype) !void {
    const fd = std.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.LearnedDirectoryRead;
    const d = std.c.fdopendir(fd) orelse {
        _ = std.c.close(fd);
        return error.LearnedDirectoryRead;
    };
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        if (ent.type != std.c.DT.REG) continue;
        const name = std.mem.sliceTo(&ent.name, 0);
        const key = stateKey(name) orelse continue;
        const indexed = for (metas) |x| {
            if (x.key == key) break true;
        } else false;
        if (indexed) continue;
        try unlinkAt(fd, @ptrCast(&ent.name));
    }
}
pub fn bytes(dir: [:0]const u8) !u64 {
    const d = std.c.opendir(dir) orelse return if (std.c.errno(-1) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = std.c.closedir(d);
    var n: u64 = 0;
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        if (ent.type != std.c.DT.REG) return error.LearnedDirectoryType;
        var buf: [1400]u8 = undefined;
        const f = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0);
        const fd = std.c.open(f, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.LearnedDirectoryRead;
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        _ = std.c.close(fd);
        if (end < 0) return error.LearnedDirectoryRead;
        n = try std.math.add(u64, n, @intCast(end));
    }
    return n;
}
pub fn fileBytes(file: [:0]const u8) !u64 {
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c.errno(fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.LearnedDirectoryRead;
    return @intCast(end);
}
pub fn otherBytes(root: []const u8, keep: []const u8, id: u64) !u64 {
    if (isCurrent(keep, id)) return 0;
    var buf: [1100]u8 = undefined;
    return bytes(try path(&buf, root, id));
}
pub fn totalOthers(root: []const u8, keep: []const u8) !u64 {
    var cursor: ?u64 = null;
    var n: u64 = 0;
    while (try next(root, keep, cursor)) |id| {
        n = try std.math.add(u64, n, try otherBytes(root, keep, id));
        cursor = id;
    }
    return n;
}
pub fn total(root: []const u8) !u64 {
    return totalOthers(root, "-");
}
pub fn used(root: []const u8, id: u64) !u64 {
    var buf: [1100]u8 = undefined;
    const f = try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}/used", .{ root, id }, 0);
    const fd = std.c.open(f, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c.errno(fd) == .NOENT) 0 else error.LearnedDirectoryRead;
    defer _ = std.c.close(fd);
    var n: u64 = 0;
    const data = std.mem.asBytes(&n);
    var done: usize = 0;
    while (done < data.len) {
        const count = std.c.read(fd, data.ptr + done, data.len - done);
        if (count < 0) {
            if (std.c.errno(count) == .INTR) continue;
            return error.LearnedDirectoryRead;
        }
        if (count == 0) return 0; // A torn recency hint is oldest; it cannot make a learned payload valid.
        done += @intCast(count);
    }
    return n;
}
pub fn removeOther(root: []const u8, keep: []const u8, id: u64) !void {
    if (isCurrent(keep, id)) return;
    var buf: [1100]u8 = undefined;
    const dir = try path(&buf, root, id);
    _ = try bytes(dir);
    const d = std.c.opendir(dir) orelse return if (std.c.errno(-1) == .NOENT) {} else error.LearnedDirectoryRead;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        if (ent.type != std.c.DT.REG) return error.LearnedDirectoryType;
        var f: [1400]u8 = undefined;
        try unlink(try std.fmt.bufPrintSentinel(&f, "{s}/{s}", .{ dir, name }, 0));
    }
    const rc = std.c.rmdir(dir);
    if (rc != 0 and std.c.errno(rc) != .NOENT) return error.LearnedUnlink;
}
test "acknowledged deletion refuses a directory and accepts an absent file" {
    var buf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&buf, "/tmp/tf-unlink-{d}", .{std.c.getpid()}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(dir, 0o700));
    defer _ = std.c.rmdir(dir);
    try std.testing.expectError(error.LearnedUnlink, unlink(dir));
    var missing: [160]u8 = undefined;
    try unlink(try std.fmt.bufPrintSentinel(&missing, "{s}/absent", .{dir}, 0));
}
