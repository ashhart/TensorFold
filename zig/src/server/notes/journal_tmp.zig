//! A journal rewrite's temporary file: a fresh name each time, made exclusively, and only ours ever removed.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

/// A temporary file this process made: its path, descriptor, and the file it is (device, inode).
pub const Tmp = struct {
    path: [:0]u8,
    fd: c_int,
    dev: i128,
    ino: i128,
};

/// What a file is: its device and inode, and whether it is a folder.
pub const Id = struct { dev: i128, ino: i128, dir: bool };

/// The open file `fd`, or with `fd` < 0 the name `path` itself (a link not followed); null when it cannot be read.
pub fn idOf(fd: c_int, path: [*:0]const u8) ?Id {
    return if (builtin.os.tag == .linux) idStatx(fd, path) else idStat(fd, path);
}

/// Zig 0.17's std.c has no fstat or fstatat on Linux: statx, of the descriptor (an empty path) or of the name.
fn idStatx(fd: c_int, path: [*:0]const u8) ?Id {
    var x: std.os.linux.Statx = undefined;
    const want: std.os.linux.STATX = .{ .TYPE = true, .INO = true };
    const rc = if (fd >= 0)
        std.c.statx(fd, "", std.c.AT.EMPTY_PATH, want, &x)
    else
        std.c.statx(std.c.AT.FDCWD, path, std.c.AT.SYMLINK_NOFOLLOW, want, &x);
    if (rc != 0 or !x.mask.TYPE or !x.mask.INO) return null;
    return .{ .dev = (@as(i128, x.dev_major) << 32) | x.dev_minor, .ino = x.ino, .dir = x.mode & std.c.S.IFMT == std.c.S.IFDIR };
}

/// Elsewhere (macOS): fstat of the descriptor, or fstatat of the name without following a link, as before.
fn idStat(fd: c_int, path: [*:0]const u8) ?Id {
    var st: std.c.Stat = undefined;
    const rc = if (fd >= 0) std.c.fstat(fd, &st) else std.c.fstatat(std.c.AT.FDCWD, path, &st, std.c.AT.SYMLINK_NOFOLLOW);
    if (rc != 0) return null;
    return .{ .dev = st.dev, .ino = st.ino, .dir = st.mode & std.c.S.IFMT == std.c.S.IFDIR };
}

var counter: std.atomic.Value(u64) = .init(0);

/// Tests only: the random part of every name, so a test can plant links at the names a rewrite will try.
pub var fixed_random: ?u64 = null;

/// The counter the next name gets (tests: the names a rewrite will try are NEXT, NEXT+1, ...).
pub fn next() u64 {
    return counter.load(.monotonic);
}

/// 8 bytes from the system's random source: arc4random_buf, else getentropy, else the clock and the counter.
fn random() u64 {
    if (fixed_random) |r| return r;
    var x: u64 = 0;
    const bytes: [*]u8 = @ptrCast(&x);
    if (comptime @TypeOf(std.c.arc4random_buf) != void) {
        std.c.arc4random_buf(bytes, @sizeOf(u64));
        return x;
    } else if (comptime @TypeOf(std.c.getentropy) != void) {
        if (std.c.getentropy(bytes, @sizeOf(u64)) == 0) return x;
    }
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @bitCast(@as(i64, ts.nsec))) *% 0x9E3779B97F4A7C15 ^ counter.load(.monotonic);
}

/// PATH.tmp.PID.COUNTER.RANDOM, opened O_CREAT|O_EXCL|O_NOFOLLOW at 0600: a link or file planted there is refused.
pub fn create(a: Allocator, path: []const u8) error{ OutOfMemory, JournalIo }!Tmp {
    var tries: usize = 0;
    while (tries < 16) : (tries += 1) {
        const n = counter.fetchAdd(1, .monotonic);
        const name = try std.fmt.allocPrintSentinel(a, "{s}.tmp.{d}.{d}.{x:0>16}", .{ path, std.c.getpid(), n, random() }, 0);
        const fd = std.c.open(name, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd >= 0) {
            const id = idOf(fd, "") orelse {
                _ = std.c.close(fd);
                _ = std.c.unlink(name); // it is ours: made exclusively a moment ago
                a.free(name);
                return error.JournalIo;
            };
            return .{ .path = name, .fd = fd, .dev = id.dev, .ino = id.ino };
        }
        a.free(name);
        if (std.c.errno(fd) != .EXIST) return error.JournalIo;
    }
    return error.JournalIo;
}

/// Closes it and removes its name, only while that name is still this very file (never what replaced it).
pub fn removeOwn(t: Tmp) void {
    _ = std.c.close(t.fd);
    const id = idOf(-1, t.path) orelse return;
    if (id.dev == t.dev and id.ino == t.ino) _ = std.c.unlink(t.path);
}

/// Whether `name` is one of ours for the journal named `base`: base.tmp (before unique names) or base.tmp.P.C.R.
pub fn isOurs(base: []const u8, name: []const u8) bool {
    if (!std.mem.startsWith(u8, name, base) or !std.mem.startsWith(u8, name[base.len..], ".tmp")) return false;
    const rest = name[base.len + ".tmp".len ..];
    if (rest.len == 0) return true;
    var parts = std.mem.splitScalar(u8, rest, '.');
    if ((parts.next() orelse return false).len != 0) return false;
    for (0..2) |_| {
        const p = parts.next() orelse return false;
        if (p.len == 0) return false;
        for (p) |c| if (!std.ascii.isDigit(c)) return false;
    }
    const r = parts.next() orelse return false;
    if (r.len != 16 or parts.next() != null) return false;
    for (r) |c| if (!std.ascii.isHex(c) or std.ascii.isUpper(c)) return false;
    return true;
}

/// Removes the stale temporaries of `path` (a rewrite that died); only with its lock held; links not followed.
pub fn cleanStale(a: Allocator, path: []const u8) error{OutOfMemory}!usize {
    const dir = std.fs.path.dirname(path) orelse ".";
    const base = std.fs.path.basename(path);
    const dirz = try a.dupeSentinel(u8, dir, 0);
    defer a.free(dirz);
    const d = std.c.opendir(dirz) orelse return 0;
    defer _ = std.c.closedir(d);
    var gone: std.ArrayList([:0]u8) = .empty;
    defer {
        for (gone.items) |g| a.free(g);
        gone.deinit(a);
    }
    while (std.c.readdir(d)) |e| {
        const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&e.name)), 0);
        if (!isOurs(base, name)) continue;
        try gone.append(a, try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0));
    }
    var n: usize = 0;
    for (gone.items) |g| {
        const id = idOf(-1, g) orelse continue;
        if (id.dir) continue;
        if (std.c.unlink(g) == 0) n += 1;
    }
    return n;
}

test "our temporary names, and only them" {
    try std.testing.expect(isOurs("snap.json.journal", "snap.json.journal.tmp"));
    try std.testing.expect(isOurs("snap.json.journal", "snap.json.journal.tmp.123.0.0123456789abcdef"));
    try std.testing.expect(!isOurs("snap.json.journal", "snap.json.journal"));
    try std.testing.expect(!isOurs("snap.json.journal", "snap.json.journal.tmpx"));
    try std.testing.expect(!isOurs("snap.json.journal", "snap.json.journal.tmp.123.0.0123456789ABCDEF"));
    try std.testing.expect(!isOurs("snap.json.journal", "snap.json.journal.tmp.123.0.0123456789abcdef.x"));
    try std.testing.expect(!isOurs("snap.json.journal", "other.journal.tmp"));
}
