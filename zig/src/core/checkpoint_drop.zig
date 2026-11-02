//! Linux advice waits for TP load readiness; mappings can discard their own PTEs while peers still read the file.
const std = @import("std");
const Io = std.Io;
const linux = std.os.linux;
const enabled = @import("builtin").os.tag == .linux;

/// Process-wide, so loader helper threads share the same deferred advice and concurrent opens fail closed.
fn Queue(comptime Ops: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { id: u128, file: Ops.File };
        mutex: std.atomic.Mutex = .unlocked,
        users: usize = 0,
        failed: bool = false,
        files: std.ArrayList(Entry) = .empty,
        fn begin(q: *Self) void {
            while (!q.mutex.tryLock()) std.atomic.spinLoopHint();
            defer q.mutex.unlock();
            if (q.users == 0) {
                q.failed = false;
            }
            q.users += 1;
        }
        fn hold(q: *Self, file: Ops.File) bool {
            while (!q.mutex.tryLock()) std.atomic.spinLoopHint();
            defer q.mutex.unlock();
            if (q.users == 0) return false;
            const id = Ops.identity(file) catch return true;
            for (q.files.items) |entry| if (entry.id == id) return true;
            const copy = Ops.copy(file) catch return true;
            q.files.append(std.heap.page_allocator, .{ .id = id, .file = copy }) catch Ops.close(copy);
            return true;
        }
        fn deferring(q: *Self) bool {
            while (!q.mutex.tryLock()) std.atomic.spinLoopHint();
            defer q.mutex.unlock();
            return q.users > 0;
        }
        fn end(q: *Self, success: bool) void {
            while (!q.mutex.tryLock()) std.atomic.spinLoopHint();
            defer q.mutex.unlock();
            std.debug.assert(q.users > 0);
            q.failed = q.failed or !success;
            q.users -= 1;
            if (q.users > 0) return;
            for (q.files.items) |entry| {
                if (!q.failed) Ops.advise(entry.file);
                Ops.close(entry.file);
            }
            q.files.deinit(std.heap.page_allocator);
            q.files = .empty;
        }
    };
}
const Files = struct {
    pub const File = Io.File;
    fn identity(file: File) !u128 {
        var stat: linux.Statx = undefined;
        if (linux.errno(linux.statx(file.handle, "", linux.AT.EMPTY_PATH, .{ .INO = true }, &stat)) != .SUCCESS or
            !stat.mask.INO) return error.StatFailed;
        return (@as(u128, stat.dev_major) << 96) | (@as(u128, stat.dev_minor) << 64) | stat.ino;
    }
    fn copy(file: File) !File {
        const fd = linux.fcntl(file.handle, linux.F.DUPFD_CLOEXEC, 0);
        if (linux.errno(fd) != .SUCCESS) return error.DuplicateFailed;
        return .{ .handle = @intCast(fd), .flags = file.flags };
    }
    fn close(file: File) void {
        _ = linux.close(file.handle);
    }
    fn adviseRange(file: File, offset: u64, len: usize) void {
        _ = linux.fadvise(file.handle, @intCast(offset), @intCast(len), linux.POSIX_FADV.DONTNEED);
    }
    fn advise(file: File) void {
        _ = linux.fadvise(file.handle, 0, 0, linux.POSIX_FADV.DONTNEED);
    }
};
var pending: Queue(Files) = .{};

pub const Load = struct {
    active: bool,
    pub fn begin(_: Io, tp: bool) Load {
        if (!enabled or !tp) return .{ .active = false };
        pending.begin();
        return .{ .active = true };
    }
    /// Only after the all-rank load-ready exchange succeeds, including side heads and vision reads.
    pub fn finish(l: *Load) void {
        if (!l.active) return;
        l.active = false;
        pending.end(true);
    }
    pub fn finishAfter(l: *Load, args: anytype, comptime wait: anytype) !void {
        try after(l, args, wait);
    }
    /// A refusal or failed rendezvous keeps the page cache for ranks still loading; descriptors still close.
    pub fn deinit(l: *Load) void {
        if (!l.active) return;
        l.active = false;
        pending.end(false);
    }
};
fn after(load: anytype, args: anytype, comptime wait: anytype) !void {
    _ = try @call(.auto, wait, args);
    load.finish();
}
pub fn drop(file: Io.File) void {
    if (!enabled) return;
    if (!pending.hold(file)) Files.advise(file);
}

/// Range hash/read advice is suppressed during TP; the final owner selects only load-only files for the queue.
pub fn range(file: Io.File, offset: u64, len: usize) void {
    if (!enabled) return;
    rangeWhenReady(&pending, file, offset, len, Files.adviseRange);
}
fn rangeWhenReady(q: anytype, file: anytype, offset: u64, len: usize, comptime advise: anytype) void {
    if (q.deferring()) return;
    advise(file, offset, len);
}

test "checkpoint TP: two fake ranks finish all reads before advice, duplicate inodes close once" {
    const Fake = struct {
        const File = u32;
        var reads: [2]u32 = .{ 0, 0 };
        var advised: usize = 0;
        var ranged: usize = 0;
        var closed: usize = 0;
        fn identity(file: File) !u128 {
            if (file == 99) return error.StatFailed;
            return file;
        }
        fn copy(file: File) !File {
            if (file == 88) return error.DuplicateFailed;
            return file;
        }
        fn close(_: File) void {
            closed += 1;
        }
        fn adviseRange(_: File, _: u64, _: usize) void {
            ranged += 1;
        }
        fn advise(_: File) void {
            std.debug.assert(reads[0] == 3 and reads[1] == 3);
            advised += 1;
        }
    };
    Fake.advised = 0;
    Fake.closed = 0;
    Fake.ranged = 0;
    Fake.reads = .{ 0, 0 };
    var q: Queue(Fake) = .{};
    q.begin();
    q.begin();
    try std.testing.expect(q.deferring());
    rangeWhenReady(&q, @as(u32, 7), 0, 4096, Fake.adviseRange);
    try std.testing.expectEqual(@as(usize, 0), Fake.ranged);
    Fake.reads[0] = 3;
    try std.testing.expect(q.hold(7));
    try std.testing.expect(q.hold(7));
    try std.testing.expect(q.hold(99));
    try std.testing.expect(q.hold(88));
    q.end(true);
    try std.testing.expectEqual(@as(usize, 0), Fake.advised);
    Fake.reads[1] = 3;
    try std.testing.expect(q.hold(7));
    try std.testing.expect(q.hold(11));
    q.end(true);
    try std.testing.expectEqual(@as(usize, 2), Fake.advised);
    try std.testing.expectEqual(@as(usize, 2), Fake.closed);
    q.begin();
    q.begin();
    try std.testing.expect(q.hold(7));
    q.end(false);
    q.end(true);
    try std.testing.expectEqual(@as(usize, 2), Fake.advised);
    try std.testing.expectEqual(@as(usize, 3), Fake.closed);
    try std.testing.expect(!q.hold(11));
    try std.testing.expect(!q.deferring());
    rangeWhenReady(&q, @as(u32, 7), 0, 4096, Fake.adviseRange);
    try std.testing.expectEqual(@as(usize, 1), Fake.ranged);
}

test "checkpoint TP: separate fake rank queues wait for peer head reads and never advise on barrier failure" {
    const Fake = struct {
        const File = u32;
        var read: [2]bool = .{ false, false };
        var advised: usize = 0;
        var closed: usize = 0;
        fn identity(file: File) !u128 {
            return file;
        }
        fn copy(file: File) !File {
            return file;
        }
        fn close(_: File) void {
            closed += 1;
        }
        fn advise(_: File) void {
            std.debug.assert(read[0] and read[1]);
            advised += 1;
        }
        fn wait(fail: bool) !void {
            if (fail) return error.PeerFailed;
            if (!read[0] or !read[1]) return error.PeerStillReading;
        }
    };
    const Rank = struct {
        q: Queue(Fake) = .{},
        pub fn finish(rank: *@This()) void {
            rank.q.end(true);
        }
    };
    Fake.read = .{ true, false };
    Fake.advised = 0;
    Fake.closed = 0;
    var ranks = [_]Rank{ .{}, .{} };
    for (&ranks) |*rank| {
        rank.q.begin();
        try std.testing.expect(rank.q.hold(7));
    }
    try std.testing.expectError(error.PeerStillReading, after(&ranks[0], .{false}, Fake.wait));
    try std.testing.expectEqual(@as(usize, 0), Fake.advised);
    Fake.read[1] = true;
    try after(&ranks[0], .{false}, Fake.wait);
    try std.testing.expectError(error.PeerFailed, after(&ranks[1], .{true}, Fake.wait));
    ranks[1].q.end(false);
    try std.testing.expectEqual(@as(usize, 1), Fake.advised);
    try std.testing.expectEqual(@as(usize, 2), Fake.closed);
}
