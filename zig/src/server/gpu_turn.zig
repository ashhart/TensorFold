//! Speech holds a shared flock; batch work yields between short command buffers and tracks speech generations.
const std = @import("std");

pub const path = "/tmp/tensorfold-speech.lock";

/// Where batch work gives way: the speech lock (tests point `at` at their own).
pub const Turn = struct { io: std.Io, at: []const u8 = path };

/// How often waiting work looks again; one poll is the most speech's end can be missed by.
const poll: std.Io.Duration = .fromMilliseconds(20);

/// Longest speech waits behind an exclusive hold: batch probes take microseconds, so a longer one is not a probe.
const patience: std.Io.Duration = .fromSeconds(1);

/// Shared hold; best effort, so speech goes on without one when the lock cannot be opened or stays held.
pub fn speak(io: std.Io, at: []const u8) ?std.Io.File {
    defer stamp(io, at);
    const file = open(io, at) catch return null;
    const until = std.Io.Clock.awake.now(io).addDuration(patience);
    while (true) {
        if (file.tryLock(io, .shared) catch break) return file;
        if (std.Io.Clock.awake.now(io).compare(.gte, until)) break;
        io.sleep(poll, .awake) catch break;
    }
    file.close(io);
    return null;
}

pub fn done(io: std.Io, held: ?std.Io.File) void {
    if (held) |file| file.close(io);
}

/// Seconds actually waited, not an overlap detector: a whole request may finish between two polls.
pub fn yield(io: std.Io, at: []const u8) f64 {
    const file = open(io, at) catch return 0;
    defer file.close(io);
    if (free(io, file)) return 0;
    const started = std.Io.Clock.awake.now(io);
    while (!free(io, file)) io.sleep(poll, .awake) catch break;
    return @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
}

/// Whether no speech holds the turn. Speech only shares it, so a hold that also refuses a shared probe is no voice
/// and is not waited for; an error counts as free, as speech never depends on batch work seeing it.
fn free(io: std.Io, file: std.Io.File) bool {
    if (file.tryLock(io, .exclusive) catch return true) {
        file.unlock(io);
        return true;
    }
    if (!(file.tryLock(io, .shared) catch return true)) return true;
    file.unlock(io);
    return false;
}

/// Read-only so the speech server's and a render's users can share one file; flock needs no write access.
/// Created only when absent, else `existing`: a name planted in the shared directory stays inert.
fn open(io: std.Io, at: []const u8) !std.Io.File {
    const dir = std.Io.Dir.cwd();
    return existing(io, at, .read_only) catch |e| switch (e) {
        error.FileNotFound => {
            const file = dir.createFile(io, at, .{ .read = true, .exclusive = true }) catch |create_err| {
                if (create_err == error.PathAlreadyExists) return existing(io, at, .read_only);
                return create_err;
            };
            errdefer file.close(io);
            try file.setPermissions(io, .fromMode(0o644)); // past the creator's umask, so the other user opens it
            return file;
        },
        else => e,
    };
}

pub const Mark = struct { inode: std.Io.File.INode, bytes: [16]u8 };

/// Snapshot before a timed region; missing instrumentation or active speech cannot certify a clean sample.
pub fn mark(io: std.Io, at: []const u8) ?Mark {
    const file = marker(io, at) catch return null;
    defer file.close(io);
    const stat = file.stat(io) catch return null;
    if (stat.permissions.toMode() & 0o666 != 0o666) return null;
    var result: Mark = .{ .inode = stat.inode, .bytes = undefined };
    if ((file.readPositionalAll(io, &result.bytes, 0) catch return null) != result.bytes.len) return null;
    const lock = open(io, at) catch return null;
    defer lock.close(io);
    if (!(lock.tryLock(io, .exclusive) catch return null)) return null;
    lock.unlock(io);
    return result;
}

/// False also means unknown: never publish a timing as clean when the speech signal cannot be read.
pub fn clean(io: std.Io, at: []const u8, before: ?Mark) bool {
    const first = before orelse return false;
    const last = mark(io, at) orelse return false;
    return first.inode == last.inode and std.mem.eql(u8, &first.bytes, &last.bytes);
}

fn stamp(io: std.Io, at: []const u8) void {
    const file = marker(io, at) catch return;
    defer file.close(io);
    var bytes: [16]u8 = undefined;
    io.random(&bytes);
    file.writePositionalAll(io, &bytes, 0) catch {
        file.setLength(io, 0) catch {}; // leave an unreadable generation rather than certify the stale one
    };
}

// Separate from the read-only flock: both users can publish activity without replacing a live lock's inode.
fn marker(io: std.Io, at: []const u8) !std.Io.File {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}.stamp", .{at});
    const dir = std.Io.Dir.cwd();
    return existing(io, name, .read_write) catch |e| switch (e) {
        error.FileNotFound => {
            const file = dir.createFile(io, name, .{ .read = true, .exclusive = true }) catch |create_err| {
                if (create_err == error.PathAlreadyExists) return existing(io, name, .read_write);
                return create_err;
            };
            errdefer file.close(io);
            try file.setPermissions(io, .fromMode(0o666));
            var bytes: [16]u8 = undefined;
            io.random(&bytes);
            try file.writePositionalAll(io, &bytes, 0);
            return file;
        },
        else => e,
    };
}

/// A file already at `name`, only when it is one regular file with no other name: never through a symlink, and
/// nonblocking, so a planted FIFO cannot stall the open and a planted hard link cannot redirect the stamp's writes.
fn existing(io: std.Io, name: []const u8, mode: std.Io.Dir.OpenFileOptions.Mode) !std.Io.File {
    const file: std.Io.File = if (@import("builtin").os.tag == .windows)
        try std.Io.Dir.cwd().openFile(io, name, .{ .mode = mode, .follow_symlinks = false })
    else
        .{ .handle = try std.posix.openat(std.posix.AT.FDCWD, name, .{
            .ACCMODE = if (mode == .read_write) .RDWR else .RDONLY,
            .NOFOLLOW = true,
            .NONBLOCK = true,
            .CLOEXEC = true,
        }, 0), .flags = .{ .nonblocking = false } };
    errdefer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file or stat.nlink != 1) return error.NotRegularFile;
    return file;
}

/// Committed command buffers, waited in commit order. No error returns before every one has completed, so a
/// failure never frees or reuses buffers the GPU still reads. `C` has `wait`, `failure` and `gpuSeconds`.
pub fn Queue(comptime C: type, comptime capacity: usize) type {
    return struct {
        ring: [capacity]C = undefined,
        first: usize = 0,
        n: usize = 0,
        /// GPU seconds of the buffers waited so far.
        gpu: f64 = 0,
        failed: bool = false,

        const Self = @This();

        /// Keeps `cb`, then waits the oldest until at most `ahead` remain unwaited.
        pub fn add(q: *Self, cb: C, ahead: usize) !void {
            if (q.n == capacity) q.waitOldest();
            q.ring[(q.first + q.n) % capacity] = cb;
            q.n += 1;
            while (q.n > ahead) q.waitOldest();
            if (q.failed) return q.drain();
        }

        /// Waits every buffer still kept; GpuFailed if any of them, or any waited earlier, failed.
        pub fn drain(q: *Self) !void {
            while (q.n > 0) q.waitOldest();
            if (q.failed) return error.GpuFailed;
        }

        fn waitOldest(q: *Self) void {
            const cb = q.ring[q.first];
            q.first = (q.first + 1) % capacity;
            q.n -= 1;
            cb.wait();
            if (cb.failure()) |text| {
                std.debug.print("command buffer failed: {s}\n", .{text});
                q.failed = true;
            } else q.gpu += cb.gpuSeconds();
        }
    };
}

test "batch work waits for every speech hold to end, and not at all without one" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const at = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(at);
    const lock = try std.fs.path.join(std.testing.allocator, &.{ at, "speech.lock" });
    defer std.testing.allocator.free(lock);

    try std.testing.expectEqual(@as(f64, 0), yield(io, lock));
    const first = speak(io, lock) orelse return error.NoHold;
    const second = speak(io, lock) orelse return error.NoHold; // two voices at once share the turn
    const Ender = struct {
        fn run(h: [2]std.Io.File) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(150), .awake) catch {};
            done(std.testing.io, h[0]);
            std.Io.sleep(std.testing.io, .fromMilliseconds(150), .awake) catch {};
            done(std.testing.io, h[1]);
        }
    };
    const ender = try std.Thread.spawn(.{}, Ender.run, .{[2]std.Io.File{ first, second }});
    const waited = yield(io, lock);
    ender.join();
    try std.testing.expect(waited >= 0.29);
    try std.testing.expectEqual(@as(f64, 0), yield(io, lock));
}

test "speech waits out a batch probe, and the file it makes is open to every user" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const at = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(at);
    const lock = try std.fs.path.join(std.testing.allocator, &.{ at, "speech.lock" });
    defer std.testing.allocator.free(lock);

    const prober = try open(io, lock);
    defer prober.close(io);
    if (@import("builtin").os.tag != .windows) try std.testing.expectEqual(@as(u32, 0o644), (try prober.stat(io)).permissions.toMode() & 0o777);
    const before = mark(io, lock);
    // a probe as `free` and `mark` take it, descheduled for 200 ms while speech arrives
    try prober.lock(io, .exclusive);
    const Release = struct {
        fn run(file: std.Io.File) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(200), .awake) catch {};
            file.unlock(std.testing.io);
        }
    };
    const thread = try std.Thread.spawn(.{}, Release.run, .{prober});
    const turn = speak(io, lock);
    thread.join();
    const held = turn orelse return error.NoHold;
    defer done(io, held);
    try std.testing.expect(!free(io, prober));
    try std.testing.expect(!clean(io, lock, before));
}

test "a planted lock path neither redirects the files speech makes nor holds speech up for long" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const at = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(at);
    const lock = try std.fs.path.join(std.testing.allocator, &.{ at, "speech.lock" });
    defer std.testing.allocator.free(lock);

    // dangling symlinks where the lock and its stamp go: nothing is created or chmodded at their targets
    try tmp.dir.symLink(io, "victim", "speech.lock", .{});
    try tmp.dir.symLink(io, "victim.stamp", "speech.lock.stamp", .{});
    try std.testing.expect(speak(io, lock) == null);
    try std.testing.expectEqual(@as(f64, 0), yield(io, lock));
    try std.testing.expect(mark(io, lock) == null);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "victim", .{}));
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "victim.stamp", .{}));

    // a FIFO for the lock and a hard link for the stamp: neither stalls speech, and the link's target keeps its bytes
    try tmp.dir.deleteFile(io, "speech.lock");
    try tmp.dir.deleteFile(io, "speech.lock.stamp");
    var fifo_buf: [std.fs.max_path_bytes:0]u8 = undefined;
    try std.testing.expectEqual(@as(c_int, 0), mkfifo(try std.fmt.bufPrintSentinel(&fifo_buf, "{s}", .{lock}, 0), 0o666));
    try tmp.dir.writeFile(io, .{ .sub_path = "victim", .data = "kept" });
    try tmp.dir.hardLink("victim", tmp.dir, "speech.lock.stamp", io, .{});
    const planted = std.Io.Clock.awake.now(io);
    try std.testing.expect(speak(io, lock) == null);
    try std.testing.expectEqual(@as(f64, 0), yield(io, lock));
    try std.testing.expect(mark(io, lock) == null);
    try std.testing.expect(planted.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds() < std.time.ns_per_s);
    var kept: [8]u8 = undefined;
    try std.testing.expectEqualStrings("kept", try tmp.dir.readFile(io, "victim", &kept));

    // a real lock someone else holds exclusively and never lets go: speech still starts, without the turn, and
    // batch work does not wait for a holder that is no voice
    try tmp.dir.deleteFile(io, "speech.lock");
    try tmp.dir.deleteFile(io, "speech.lock.stamp");
    const squatter = try open(io, lock);
    defer squatter.close(io);
    try squatter.lock(io, .exclusive);
    const started = std.Io.Clock.awake.now(io);
    try std.testing.expect(speak(io, lock) == null);
    try std.testing.expect(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds() < 3 * std.time.ns_per_s);
    try std.testing.expectEqual(@as(f64, 0), yield(io, lock));
}

extern "c" fn mkfifo(path: [*:0]const u8, mode: std.c.mode_t) c_int;

test "timed work excludes speech completed entirely between polls and preexisting speech" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const at = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(at);
    const lock = try std.fs.path.join(std.testing.allocator, &.{ at, "speech.lock" });
    defer std.testing.allocator.free(lock);

    const before = mark(io, lock);
    try std.testing.expect(clean(io, lock, before));
    const held = speak(io, lock) orelse return error.NoHold;
    const during = mark(io, lock);
    done(io, held);
    try std.testing.expectEqual(@as(f64, 0), yield(io, lock));
    try std.testing.expect(!clean(io, lock, before));
    try std.testing.expect(!clean(io, lock, during));
    try std.testing.expect(clean(io, lock, mark(io, lock)));
    const Voice = struct {
        fn run(path_: []const u8) void {
            for (0..8) |_| {
                const turn = speak(std.testing.io, path_);
                done(std.testing.io, turn);
            }
        }
    };
    const concurrent = mark(io, lock);
    const first = try std.Thread.spawn(.{}, Voice.run, .{lock});
    const second = try std.Thread.spawn(.{}, Voice.run, .{lock});
    first.join();
    second.join();
    try std.testing.expect(!clean(io, lock, concurrent));

    const last = mark(io, lock);
    try tmp.dir.deleteFile(io, "speech.lock.stamp");
    try std.testing.expect(!clean(io, lock, last));
    try std.testing.expect(!clean(io, "/nonexistent/speech.lock", null));
    const file = try marker(io, lock);
    defer file.close(io);
    try file.setLength(io, 0);
    try std.testing.expect(!clean(io, lock, mark(io, lock)));
    if (@import("builtin").os.tag != .windows) {
        try file.setPermissions(io, .fromMode(0o600));
        try std.testing.expect(!clean(io, lock, mark(io, lock)));
    }
}

test "a failed command buffer is reported only after every buffer committed with it has completed" {
    const Fake = struct {
        bad: bool = false,
        waited: *usize,
        fn wait(c: @This()) void {
            c.waited.* += 1;
        }
        fn failure(c: @This()) ?[*:0]const u8 {
            return if (c.bad) "fault" else null;
        }
        fn gpuSeconds(_: @This()) f64 {
            return 1;
        }
    };
    var waited: usize = 0;
    var q: Queue(Fake, 4) = .{};
    for (0..5) |_| try q.add(.{ .waited = &waited }, 3); // at most three left unwaited
    try std.testing.expectEqual(@as(usize, 2), waited);
    try std.testing.expectEqual(@as(f64, 2), q.gpu);
    // the oldest fails while three committed after it still run: the error waits for all of them
    try q.drain();
    try q.add(.{ .waited = &waited, .bad = true }, 3);
    for (0..2) |_| try q.add(.{ .waited = &waited }, 3);
    try std.testing.expectEqual(@as(usize, 5), waited);
    try std.testing.expectError(error.GpuFailed, q.add(.{ .waited = &waited }, 3));
    try std.testing.expectEqual(@as(usize, 9), waited);
    try std.testing.expectEqual(@as(usize, 0), q.n);

    var late: Queue(Fake, 4) = .{};
    try late.add(.{ .waited = &waited, .bad = true }, 3);
    try late.add(.{ .waited = &waited }, 3);
    try std.testing.expectError(error.GpuFailed, late.drain());
    try std.testing.expectEqual(@as(usize, 11), waited);
}
