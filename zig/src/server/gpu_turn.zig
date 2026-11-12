//! Speech holds a shared flock; batch work yields between short command buffers and tracks speech generations.
const std = @import("std");

pub const path = "/tmp/tensorfold-speech.lock";

/// Where batch work gives way: the speech lock (tests point `at` at their own).
pub const Turn = struct { io: std.Io, at: []const u8 = path };

/// How often waiting batch work looks again; one poll is the most speech's end can be missed by.
const poll: std.Io.Duration = .fromMilliseconds(20);

/// Shared hold; best effort only when the lock file cannot be opened. It blocks only behind batch work's exclusive
/// probes, which `free` and `mark` take with tryLock and release at once, never across GPU work.
pub fn speak(io: std.Io, at: []const u8) ?std.Io.File {
    defer stamp(io, at);
    const file = open(io, at) catch return null;
    file.lock(io, .shared) catch {
        file.close(io);
        return null;
    };
    return file;
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

/// Whether no speech holds the turn; an error counts as free, as speech never depends on batch work seeing it.
fn free(io: std.Io, file: std.Io.File) bool {
    const got = file.tryLock(io, .exclusive) catch return true;
    if (got) file.unlock(io);
    return got;
}

/// Read-only so the speech server's and a render's users can share one file; flock needs no write access.
fn open(io: std.Io, at: []const u8) !std.Io.File {
    const dir = std.Io.Dir.cwd();
    return dir.openFile(io, at, .{}) catch |e| switch (e) {
        error.FileNotFound => {
            const file = try dir.createFile(io, at, .{ .read = true, .truncate = false });
            file.setPermissions(io, .fromMode(0o644)) catch {}; // past the creator's umask, so the other user opens it
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
    if (stat.kind != .file or stat.permissions.toMode() & 0o666 != 0o666) return null;
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
    return dir.openFile(io, name, .{ .mode = .read_write, .follow_symlinks = false }) catch |e| switch (e) {
        error.FileNotFound => {
            const file = dir.createFile(io, name, .{ .read = true, .exclusive = true }) catch |create_err| {
                if (create_err == error.PathAlreadyExists) return dir.openFile(io, name, .{ .mode = .read_write, .follow_symlinks = false });
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

test "speech waits out a batch probe however long it is held, and the file it makes is open to every user" {
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
