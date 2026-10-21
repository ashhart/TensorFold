//! Real file mutations, corrupt entries and competing processes exercise the shared digest cache.
const std = @import("std");
const cache = @import("digest_cache.zig");
const t = std.testing;
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

test {
    _ = @import("digest_parallel_test.zig");
}

const Reader = struct {
    io: Io = t.io,
    calls: ?*usize = null,
    change: bool = false,
    pub fn read(r: Reader, file: Io.File, size: u64) !cache.Value {
        if (r.calls) |n| n.* += 1;
        var h = Sha256.init(.{});
        var at: u64 = 0;
        var bytes: [4096]u8 = undefined;
        while (at < size) {
            const n = try file.readPositionalAll(r.io, bytes[0..@intCast(@min(bytes.len, size - at))], at);
            if (n == 0) return error.FileShrank;
            h.update(bytes[0..n]);
            at += n;
        }
        if (r.change) try file.setTimestamps(r.io, .{ .modify_timestamp = .{ .new = .fromNanoseconds(1) } });
        return .{ .sha256 = h.finalResult() };
    }
};

fn path(a: std.mem.Allocator, tmp: *const t.TmpDir, sub: []const u8) ![]u8 {
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    return std.fs.path.join(a, &.{ root, sub });
}

fn get(c: cache.Cache, file: []const u8, calls: *usize) !cache.Result {
    return c.get(t.allocator, t.io, file, false, Reader{ .calls = calls });
}

test "digest cache reads once until mtime or size changes" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "file", .data = "first" });
    const root = try path(t.allocator, &tmp, "cache");
    defer t.allocator.free(root);
    const file = try path(t.allocator, &tmp, "file");
    defer t.allocator.free(file);
    const c = cache.Cache{ .root = root };
    var calls: usize = 0;
    const first = try get(c, file, &calls);
    try t.expect(!first.hit);
    try t.expect((try get(c, file, &calls)).hit);
    try t.expectEqual(@as(usize, 1), calls);
    const f = try tmp.dir.openFile(t.io, "file", .{ .mode = .read_write });
    defer f.close(t.io);
    try f.setTimestamps(t.io, .{ .modify_timestamp = .{ .new = .fromNanoseconds(@intCast(first.key.mtime + 1_000_000_000)) } });
    try t.expect(!(try get(c, file, &calls)).hit);
    try t.expectEqual(@as(usize, 2), calls);
    try f.writePositionalAll(t.io, "more", 5);
    const bigger = try get(c, file, &calls);
    try t.expect(!bigger.hit);
    try t.expectEqual(@as(usize, 3), calls);
    try t.expect(!std.mem.eql(u8, &first.value.sha256, &bigger.value.sha256));
}

test "digest cache detects same-size rewrites even when mtime is restored" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "file", .data = "first" });
    const root = try path(t.allocator, &tmp, "cache");
    defer t.allocator.free(root);
    const file = try path(t.allocator, &tmp, "file");
    defer t.allocator.free(file);
    const c = cache.Cache{ .root = root };
    var calls: usize = 0;
    const old = try get(c, file, &calls);
    const f = try tmp.dir.openFile(t.io, "file", .{ .mode = .read_write });
    defer f.close(t.io);
    try f.writePositionalAll(t.io, "other", 0);
    try f.setTimestamps(t.io, .{ .modify_timestamp = .{ .new = .fromNanoseconds(@intCast(old.key.mtime)) } });
    const now = try get(c, file, &calls);
    try t.expectEqual(old.key.mtime, now.key.mtime);
    try t.expect(!now.hit and calls == 2);
    try t.expect(!std.mem.eql(u8, &old.value.sha256, &now.value.sha256));
}

test "digest cache refuses changed device inode or key fields with a valid checksum" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "file", .data = "first" });
    const root = try path(t.allocator, &tmp, "cache");
    defer t.allocator.free(root);
    const file = try path(t.allocator, &tmp, "file");
    defer t.allocator.free(file);
    const c = cache.Cache{ .root = root };
    var calls: usize = 0;
    const good = try get(c, file, &calls);
    var dir = try Io.Dir.cwd().openDir(t.io, root, .{});
    defer dir.close(t.io);
    inline for (.{ "device", "inode", "size", "mtime", "ctime" }) |field| {
        var changed = good.key;
        @field(changed, field) += 1;
        const bytes = try cache.encode(t.allocator, &.{.{ .key = changed, .value = .{ .sha256 = @splat(9) } }});
        defer t.allocator.free(bytes);
        try dir.writeFile(t.io, .{ .sub_path = "index.bin", .data = bytes });
        const now = try get(c, file, &calls);
        try t.expect(!now.hit);
        try t.expectEqual(good.value.sha256, now.value.sha256);
    }
    try t.expectEqual(@as(usize, 6), calls);
}

test "digest cache refuses corruption without returning or recomputing a trusted value" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "file", .data = "first" });
    const root = try path(t.allocator, &tmp, "cache");
    defer t.allocator.free(root);
    const file = try path(t.allocator, &tmp, "file");
    defer t.allocator.free(file);
    const c = cache.Cache{ .root = root };
    var calls: usize = 0;
    _ = try get(c, file, &calls);
    var dir = try Io.Dir.cwd().openDir(t.io, root, .{});
    defer dir.close(t.io);
    const bytes = try dir.readFileAlloc(t.io, "index.bin", t.allocator, .limited(cache.max_bytes));
    defer t.allocator.free(bytes);
    bytes[12 + 56] ^= 1;
    try dir.writeFile(t.io, .{ .sub_path = "index.bin", .data = bytes });
    try t.expectError(error.CorruptDigestCache, get(c, file, &calls));
    try t.expectEqual(@as(usize, 1), calls);
    for ([_]usize{ 0, 7, bytes.len - 1 }) |n| try t.expectError(error.CorruptDigestCache, cache.decode(t.allocator, bytes[0..n]));
}

test "digest cache refuses a file that changes while it is hashed" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "file", .data = "first" });
    const root = try path(t.allocator, &tmp, "cache");
    defer t.allocator.free(root);
    const file = try path(t.allocator, &tmp, "file");
    defer t.allocator.free(file);
    try t.expectError(error.DigestFileChanged, (cache.Cache{ .root = root }).get(t.allocator, t.io, file, false, Reader{ .change = true }));
    var dir = try Io.Dir.cwd().openDir(t.io, root, .{});
    defer dir.close(t.io);
    try t.expectError(error.FileNotFound, dir.openFile(t.io, "index.bin", .{}));
}

test "digest cache stays bounded and evicts old identities" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try path(t.allocator, &tmp, "cache");
    defer t.allocator.free(root);
    const c = cache.Cache{ .root = root, .limit = 3 };
    var calls: usize = 0;
    for (0..7) |i| {
        var name: [32]u8 = undefined;
        const sub = try std.fmt.bufPrint(&name, "file-{d}", .{i});
        try tmp.dir.writeFile(t.io, .{ .sub_path = sub, .data = sub });
        const file = try path(t.allocator, &tmp, sub);
        defer t.allocator.free(file);
        _ = try get(c, file, &calls);
    }
    var dir = try Io.Dir.cwd().openDir(t.io, root, .{ .iterate = true });
    defer dir.close(t.io);
    const bytes = try dir.readFileAlloc(t.io, "index.bin", t.allocator, .limited(cache.max_bytes));
    defer t.allocator.free(bytes);
    var records = try cache.decode(t.allocator, bytes);
    defer records.deinit(t.allocator);
    try t.expectEqual(@as(usize, 3), records.items.len);
    var it = dir.iterate();
    var count: usize = 0;
    while (try it.next(t.io)) |entry| {
        try t.expect(std.mem.eql(u8, entry.name, "index.bin") or std.mem.eql(u8, entry.name, "index.lock") or
            (std.mem.startsWith(u8, entry.name, "hash-") and std.mem.endsWith(u8, entry.name, ".lock")));
        count += 1;
    }
    try t.expect(count <= 2 + @import("digest_lock.zig").stripes);
    const first = try path(t.allocator, &tmp, "file-0");
    defer t.allocator.free(first);
    try t.expect(!(try get(c, first, &calls)).hit);
}

const Scan = struct {
    root: []const u8,
    ready: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    bad: std.atomic.Value(bool) = .init(false),
    reads: std.atomic.Value(usize) = .init(0),
    fn run(s: *Scan) void {
        s.ready.store(true, .release);
        var dir = Io.Dir.cwd().openDir(t.io, s.root, .{}) catch {
            s.bad.store(true, .release);
            return;
        };
        defer dir.close(t.io);
        while (!s.stop.load(.acquire)) {
            const bytes = dir.readFileAlloc(t.io, "index.bin", std.heap.page_allocator, .limited(cache.max_bytes)) catch |err| {
                if (err != error.FileNotFound) s.bad.store(true, .release);
                continue;
            };
            defer std.heap.page_allocator.free(bytes);
            var records = cache.decode(std.heap.page_allocator, bytes) catch {
                s.bad.store(true, .release);
                continue;
            };
            records.deinit(std.heap.page_allocator);
            _ = s.reads.fetchAdd(1, .monotonic);
        }
    }
};

const Writer = struct {
    root: []const u8,
    files: []const u8,
    start: *std.atomic.Value(bool),
    failed: bool = false,
    reads: usize = 0,
    fn run(w: *Writer) void {
        while (!w.start.load(.acquire)) std.atomic.spinLoopHint();
        const result = std.process.run(std.heap.page_allocator, t.io, .{
            .argv = &.{ @import("digest_test_options").writer, w.root, w.files },
            .stdout_limit = .limited(1024),
            .stderr_limit = .limited(8192),
        }) catch {
            w.failed = true;
            return;
        };
        defer std.heap.page_allocator.free(result.stdout);
        defer std.heap.page_allocator.free(result.stderr);
        if (result.term != .exited or result.term.exited != 0) {
            std.debug.print("digest child: {s}\n", .{result.stderr});
            w.failed = true;
            return;
        }
        w.reads = std.fmt.parseInt(usize, std.mem.trim(u8, result.stdout, "\n"), 10) catch {
            w.failed = true;
            return;
        };
    }
};

test "concurrent rank processes read shards once and publish complete indexes atomically" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    if (@import("builtin").os.tag != .macos and @import("builtin").os.tag != .linux) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try path(a, &tmp, "cache");
    try Io.Dir.cwd().createDirPath(t.io, root);
    for (0..16) |i| {
        var name: [32]u8 = undefined;
        const sub = try std.fmt.bufPrint(&name, "file-{d}", .{i});
        try tmp.dir.writeFile(t.io, .{ .sub_path = sub, .data = sub });
    }
    const files = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    var start = std.atomic.Value(bool).init(false);
    var scanner = Scan{ .root = root };
    const scan = try std.Thread.spawn(.{}, Scan.run, .{&scanner});
    var writers = [_]Writer{ .{ .root = root, .files = files, .start = &start }, .{ .root = root, .files = files, .start = &start } };
    var threads: [2]std.Thread = undefined;
    for (&writers, &threads) |*w, *thread| thread.* = try std.Thread.spawn(.{}, Writer.run, .{w});
    while (!scanner.ready.load(.acquire)) std.atomic.spinLoopHint();
    start.store(true, .release);
    for (threads) |thread| thread.join();
    scanner.stop.store(true, .release);
    scan.join();
    for (writers) |writer| try t.expect(!writer.failed);
    try t.expectEqual(@as(usize, 16), writers[0].reads + writers[1].reads);
    try t.expect(!scanner.bad.load(.acquire));
    try t.expect(scanner.reads.load(.acquire) > 0);
}

test "digest cache shares whole hashes and retains the checkpoint projection" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "file", .data = "first" });
    const root = try path(t.allocator, &tmp, "cache");
    defer t.allocator.free(root);
    const file = try path(t.allocator, &tmp, "file");
    defer t.allocator.free(file);
    const Projector = struct {
        calls: *usize,
        pub fn read(r: @This(), f: Io.File, size: u64) !cache.Value {
            var value = try (Reader{ .calls = r.calls }).read(f, size);
            value.shared = @splat(7);
            return value;
        }
    };
    var calls: usize = 0;
    const c = cache.Cache{ .root = root };
    const first = try c.get(t.allocator, t.io, file, true, Projector{ .calls = &calls });
    const whole = try get(c, file, &calls);
    try t.expect(whole.hit);
    try t.expectEqual(first.value.sha256, whole.value.sha256);
    const shared = try c.get(t.allocator, t.io, file, true, Projector{ .calls = &calls });
    try t.expect(shared.hit);
    try t.expectEqual(first.value.shared.?, shared.value.shared.?);
    try t.expectEqual(@as(usize, 1), calls);
    try t.expectError(error.DigestProjectionMissing, (cache.Cache{ .root = null }).get(t.allocator, t.io, file, true, Reader{}));
}

test "digest cache schema is bounded versioned and has no implicit home in tests" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    try t.expect((try cache.environmentRoot(t.allocator)) == null);
    try t.expectError(error.CacheHomeMissing, cache.defaultRoot(t.allocator, "relative"));
    var bytes: [44]u8 = @splat(0);
    @memcpy(bytes[0..8], "TFDG0001");
    std.mem.writeInt(u32, bytes[8..12], cache.max_entries + 1, .little);
    try t.expectError(error.CorruptDigestCache, cache.decode(t.allocator, &bytes));
    const key = cache.Key{ .device = 1, .inode = 2, .size = 3, .mtime = 4, .ctime = 5 };
    const record = cache.Record{ .key = key, .value = .{ .sha256 = @splat(1) } };
    const duplicate = try cache.encode(t.allocator, &.{ record, record });
    defer t.allocator.free(duplicate);
    try t.expectError(error.CorruptDigestCache, cache.decode(t.allocator, duplicate));
}
