//! The libraries a load builds while someone watches: each one's source or metallib and its compile options, hashed.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The watcher every library build reports to, or null; an execution witness sets it for one engine's load.
pub var watcher: std.atomic.Value(?*Built) = .init(null);

/// One load's builds, in any order.
pub const Built = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    digests: std.ArrayList([32]u8) = .empty,
    /// A build it could not record (no memory, or a metallib from a file): its digest would cover less than the load.
    missed: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: std.Io) Built {
        return .{ .gpa = gpa, .io = io };
    }

    pub fn deinit(b: *Built) void {
        b.digests.deinit(b.gpa);
    }

    /// One build: the source or metallib bytes, and the options a source compiled with (null: a metallib).
    pub fn add(b: *Built, bytes: []const u8, options: ?[3]u64) void {
        var h = Sha256.init(.{});
        field(&h, if (options) |o| std.mem.sliceAsBytes(&o) else "metallib");
        field(&h, bytes);
        const d = h.finalResult();
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        b.digests.append(b.gpa, d) catch {
            b.missed = true;
        };
    }

    pub fn miss(b: *Built) void {
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        b.missed = true;
    }

    /// Every distinct build in one digest, the same in whatever order the load built them; null when one was missed.
    pub fn digest(b: *Built) ?[32]u8 {
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        if (b.missed) return null;
        std.mem.sort([32]u8, b.digests.items, {}, less);
        var h = Sha256.init(.{});
        for (b.digests.items, 0..) |d, i| {
            if (i > 0 and std.mem.eql(u8, &d, &b.digests.items[i - 1])) continue;
            h.update(&d);
        }
        return h.finalResult();
    }
};

/// Tells the watcher, if there is one, about a build.
pub fn report(bytes: []const u8, options: ?[3]u64) void {
    if (watcher.load(.acquire)) |b| b.add(bytes, options);
}

/// Tells the watcher, if there is one, about a build it cannot hash.
pub fn missed() void {
    if (watcher.load(.acquire)) |b| b.miss();
}

fn less(_: void, x: [32]u8, y: [32]u8) bool {
    return std.mem.order(u8, &x, &y) == .lt;
}

fn field(h: *Sha256, bytes: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, bytes.len, .little);
    h.update(&len);
    h.update(bytes);
}

test "built: the digest ignores build order and repeats, and changes with a source, its options or a missed build" {
    const t = std.testing;
    var x = Built.init(t.allocator, t.io);
    defer x.deinit();
    x.add("kernel a", .{ 0, 1, 2 });
    x.add("kernel b", .{ 0, 1, 2 });
    var y = Built.init(t.allocator, t.io);
    defer y.deinit();
    y.add("kernel b", .{ 0, 1, 2 });
    y.add("kernel a", .{ 0, 1, 2 });
    y.add("kernel a", .{ 0, 1, 2 });
    try t.expectEqualSlices(u8, &x.digest().?, &y.digest().?);
    var z = Built.init(t.allocator, t.io);
    defer z.deinit();
    z.add("kernel a", .{ 0, 1, 3 });
    z.add("kernel b", .{ 0, 1, 2 });
    try t.expect(!std.mem.eql(u8, &x.digest().?, &z.digest().?));
    z.add("metallib bytes", null);
    try t.expect(z.digest() != null);
    z.miss();
    try t.expect(z.digest() == null);
}

test "built: builds report to the watcher only while one is set" {
    const t = std.testing;
    var b = Built.init(t.allocator, t.io);
    defer b.deinit();
    report("before", .{ 0, 0, 0 });
    watcher.store(&b, .release);
    report("during", .{ 0, 0, 0 });
    watcher.store(null, .release);
    report("after", .{ 0, 0, 0 });
    try t.expectEqual(@as(usize, 1), b.digests.items.len);
}
