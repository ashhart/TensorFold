//! The libraries one load builds while it is watched: each one's source or metallib and its compile options, hashed.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// Guards `owner` and every append to it, so a stop waits for a report that already found its watcher.
var lock: std.atomic.Mutex = .unlocked;
/// The one load being watched, or null; only its thread's builds count, any other build invalidates it.
var owner: ?*Built = null;

/// One load's builds, in any order.
pub const Built = struct {
    gpa: std.mem.Allocator,
    thread: std.Thread.Id = 0,
    digests: std.ArrayList([32]u8) = .empty,
    /// A build it could not record, another thread's build, or a lost or refused watch: its digest is void.
    missed: bool = false,

    pub fn init(gpa: std.mem.Allocator) Built {
        return .{ .gpa = gpa };
    }

    pub fn deinit(b: *Built) void {
        b.digests.deinit(b.gpa);
    }

    /// Watches this thread's builds from now; false (and void) while another load is watched.
    pub fn watch(b: *Built) bool {
        hold();
        defer lock.unlock();
        if (owner != null) {
            b.missed = true;
            return false;
        }
        b.thread = std.Thread.getCurrentId();
        owner = b;
        return true;
    }

    /// Stops watching, after any report in flight; every distinct build in one digest, null when void.
    pub fn stop(b: *Built) ?[32]u8 {
        hold();
        defer lock.unlock();
        if (owner == b) owner = null else b.missed = true;
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

/// Tells the watched load, if there is one, about a build: the source or metallib bytes, and a source's options.
pub fn report(bytes: []const u8, options: ?[3]u64) void {
    var h = Sha256.init(.{});
    field(&h, if (options) |o| std.mem.sliceAsBytes(&o) else "metallib");
    field(&h, bytes);
    const d = h.finalResult();
    hold();
    defer lock.unlock();
    const b = owner orelse return;
    if (b.thread != std.Thread.getCurrentId()) {
        b.missed = true;
        return;
    }
    b.digests.append(b.gpa, d) catch {
        b.missed = true;
    };
}

/// Tells the watched load, if there is one, about a build it cannot hash.
pub fn missed() void {
    hold();
    defer lock.unlock();
    if (owner) |b| b.missed = true;
}

fn hold() void {
    while (!lock.tryLock()) std.atomic.spinLoopHint();
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

const t = std.testing;

fn watched(sources: []const []const u8, options: [3]u64) !?[32]u8 {
    var b = Built.init(t.allocator);
    defer b.deinit();
    try t.expect(b.watch());
    for (sources) |s| report(s, options);
    return b.stop();
}

test "built: the digest ignores build order and repeats, and changes with a source or its options" {
    const x = (try watched(&.{ "kernel a", "kernel b" }, .{ 0, 1, 2 })).?;
    try t.expectEqualSlices(u8, &x, &(try watched(&.{ "kernel b", "kernel a", "kernel a" }, .{ 0, 1, 2 })).?);
    try t.expect(!std.mem.eql(u8, &x, &(try watched(&.{ "kernel a", "kernel b" }, .{ 0, 1, 3 })).?));
    try t.expect(!std.mem.eql(u8, &x, &(try watched(&.{"kernel a"}, .{ 0, 1, 2 })).?));
}

test "built: builds count only while watched; a build it cannot hash voids the digest" {
    var b = Built.init(t.allocator);
    defer b.deinit();
    report("before", .{ 0, 0, 0 });
    try t.expect(b.watch());
    report("during", .{ 0, 0, 0 });
    try t.expectEqual(@as(usize, 1), b.digests.items.len);
    missed();
    try t.expect(b.stop() == null);
    report("after", .{ 0, 0, 0 });
    try t.expectEqual(@as(usize, 1), b.digests.items.len);
}

test "built: a second watch is refused and void, and the first keeps only its own builds" {
    var first = Built.init(t.allocator);
    defer first.deinit();
    var second = Built.init(t.allocator);
    defer second.deinit();
    try t.expect(first.watch());
    try t.expect(!second.watch());
    report("first's build", .{ 0, 0, 0 });
    try t.expect(second.stop() == null); // never the owner: void, and the first's watch stays
    try t.expectEqual(@as(usize, 0), second.digests.items.len);
    try t.expect(first.stop() != null);
    try t.expect(second.stop() == null);
}

test "built: another thread's build during a watched load voids it" {
    var b = Built.init(t.allocator);
    defer b.deinit();
    try t.expect(b.watch());
    const other = try std.Thread.spawn(.{}, report, .{ "another thread's build", @as(?[3]u64, .{ 0, 0, 0 }) });
    other.join();
    try t.expectEqual(@as(usize, 0), b.digests.items.len);
    try t.expect(b.stop() == null);
}

test "built: a stop waits for a report that holds the watcher, so no append outlives it" {
    var b = Built.init(t.allocator);
    defer b.deinit();
    try t.expect(b.watch());
    hold(); // a report that found its watcher and has not appended yet
    const Stop = struct {
        fn run(x: *Built, done: *std.atomic.Value(bool), got: *?[32]u8) void {
            got.* = x.stop();
            done.store(true, .release);
        }
    };
    var done: std.atomic.Value(bool) = .init(false);
    var got: ?[32]u8 = null;
    const stopper = try std.Thread.spawn(.{}, Stop.run, .{ &b, &done, &got });
    for (0..1_000_000) |_| std.atomic.spinLoopHint();
    try t.expect(!done.load(.acquire));
    try b.digests.append(b.gpa, @splat(7)); // the held report's append
    lock.unlock();
    stopper.join();
    try t.expect(done.load(.acquire));
    try t.expect(got != null);
    try t.expect(owner == null);
}
