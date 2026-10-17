//! The graph caches' counts as other threads read them (/health): fixed slots under one lock, each name copied in.

const std = @import("std");

/// A name's bytes kept per slot; a longer name is cut to this many.
pub const name_max = 32;

pub const Counts = struct { held: u32 = 0, captured: u64 = 0, replayed: u64 = 0, eager: u64 = 0, floor_hits: u64 = 0, under: bool = false };

/// One cache's name and counts at one instant; it owns its name, so it outlives the cache it was taken from.
pub const Snapshot = struct {
    name_buf: [name_max]u8 = undefined,
    name_len: u8 = 0,
    counts: Counts = .{},

    pub fn name(s: *const Snapshot) []const u8 {
        return s.name_buf[0..s.name_len];
    }
};

const Slot = struct { taken: bool = false, snap: Snapshot = .{} };

var slots: [8]Slot = @splat(.{});
var mutex: std.atomic.Mutex = .unlocked;

/// Test seams: `contended` runs when a lock attempt finds it held, `reading` between a slot's name and counts copies.
var contended: ?*const fn () void = null;
var reading: ?*const fn () void = null;

fn lock() void {
    if (mutex.tryLock()) return;
    if (contended) |f| f();
    while (!mutex.tryLock()) std.Thread.yield() catch {};
}

/// A free slot holding a copy of `name_text`, or null when all are taken (that cache is then not reported).
pub fn claim(name_text: []const u8) ?usize {
    lock();
    defer mutex.unlock();
    for (&slots, 0..) |*s, i| if (!s.taken) {
        const n = @min(name_text.len, name_max);
        s.* = .{ .taken = true };
        @memcpy(s.snap.name_buf[0..n], name_text[0..n]);
        s.snap.name_len = @intCast(n);
        return i;
    };
    return null;
}

pub fn publish(i: usize, c: Counts) void {
    lock();
    defer mutex.unlock();
    slots[i].snap.counts = c;
}

pub fn release(i: usize) void {
    lock();
    defer mutex.unlock();
    slots[i] = .{};
}

/// The slots in use, at most `out.len`; returns how many it filled. Each copy is one cache's name and counts.
pub fn snapshots(out: []Snapshot) usize {
    lock();
    defer mutex.unlock();
    var n: usize = 0;
    for (&slots) |*s| {
        if (n == out.len) break;
        if (!s.taken) continue;
        out[n].name_buf = s.snap.name_buf;
        out[n].name_len = s.snap.name_len;
        if (reading) |f| f();
        out[n].counts = s.snap.counts;
        n += 1;
    }
    return n;
}

// -- test: a reader overlapping a release and a reuse of its slot, ordered by the seams (no timing) ---------------

const Overlap = struct {
    var slot: usize = 0;
    var reused: usize = 0;
    var thread: ?std.Thread = null;
    var blocked: std.atomic.Value(bool) = .init(false);
    var done: std.atomic.Value(bool) = .init(false);

    /// Another thread frees the reader's slot and claims it for a new cache, as a deinit and a new cache's first step.
    fn other() void {
        release(slot);
        reused = claim("second").?;
        publish(reused, .{ .captured = 2 });
        done.store(true, .release);
    }

    /// Mid-copy: start the other thread and wait until it is blocked on the lock or (without one) has finished.
    fn mid() void {
        if (thread != null) return;
        thread = std.Thread.spawn(.{}, other, .{}) catch unreachable;
        while (!blocked.load(.acquire) and !done.load(.acquire)) std.Thread.yield() catch {};
    }

    fn onContended() void {
        blocked.store(true, .release);
    }
};

test "a snapshot overlapping its slot's release and reuse copies one cache's name and counts, never a mix" {
    const t = std.testing;
    Overlap.slot = claim("first").?;
    publish(Overlap.slot, .{ .captured = 1 });
    reading = Overlap.mid;
    contended = Overlap.onContended;
    var out: [8]Snapshot = undefined;
    const n = snapshots(&out);
    reading = null;
    Overlap.thread.?.join();
    contended = null;
    defer release(Overlap.reused); // a failed check leaves no slot taken for later tests
    try t.expectEqual(@as(usize, 1), n);
    try t.expectEqualStrings("first", out[0].name());
    try t.expectEqual(@as(u64, 1), out[0].counts.captured);
    try t.expect(Overlap.blocked.load(.acquire)); // the other thread met the lock while the copy was in progress
    // after the release and reuse: the new cache, with its own name and counts
    try t.expectEqual(@as(usize, 1), snapshots(&out));
    try t.expectEqualStrings("second", out[0].name());
    try t.expectEqual(@as(u64, 2), out[0].counts.captured);
    release(Overlap.reused);
    try t.expectEqual(@as(usize, 0), snapshots(&out));
}

test "a long name is cut to name_max and the copy outlives the caller's buffer" {
    var buf: [40]u8 = @splat('n');
    const i = claim(&buf).?;
    defer release(i);
    buf = @splat('x');
    var out: [8]Snapshot = undefined;
    try std.testing.expectEqual(@as(usize, 1), snapshots(&out));
    const want: [name_max]u8 = @splat('n');
    try std.testing.expectEqualStrings(&want, out[0].name());
}
