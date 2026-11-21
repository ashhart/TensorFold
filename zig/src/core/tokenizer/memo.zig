//! Token ids of text segments between added tokens: a chat's earlier messages are tokenized once, not each render.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Segments shorter than this are tokenized each time: their lookup would cost about what their encode does.
pub const min_bytes = 64;

const Entry = struct { text: []u8, first: bool, ids: []u32, previous: ?u64 = null, next: ?u64 = null };
/// Bytes a kept segment costs beyond its text and ids: its entry and its key in the map.
pub const overhead = @sizeOf(Entry) + @sizeOf(u64);

/// A segment's bytes against the cap: its text, its ids and the fixed `overhead`; null past usize.
pub fn bytes(text: usize, ids: usize) ?usize {
    const id_bytes = std.math.mul(usize, ids, @sizeOf(u32)) catch return null;
    return std.math.add(usize, text, std.math.add(usize, id_bytes, overhead) catch return null) catch null;
}

/// A bounded segment cache; successful reads refresh recency under the same lock as saves and eviction.
pub const Memo = struct {
    gpa: Allocator,
    cap: usize,
    held: usize = 0,
    busy: std.atomic.Value(bool) = .init(false),
    map: std.AutoHashMapUnmanaged(u64, Entry) = .empty,
    oldest: ?u64 = null,
    newest: ?u64 = null,
    hits: u64 = 0,

    pub fn create(gpa: Allocator, cap: usize) !*Memo {
        const m = try gpa.create(Memo);
        m.* = .{ .gpa = gpa, .cap = cap };
        return m;
    }

    pub fn destroy(m: *Memo) void {
        var it = m.map.valueIterator();
        while (it.next()) |e| free(m.gpa, e.*);
        m.map.deinit(m.gpa);
        m.gpa.destroy(m);
    }

    /// Appends `text`'s kept ids to `ids` and returns true, or returns false when none are kept.
    pub fn append(m: *Memo, text: []const u8, first: bool, a: Allocator, ids: *std.ArrayList(u32)) !bool {
        m.lock();
        defer m.unlock();
        const k = key(text, first);
        const e = m.map.get(k) orelse return false;
        if (e.first != first or !std.mem.eql(u8, e.text, text)) return false;
        try ids.appendSlice(a, e.ids);
        m.touch(k);
        m.hits +|= 1;
        return true;
    }

    /// Copies a completed segment; concurrent duplicate saves retain the existing copy and refresh its recency.
    pub fn put(m: *Memo, text: []const u8, first: bool, ids: []const u32) void {
        const size = bytes(text.len, ids.len) orelse return;
        if (size > m.cap) return;
        const k = key(text, first);
        m.lock();
        const present = m.same(k, text, first);
        if (present) m.touch(k);
        m.unlock();
        if (present) return;
        const copy = m.gpa.dupe(u8, text) catch return;
        const e: Entry = .{ .text = copy, .first = first, .ids = m.gpa.dupe(u32, ids) catch return m.gpa.free(copy) };
        m.lock();
        defer m.unlock();
        if (m.same(k, text, first)) {
            m.touch(k);
            return free(m.gpa, e);
        }
        m.map.ensureUnusedCapacity(m.gpa, 1) catch return free(m.gpa, e);
        if (m.map.contains(k)) m.remove(k);
        while (size > m.cap - m.held) m.remove(m.oldest.?);
        m.map.putAssumeCapacity(k, e);
        m.linkNewest(k);
        m.held += size;
    }

    fn same(m: *const Memo, k: u64, text: []const u8, first: bool) bool {
        const e = m.map.get(k) orelse return false;
        return e.first == first and std.mem.eql(u8, e.text, text);
    }

    fn unlink(m: *Memo, k: u64) void {
        const e = m.map.get(k).?;
        if (e.previous) |p| m.map.getPtr(p).?.next = e.next else m.oldest = e.next;
        if (e.next) |n| m.map.getPtr(n).?.previous = e.previous else m.newest = e.previous;
    }

    fn linkNewest(m: *Memo, k: u64) void {
        const e = m.map.getPtr(k).?;
        e.previous = m.newest;
        e.next = null;
        if (m.newest) |n| m.map.getPtr(n).?.next = k else m.oldest = k;
        m.newest = k;
    }

    fn touch(m: *Memo, k: u64) void {
        if (m.newest == k) return;
        m.unlink(k);
        m.linkNewest(k);
    }

    fn remove(m: *Memo, k: u64) void {
        m.unlink(k);
        const e = m.map.fetchRemove(k).?.value;
        m.held -= bytes(e.text.len, e.ids.len).?;
        free(m.gpa, e);
    }

    fn lock(m: *Memo) void {
        while (m.busy.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlock(m: *Memo) void {
        m.busy.store(false, .release);
    }
};

fn key(text: []const u8, first: bool) u64 {
    return std.hash.Wyhash.hash(@intFromBool(first), text);
}

fn free(gpa: Allocator, e: Entry) void {
    gpa.free(e.text);
    gpa.free(e.ids);
}

test {
    _ = @import("memo_reuse_test.zig");
}

test "a kept segment comes back only for the same text and first flag, and the oldest goes past the cap" {
    const gpa = std.testing.allocator;
    const m = try Memo.create(gpa, 60 + overhead);
    defer m.destroy();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    m.put("abcd", false, &.{ 1, 2 });
    try std.testing.expect(!try m.append("abcd", true, gpa, &ids));
    try std.testing.expect(try m.append("abcd", false, gpa, &ids));
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, ids.items);
    m.put("0123456789012345678901234567890123456789", false, &.{ 3, 4, 5 }); // 52 bytes and an overhead: "abcd" goes
    try std.testing.expect(!try m.append("abcd", false, gpa, &ids));
    try std.testing.expectEqual(52 + overhead, m.held);
}

test "each kept segment counts its entry against the cap, so short segments cannot outgrow it" {
    const gpa = std.testing.allocator;
    const payload = min_bytes + 16 * @sizeOf(u32); // a segment's text and ids alone
    const m = try Memo.create(gpa, 10 * payload);
    defer m.destroy();
    var text: [min_bytes]u8 = @splat('x');
    for (0..20) |i| {
        text[0] = @intCast(i);
        m.put(&text, false, &(@as([16]u32, @splat(@intCast(i)))));
    }
    // ten segments' text and ids no longer fit: their entries and keys count too
    try std.testing.expect(m.map.count() < 10);
    try std.testing.expectEqual(10 * payload / bytes(min_bytes, 16).?, m.map.count());
    try std.testing.expect(m.held <= m.cap);
}
