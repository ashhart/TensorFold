//! Token ids of long text segments between added tokens: a chat's earlier messages are tokenized once, not each render.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Segments shorter than this are tokenized each time: their lookup would cost about what their encode does.
pub const min_bytes = 1024;

const Entry = struct { text: []u8, first: bool, ids: []u32, previous: ?u64 = null, next: ?u64 = null };

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
        const id_bytes = std.math.mul(usize, ids.len, @sizeOf(u32)) catch return;
        const size = std.math.add(usize, text.len, id_bytes) catch return;
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
        m.held -= e.text.len + e.ids.len * @sizeOf(u32);
        free(m.gpa, e);
    }

    pub fn clear(m: *Memo) void {
        m.lock();
        defer m.unlock();
        var it = m.map.valueIterator();
        while (it.next()) |e| {
            std.crypto.secureZero(u8, e.text);
            std.crypto.secureZero(u32, e.ids);
            free(m.gpa, e.*);
        }
        m.map.clearRetainingCapacity();
        m.oldest = null;
        m.newest = null;
        m.held = 0;
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

test "clear drops every kept segment, zeroed, and the memo keeps working" {
    const a = std.testing.allocator;
    const m = try Memo.create(a, 1024);
    defer m.destroy();
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    m.put("fictional cobalt meadow", false, &.{ 1, 2, 3 });
    try std.testing.expect(try m.append("fictional cobalt meadow", false, a, &out));
    m.clear();
    try std.testing.expectEqual(@as(usize, 0), m.map.count());
    try std.testing.expectEqual(@as(usize, 0), m.held);
    try std.testing.expect(!try m.append("fictional cobalt meadow", false, a, &out));
    m.put("fresh", false, &.{4});
    try std.testing.expect(try m.append("fresh", false, a, &out));
}

test "a kept segment comes back only for the same text and first flag, and the oldest goes past the cap" {
    const gpa = std.testing.allocator;
    const m = try Memo.create(gpa, 60);
    defer m.destroy();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    m.put("abcd", false, &.{ 1, 2 });
    try std.testing.expect(!try m.append("abcd", true, gpa, &ids));
    try std.testing.expect(try m.append("abcd", false, gpa, &ids));
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, ids.items);
    m.put("0123456789012345678901234567890123456789", false, &.{ 3, 4, 5 }); // 52 bytes: "abcd" (12) goes
    try std.testing.expect(!try m.append("abcd", false, gpa, &ids));
    try std.testing.expectEqual(@as(usize, 52), m.held);
}
