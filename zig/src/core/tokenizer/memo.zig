//! Token ids of long text segments between added tokens: a chat's earlier messages are tokenized once, not each render.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Segments shorter than this are tokenized each time: their lookup would cost about what their encode does.
pub const min_bytes = 1024;

const Entry = struct { text: []u8, first: bool, ids: []u32 };

/// Kept segments up to `cap` bytes of text and ids, the oldest dropped first; a spin lock guards the lookups and saves.
pub const Memo = struct {
    gpa: Allocator,
    cap: usize,
    held: usize = 0,
    busy: std.atomic.Value(bool) = .init(false),
    map: std.AutoHashMapUnmanaged(u64, Entry) = .empty,
    order: std.ArrayList(u64) = .empty, // keys oldest first
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
        m.order.deinit(m.gpa);
        m.gpa.destroy(m);
    }

    /// Appends `text`'s kept ids to `ids` and returns true, or returns false when none are kept.
    pub fn append(m: *Memo, text: []const u8, first: bool, a: Allocator, ids: *std.ArrayList(u32)) !bool {
        m.lock();
        defer m.unlock();
        const e = m.map.get(key(text, first)) orelse return false;
        if (e.first != first or !std.mem.eql(u8, e.text, text)) return false;
        try ids.appendSlice(a, e.ids);
        m.hits += 1;
        return true;
    }

    /// Keeps `ids` as `text`'s, past the oldest segments while the total passes the cap; a failed copy keeps nothing.
    pub fn put(m: *Memo, text: []const u8, first: bool, ids: []const u32) void {
        const size = text.len + ids.len * 4;
        if (size > m.cap) return;
        const copy = m.gpa.dupe(u8, text) catch return;
        const e: Entry = .{ .text = copy, .first = first, .ids = m.gpa.dupe(u32, ids) catch return m.gpa.free(copy) };
        m.lock();
        defer m.unlock();
        const k = key(text, first);
        if (m.map.fetchRemove(k)) |old| {
            m.held -= old.value.text.len + old.value.ids.len * 4;
            free(m.gpa, old.value);
            if (std.mem.indexOfScalar(u64, m.order.items, k)) |i| _ = m.order.orderedRemove(i);
        }
        while (m.held + size > m.cap and m.order.items.len > 0) {
            const gone = m.map.fetchRemove(m.order.orderedRemove(0)).?;
            m.held -= gone.value.text.len + gone.value.ids.len * 4;
            free(m.gpa, gone.value);
        }
        m.order.append(m.gpa, k) catch return free(m.gpa, e);
        m.map.put(m.gpa, k, e) catch {
            _ = m.order.pop();
            return free(m.gpa, e);
        };
        m.held += size;
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
