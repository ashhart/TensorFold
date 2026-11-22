const std = @import("std");
const sessions = @import("kv").sessions;
const Sessions = @import("sessions_gpu.zig").Sessions;
const eh = @import("engram_host.zig");
const ced = @import("ced.zig");
const Blob = sessions.disk.Blob;
const slot_bytes = Sessions.slot_bytes;
const ced_at = Sessions.ced_at;
const ds_at = Sessions.ds_at;
pub const Snap = struct {
    owner: *Sessions,
    bytes: []u8,
    pos: u64,
    tail: [eh.max_ngram]u32,
    tail_len: usize,
    stash: ced.Stash,
    /// serialized on parking; the slot blob borrows this storage alongside the role views
    slot: [slot_bytes]u8 = undefined,
    /// the drafter's rows the snapshot holds (a restore's context start is pos - ds_rows)
    ds_rows: u64,
    /// `Bounded.nbytes` of this snapshot: the RAM tier's measure
    measure: u64,

    pub fn bounded(p: *Snap) sessions.store.Bounded {
        return .{ .ptr = p, .vtable = &.{ .nbytes = nbytes, .blobs = blobs, .release = release } };
    }

    pub fn nbytes(ptr: *anyopaque) u64 {
        const p: *Snap = @ptrCast(@alignCast(ptr));
        return @sizeOf(Snap) + p.bytes.len;
    }

    /// Blobs borrow Snap storage through the disk's last read; only descriptors are returned-owned.
    pub fn blobs(ptr: *anyopaque, gpa: std.mem.Allocator) anyerror![]Blob {
        const p: *Snap = @ptrCast(@alignCast(ptr));
        const ps = p.owner.pieces;
        const out = try gpa.alloc(Blob, ps.len + 1);
        errdefer gpa.free(out);
        var at: usize = 0;
        for (ps, out[0..ps.len]) |pc, *b| {
            b.* = .{ .name = pc.role, .bytes = p.bytes[at..][0..pc.len], .owned = false };
            at += pc.len;
        }
        const slot = &p.slot;
        std.mem.writeInt(u64, slot[0..8], p.pos, .little);
        std.mem.writeInt(u64, slot[8..16], p.tail_len, .little);
        for (0..eh.max_ngram) |i| std.mem.writeInt(u32, slot[16 + 4 * i ..][0..4], if (i < p.tail_len) p.tail[i] else 0, .little);
        // CED: the stash's positions and ids (a full-mode entry's are zeros)
        const c = slot[ced_at..];
        std.mem.writeInt(u64, c[0..8], p.stash.from, .little);
        std.mem.writeInt(u64, c[8..16], p.stash.hi, .little);
        for (p.stash.ids, 0..) |t, i| std.mem.writeInt(u32, c[16 + 4 * i ..][0..4], t, .little);
        std.mem.writeInt(u64, slot[ds_at..][0..8], p.ds_rows, .little);
        out[ps.len] = .{ .name = "slot", .bytes = slot, .owned = false };
        return out;
    }

    pub fn release(ptr: *anyopaque) void {
        const p: *Snap = @ptrCast(@alignCast(ptr));
        const gpa = p.owner.gpa;
        gpa.free(p.bytes);
        gpa.destroy(p);
    }
};

fn freeBlobs(a: std.mem.Allocator, out: []Blob) void {
    for (out) |b| b.deinit(a);
    a.free(out);
}

fn expectBlobs(p: *const Snap, pieces: []const @import("sessions_gpu.zig").Piece, source: []const u8, out: []const Blob) !void {
    const t = std.testing;
    try t.expectEqual(pieces.len + 1, out.len);
    var at: usize = 0;
    for (pieces, out[0..pieces.len]) |pc, b| {
        try t.expectEqualStrings(pc.role, b.name);
        try t.expectEqualSlices(u8, source[at..][0..pc.len], b.bytes);
        try t.expect(b.name.ptr == pc.role.ptr and b.bytes.ptr == source[at..].ptr);
        try t.expect(!b.owned);
        at += pc.len;
    }
    const slot = out[pieces.len];
    try t.expectEqualStrings("slot", slot.name);
    try t.expectEqual(slot_bytes, slot.bytes.len);
    try t.expect(slot.bytes.ptr == p.slot[0..].ptr and !slot.owned);
    try t.expectEqual(p.pos, std.mem.readInt(u64, slot.bytes[0..8], .little));
    try t.expectEqual(@as(u64, @intCast(p.tail_len)), std.mem.readInt(u64, slot.bytes[8..16], .little));
    for (0..eh.max_ngram) |i| try t.expectEqual(if (i < p.tail_len) p.tail[i] else 0, std.mem.readInt(u32, slot.bytes[16 + 4 * i ..][0..4], .little));
    const c = slot.bytes[ced_at..];
    try t.expectEqual(p.stash.from, std.mem.readInt(u64, c[0..8], .little));
    try t.expectEqual(p.stash.hi, std.mem.readInt(u64, c[8..16], .little));
    for (p.stash.ids, 0..) |id, i| try t.expectEqual(id, std.mem.readInt(u32, c[16 + 4 * i ..][0..4], .little));
    try t.expectEqual(p.ds_rows, std.mem.readInt(u64, slot.bytes[ds_at..][0..8], .little));
}

fn allocationCase(a: std.mem.Allocator) !void {
    const chunk = 4096;
    var owner: Sessions = undefined;
    var pieces = [_]@import("sessions_gpu.zig").Piece{ .{ .role = "s.L2.carry", .len = chunk + 1 }, .{ .role = "s.L3.ring", .len = 7 } };
    owner.pieces = &pieces;
    var bytes: [chunk + 8]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @truncate(i *% 73 +% 19);
    const before = bytes;
    var tail: [eh.max_ngram]u32 = @splat(0);
    tail[0] = 11;
    tail[1] = 12;
    var stash: ced.Stash = .{ .from = 3, .hi = 9 };
    stash.ids[0] = 21;
    stash.ids[stash.ids.len - 1] = 22;
    var snapshot: Snap = .{ .owner = &owner, .bytes = &bytes, .pos = 9, .tail = tail, .tail_len = 2, .stash = stash, .ds_rows = 4, .measure = bytes.len };
    const result = snapshot.bounded().vtable.blobs(&snapshot, a);
    try std.testing.expectEqualSlices(u8, &before, &bytes);
    const out = try result;
    defer freeBlobs(a, out);
    try expectBlobs(&snapshot, &pieces, &bytes, out);
}

test "Snapshot state: blob allocation failures release descriptors and preserve the source across a chunk" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}

test "Snapshot state: parking peak is returned descriptors, not a temporary payload copy" {
    var peak: @import("engine_api").rows_peak_allocator.Peak = .{ .child = std.testing.allocator };
    try allocationCase(peak.allocator());
    // Two role views plus slot are retained output; their borrowed payload costs no allocation.
    try std.testing.expectEqual(3 * @sizeOf(Blob), peak.peak);
    try std.testing.expectEqual(@as(usize, 0), peak.live);
}
