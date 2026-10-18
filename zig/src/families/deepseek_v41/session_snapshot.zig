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

    /// The park's blobs: each role's bytes under its name, then "slot" (position, tail).
    pub fn blobs(ptr: *anyopaque, gpa: std.mem.Allocator) anyerror![]Blob {
        const p: *Snap = @ptrCast(@alignCast(ptr));
        const ps = p.owner.pieces;
        const out = try gpa.alloc(Blob, ps.len + 1);
        var made: usize = 0;
        errdefer {
            for (out[0..made]) |b| {
                gpa.free(b.name);
                gpa.free(b.bytes);
            }
            gpa.free(out);
        }
        var at: usize = 0;
        for (ps, out[0..ps.len]) |pc, *b| {
            const name = try gpa.dupe(u8, pc.role);
            errdefer gpa.free(name);
            b.* = .{ .name = name, .bytes = try gpa.dupe(u8, p.bytes[at..][0..pc.len]) };
            made += 1;
            at += pc.len;
        }
        var slot: [slot_bytes]u8 = undefined;
        std.mem.writeInt(u64, slot[0..8], p.pos, .little);
        std.mem.writeInt(u64, slot[8..16], p.tail_len, .little);
        for (0..eh.max_ngram) |i| std.mem.writeInt(u32, slot[16 + 4 * i ..][0..4], if (i < p.tail_len) p.tail[i] else 0, .little);
        // CED: the stash's positions and ids (a full-mode entry's are zeros)
        const c = slot[ced_at..];
        std.mem.writeInt(u64, c[0..8], p.stash.from, .little);
        std.mem.writeInt(u64, c[8..16], p.stash.hi, .little);
        for (p.stash.ids, 0..) |t, i| std.mem.writeInt(u32, c[16 + 4 * i ..][0..4], t, .little);
        std.mem.writeInt(u64, slot[ds_at..][0..8], p.ds_rows, .little);
        const slot_name = try gpa.dupe(u8, "slot");
        errdefer gpa.free(slot_name);
        out[ps.len] = .{ .name = slot_name, .bytes = try gpa.dupe(u8, &slot) };
        return out;
    }

    pub fn release(ptr: *anyopaque) void {
        const p: *Snap = @ptrCast(@alignCast(ptr));
        const gpa = p.owner.gpa;
        gpa.free(p.bytes);
        gpa.destroy(p);
    }
};

fn allocationCase(a: std.mem.Allocator) !void {
    var owner: Sessions = undefined;
    var pieces = [_]@import("sessions_gpu.zig").Piece{.{ .role = "s.L2.carry", .len = 8 }};
    owner.pieces = &pieces;
    var bytes: [8]u8 = @splat(7);
    var snapshot: Snap = .{ .owner = &owner, .bytes = &bytes, .pos = 9, .tail = @splat(0), .tail_len = 0, .stash = .{}, .ds_rows = 0, .measure = 8 };
    const out = try snapshot.bounded().vtable.blobs(&snapshot, a);
    defer a.free(out);
    for (out) |b| {
        a.free(b.name);
        a.free(b.bytes);
    }
}
test "Snapshot state: snapshot blob allocation failures release every partial owner" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}
