//! TF_FLASHNEXT_SCALE_INDEX: an expert projection's scales and biases as device index and table buffers (scale_pairs).
const std = @import("std");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const pairs = @import("scale_pairs.zig");

/// A scales buffer's identity: the map from it to its index and table.
pub const Key = struct { id: @FieldType(@FieldType(Buf, "b"), "id"), off: usize };

pub fn keyOf(b: Buf) Key {
    return .{ .id = b.b.id, .off = b.off };
}

/// The index and table buffers of the scales `s` and biases `b` (whole buffers of bf16), kept under `s`'s key.
pub fn add(r: *Run, s: Buf, b: Buf) !void {
    const groups = (s.b.length() - s.off) / 2;
    const gpa = std.heap.page_allocator;
    const table = try gpa.create([pairs.most]u32);
    defer gpa.destroy(table);
    const seen = try gpa.create([pairs.slots]u32);
    defer gpa.destroy(seen);
    const ix = try r.buffer(groups * 2);
    const sp: [*]align(1) const u16 = @ptrCast(s.b.contents() + s.off);
    const bp: [*]align(1) const u16 = @ptrCast(b.b.contents() + b.off);
    const ip: [*]align(1) u16 = @ptrCast(ix.contents());
    const n = try pairs.index(sp[0..groups], bp[0..groups], ip[0..groups], table, seen);
    const t = try r.buffer(n * 4);
    @memcpy(t.contents()[0 .. n * 4], std.mem.sliceAsBytes(table[0..n]));
    try r.sbix.put(r.arena, keyOf(s), .{ .{ .b = ix }, .{ .b = t } });
    r.sb_pairs = @max(r.sb_pairs, n);
}

/// Every projection of an expert set (weight, scales, biases a projection) indexed; a table past 16 bits turns it off.
pub fn addAll(r: *Run, ex: *const [18]Buf) !void {
    for (0..6) |j| add(r, ex[j * 3 + 1], ex[j * 3 + 2]) catch |err| switch (err) {
        error.TooManyPairs => {
            std.log.warn("TF_FLASHNEXT_SCALE_INDEX: a tensor has more than {d} scale and bias pairs; the index is off", .{pairs.most});
            r.sb_index = false;
            return;
        },
        else => return err,
    };
}
