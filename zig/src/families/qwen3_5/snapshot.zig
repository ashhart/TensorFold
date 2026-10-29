//! A committed prompt state in host memory, independent of stream capacity and attention page placement.
const std = @import("std");
const c = @import("config.zig");
const st = @import("state.zig");
const page_tokens = @import("../../core/metal_kv_pages.zig").page_tokens;

pub const Snap = struct { at: u32, bytes: []u8 };

pub fn bytes(at: u32) u64 {
    return st.cacheBytes(at);
}

pub fn save(gpa: std.mem.Allocator, cache: *const st.Cache, at: u32) !*Snap {
    if (cache.len != at or at > cache.capacity) return error.QwenSnapshotPosition;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .at = at, .bytes = try gpa.alloc(u8, bytes(at)) };
    transfer(cache, snap, false);
    return snap;
}

pub fn restore(cache: *st.Cache, snap: *const Snap) !void {
    if (cache.len != 0 or snap.at > cache.capacity or snap.bytes.len != bytes(snap.at))
        return error.QwenSnapshotPosition;
    if (cache.paged) |p| try p.pages.cover(p.lane, snap.at);
    transfer(cache, snap, true);
    cache.len = snap.at;
    cache.last = null;
}

pub fn drop(gpa: std.mem.Allocator, snap: *Snap) void {
    gpa.free(snap.bytes);
    gpa.destroy(snap);
}

fn copy(live: []u8, saved: []u8, comptime restoring: bool) void {
    if (restoring) @memcpy(live, saved) else @memcpy(saved, live);
}

fn transfer(cache: *const st.Cache, snap: *const Snap, comptime restoring: bool) void {
    var off: usize = c.vocab * 2;
    copy(cache.logits.contents()[0..off], snap.bytes[0..off], restoring);
    var attention: usize = 0;
    for (cache.blocks) |block| switch (block) {
        .delta => |d| {
            copy(d.recurrence.contents()[0..st.delta_bytes], snap.bytes[off..][0..st.delta_bytes], restoring);
            off += st.delta_bytes;
            copy(d.conv.contents()[0..st.conv_bytes], snap.bytes[off..][0..st.conv_bytes], restoring);
            off += st.conv_bytes;
        },
        .attention => |kv| {
            for ([_]@TypeOf(kv.keys){ kv.keys, kv.values }, 0..) |buf, value| for (0..c.kv_heads) |h| {
                var row: usize = 0;
                while (row < snap.at) {
                    const n = if (cache.paged != null) @min(snap.at - row, page_tokens - row % page_tokens) else snap.at;
                    const start = if (cache.paged) |p| blk: {
                        const id = p.pages.ids(p.lane)[row / page_tokens];
                        break :blk id * p.pages.layout.pageBytes() + st.pagePart(attention, value == 1) +
                            (h * page_tokens + row % page_tokens) * c.head_dim * 2;
                    } else (h * cache.capacity + row) * c.head_dim * 2;
                    const size = n * c.head_dim * 2;
                    copy((buf.contents() + @as(usize, @intCast(start)))[0..size], snap.bytes[off..][0..size], restoring);
                    off += size;
                    row += n;
                }
            };
            attention += 1;
        },
    };
    std.debug.assert(off == snap.bytes.len);
}
