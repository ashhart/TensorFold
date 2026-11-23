//! Compact retained snapshots: the rows the shared kept-state walk names, then all convolution states.
const std = @import("std");
const mtl = @import("metal");
const state = @import("state.zig");
const ap = @import("attention_plan.zig");
const kept = @import("../kept_state.zig");

/// Eight global layers keep [0, at); forty sliding ones the window less one that their next row reads.
pub fn bytes(at: usize) u64 {
    const rows = @as(u64, 8) * kept.kept(true, state.sliding_window, at) + @as(u64, 40) * kept.kept(false, state.sliding_window, at);
    return rows * @as(u64, state.kv_width) * 4 + state.conv_bytes;
}

/// A cache's visits as the shared walk reads them: a slot's K and V, each head-major, and its ring.
fn visits(cache: *const state.Cache) [state.visit_slots]kept.Visit {
    var out: [state.visit_slots]kept.Visit = undefined;
    for (&out, 0..) |*v, slot| v.* = .{ .k = cache.key_at[slot], .v = cache.value_at[slot], .heads = state.kv_width / ap.head_dim, .ring = cache.ring(slot), .row = ap.head_dim * 2, .global = cache.globals[slot] };
    return out;
}

pub const Snap = struct {
    buf: mtl.Buffer,
    at: usize,
    bytes: u64,
    freed: bool = false,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn free(s: *Snap) void {
        if (!s.freed) {
            s.setResident(null);
            s.buf.deinit();
            s.freed = true;
        }
    }

    pub fn deinit(s: *Snap) void {
        s.free();
    }

    pub fn setResident(s: *Snap, hook: ?mtl.ResidencySet.Follow) void {
        if (s.freed) return;
        if (s.follow) |f| f.leave(f.ctx, s.buf);
        s.follow = hook;
        if (hook) |f| f.join(f.ctx, s.buf);
    }
};

fn encode(queue: mtl.Queue, cache: *const state.Cache, snap_buf: mtl.Buffer, snap: ?*const Snap, comptime what: []const u8) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    const len = if (snap) |image| image.at else cache.len;
    // Rows go by position, so a state restores into rings of another size; a wrapped ring is two pieces a head.
    const all = visits(cache);
    var walk = kept.Walk.init(&all, state.sliding_window, len);
    while (walk.next()) |p| {
        const in_cache: usize = @intCast(p.cache);
        const in_state: usize = @intCast(p.offset);
        if (snap) |image| e.copy(image.buf, in_state, cache.kv, in_cache, @intCast(p.len)) else e.copy(cache.kv, in_cache, snap_buf, in_state, @intCast(p.len));
    }
    const at: usize = @intCast(walk.offset);
    if (snap) |image| e.copy(image.buf, at, cache.conv, 0, state.conv_bytes) else e.copy(cache.conv, 0, snap_buf, at, state.conv_bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop snapshot {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

/// GPU-private snapshots keep only the rows a later row can read, never full-capacity allocations.
pub fn save(device: mtl.Device, queue: mtl.Queue, cache: *const state.Cache) !Snap {
    if (cache.pending != null) return error.Inkling2CacheTransaction;
    const size = bytes(cache.len);
    const buf = try device.buffer(@intCast(size), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try encode(queue, cache, buf, null, "save");
    return .{ .buf = buf, .at = cache.len, .bytes = size };
}

/// Restore into an empty cache whose rings hold the kept rows; they may be of another size than the saver's.
pub fn restore(cache: *state.Cache, queue: mtl.Queue, snap: *const Snap) !void {
    if (snap.freed or cache.len != 0 or cache.pending != null or snap.at > cache.capacity or snap.bytes < bytes(snap.at)) return error.Inkling2SnapshotGeometry;
    if (!kept.fits(&visits(cache), state.sliding_window, snap.at)) return error.Inkling2SnapshotGeometry;
    try encode(queue, cache, snap.buf, snap, "restore");
    cache.len = snap.at;
}
