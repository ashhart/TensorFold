//! Compact retained snapshots of every live KV ring prefix and all convolution states.
const std = @import("std");
const mtl = @import("metal");
const state = @import("state.zig");
const ap = @import("attention_plan.zig");

/// Eight global layers retain [0,at); forty sliding layers retain ring prefixes, with independent K/V pieces per visit.
pub fn bytes(at: usize) u64 {
    const keys = @as(u64, 8) * @as(u64, @intCast(at)) + @as(u64, 40) * @as(u64, @intCast(@min(at, state.sliding_ring)));
    return keys * @as(u64, state.kv_width) * 4 + state.conv_bytes;
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
    var at: usize = 0;
    const len = if (snap) |image| image.at else cache.len;
    for (0..state.visit_slots) |slot| {
        const live = if (cache.globals[slot]) len else @min(len, cache.ring(slot));
        // A slot holds its heads one after another, each the ring's rows: a head's live rows are its first `live`.
        const n = live * ap.head_dim * 2;
        const head = cache.ring(slot) * ap.head_dim * 2;
        if (n != 0) for ([_]usize{ cache.key_at[slot], cache.value_at[slot] }) |base| for (0..state.kv_width / ap.head_dim) |h| {
            if (snap) |image| e.copy(image.buf, at, cache.kv, base + h * head, n) else e.copy(cache.kv, base + h * head, snap_buf, at, n);
            at += n;
        };
    }
    if (snap) |image| e.copy(image.buf, at, cache.conv, 0, state.conv_bytes) else e.copy(cache.conv, 0, snap_buf, at, state.conv_bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop snapshot {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

/// GPU-private snapshots retain only live global and sliding prefixes, never full-capacity allocations.
pub fn save(device: mtl.Device, queue: mtl.Queue, cache: *const state.Cache) !Snap {
    try cache.ensureUsable();
    if (cache.pending != null) return error.Inkling2CacheTransaction;
    const size = bytes(cache.len);
    const buf = try device.buffer(@intCast(size), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try encode(queue, cache, buf, null, "save");
    return .{ .buf = buf, .at = cache.len, .bytes = size };
}

/// Restore into an empty, large-enough cache; global capacity may grow while sliding positions stay unchanged.
pub fn restore(cache: *state.Cache, queue: mtl.Queue, snap: *const Snap) !void {
    try cache.ensureUsable();
    if (snap.freed or cache.len != 0 or cache.pending != null or snap.at > cache.capacity or snap.bytes < bytes(snap.at)) return error.Inkling2SnapshotGeometry;
    try encode(queue, cache, snap.buf, snap, "restore");
    cache.len = snap.at;
}
