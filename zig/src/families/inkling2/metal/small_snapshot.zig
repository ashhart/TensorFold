//! Compact retained snapshots of every live KV ring prefix and all convolution states for Inkling 2 Small.
const std = @import("std");
const mtl = @import("metal");
const state = @import("small_state.zig");
const ap = @import("attention_plan.zig");

/// Full 48-layer Small snapshot size at `at` tokens: 8 global prefixes, 40 sliding rings, 4 conv histories per layer.
pub fn bytes(at: usize) u64 {
    const keys = @as(u64, 8) * @as(u64, @intCast(at)) + @as(u64, 40) * @as(u64, @intCast(@min(at, state.sliding_ring)));
    return keys * @as(u64, state.kv_width) * 4 + state.conv_bytes;
}

/// Exact snapshot bytes for `cache` (supports both full 48-layer models and smaller synthetic test caches).
pub fn bytesFor(cache: *const state.Cache, at: usize) u64 {
    var keys: u64 = 0;
    for (0..cache.slots) |slot| {
        const live = if (cache.globals[slot]) at else @min(at, cache.ring(slot));
        keys += @as(u64, @intCast(live));
    }
    return keys * @as(u64, state.kv_width) * 4 + cache.conv_total_bytes;
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
    for (0..cache.slots) |slot| {
        const live = if (cache.globals[slot]) len else @min(len, cache.ring(slot));
        const n = live * ap.head_dim * 2;
        const head = cache.ring(slot) * ap.head_dim * 2;
        if (n != 0) for ([_]usize{ cache.key_at[slot], cache.value_at[slot] }) |base| for (0..state.kv_width / ap.head_dim) |h| {
            if (snap) |image| e.copy(image.buf, at, cache.kv, base + h * head, n) else e.copy(cache.kv, base + h * head, snap_buf, at, n);
            at += n;
        };
    }
    if (snap) |image| e.copy(image.buf, at, cache.conv, 0, cache.conv_total_bytes) else e.copy(cache.conv, 0, snap_buf, at, cache.conv_total_bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_small snapshot {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

/// GPU-private snapshots retain only live global and sliding prefixes, never full-capacity allocations.
pub fn save(device: mtl.Device, queue: mtl.Queue, cache: *const state.Cache) !Snap {
    if (cache.pending != null) return error.Inkling2CacheTransaction;
    const size = bytesFor(cache, cache.len);
    const buf = try device.buffer(@intCast(@max(size, 16)), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try encode(queue, cache, buf, null, "save");
    return .{ .buf = buf, .at = cache.len, .bytes = size };
}

/// Restore into an empty, large-enough cache; global capacity may grow while sliding positions stay unchanged.
pub fn restore(cache: *state.Cache, queue: mtl.Queue, snap: *const Snap) !void {
    if (snap.freed or cache.len != 0 or cache.pending != null or snap.at > cache.capacity or snap.bytes < bytesFor(cache, snap.at)) {
        return error.Inkling2SnapshotGeometry;
    }
    try encode(queue, cache, snap.buf, snap, "restore");
    cache.len = snap.at;
}
