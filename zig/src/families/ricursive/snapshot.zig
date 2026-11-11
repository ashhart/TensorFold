//! A stream's prompt state kept for reuse on Metal: each layer's held keys and values (a sliding layer's ring).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");

/// One kept state: its bytes on the GPU (private) and the prompt tokens it holds.
pub const Snap = struct {
    buf: mtl.Buffer,
    at: u32,
    bytes: u64,
    freed: bool = false, // its storage given back early (a stream needed the room); its entry goes at the pass's report
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn free(self: *Snap) void {
        if (!self.freed) {
            if (self.follow) |f| f.leave(f.ctx, self.buf);
            self.buf.deinit();
        }
        self.freed = true;
    }
};

/// Slots of layer `l` a state after `at` tokens holds: every position, or a sliding layer's full ring past it.
fn held(c: *const cfg.Config, l: usize, at: usize) usize {
    return if (c.global[l]) at else @min(at, st.ring(c, st.max_capacity));
}

/// Bytes a state after `at` prompt tokens takes.
pub fn bytes(c: *const cfg.Config, at: usize) u64 {
    var total: u64 = 0;
    for (0..c.layers) |l| total += @as(u64, held(c, l, at)) * st.kvBytes(c);
    return total;
}

/// Copy `cache`'s state at its length into a new snapshot.
pub fn save(device: mtl.Device, queue: mtl.Queue, c: *const cfg.Config, cache: *const st.Cache) !Snap {
    const n = bytes(c, cache.len);
    const buf = try device.buffer(@intCast(@max(n, 16)), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try copy(queue, c, cache, cache.len, buf, true);
    if (cache.follow) |f| f.join(f.ctx, buf);
    return .{ .buf = buf, .at = @intCast(cache.len), .bytes = n, .follow = cache.follow };
}

/// Make `cache` (empty, any capacity past the state) hold `snap`'s state: a ring past its size is a full ring on both.
pub fn restore(queue: mtl.Queue, c: *const cfg.Config, snap: *const Snap, cache: *st.Cache) !void {
    if (snap.freed or snap.at >= cache.capacity) return error.RicursiveSnapshotGone;
    for (0..c.layers) |l| if (held(c, l, snap.at) > cache.layer_slots[l]) return error.RicursiveSnapshotGone;
    try copy(queue, c, cache, snap.at, snap.buf, false);
    cache.len = snap.at;
}

/// The pieces between the cache and a snapshot buffer (a layer's keys, then its values), one blit pass, waited on.
fn copy(queue: mtl.Queue, c: *const cfg.Config, cache: *const st.Cache, at: usize, buf: mtl.Buffer, out: bool) !void {
    const row = cache.kv_width * 2;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var off: usize = 0;
    for (0..cache.layers) |l| {
        const len = held(c, l, at) * row;
        if (len == 0) continue;
        for ([_]usize{ cache.keys(l), cache.values(l) }) |from| {
            if (out) e.copy(cache.kv, from, buf, off, len) else e.copy(buf, off, cache.kv, from, len);
            off += len;
        }
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("ricursive prompt state copy: {s}", .{msg});
        return error.GpuFailed;
    }
}

test "ricursive snapshot: global layers keep every position, sliding layers at most their ring" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    try std.testing.expectEqual(@as(u64, 0), bytes(&c, 0));
    try std.testing.expectEqual(@as(u64, 80 * 1000 * 4096), bytes(&c, 1000));
    try std.testing.expectEqual(@as(u64, (16 * 100000 + 64 * 3072) * 4096), bytes(&c, 100000));
    try std.testing.expectEqual(st.cacheBytes(&c, 262144) - c.vocab * 2, bytes(&c, 262144));
}
