//! A stream's prompt state kept for reuse on Metal: every layer's first `at` keys and values.
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

/// Bytes a state after `at` prompt tokens takes: the keys and values of `at` positions in every layer.
pub fn bytes(c: *const cfg.Config, at: usize) u64 {
    return @as(u64, c.layers) * st.kvBytes(c) * at;
}

/// Copy `cache`'s state at its length into a new snapshot.
pub fn save(device: mtl.Device, queue: mtl.Queue, c: *const cfg.Config, cache: *const st.Cache) !Snap {
    const n = bytes(c, cache.len);
    const buf = try device.buffer(@intCast(@max(n, 16)), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try copy(queue, cache, buf, true);
    if (cache.follow) |f| f.join(f.ctx, buf);
    return .{ .buf = buf, .at = @intCast(cache.len), .bytes = n, .follow = cache.follow };
}

/// Make `cache` (empty, any capacity past the state) hold `snap`'s state.
pub fn restore(queue: mtl.Queue, snap: *const Snap, cache: *st.Cache) !void {
    if (snap.freed or snap.at >= cache.capacity) return error.LlamaSnapshotGone;
    cache.len = snap.at;
    errdefer cache.len = 0;
    try copy(queue, cache, snap.buf, false);
}

/// The pieces between the cache and a snapshot buffer (a layer's keys, then its values), one blit pass, waited on.
fn copy(queue: mtl.Queue, cache: *const st.Cache, buf: mtl.Buffer, out: bool) !void {
    const row = cache.kv_width * 2;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var at: usize = 0;
    for (0..cache.layers) |l| {
        for ([_]usize{ cache.keys(l), cache.values(l) }) |off| {
            const len = cache.len * row;
            if (len == 0) continue;
            if (out) e.copy(cache.kv, off, buf, at, len) else e.copy(buf, at, cache.kv, off, len);
            at += len;
        }
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("llama prompt state copy: {s}", .{msg});
        return error.GpuFailed;
    }
}

test "llama snapshot: a state's bytes are every layer's keys and values to its length" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    try std.testing.expectEqual(@as(u64, 0), bytes(&c, 0));
    try std.testing.expectEqual(@as(u64, 52 * 4096 * 1000), bytes(&c, 1000));
    try std.testing.expectEqual(st.cacheBytes(&c, 4096) - c.vocab * 2, bytes(&c, 4096));
}
