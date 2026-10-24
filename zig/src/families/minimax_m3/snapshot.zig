//! A stream's prompt state kept for reuse: every layer's first `at` keys, values and index keys.
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

/// One contiguous piece of a cache's kv buffer (offset) and its length.
const Piece = struct { off: usize, len: usize };

/// Most pieces a state holds: a layer's keys, values, their scales (int8) and index keys.
const max_pieces = 5 * cfg.max_layers;

/// A state's pieces at `at`, in order: every layer's first `at` keys, values, scales and index keys.
fn pieces(c: *const cfg.Config, cache: *const st.Cache, at: usize, out: *std.ArrayList(Piece)) void {
    const row: usize = c.kvWidth() * c.kvBytes();
    for (0..c.layers) |i| {
        out.appendAssumeCapacity(.{ .off = cache.keys(i), .len = at * row });
        out.appendAssumeCapacity(.{ .off = cache.values(c, i), .len = at * row });
        if (c.kv8) {
            out.appendAssumeCapacity(.{ .off = cache.keyScales(c, i), .len = at * @as(usize, c.kv_heads) * 4 });
            out.appendAssumeCapacity(.{ .off = cache.valueScales(c, i), .len = at * @as(usize, c.kv_heads) * 4 });
        }
        out.appendAssumeCapacity(.{ .off = cache.indexKeys(c, i), .len = at * @as(usize, c.index_dim) * 2 });
    }
}

/// Bytes a state after `at` prompt tokens takes.
pub fn bytes(c: *const cfg.Config, at: usize) u64 {
    return @as(u64, c.layers) * at * st.rowBytes(c);
}

/// Copy `cache`'s state at its length into a new snapshot.
pub fn save(device: mtl.Device, queue: mtl.Queue, c: *const cfg.Config, cache: *const st.Cache) !Snap {
    const n = bytes(c, cache.len);
    const buf = try device.buffer(@intCast(@max(n, 16)), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try copy(queue, c, cache, buf, true);
    if (cache.follow) |f| f.join(f.ctx, buf);
    return .{ .buf = buf, .at = @intCast(cache.len), .bytes = n, .follow = cache.follow };
}

/// Make `cache` (empty, any capacity past the state) hold `snap`'s state.
pub fn restore(queue: mtl.Queue, c: *const cfg.Config, snap: *const Snap, cache: *st.Cache) !void {
    if (snap.freed or snap.at >= cache.capacity) return error.MiniMaxSnapshotGone;
    cache.len = snap.at;
    errdefer cache.len = 0;
    try copy(queue, c, cache, snap.buf, false);
}

/// The pieces between the cache and a snapshot buffer, one blit pass, waited on.
fn copy(queue: mtl.Queue, c: *const cfg.Config, cache: *const st.Cache, buf: mtl.Buffer, out: bool) !void {
    var store: [max_pieces]Piece = undefined;
    var list: std.ArrayList(Piece) = .initBuffer(&store);
    pieces(c, cache, cache.len, &list);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var at: usize = 0;
    for (list.items) |p| {
        if (p.len == 0) continue;
        if (out) e.copy(cache.kv, p.off, buf, at, p.len) else e.copy(buf, at, cache.kv, p.off, p.len);
        at += p.len;
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("minimax_m3 prompt state copy: {s}", .{msg});
        return error.GpuFailed;
    }
}

test "minimax_m3 snapshot: a state's bytes match its pieces" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("config_fixture.json"));
    const cache = st.Cache{ .kv = undefined, .logits = undefined, .capacity = 1 << 20 };
    for ([_]usize{ 5, 128, 2049, 100_003 }) |at| {
        var store: [max_pieces]Piece = undefined;
        var list: std.ArrayList(Piece) = .initBuffer(&store);
        pieces(&c, &cache, at, &list);
        var sum: u64 = 0;
        for (list.items) |p| sum += p.len;
        try std.testing.expectEqual(bytes(&c, at), sum);
    }
}
