//! A stream's prompt state kept for reuse: full layers' first `at` keys and values, the sliding rings' live rows.
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

/// Most pieces a state holds: two a sliding layer's keys and values (a ring may wrap), one a full layer's.
const max_pieces = 4 * cfg.max_layers;

/// Positions [lo, hi) of a ring of `ring` slots, `width` bytes each, from `base`: one piece, or two where it wraps.
fn ring(base: usize, ring_: usize, width: usize, lo: usize, hi: usize, out: *std.ArrayList(Piece)) void {
    if (hi <= lo) return;
    const first = lo % ring_;
    const n = @min(hi - lo, ring_ - first);
    out.appendAssumeCapacity(.{ .off = base + first * width, .len = n * width });
    if (n < hi - lo) out.appendAssumeCapacity(.{ .off = base, .len = (hi - lo - n) * width });
}

/// The pieces a state at `at` holds, in the snapshot's order: what the next rows read, nothing older.
fn pieces(c: *const cfg.Config, cache: *const st.Cache, at: usize, out: *std.ArrayList(Piece)) void {
    const row: usize = c.kvWidth() * 2;
    for (c.layer[0..c.layers], 0..) |l, i| {
        if (l.kind == .sliding) { // the window behind the next row, from its 64-key block: masked keys' values stay real
            const lo = slidingFrom(c, at);
            ring(cache.keys(i), st.slidingRing(c), row, lo, at, out);
            ring(cache.values(c, i), st.slidingRing(c), row, lo, at, out);
            continue;
        }
        out.appendAssumeCapacity(.{ .off = cache.keys(i), .len = at * row });
        out.appendAssumeCapacity(.{ .off = cache.values(c, i), .len = at * row });
    }
}

/// A sliding layer's first kept position: its window's 64-key block start (masked keys' values meet a 0, never NaN).
fn slidingFrom(c: *const cfg.Config, at: usize) usize {
    return std.mem.alignBackward(usize, at -| (c.window - 1), 64);
}

/// Bytes a state after `at` prompt tokens takes.
pub fn bytes(c: *const cfg.Config, at: usize) u64 {
    const row: u64 = c.kvWidth() * 2;
    var total: u64 = 0;
    for (c.layer[0..c.layers]) |l| total += 2 * row * (if (l.kind == .sliding) at - slidingFrom(c, at) else at);
    return total;
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
    if (snap.freed or snap.at >= cache.capacity) return error.FlintSnapshotGone;
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
        std.log.err("flint prompt state copy: {s}", .{msg});
        return error.GpuFailed;
    }
}

test "flint snapshot: a state's bytes match its pieces, rings wrapped or not" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("config_fixture.json"));
    const cache = st.Cache{ .kv = undefined, .logits = undefined, .capacity = 1 << 20 };
    for ([_]usize{ 5, 300, 512, 513, 1000, 2600, 100_003 }) |at| {
        var store: [max_pieces]Piece = undefined;
        var list: std.ArrayList(Piece) = .initBuffer(&store);
        pieces(&c, &cache, at, &list);
        var sum: u64 = 0;
        for (list.items) |p| sum += p.len;
        try std.testing.expectEqual(bytes(&c, at), sum);
    }
}
