//! A stream's prompt state kept for reuse on Metal: full layers' first `at` keys and values, the rings' live rows.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("metal_state.zig");

const geo = cfg.spark4b;

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
const max_pieces = 4 * geo.layers;

/// Bytes a position takes in one layer's keys (or values).
const row: usize = geo.kvWidth() * 2;

/// Positions [lo, hi) of the ring from `base`: one piece, or two where it wraps.
fn ring(base: usize, lo: usize, hi: usize, out: *std.ArrayList(Piece)) void {
    if (hi <= lo) return;
    const first = lo % st.sliding_ring;
    const n = @min(hi - lo, st.sliding_ring - first);
    out.appendAssumeCapacity(.{ .off = base + first * row, .len = n * row });
    if (n < hi - lo) out.appendAssumeCapacity(.{ .off = base, .len = (hi - lo - n) * row });
}

/// The pieces a state at `at` holds, in the snapshot's order: what the next rows read, nothing older.
fn pieces(cache: *const st.Cache, at: usize, out: *std.ArrayList(Piece)) void {
    for (0..geo.layers) |i| {
        if (!cfg.full(i)) { // the window behind the next row, from its 64-key block: masked keys' values stay real
            const lo = slidingFrom(at);
            ring(cache.keys(i), lo, at, out);
            ring(cache.values(i), lo, at, out);
            continue;
        }
        out.appendAssumeCapacity(.{ .off = cache.keys(i), .len = at * row });
        out.appendAssumeCapacity(.{ .off = cache.values(i), .len = at * row });
    }
}

/// A sliding layer's first kept position: its window's 64-key block start (masked keys' values meet a 0, never NaN).
fn slidingFrom(at: usize) usize {
    return std.mem.alignBackward(usize, at -| (cfg.window - 1), 64);
}

/// Bytes a state after `at` prompt tokens takes.
pub fn bytes(at: usize) u64 {
    var total: u64 = 0;
    for (0..geo.layers) |i| total += 2 * row * (if (cfg.full(i)) at else at - slidingFrom(at));
    return total;
}

/// Copy `cache`'s state at its length into a new snapshot.
pub fn save(device: mtl.Device, queue: mtl.Queue, cache: *const st.Cache) !Snap {
    const n = bytes(cache.len);
    const buf = try device.buffer(@intCast(@max(n, 16)), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try copy(queue, cache, buf, true);
    if (cache.follow) |f| f.join(f.ctx, buf);
    return .{ .buf = buf, .at = @intCast(cache.len), .bytes = n, .follow = cache.follow };
}

/// Make `cache` (empty, any capacity past the state) hold `snap`'s state.
pub fn restore(queue: mtl.Queue, snap: *const Snap, cache: *st.Cache) !void {
    if (snap.freed or snap.at >= cache.capacity) return error.SparkSnapshotGone;
    cache.len = snap.at;
    errdefer cache.len = 0;
    try copy(queue, cache, snap.buf, false);
}

/// The pieces between the cache and a snapshot buffer, one blit pass, waited on.
fn copy(queue: mtl.Queue, cache: *const st.Cache, buf: mtl.Buffer, out: bool) !void {
    var store: [max_pieces]Piece = undefined;
    var list: std.ArrayList(Piece) = .initBuffer(&store);
    pieces(cache, cache.len, &list);
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
        std.log.err("spark2_5 prompt state copy: {s}", .{msg});
        return error.GpuFailed;
    }
}

test "spark metal snapshot: a state's bytes match its pieces, rings wrapped or not" {
    const cache = st.Cache{ .kv = undefined, .logits = undefined, .capacity = 1 << 20 };
    for ([_]usize{ 5, 300, 511, 512, 513, 1000, 2600, 100_003 }) |at| {
        var store: [max_pieces]Piece = undefined;
        var list: std.ArrayList(Piece) = .initBuffer(&store);
        pieces(&cache, at, &list);
        var sum: u64 = 0;
        for (list.items) |p| sum += p.len;
        try std.testing.expectEqual(bytes(at), sum);
    }
}
