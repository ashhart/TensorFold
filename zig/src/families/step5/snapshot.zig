//! A stream's prompt state kept for reuse: full layers' first `at` keys, values and summaries, every ring whole.
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

    pub fn free(self: *Snap) void {
        if (!self.freed) self.buf.deinit();
        self.freed = true;
    }
};

/// One contiguous piece of a cache (offset into its kv or index buffer) and its length.
const Piece = struct { index: bool, off: usize, len: usize };

/// The pieces a state at `at` holds, in the snapshot's order (`cache` null: lengths only, offsets 0).
fn pieces(c: *const cfg.Config, cache: ?*const st.Cache, at: usize, out: *std.ArrayList(Piece), gpa: std.mem.Allocator) !void {
    const row: usize = c.kvWidth() * 2;
    for (c.layer[0 .. c.layers + c.mtp], 0..) |l, i| {
        if (l.kind == .sliding) {
            try out.append(gpa, .{ .index = false, .off = if (cache) |k| k.keys(i) else 0, .len = st.slidingRing(c) * row * 2 });
            continue;
        }
        try out.append(gpa, .{ .index = false, .off = if (cache) |k| k.keys(i) else 0, .len = at * row });
        try out.append(gpa, .{ .index = false, .off = if (cache) |k| k.values(c, i) else 0, .len = at * row });
        try out.append(gpa, .{ .index = true, .off = if (cache) |k| k.summaries(i) else 0, .len = at / c.region * c.proxy * 2 });
        try out.append(gpa, .{ .index = true, .off = if (cache) |k| k.indexKeys(c, i) else 0, .len = 2 * st.index_ring * c.proxy * 2 });
    }
}

/// Bytes a state after `at` prompt tokens takes.
pub fn bytes(c: *const cfg.Config, at: usize) u64 {
    const row: u64 = c.kvWidth() * 2;
    var total: u64 = 0;
    for (c.layer[0 .. c.layers + c.mtp]) |l| total += if (l.kind == .sliding) st.slidingRing(c) * row * 2 else 2 * at * row + at / c.region * c.proxy * 2 + 2 * st.index_ring * c.proxy * 2;
    return total;
}

/// Copy `cache`'s state at its length into a new snapshot.
pub fn save(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, c: *const cfg.Config, cache: *const st.Cache) !Snap {
    const n = bytes(c, cache.len);
    const buf = try device.buffer(@intCast(n), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try copy(gpa, queue, c, cache, buf, true);
    return .{ .buf = buf, .at = @intCast(cache.len), .bytes = n };
}

/// Make `cache` (empty, any capacity past the state) hold `snap`'s state.
pub fn restore(gpa: std.mem.Allocator, queue: mtl.Queue, c: *const cfg.Config, snap: *const Snap, cache: *st.Cache) !void {
    if (snap.freed or snap.at >= cache.capacity) return error.Step5SnapshotGone;
    cache.len = snap.at;
    errdefer cache.len = 0;
    try copy(gpa, queue, c, cache, snap.buf, false);
}

/// The pieces between the cache and a snapshot buffer, one blit pass, waited on.
fn copy(gpa: std.mem.Allocator, queue: mtl.Queue, c: *const cfg.Config, cache: *const st.Cache, buf: mtl.Buffer, out: bool) !void {
    var list: std.ArrayList(Piece) = .empty;
    defer list.deinit(gpa);
    try pieces(c, cache, cache.len, &list, gpa);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var at: usize = 0;
    for (list.items) |p| {
        if (p.len == 0) continue;
        const live = if (p.index) cache.index else cache.kv;
        if (out) e.copy(live, p.off, buf, at, p.len) else e.copy(buf, at, live, p.off, p.len);
        at += p.len;
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("step5 prompt state copy: {s}", .{msg});
        return error.GpuFailed;
    }
}

test "a state's bytes match its pieces" {
    var c = cfg.Config{};
    c.layers = 4;
    c.mtp = 0;
    for (c.layer[0..4], 0..) |*l, i| l.* = .{ .kind = if (i == 3) .full else .sliding, .span = 0, .theta = 0, .limit = 0, .shared_limit = 0, .moe = false };
    var list: std.ArrayList(Piece) = .empty;
    defer list.deinit(std.testing.allocator);
    try pieces(&c, null, 1000, &list, std.testing.allocator);
    var sum: u64 = 0;
    for (list.items) |p| sum += p.len;
    try std.testing.expectEqual(bytes(&c, 1000), sum);
}
