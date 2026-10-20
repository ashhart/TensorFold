//! A stream's prompt state kept for reuse: each sliding ring's last window of keys and values, the global layers' all.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");

/// One kept state: its bytes on the GPU (private) and the prompt tokens it holds.
pub const Snap = struct {
    buf: mtl.Buffer,
    at: u32,
    bytes: u64,
    /// The DFlash drafter's context for the same stream at `at` (dflash.zig keepLane), when it had one there.
    draft: ?mtl.Buffer = null,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn free(self: *Snap) void {
        if (self.follow) |f| {
            f.leave(f.ctx, self.buf);
            if (self.draft) |b| f.leave(f.ctx, b);
        }
        self.buf.deinit();
        if (self.draft) |b| b.deinit();
    }

    pub fn setDraft(self: *Snap, buffer: mtl.Buffer) void {
        if (self.draft) |old| {
            if (self.follow) |f| f.leave(f.ctx, old);
            old.deinit();
        }
        self.draft = buffer;
        if (self.follow) |f| f.join(f.ctx, buffer);
    }
};

/// One contiguous piece of a layer's key (or value) buffer: its offset and length in bytes.
const Piece = struct { layer: usize, value: bool, off: usize, len: usize };

/// Positions [lo, hi) of one head's ring of `width`-byte slots from `base`: one piece, or two where it wraps.
fn ring(layer: usize, value: bool, base: usize, width: usize, lo: usize, hi: usize, out: *std.ArrayList(Piece), gpa: std.mem.Allocator) !void {
    if (hi <= lo) return;
    const first = lo % st.ring;
    const n = @min(hi - lo, st.ring - first);
    try out.append(gpa, .{ .layer = layer, .value = value, .off = base + first * width, .len = n * width });
    if (n < hi - lo) try out.append(gpa, .{ .layer = layer, .value = value, .off = base, .len = (hi - lo - n) * width });
}

/// The pieces of `cache` (its capacity places the global heads) a state at `at` holds: what the next rows read.
fn pieces(cache: *const st.Cache, at: usize, out: *std.ArrayList(Piece), gpa: std.mem.Allocator) !void {
    for (0..cache.geo.layers) |i| {
        const width = cfg.headDim(i) * 2;
        const slots = st.slotsOf(i, cache.capacity);
        for (0..cache.geo.kvHeads(i)) |h| for ([_]bool{ false, true }) |value| {
            const base = h * slots * width;
            if (cfg.global(i)) {
                try out.append(gpa, .{ .layer = i, .value = value, .off = base, .len = at * width });
            } else try ring(i, value, base, width, at -| cfg.window, at, out, gpa);
        };
    }
}

/// Bytes a state after `at` prompt tokens takes.
pub fn bytes(geo: cfg.Geometry, at: usize) u64 {
    var total: u64 = 0;
    for (0..geo.layers) |i| {
        const rows = if (cfg.global(i)) at else @min(at, cfg.window);
        total += 2 * geo.kvHeads(i) * rows * cfg.headDim(i) * 2;
    }
    return total;
}

/// Copy `cache`'s state at its length into a new snapshot.
pub fn save(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, cache: *const st.Cache) !Snap {
    const n = bytes(cache.geo, cache.len);
    const buf = try device.buffer(@intCast(@max(n, 16)), mtl.ResourceOptions.private);
    errdefer buf.deinit();
    try copy(gpa, queue, cache, cache.len, buf, true);
    if (cache.follow) |f| f.join(f.ctx, buf);
    return .{ .buf = buf, .at = @intCast(cache.len), .bytes = n, .follow = cache.follow };
}

/// Make `cache` (empty, its capacity past the state) hold `snap`'s state.
pub fn restore(gpa: std.mem.Allocator, queue: mtl.Queue, snap: *const Snap, cache: *st.Cache) !void {
    if (snap.at >= cache.capacity) return error.GemmaSnapshotTooLong;
    try copy(gpa, queue, cache, snap.at, snap.buf, false);
    cache.len = snap.at;
}

/// The pieces between the cache and a snapshot buffer, one blit pass, waited on.
fn copy(gpa: std.mem.Allocator, queue: mtl.Queue, cache: *const st.Cache, at: usize, buf: mtl.Buffer, out: bool) !void {
    var list: std.ArrayList(Piece) = .empty;
    defer list.deinit(gpa);
    try pieces(cache, at, &list, gpa);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var to: usize = 0;
    for (list.items) |p| {
        if (p.len == 0) continue;
        const live = if (p.value) cache.v[p.layer] else cache.k[p.layer];
        if (out) e.copy(live, p.off, buf, to, p.len) else e.copy(buf, to, live, p.off, p.len);
        to += p.len;
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("gemma4 prompt state copy: {s}", .{msg});
        return error.GpuFailed;
    }
}

test "a state's bytes match its pieces, rings wrapped or not" {
    for ([_]cfg.Geometry{ cfg.dense31b, cfg.unified12b }) |geo| for ([_]usize{ 1, 5, 1000, 1024, 1500, 1536, 2600, 100_003 }) |at| {
        var cache = st.Cache{ .k = undefined, .v = undefined, .geo = geo, .logits = undefined, .capacity = 1 << 18 };
        var list: std.ArrayList(Piece) = .empty;
        defer list.deinit(std.testing.allocator);
        try pieces(&cache, at, &list, std.testing.allocator);
        var sum: u64 = 0;
        for (list.items) |p| {
            sum += p.len;
            const slots = st.slotsOf(p.layer, cache.capacity);
            try std.testing.expect(p.off + p.len <= geo.kvHeads(p.layer) * slots * cfg.headDim(p.layer) * 2);
        }
        try std.testing.expectEqual(bytes(geo, at), sum);
    };
}
