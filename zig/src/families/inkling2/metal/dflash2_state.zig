//! Fixed DFlash 2 scratch, per-stream sliding K/V rings, and GPU-private prompt-cache snapshots.
const std = @import("std");
const mtl = @import("metal");
const w = @import("dflash2_weights.zig");
const st = @import("state.zig");

pub const absorb_rows: usize = 256;
pub const max_chunks: usize = (w.window_size + w.block_size + 127) / 128;
pub const max_context: u64 = 1048576;
pub const tap_row_bytes: usize = st.feature_row_bytes;
pub const row_bytes: usize = w.kv_width * 2;
pub const layer_row_bytes: u64 = @as(u64, w.layers_n) * 2 * row_bytes;
pub const Tail = struct { buf: mtl.Buffer, off: usize = 0 };

const scratch_fields = [_][]const u8{
    "ids",  "resid", "x",  "xc",   "dyn", "q_raw", "k_raw",  "v_raw", "q",    "kb",  "vb",  "part",
    "attn", "y",     "yc", "gate", "up",  "act",   "logits", "hid",   "tval", "tid", "seq", "fcx",
    "hctx", "ck",    "cv",
};

fn scratchSizes() [scratch_fields.len]usize {
    const n = w.block_size;
    const d = n - 1;
    const a = absorb_rows;
    const bf = 2;
    return .{
        n * 4,
        n * w.hidden * 4,
        n * w.hidden * bf,
        n * w.hidden * bf,
        n * w.dyn_width * bf,
        n * w.query_width * bf,
        n * w.kv_width * bf,
        n * w.kv_width * bf,
        n * w.query_width * bf,
        n * w.kv_width * bf,
        n * w.kv_width * bf,
        n * w.q_heads * max_chunks * 132 * 4,
        n * w.query_width * bf,
        n * w.hidden * bf,
        n * w.hidden * bf,
        n * w.intermediate * bf,
        n * w.intermediate * bf,
        n * w.intermediate * bf,
        d * w.padded_vocab * bf,
        d * w.rank * bf,
        d * w.top_k * 4,
        d * w.top_k * 4,
        n * 4,
        a * w.hidden * bf,
        a * w.hidden * bf,
        a * w.kv_width * bf,
        a * w.kv_width * bf,
    };
}

pub fn scratchBytes() u64 {
    var total: u64 = 0;
    for (scratchSizes()) |s| total += s;
    return total;
}

pub fn streamBytes() u64 {
    return @as(u64, w.layers_n) * 2 * w.ring_size * row_bytes;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer,
    x: mtl.Buffer,
    xc: mtl.Buffer,
    dyn: mtl.Buffer,
    q_raw: mtl.Buffer,
    k_raw: mtl.Buffer,
    v_raw: mtl.Buffer,
    q: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    part: mtl.Buffer,
    attn: mtl.Buffer,
    y: mtl.Buffer,
    yc: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    logits: mtl.Buffer,
    hid: mtl.Buffer,
    tval: mtl.Buffer,
    tid: mtl.Buffer,
    seq: mtl.Buffer,
    fcx: mtl.Buffer,
    hctx: mtl.Buffer,
    ck: mtl.Buffer,
    cv: mtl.Buffer,

    pub fn init(device: mtl.Device) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = scratchSizes();
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (scratch_fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (scratch_fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(ns[i], 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (scratch_fields) |f| @field(s, f).deinit();
        s.* = undefined;
    }
};

pub const Lane = struct {
    k: [w.layers_n]mtl.Buffer,
    v: [w.layers_n]mtl.Buffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [w.block_size - 1]u32 = @splat(0),
    held_len: u32 = 0,

    pub fn init(device: mtl.Device) !Lane {
        var l: Lane = .{ .k = undefined, .v = undefined };
        const bytes = w.ring_size * row_bytes;
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..w.layers_n) |i| {
            l.k[i] = try device.buffer(bytes, mtl.ResourceOptions.private);
            l.v[i] = device.buffer(bytes, mtl.ResourceOptions.private) catch |err| {
                l.k[i].deinit();
                return err;
            };
            made += 1;
        }
        return l;
    }

    pub fn deinit(l: *Lane) void {
        for (0..w.layers_n) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        l.* = undefined;
    }
};

/// Reset a lane back to position zero with no held drafts after a reset or failed restore.
pub fn clearLane(lane: *Lane) void {
    lane.from = 0;
    lane.end = 0;
    lane.held_len = 0;
}

/// Resolve an unabsorbed prompt-end or multi-stream tail tap row when `lane_end + 1 == at_pos`.
pub fn resolveTailTap(tap: st.Tap, at_pos: u32, lane_end: u64, tap_gen: u64) !?Tail {
    if (lane_end == at_pos) return null;
    if (lane_end + 1 != at_pos) return error.SnapshotOutOfStep;
    const buf = tap.buf orelse return error.SnapshotOutOfStep;
    if (tap.gen) |g| if (g != tap_gen) return error.StaleInkling2Taps;
    if (lane_end < tap.start) return error.SnapshotOutOfStep;
    const rows_before = std.math.cast(usize, lane_end - tap.start) orelse return error.SnapshotOutOfStep;
    if (rows_before >= tap.rows) return error.SnapshotOutOfStep;
    const delta = std.math.mul(usize, rows_before, tap_row_bytes) catch return error.SnapshotOutOfStep;
    const src_off = std.math.add(usize, tap.off, delta) catch return error.SnapshotOutOfStep;
    const src_end = std.math.add(usize, src_off, tap_row_bytes) catch return error.SnapshotOutOfStep;
    if (src_end > buf.length()) return error.SnapshotOutOfStep;
    return .{ .buf = buf, .off = src_off };
}

pub fn exactSnapBytes(rows: usize, has_tail: bool) u64 {
    return @as(u64, rows) * layer_row_bytes + if (has_tail) @as(u64, tap_row_bytes) else 0;
}

/// Upper bound on GPU-private snapshot bytes at `at` tokens (includes optional unabsorbed prompt-end tap row).
pub fn snapBytes(at: usize) u64 {
    return exactSnapBytes(@min(at, w.window_size), true);
}

/// Compact GPU-private snapshot of DFlash 2's active sliding K/V window [from, end) and optional tail tap.
pub const Snap = struct {
    buf: ?mtl.Buffer = null,
    from: u64,
    end: u64,
    bytes_held: u64,
    tail_off: ?usize = null,
    freed: bool = false,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn bytes(at: usize) u64 {
        return snapBytes(at);
    }

    pub fn free(s: *Snap) void {
        if (!s.freed) {
            s.setResident(null);
            if (s.buf) |b| b.deinit();
            s.buf = null;
            s.freed = true;
        }
    }

    pub fn deinit(s: *Snap) void {
        s.free();
    }

    pub fn setResident(s: *Snap, hook: ?mtl.ResidencySet.Follow) void {
        if (s.freed) return;
        const b = s.buf orelse return;
        if (s.follow) |f| f.leave(f.ctx, b);
        s.follow = hook;
        if (hook) |f| f.join(f.ctx, b);
    }

    pub fn save(device: mtl.Device, queue: mtl.Queue, lane: *const Lane, tail: ?Tail) !Snap {
        if (lane.from > lane.end or lane.end > max_context) return error.DFlash2SnapshotGeometry;
        for (0..w.layers_n) |i| {
            if (lane.k[i].length() < w.ring_size * row_bytes or lane.v[i].length() < w.ring_size * row_bytes) {
                return error.DFlash2SnapshotGeometry;
            }
        }
        if (tail) |t| {
            const tap_end = std.math.add(usize, t.off, tap_row_bytes) catch return error.DFlash2SnapshotGeometry;
            if (tap_end > t.buf.length()) return error.DFlash2SnapshotGeometry;
        }
        const save_from: u64 = @max(lane.from, lane.end -| w.window_size);
        const live_u64 = lane.end - save_from;
        if (live_u64 > w.window_size) return error.DFlash2SnapshotGeometry;
        const live_rows: usize = @intCast(live_u64);
        const has_tail = tail != null;
        const size = exactSnapBytes(live_rows, has_tail);
        if (size == 0) {
            return .{ .buf = null, .from = save_from, .end = lane.end, .bytes_held = 0, .tail_off = null };
        }
        const tail_off: ?usize = if (has_tail) @intCast(size - tap_row_bytes) else null;
        const buf = try device.buffer(@intCast(size), mtl.ResourceOptions.private);
        errdefer buf.deinit();
        try blitSnap(queue, lane, buf, save_from, live_rows, tail, tail_off, false, "save");
        return .{ .buf = buf, .from = save_from, .end = lane.end, .bytes_held = size, .tail_off = tail_off };
    }

    pub fn restore(lane: *Lane, queue: mtl.Queue, snap: *const Snap) !void {
        if (snap.freed or lane.end != 0 or lane.from != 0 or snap.from > snap.end or snap.end > max_context) {
            return error.DFlash2SnapshotGeometry;
        }
        const live_u64 = snap.end - snap.from;
        if (live_u64 > w.window_size) return error.DFlash2SnapshotGeometry;
        const live_rows: usize = @intCast(live_u64);
        const kv_bytes = exactSnapBytes(live_rows, false);
        const need = exactSnapBytes(live_rows, snap.tail_off != null);
        if (snap.bytes_held < need) return error.DFlash2SnapshotGeometry;
        for (0..w.layers_n) |i| {
            if (lane.k[i].length() < w.ring_size * row_bytes or lane.v[i].length() < w.ring_size * row_bytes) {
                return error.DFlash2SnapshotGeometry;
            }
        }
        if (need > 0) {
            const buf = snap.buf orelse return error.DFlash2SnapshotGeometry;
            if (buf.length() < need) return error.DFlash2SnapshotGeometry;
            if (snap.tail_off) |to| {
                if (to < kv_bytes) return error.DFlash2SnapshotGeometry;
                const tail_end = std.math.add(usize, to, tap_row_bytes) catch return error.DFlash2SnapshotGeometry;
                if (tail_end > snap.bytes_held or tail_end > buf.length()) return error.DFlash2SnapshotGeometry;
            }
            if (live_rows > 0) {
                try blitSnap(queue, lane, buf, snap.from, live_rows, null, null, true, "restore");
            }
        } else if (snap.tail_off != null) {
            return error.DFlash2SnapshotGeometry;
        }
        lane.from = snap.from;
        lane.end = snap.end;
        lane.held_len = 0;
    }
};

fn blitSnap(
    queue: mtl.Queue,
    lane: *const Lane,
    snap_buf: mtl.Buffer,
    save_from: u64,
    live_rows: usize,
    tail: ?Tail,
    tail_off: ?usize,
    restoring: bool,
    comptime what: []const u8,
) !void {
    if (live_rows == 0 and (restoring or tail == null)) return;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var at: usize = 0;
    if (live_rows > 0) {
        const start_slot: usize = @intCast(save_from % w.ring_size);
        const first_rows = @min(live_rows, w.ring_size - start_slot);
        const second_rows = live_rows - first_rows;
        const first_off = start_slot * row_bytes;
        const first_bytes = first_rows * row_bytes;
        const second_bytes = second_rows * row_bytes;
        for (0..w.layers_n) |i| {
            for ([_]mtl.Buffer{ lane.k[i], lane.v[i] }) |ring_buf| {
                if (restoring) {
                    e.copy(snap_buf, at, ring_buf, first_off, first_bytes);
                    if (second_bytes > 0) e.copy(snap_buf, at + first_bytes, ring_buf, 0, second_bytes);
                } else {
                    e.copy(ring_buf, first_off, snap_buf, at, first_bytes);
                    if (second_bytes > 0) e.copy(ring_buf, 0, snap_buf, at + first_bytes, second_bytes);
                }
                at += first_bytes + second_bytes;
            }
        }
    }
    if (!restoring) {
        if (tail) |t| e.copy(t.buf, t.off, snap_buf, tail_off.?, tap_row_bytes);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop DFlash2 snapshot {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

comptime {
    if (w.window_size != 2048 or w.ring_size != 2176 or layer_row_bytes != 20480) {
        @compileError("DFlash2 ring geometry drift");
    }
    if (st.feature_row_bytes != tap_row_bytes) {
        @compileError("DFlash2 tail tap row bytes mismatch");
    }
}

test "DFlash2 stream, scratch, and snapshot byte accounting fit fixed ring geometry" {
    try std.testing.expectEqual(@as(usize, 2176), w.ring_size);
    try std.testing.expectEqual(@as(u64, 5 * 2 * 2176 * 1024 * 2), streamBytes());
    try std.testing.expect(scratchBytes() > 0);
    try std.testing.expect(Snap.bytes(0) >= exactSnapBytes(0, true));
    try std.testing.expect(Snap.bytes(64) >= exactSnapBytes(64, false));
    try std.testing.expect(Snap.bytes(2048) >= exactSnapBytes(2048, true));
    try std.testing.expect(Snap.bytes(4096) >= exactSnapBytes(w.window_size, true));
    try std.testing.expectEqual(exactSnapBytes(w.window_size, true), Snap.bytes(2048));
    try std.testing.expectEqual(Snap.bytes(2048), Snap.bytes(8192));
}

test "DFlash2 resolveTailTap preserves non-first-stream tap offset and validates generation and bounds" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const buf = try device.buffer(6 * tap_row_bytes, mtl.ResourceOptions.shared);
    defer buf.deinit();
    const shared_tap: st.Tap = .{ .buf = buf, .off = 2 * tap_row_bytes, .start = 10, .rows = 3, .gen = 7 };
    try std.testing.expectEqual(@as(?Tail, null), try resolveTailTap(shared_tap, 13, 13, 7));
    const tail = (try resolveTailTap(shared_tap, 13, 12, 7)).?;
    try std.testing.expectEqual(buf.id, tail.buf.id);
    try std.testing.expectEqual(4 * tap_row_bytes, tail.off);
    const stable_tap: st.Tap = .{ .buf = buf, .off = 2 * tap_row_bytes, .start = 10, .rows = 3, .gen = null };
    try std.testing.expectEqual(4 * tap_row_bytes, (try resolveTailTap(stable_tap, 13, 12, 99)).?.off);
    try std.testing.expectError(error.StaleInkling2Taps, resolveTailTap(shared_tap, 13, 12, 8));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 14, 13, 7));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 13, 11, 7));
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(shared_tap, 10, 9, 7));
    const truncated: st.Tap = .{ .buf = buf, .off = 5 * tap_row_bytes + 1, .start = 10, .rows = 1, .gen = 7 };
    try std.testing.expectError(error.SnapshotOutOfStep, resolveTailTap(truncated, 11, 10, 7));
}

fn blitCopy(queue: mtl.Queue, src: mtl.Buffer, src_off: usize, dst: mtl.Buffer, dst_off: usize, bytes: usize) !void {
    if (bytes == 0) return;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    e.copy(src, src_off, dst, dst_off, bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

fn copyRingSpan(
    queue: mtl.Queue,
    ring_buf: mtl.Buffer,
    flat: mtl.Buffer,
    flat_off: usize,
    from: u64,
    rows: usize,
    to_ring: bool,
) !void {
    if (rows == 0) return;
    const start_slot: usize = @intCast(from % w.ring_size);
    const first_rows = @min(rows, w.ring_size - start_slot);
    const second_rows = rows - first_rows;
    const first_off = start_slot * row_bytes;
    const first_bytes = first_rows * row_bytes;
    const second_bytes = second_rows * row_bytes;
    if (to_ring) {
        try blitCopy(queue, flat, flat_off, ring_buf, first_off, first_bytes);
        try blitCopy(queue, flat, flat_off + first_bytes, ring_buf, 0, second_bytes);
    } else {
        try blitCopy(queue, ring_buf, first_off, flat, flat_off, first_bytes);
        try blitCopy(queue, ring_buf, 0, flat, flat_off + first_bytes, second_bytes);
    }
}

test "DFlash2 snapshot preserves ring wrap-around, non-first-stream tail tap, and rejects active lanes" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    var from = try Lane.init(device);
    defer from.deinit();
    var to = try Lane.init(device);
    defer to.deinit();

    const start_pos: u64 = 2100;
    const end_pos: u64 = 2300;
    const rows: usize = @intCast(end_pos - start_pos);
    const total_bytes = rows * layer_row_bytes;
    const want = try device.buffer(total_bytes, mtl.ResourceOptions.shared);
    defer want.deinit();
    const got = try device.buffer(total_bytes, mtl.ResourceOptions.shared);
    defer got.deinit();
    var prng = std.Random.DefaultPrng.init(31);
    prng.random().bytes(want.slice(u8, total_bytes));

    // Poison destination ring bytes while metadata stays cleared to prove restore overwrites active slots cleanly.
    const dirty = try device.buffer(w.ring_size * row_bytes, mtl.ResourceOptions.shared);
    defer dirty.deinit();
    @memset(dirty.slice(u8, w.ring_size * row_bytes), 0xa5);
    for (0..w.layers_n) |i| {
        try blitCopy(queue, dirty, 0, from.k[i], 0, w.ring_size * row_bytes);
        try blitCopy(queue, dirty, 0, from.v[i], 0, w.ring_size * row_bytes);
        try blitCopy(queue, dirty, 0, to.k[i], 0, w.ring_size * row_bytes);
        try blitCopy(queue, dirty, 0, to.v[i], 0, w.ring_size * row_bytes);
    }
    var off: usize = 0;
    for (0..w.layers_n) |i| {
        try copyRingSpan(queue, from.k[i], want, off, start_pos, rows, true);
        off += rows * row_bytes;
        try copyRingSpan(queue, from.v[i], want, off, start_pos, rows, true);
        off += rows * row_bytes;
    }
    from.from = start_pos;
    from.end = end_pos;

    const taps = try device.buffer(5 * tap_row_bytes, mtl.ResourceOptions.shared);
    defer taps.deinit();
    const tap_bytes = taps.slice(u8, 5 * tap_row_bytes);
    prng.random().bytes(tap_bytes);
    const tail_off_src = 3 * tap_row_bytes;
    try std.testing.expect(!std.mem.eql(u8, tap_bytes[0..tap_row_bytes], tap_bytes[tail_off_src..][0..tap_row_bytes]));

    var snap = try Snap.save(device, queue, &from, .{ .buf = taps, .off = tail_off_src });
    defer snap.free();
    try std.testing.expectEqual(start_pos, snap.from);
    try std.testing.expectEqual(end_pos, snap.end);
    try std.testing.expectEqual(exactSnapBytes(rows, true), snap.bytes_held);
    try std.testing.expectEqual(@as(?usize, @intCast(total_bytes)), snap.tail_off);

    // Restoring into an active lane is refused; after clearLane it succeeds bit-identically.
    to.end = 17;
    try std.testing.expectError(error.DFlash2SnapshotGeometry, Snap.restore(&to, queue, &snap));
    clearLane(&to);
    try Snap.restore(&to, queue, &snap);
    try std.testing.expectEqual(start_pos, to.from);
    try std.testing.expectEqual(end_pos, to.end);
    try std.testing.expectEqual(@as(u32, 0), to.held_len);

    off = 0;
    for (0..w.layers_n) |i| {
        try copyRingSpan(queue, to.k[i], got, off, start_pos, rows, false);
        off += rows * row_bytes;
        try copyRingSpan(queue, to.v[i], got, off, start_pos, rows, false);
        off += rows * row_bytes;
    }
    try std.testing.expectEqualSlices(u8, want.slice(u8, total_bytes), got.slice(u8, total_bytes));

    const got_tail = try device.buffer(tap_row_bytes, mtl.ResourceOptions.shared);
    defer got_tail.deinit();
    try blitCopy(queue, snap.buf.?, snap.tail_off.?, got_tail, 0, tap_row_bytes);
    try std.testing.expectEqualSlices(u8, tap_bytes[tail_off_src..][0..tap_row_bytes], got_tail.slice(u8, tap_row_bytes));
    try std.testing.expect(!std.mem.eql(u8, tap_bytes[0..tap_row_bytes], got_tail.slice(u8, tap_row_bytes)));
    try std.testing.expectError(error.DFlash2SnapshotGeometry, Snap.save(device, queue, &from, .{ .buf = taps, .off = 4 * tap_row_bytes + 1 }));
}

test "DFlash2 snapshot clamps retention to window_size across wrap-around and validates geometry" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    var from = try Lane.init(device);
    defer from.deinit();
    var to = try Lane.init(device);
    defer to.deinit();

    // Empty lane snapshot works cleanly without GPU buffer allocation.
    from.from = 128;
    from.end = 128;
    var empty = try Snap.save(device, queue, &from, null);
    defer empty.free();
    try std.testing.expect(empty.buf == null);
    try std.testing.expectEqual(@as(u64, 0), empty.bytes_held);
    try Snap.restore(&to, queue, &empty);
    try std.testing.expectEqual(@as(u64, 128), to.from);
    try std.testing.expectEqual(@as(u64, 128), to.end);
    clearLane(&to);

    // End past ring_size with from == 0 saves [2300 - 2048 .. 2300) = [252 .. 2300) wrapping ring_size = 2176.
    const save_from: u64 = 2300 - w.window_size;
    const full_bytes = w.window_size * layer_row_bytes;
    const want = try device.buffer(full_bytes, mtl.ResourceOptions.shared);
    defer want.deinit();
    const got = try device.buffer(full_bytes, mtl.ResourceOptions.shared);
    defer got.deinit();
    var prng = std.Random.DefaultPrng.init(77);
    prng.random().bytes(want.slice(u8, full_bytes));
    var off: usize = 0;
    for (0..w.layers_n) |i| {
        try copyRingSpan(queue, from.k[i], want, off, save_from, w.window_size, true);
        off += w.window_size * row_bytes;
        try copyRingSpan(queue, from.v[i], want, off, save_from, w.window_size, true);
        off += w.window_size * row_bytes;
    }
    from.from = 0;
    from.end = 2300;
    var snap = try Snap.save(device, queue, &from, null);
    defer snap.free();
    try std.testing.expectEqual(save_from, snap.from);
    try std.testing.expectEqual(@as(u64, 2300), snap.end);
    try std.testing.expectEqual(exactSnapBytes(w.window_size, false), snap.bytes_held);
    try Snap.restore(&to, queue, &snap);
    try std.testing.expectEqual(save_from, to.from);
    try std.testing.expectEqual(@as(u64, 2300), to.end);
    off = 0;
    for (0..w.layers_n) |i| {
        try copyRingSpan(queue, to.k[i], got, off, save_from, w.window_size, false);
        off += w.window_size * row_bytes;
        try copyRingSpan(queue, to.v[i], got, off, save_from, w.window_size, false);
        off += w.window_size * row_bytes;
    }
    try std.testing.expectEqualSlices(u8, want.slice(u8, full_bytes), got.slice(u8, full_bytes));

    // Geometry error guards: bad lane bounds, truncated bytes, overlapping tail_off, out-of-range end, and freed snap.
    from.from = 500;
    from.end = 400;
    try std.testing.expectError(error.DFlash2SnapshotGeometry, Snap.save(device, queue, &from, null));
    from.from = 0;
    from.end = max_context + 1;
    try std.testing.expectError(error.DFlash2SnapshotGeometry, Snap.save(device, queue, &from, null));
    clearLane(&to);
    var bad = snap;
    bad.bytes_held -= 1;
    try std.testing.expectError(error.DFlash2SnapshotGeometry, Snap.restore(&to, queue, &bad));
    var bad_tail = snap;
    bad_tail.tail_off = 0;
    try std.testing.expectError(error.DFlash2SnapshotGeometry, Snap.restore(&to, queue, &bad_tail));
    var bad_end = snap;
    bad_end.from = max_context;
    bad_end.end = max_context + 1;
    try std.testing.expectError(error.DFlash2SnapshotGeometry, Snap.restore(&to, queue, &bad_end));
    snap.free();
    try std.testing.expectError(error.DFlash2SnapshotGeometry, Snap.restore(&to, queue, &snap));
}
