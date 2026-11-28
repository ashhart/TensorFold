//! Fixed eight-row DSpark scratch and committed-only per-stream context rings.
const std = @import("std");
const mtl = @import("metal");

pub const hidden = 2304;
pub const query = 3072;
pub const kv = 1024;
pub const intermediate = 9216;
pub const rank = 256;
pub const vocabulary = 200058;
pub const block = 8;
pub const max_streams = 8;
pub const absorb_rows = 64;
pub const local_ring = std.mem.alignForward(usize, 1024 + block, 128);
pub const logit_groups = (vocabulary + 255) / 256;
pub const tap_row_bytes: usize = 4 * hidden * 2;
pub const Tail = struct { buf: mtl.Buffer, off: usize = 0 };

const fields = [_][]const u8{
    "ids",    "resid", "next_resid", "x",           "y",          "q_raw",     "k_raw",     "v_raw", "q",     "kb",         "vb", "attn",
    "gate",   "up",    "act",        "base",        "fused",      "ck",        "cv",        "kappa", "state", "next_state", "z",  "lambda",
    "logits", "held",  "probs",      "group_stats", "sample_tok", "sample_ck", "sample_ci",
};

fn sizes() [fields.len]usize {
    const r = max_streams * block;
    const s = max_streams;
    const a = absorb_rows;
    const b = 2;
    return .{
        s * 4,              r * hidden * b,     r * hidden * b,        r * hidden * b,       r * hidden * b,
        r * query * b,      r * kv * b,         r * kv * b,            r * query * b,        r * kv * b,
        r * kv * b,         r * query * b,      r * intermediate * b,  r * intermediate * b, r * intermediate * b,
        r * vocabulary * b, 4 * a * hidden * b, 3 * a * kv * b,        3 * a * kv * b,       (block - 1) * s * 3 * rank * 4,
        s * rank * b,       s * rank * b,       s * 3 * rank * b,      s * rank * b,         s * vocabulary * b,
        s * block * 4,      s * block * 4,      s * logit_groups * 16, s * 4,                s * 32 * 64 * 4,
        s * 32 * 64 * 4,
    };
}

pub fn scratchBytes() usize {
    var bytes: usize = 0;
    for (sizes()) |n| bytes += n;
    return bytes;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer,
    next_resid: mtl.Buffer,
    x: mtl.Buffer,
    y: mtl.Buffer,
    q_raw: mtl.Buffer,
    k_raw: mtl.Buffer,
    v_raw: mtl.Buffer,
    q: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    attn: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    base: mtl.Buffer,
    fused: mtl.Buffer,
    ck: mtl.Buffer,
    cv: mtl.Buffer,
    kappa: mtl.Buffer,
    state: mtl.Buffer,
    next_state: mtl.Buffer,
    z: mtl.Buffer,
    lambda: mtl.Buffer,
    logits: mtl.Buffer,
    held: mtl.Buffer,
    probs: mtl.Buffer,
    group_stats: mtl.Buffer,
    sample_tok: mtl.Buffer,
    sample_ck: mtl.Buffer,
    sample_ci: mtl.Buffer,

    pub fn init(device: mtl.Device) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = sizes();
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(ns[i], 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        s.* = undefined;
    }
};
/// A global ring's rows come this many at a time.
pub const global_step = 128;
pub fn globalRing(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, global_step);
}

pub fn scoreBytes(_: usize) u64 {
    return 0;
}

/// Two sliding layers hold two fixed K/V rings; layer 1 holds two global K/V rings.
pub fn laneFixedBytes() u64 {
    return 2 * 2 * local_ring * kv * 2 + hidden * 2;
}

pub fn laneTokenBytes() u64 {
    return 2 * kv * 2;
}

pub fn laneBytes(capacity: usize) u64 {
    return laneFixedBytes() + globalRing(capacity) * laneTokenBytes();
}

pub const Lane = struct {
    k: [3]mtl.Buffer,
    v: [3]mtl.Buffer,
    rings: [3]usize,
    newest: mtl.Buffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [7]u32 = @splat(0),
    probs: [7]f32 = @splat(0),
    held_len: u32 = 0,
    sampled: bool = false,

    pub fn init(device: mtl.Device, capacity: usize) !Lane {
        if (capacity == 0 or capacity > 1048576) return error.DSparkLaneGeometry;
        var l: Lane = .{ .k = undefined, .v = undefined, .rings = .{ local_ring, globalRing(capacity), local_ring }, .newest = undefined };
        var made: usize = 0;
        errdefer {
            for (0..made) |i| {
                l.k[i].deinit();
                l.v[i].deinit();
            }
        }
        for (l.rings, 0..) |ring, i| {
            const bytes = ring * kv * 2;
            l.k[i] = try device.buffer(bytes, mtl.ResourceOptions.private);
            l.v[i] = device.buffer(bytes, mtl.ResourceOptions.private) catch |err| {
                l.k[i].deinit();
                return err;
            };
            made += 1;
        }
        l.newest = try device.buffer(hidden * 2, mtl.ResourceOptions.private);
        return l;
    }

    pub fn deinit(l: *Lane) void {
        l.newest.deinit();
        for (0..3) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        l.* = undefined;
    }
};

pub fn exactSnapBytes(end: usize, has_tail: bool) u64 {
    const keys = @as(u64, @intCast(end)) + 2 * @as(u64, @intCast(@min(end, local_ring)));
    return keys * @as(u64, kv) * 4 + hidden * 2 + if (has_tail) @as(u64, tap_row_bytes) else 0;
}

/// Upper bound on GPU-private snapshot bytes at `at` tokens (includes optional unabsorbed prompt-end tap row).
pub fn snapBytes(at: usize) u64 {
    return exactSnapBytes(at, true);
}

/// Compact GPU-private snapshot of a DSpark lane's live K/V rings, newest fused row, and optional tail tap.
pub const Snap = struct {
    buf: mtl.Buffer,
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

    pub fn save(device: mtl.Device, queue: mtl.Queue, lane: *const Lane, tail: ?Tail) !Snap {
        const end: usize = @intCast(lane.end);
        if (end > lane.rings[1] or lane.rings[0] != local_ring or lane.rings[2] != local_ring or lane.from > lane.end) {
            return error.DSparkSnapshotGeometry;
        }
        if (tail) |t| {
            const tap_end = std.math.add(usize, t.off, tap_row_bytes) catch return error.DSparkSnapshotGeometry;
            if (tap_end > t.buf.length()) return error.DSparkSnapshotGeometry;
        }
        const has_tail = tail != null;
        const size = exactSnapBytes(end, has_tail);
        const tail_off: ?usize = if (has_tail) @intCast(size - tap_row_bytes) else null;
        const buf = try device.buffer(@intCast(size), mtl.ResourceOptions.private);
        errdefer buf.deinit();
        try blitSnap(queue, lane, buf, end, tail, tail_off, false, "save");
        return .{ .buf = buf, .from = lane.from, .end = lane.end, .bytes_held = size, .tail_off = tail_off };
    }

    pub fn restore(lane: *Lane, queue: mtl.Queue, snap: *const Snap) !void {
        const end: usize = @intCast(snap.end);
        const need = exactSnapBytes(end, snap.tail_off != null);
        if (snap.freed or lane.end != 0 or lane.from != 0 or snap.from > snap.end or end > lane.rings[1] or lane.rings[0] != local_ring or lane.rings[2] != local_ring or snap.bytes_held < need or snap.buf.length() < need) {
            return error.DSparkSnapshotGeometry;
        }
        if (snap.tail_off) |to| {
            const tail_end = std.math.add(usize, to, tap_row_bytes) catch return error.DSparkSnapshotGeometry;
            if (tail_end > snap.bytes_held or tail_end > snap.buf.length()) return error.DSparkSnapshotGeometry;
        }
        try blitSnap(queue, lane, snap.buf, end, null, null, true, "restore");
        lane.from = snap.from;
        lane.end = snap.end;
        lane.held_len = 0;
    }
};

fn blitSnap(queue: mtl.Queue, lane: *const Lane, snap_buf: mtl.Buffer, end: usize, tail: ?Tail, tail_off: ?usize, restoring: bool, comptime what: []const u8) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var at: usize = 0;
    for (0..3) |i| {
        const live = if (i == 1) end else @min(end, local_ring);
        const n = live * kv * 2;
        if (n != 0) {
            if (restoring) {
                e.copy(snap_buf, at, lane.k[i], 0, n);
                at += n;
                e.copy(snap_buf, at, lane.v[i], 0, n);
            } else {
                e.copy(lane.k[i], 0, snap_buf, at, n);
                at += n;
                e.copy(lane.v[i], 0, snap_buf, at, n);
            }
            at += n;
        }
    }
    if (restoring) {
        e.copy(snap_buf, at, lane.newest, 0, hidden * 2);
    } else {
        e.copy(lane.newest, 0, snap_buf, at, hidden * 2);
        if (tail) |t| e.copy(t.buf, t.off, snap_buf, tail_off.?, tap_row_bytes);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop DSpark snapshot {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

comptime {
    if (local_ring != 1152 or laneTokenBytes() != 4096) @compileError("DSpark ring geometry drift");
}

test "DSpark stream accounting includes sliding and global context without attention score scratch" {
    try std.testing.expectEqual(@as(u64, 9_437_184 + 4608), laneFixedBytes());
    try std.testing.expectEqual(laneFixedBytes() + 128 * 4096, laneBytes(1));
    try std.testing.expectEqual(laneFixedBytes() + 1152 * 4096, laneBytes(1025));
    try std.testing.expectEqual(@as(u64, 0), scoreBytes(1024));
    try std.testing.expect(Snap.bytes(0) >= exactSnapBytes(0, false));
    try std.testing.expect(Snap.bytes(1) >= exactSnapBytes(0, true));
    try std.testing.expect(Snap.bytes(1024) >= exactSnapBytes(1024, false));
    try std.testing.expect(Snap.bytes(2048) >= exactSnapBytes(2047, true));
    try std.testing.expectEqual(@as(u64, (2048 + 2 * local_ring) * 4096 + hidden * 2 + tap_row_bytes), Snap.bytes(2048));
}

fn blitCopy(queue: mtl.Queue, src: mtl.Buffer, src_off: usize, dst: mtl.Buffer, dst_off: usize, bytes: usize) !void {
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

test "DSpark snapshot preserves non-first-stream tail tap byte offset and rejects out-of-range offsets" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    var from = try Lane.init(device, 256);
    defer from.deinit();
    var to = try Lane.init(device, 256);
    defer to.deinit();
    const rows = 3;
    const ring_bytes = rows * kv * 2;
    const newest_bytes = hidden * 2;
    const host_lane = try device.buffer(6 * ring_bytes + newest_bytes, mtl.ResourceOptions.shared);
    defer host_lane.deinit();
    const read_lane = try device.buffer(6 * ring_bytes + newest_bytes, mtl.ResourceOptions.shared);
    defer read_lane.deinit();
    var prng = std.Random.DefaultPrng.init(19);
    prng.random().bytes(host_lane.slice(u8, 6 * ring_bytes + newest_bytes));
    var off: usize = 0;
    for (0..3) |i| {
        try blitCopy(queue, host_lane, off, from.k[i], 0, ring_bytes);
        off += ring_bytes;
        try blitCopy(queue, host_lane, off, from.v[i], 0, ring_bytes);
        off += ring_bytes;
    }
    try blitCopy(queue, host_lane, off, from.newest, 0, newest_bytes);
    from.end = rows;

    const taps = try device.buffer(5 * tap_row_bytes, mtl.ResourceOptions.shared);
    defer taps.deinit();
    const tap_bytes = taps.slice(u8, 5 * tap_row_bytes);
    prng.random().bytes(tap_bytes);
    const stream0_row = tap_bytes[0..tap_row_bytes];
    const stream1_off = 3 * tap_row_bytes;
    const stream1_row = tap_bytes[stream1_off..][0..tap_row_bytes];
    try std.testing.expect(!std.mem.eql(u8, stream0_row, stream1_row));

    var snap = try Snap.save(device, queue, &from, .{ .buf = taps, .off = stream1_off });
    defer snap.free();
    try std.testing.expect(snap.tail_off != null);
    try Snap.restore(&to, queue, &snap);
    try std.testing.expectEqual(from.end, to.end);
    try std.testing.expectEqual(from.from, to.from);
    off = 0;
    for (0..3) |i| {
        try blitCopy(queue, to.k[i], 0, read_lane, off, ring_bytes);
        off += ring_bytes;
        try blitCopy(queue, to.v[i], 0, read_lane, off, ring_bytes);
        off += ring_bytes;
    }
    try blitCopy(queue, to.newest, 0, read_lane, off, newest_bytes);
    try std.testing.expectEqualSlices(u8, host_lane.slice(u8, 6 * ring_bytes + newest_bytes), read_lane.slice(u8, 6 * ring_bytes + newest_bytes));

    const got_tail = try device.buffer(tap_row_bytes, mtl.ResourceOptions.shared);
    defer got_tail.deinit();
    try blitCopy(queue, snap.buf, snap.tail_off.?, got_tail, 0, tap_row_bytes);
    try std.testing.expectEqualSlices(u8, stream1_row, got_tail.slice(u8, tap_row_bytes));
    try std.testing.expect(!std.mem.eql(u8, stream0_row, got_tail.slice(u8, tap_row_bytes)));
    try std.testing.expectError(error.DSparkSnapshotGeometry, Snap.save(device, queue, &from, .{ .buf = taps, .off = 4 * tap_row_bytes + 1 }));
}
