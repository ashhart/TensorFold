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
pub const absorb_rows = 64;
pub const local_ring = std.mem.alignForward(usize, 1024 + block, 128);
pub const tap_row_bytes: usize = 4 * hidden * 2;

const fields = [_][]const u8{
    "ids",    "resid", "next_resid", "x",    "y",     "q_raw", "k_raw", "v_raw", "q",     "kb",         "vb", "attn",
    "gate",   "up",    "act",        "base", "fused", "ck",    "cv",    "kappa", "state", "next_state", "z",  "lambda",
    "logits",
};

fn sizes() [fields.len]usize {
    const r = block;
    const a = absorb_rows;
    const b = 2;
    return .{
        r * 4,                r * hidden * b,     r * hidden * b,       r * hidden * b,       r * hidden * b,
        r * query * b,        r * kv * b,         r * kv * b,           r * query * b,        r * kv * b,
        r * kv * b,           r * query * b,      r * intermediate * b, r * intermediate * b, r * intermediate * b,
        (r - 1) * 201024 * b, 4 * a * hidden * b, a * kv * b,           a * kv * b,           (r - 1) * 3 * rank * 4,
        rank * b,             rank * b,           3 * rank * b,         rank * b,             vocabulary * b,
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
pub fn globalRing(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, 128);
}

pub fn scoreBytes(capacity: usize) u64 {
    return @as(u64, 8 * 24 * 4) * (capacity + block);
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
    held_len: u32 = 0,

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

fn exactSnapBytes(end: usize, has_tail: bool) u64 {
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

    pub fn save(device: mtl.Device, queue: mtl.Queue, lane: *const Lane, tail_tap: ?mtl.Buffer) !Snap {
        const end: usize = @intCast(lane.end);
        if (end > lane.rings[1] or lane.rings[0] != local_ring or lane.rings[2] != local_ring or lane.from > lane.end) {
            return error.DSparkSnapshotGeometry;
        }
        const has_tail = tail_tap != null;
        const size = exactSnapBytes(end, has_tail);
        const tail_off: ?usize = if (has_tail) @intCast(size - tap_row_bytes) else null;
        const buf = try device.buffer(@intCast(size), mtl.ResourceOptions.private);
        errdefer buf.deinit();
        try blitSnap(queue, lane, buf, end, tail_tap, tail_off, false, "save");
        return .{ .buf = buf, .from = lane.from, .end = lane.end, .bytes_held = size, .tail_off = tail_off };
    }

    pub fn restore(lane: *Lane, queue: mtl.Queue, snap: *const Snap) !void {
        const end: usize = @intCast(snap.end);
        const need = exactSnapBytes(end, snap.tail_off != null);
        if (snap.freed or lane.end != 0 or lane.from != 0 or end > lane.rings[1] or lane.rings[0] != local_ring or lane.rings[2] != local_ring or snap.bytes_held < need) {
            return error.DSparkSnapshotGeometry;
        }
        try blitSnap(queue, lane, snap.buf, end, null, null, true, "restore");
        lane.from = snap.from;
        lane.end = snap.end;
        lane.held_len = 0;
    }
};

fn blitSnap(queue: mtl.Queue, lane: *const Lane, snap_buf: mtl.Buffer, end: usize, tail_src: ?mtl.Buffer, tail_off: ?usize, restoring: bool, comptime what: []const u8) !void {
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
        if (tail_src) |src| e.copy(src, 0, snap_buf, tail_off.?, tap_row_bytes);
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

test "DSpark stream accounting includes sliding, global and shared attention scratch" {
    try std.testing.expectEqual(@as(u64, 9_437_184 + 4608), laneFixedBytes());
    try std.testing.expectEqual(laneFixedBytes() + 128 * 4096, laneBytes(1));
    try std.testing.expectEqual(laneFixedBytes() + 1152 * 4096, laneBytes(1025));
    try std.testing.expectEqual(@as(u64, 768 * 1032), scoreBytes(1024));
    try std.testing.expect(Snap.bytes(0) >= exactSnapBytes(0, false));
    try std.testing.expect(Snap.bytes(1) >= exactSnapBytes(0, true));
    try std.testing.expect(Snap.bytes(1024) >= exactSnapBytes(1024, false));
    try std.testing.expect(Snap.bytes(2048) >= exactSnapBytes(2047, true));
    try std.testing.expectEqual(@as(u64, (2048 + 2 * local_ring) * 4096 + hidden * 2 + tap_row_bytes), Snap.bytes(2048));
    // Ring and scores bytes are reserved with the target cache before the lane opens.
}
