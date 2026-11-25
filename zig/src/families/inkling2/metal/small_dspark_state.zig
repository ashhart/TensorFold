//! Fixed eight-row Small DSpark scratch, 5-layer committed-only context rings and GPU-private snapshots.
const std = @import("std");
const mtl = @import("metal");
const st = @import("small_state.zig");

pub const hidden: usize = 4608;
pub const query: usize = 6144;
pub const kv: usize = 1024;
pub const intermediate: usize = 18432;
pub const rank: usize = 256;
pub const layers_count: usize = 5;
pub const global_layer: usize = 2;
pub const fusions_count: usize = 6;
pub const vocabulary: usize = 200058;
pub const block: usize = 8;
pub const max_streams: usize = 8;
pub const absorb_rows: usize = 64;
pub const local_ring: usize = std.mem.alignForward(usize, 1024 + block, 128);
pub const tap_row_bytes: usize = 8 * hidden * 2;
pub const Tail = struct { buf: mtl.Buffer, off: usize = 0 };

comptime {
    if (local_ring != 1152 or laneTokenBytes() != 4096) @compileError("Small DSpark ring geometry drift");
    if (st.feature_row_bytes != tap_row_bytes) @compileError("Small DSpark tail tap row bytes mismatch");
}

const fields = [_][]const u8{
    "ids",    "resid", "next_resid", "x",           "y",          "q_raw",     "k_raw",     "v_raw", "q",     "kb",         "vb", "attn",
    "gate",   "up",    "act",        "base",        "fused",      "ck",        "cv",        "kappa", "state", "next_state", "z",  "lambda",
    "logits", "held",  "probs",      "group_stats", "sample_tok", "sample_ck", "sample_ci",
};

fn sizesFor(vocab_n: usize) [fields.len]usize {
    const r = max_streams * block;
    const s = max_streams;
    const a = absorb_rows;
    const b = 2;
    const groups = (vocab_n + 255) / 256;
    return .{
        s * 4,           r * hidden * b,                 r * hidden * b,            r * hidden * b,            r * hidden * b,
        r * query * b,   r * kv * b,                     r * kv * b,                r * query * b,             r * kv * b,
        r * kv * b,      r * query * b,                  r * intermediate * b,      r * intermediate * b,      r * intermediate * b,
        r * vocab_n * b, fusions_count * a * hidden * b, layers_count * a * kv * b, layers_count * a * kv * b, (block - 1) * s * 3 * rank * 4,
        s * rank * b,    s * rank * b,                   s * 3 * rank * b,          s * rank * b,              s * vocab_n * b,
        s * block * 4,   s * block * 4,                  s * groups * 16,           s * 4,                     s * 32 * 64 * 4,
        s * 32 * 64 * 4,
    };
}

pub fn scratchBytes() usize {
    return scratchBytesFor(vocabulary);
}

pub fn scratchBytesFor(vocab_n: usize) usize {
    var bytes: usize = 0;
    for (sizesFor(vocab_n)) |n| bytes += n;
    return bytes;
}

pub const Scratch = struct {
    vocab_n: usize,
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
        return initFor(device, vocabulary);
    }

    pub fn initFor(device: mtl.Device, vocab_n: usize) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = sizesFor(vocab_n);
        var s: Scratch = undefined;
        s.vocab_n = vocab_n;
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

pub fn scoreBytes(_: usize) u64 {
    return 0;
}

/// Four sliding layers [0,1,3,4] hold fixed K/V rings; layer 2 holds the global K/V ring.
pub fn laneFixedBytes() u64 {
    return 4 * 2 * local_ring * kv * 2 + hidden * 2;
}

pub fn laneTokenBytes() u64 {
    return 2 * kv * 2;
}

pub fn laneBytes(capacity: usize) u64 {
    return laneFixedBytes() + globalRing(capacity) * laneTokenBytes();
}

pub const Lane = struct {
    k: [layers_count]mtl.Buffer,
    v: [layers_count]mtl.Buffer,
    rings: [layers_count]usize,
    newest: mtl.Buffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [7]u32 = @splat(0),
    probs: [7]f32 = @splat(0),
    held_len: u32 = 0,
    sampled: bool = false,

    pub fn init(device: mtl.Device, capacity: usize) !Lane {
        if (capacity == 0 or capacity > 1048576) return error.DSparkLaneGeometry;
        const g = globalRing(capacity);
        var l: Lane = .{
            .k = undefined,
            .v = undefined,
            .rings = .{ local_ring, local_ring, g, local_ring, local_ring },
            .newest = undefined,
        };
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
        for (0..layers_count) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        l.* = undefined;
    }
};

pub fn clearLane(lane: *Lane) void {
    lane.from = 0;
    lane.end = 0;
    lane.held_len = 0;
    lane.sampled = false;
}

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

pub fn exactSnapBytes(end: usize, has_tail: bool) u64 {
    const keys = @as(u64, @intCast(end)) + 4 * @as(u64, @intCast(@min(end, local_ring)));
    return keys * @as(u64, kv) * 4 + hidden * 2 + if (has_tail) @as(u64, tap_row_bytes) else 0;
}

pub fn snapBytes(at: usize) u64 {
    return exactSnapBytes(at, true);
}

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
        if (end > lane.rings[global_layer] or lane.from > lane.end) return error.DSparkSnapshotGeometry;
        for (0..layers_count) |i| if (i != global_layer and lane.rings[i] != local_ring) return error.DSparkSnapshotGeometry;
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
        if (snap.freed or lane.end != 0 or lane.from != 0 or snap.from > snap.end or end > lane.rings[global_layer] or snap.bytes_held < need or snap.buf.length() < need)
            return error.DSparkSnapshotGeometry;
        for (0..layers_count) |i| if (i != global_layer and lane.rings[i] != local_ring) return error.DSparkSnapshotGeometry;
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
    for (0..layers_count) |i| {
        const live = if (i == global_layer) end else @min(end, local_ring);
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
        std.log.err("inkling2 Small DSpark snapshot {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

test "inkling2 small: DSpark stream and snapshot accounting covers 4 local + 1 global rings" {
    try std.testing.expectEqual(@as(u64, 4 * 2 * local_ring * kv * 2 + hidden * 2), laneFixedBytes());
    try std.testing.expectEqual(laneFixedBytes() + 128 * 4096, laneBytes(1));
    try std.testing.expectEqual(laneFixedBytes() + 1152 * 4096, laneBytes(1025));
    try std.testing.expectEqual(@as(u64, (2048 + 4 * local_ring) * 4096 + hidden * 2 + tap_row_bytes), Snap.bytes(2048));
}
