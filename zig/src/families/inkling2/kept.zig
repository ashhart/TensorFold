//! Kept prompt states: a slot's sliding rings and conv states copied out; its global rows stay in the slot.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("shape.zig");
const fw = @import("forward.zig");

/// One kept state: the slot whose global rows it needs, where it stands, its copy (null once a pass overwrote it).
const Snap = struct { slot: usize, at: usize, buf: ?cuda.DeviceBuffer };

/// Bytes a state takes on this rank: every sliding layer's K and V ring of one slot, then its conv states.
pub fn bytes(c: *const cfg.Config, st: *const fw.State, conv: usize) u64 {
    var total: u64 = conv;
    for (0..c.layers) |i| {
        if (c.layer[i] == .sliding) total += 2 * ringBytes(c, st);
    }
    return total;
}

fn ringBytes(c: *const cfg.Config, st: *const fw.State) usize {
    return st.kv * st.ring[0] * c.head_dim * 2;
}

/// States by id (the same ids on every rank); each names the slot that holds its global rows.
pub const Store = struct {
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(u64, Snap) = .empty,

    pub fn deinit(s: *Store) void {
        var it = s.map.valueIterator();
        while (it.next()) |x| if (x.buf) |*b| b.free();
        s.map.deinit(s.gpa);
    }

    /// The slot's state at `at` copied under `id`.
    pub fn save(s: *Store, d: *const cuda.Driver, stream: cuda.Stream, c: *const cfg.Config, st: *const fw.State, slot: usize, id: u64, at: usize, conv: usize) !void {
        if (s.map.contains(id)) return error.BadKeep;
        var buf = try cuda.DeviceBuffer.alloc(d, bytes(c, st, conv));
        errdefer buf.free();
        try copy(d, stream, c, st, slot, buf.ptr, true);
        try stream.synchronize();
        try s.map.put(s.gpa, id, .{ .slot = slot, .at = at, .buf = buf });
    }

    /// State `id` back into `slot` (its own, its global rows still there): rings, conv states and length.
    pub fn restore(s: *Store, d: *const cuda.Driver, stream: cuda.Stream, c: *const cfg.Config, st: *fw.State, slot: usize, id: u64) !void {
        const x = s.map.get(id) orelse return error.SnapshotStale;
        const buf = x.buf orelse return error.SnapshotStale;
        if (x.slot != slot) return error.SnapshotElsewhere;
        try copy(d, stream, c, st, slot, buf.ptr, false);
        st.pos[slot] = x.at;
    }

    /// A pass on `slot` from `from` rewrites its rows past it: the states that need them go (their ids stay, stale).
    pub fn overwrite(s: *Store, slot: usize, from: usize) void {
        var it = s.map.valueIterator();
        while (it.next()) |x| if (x.slot == slot and x.at > from) if (x.buf) |*b| {
            b.free();
            x.buf = null;
        };
    }

    pub fn drop(s: *Store, id: u64) void {
        const kv = s.map.fetchRemove(id) orelse return;
        if (kv.value.buf) |b| {
            var bb = b;
            bb.free();
        }
    }

    /// Whether `id` is kept and still valid.
    pub fn holds(s: *const Store, id: u64) bool {
        const x = s.map.get(id) orelse return false;
        return x.buf != null;
    }

    /// The slot and position of a valid state `id`.
    pub fn where(s: *const Store, id: u64) ?struct { slot: usize, at: usize } {
        const x = s.map.get(id) orelse return null;
        if (x.buf == null) return null;
        return .{ .slot = x.slot, .at = x.at };
    }

    /// Valid states on `slot` (a fresh pass there ends them all).
    pub fn on(s: *const Store, slot: usize) usize {
        var n: usize = 0;
        var it = s.map.valueIterator();
        while (it.next()) |x| n += @intFromBool(x.slot == slot and x.buf != null);
        return n;
    }
};

/// The slot's rings and conv states to (out) or from `buf`, layer after layer.
fn copy(d: *const cuda.Driver, stream: cuda.Stream, c: *const cfg.Config, st: *const fw.State, slot: usize, buf: u64, out: bool) !void {
    const rb = ringBytes(c, st);
    var off: u64 = 0;
    for (0..c.layers) |i| {
        if (c.layer[i] != .sliding) continue;
        for ([_]u64{ st.kc[i], st.vc[i] }) |base| {
            const src = base + slot * rb;
            try move(d, stream, if (out) buf + off else src, if (out) src else buf + off, rb);
            off += rb;
        }
    }
    for (st.conv) |cv| for (cv, 0..) |base, j| {
        const n = 3 * (if (j < 2) st.kv * c.head_dim else @as(usize, c.hidden)) * 4;
        const src = base + slot * n;
        try move(d, stream, if (out) buf + off else src, if (out) src else buf + off, n);
        off += n;
    };
}

fn move(d: *const cuda.Driver, stream: cuda.Stream, dst: u64, src: u64, n: usize) !void {
    try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream.handle), "cuMemcpyDtoDAsync");
}

test "kept: a state at 1M holds 40 sliding layers' rings of a slot, not the global rows" {
    var c: cfg.Config = .{};
    for (0..c.layers) |i| c.layer[i] = if (i % 6 == 5) .global else .sliding;
    const st: fw.State = .{ .gpa = std.testing.allocator, .ring = .{ 3072, 1048576 }, .kv = 4 };
    // 40 layers x K and V x 4 key heads x 3,072 slots x 128 x 2 B = 240 MiB, plus the conv states
    try std.testing.expectEqual(@as(u64, 240 << 20) + 7, bytes(&c, &st, 7));
}
