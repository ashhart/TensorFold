//! Kept prompt states on every rank: a state is its slot's sliding rings; the full layers' rows stay in the slot.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("tp2_forward.zig");

const hd = @import("config.zig").head_dim;

/// A rank's copies by the id rank 0 gave them (no policy here: rank 0's lanes decide).
pub const States = struct {
    by_id: std.AutoHashMapUnmanaged(i32, cuda.DeviceBuffer) = .empty,

    pub fn deinit(st: *States, gpa: std.mem.Allocator) void {
        var it = st.by_id.valueIterator();
        while (it.next()) |b| b.free();
        st.by_id.deinit(gpa);
    }
};

/// Bytes a state takes on a rank: every sliding layer's K and V ring of one slot.
pub fn bytes(m: *const fwd.Model) u64 {
    var total: u64 = 0;
    for (m.caches, 0..) |ca, i| if (!m.c.global[i]) {
        total += 2 * ca.ring * m.hk * hd * 2;
    };
    return total;
}

fn copy(m: *fwd.Model, dst: u64, src: u64, n: usize) !void {
    try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, m.stream.handle), "cuMemcpyDtoDAsync");
}

/// K's then V's region of `slot` in a layer's cache, `n` bytes from row 0.
fn region(m: *const fwd.Model, ca: fwd.Cache, slot: usize, v: bool) u64 {
    const kw = m.hk * hd;
    return ca.kv.ptr + (@as(u64, @intFromBool(v)) * m.slots + slot) * ca.ring * kw * 2;
}

/// A kept-state header: save (slot's rings as state a), restore (a into slot), drop (a), copy (rows [0, p0) of a).
pub fn run(m: *fwd.Model, h: *const fwd.Header) !void {
    const op: fwd.Op = @fromBackingInt(@intCast(h.op));
    const slot: usize = @intCast(h.slot);
    const kw = m.hk * hd;
    if (op != .drop and slot >= m.slots) return error.RicursiveSlots;
    switch (op) {
        .save => {
            if (m.states.by_id.contains(h.a)) return error.RicursiveStateId;
            var b = try cuda.DeviceBuffer.alloc(m.d, bytes(m));
            errdefer b.free();
            var at: u64 = 0;
            for (m.caches, 0..) |ca, i| if (!m.c.global[i]) for ([_]bool{ false, true }) |v| {
                const n = ca.ring * kw * 2;
                try copy(m, b.ptr + at, region(m, ca, slot, v), n);
                at += n;
            };
            try m.states.by_id.put(m.gpa, h.a, b);
        },
        .restore => {
            const b = m.states.by_id.get(h.a) orelse return error.RicursiveStateId;
            var at: u64 = 0;
            for (m.caches, 0..) |ca, i| if (!m.c.global[i]) for ([_]bool{ false, true }) |v| {
                const n = ca.ring * kw * 2;
                try copy(m, region(m, ca, slot, v), b.ptr + at, n);
                at += n;
            };
        },
        .drop => {
            var e = m.states.by_id.fetchRemove(h.a) orelse return;
            try m.stream.synchronize(); // a restore's queued copies out of it are done
            e.value.free();
        },
        .copy => {
            const from: usize = @intCast(h.a);
            const rows: usize = @intCast(h.p0);
            if (from >= m.slots or rows > m.context) return error.RicursiveSlots;
            for (m.caches, 0..) |ca, i| if (m.c.global[i]) for ([_]bool{ false, true }) |v| {
                try copy(m, region(m, ca, slot, v), region(m, ca, from, v), rows * kw * 2);
            };
        },
        else => return error.RicursiveRows,
    }
}
