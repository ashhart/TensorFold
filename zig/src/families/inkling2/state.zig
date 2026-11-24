//! A rank's per-stream caches: K and V rings and conv states for every cache slot (Small: a layer, Mini: a visit).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("shape.zig");
const kernels = @import("kernels.zig");
const fw = @import("forward.zig");

const Buf = cuda.DeviceBuffer;

/// Bytes of one cache slot of a stream: K and V rings a visit plus four conv states (the per-stream cost).
pub fn slotBytes(c: *const cfg.Config, s: fw.Shape) u64 {
    var total: u64 = 0;
    for (0..c.visits()) |i| {
        total += 2 * @as(u64, s.layer(c, i).cacheBytes(s.kv));
        total += 2 * 3 * @as(u64, s.kv) * c.head_dim * 4 + 2 * 3 * @as(u64, c.hidden) * 4;
    }
    return total;
}

pub fn alloc(d: *const cuda.Driver, list: *std.ArrayList(Buf), gpa: std.mem.Allocator, bytes: usize) !u64 {
    var b = try Buf.alloc(d, @max(bytes, 256));
    errdefer b.free();
    try b.fill8(0, null);
    try list.append(gpa, b);
    return b.ptr;
}

/// `slots` streams' caches and conv states: K and V rings a visit, the last three inputs of each short convolution.
pub const State = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(Buf) = .empty,
    kc: []u64 = &.{},
    vc: []u64 = &.{},
    conv: [][4]u64 = &.{}, // k, v (fp32 [slots][3][kv * 128]) and attention, MLP (fp32 [slots][3][hidden]) a visit
    pos: []usize = &.{}, // positions each slot's caches hold
    ring: [2]usize = .{ 0, 0 }, // sliding and global ring slots a key head
    kv: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: *const cfg.Config, s: fw.Shape, slots: usize) !State {
        var st: State = .{ .gpa = gpa, .kv = s.kv };
        errdefer st.deinit();
        const n = c.visits();
        st.kc = try gpa.alloc(u64, n);
        st.vc = try gpa.alloc(u64, n);
        st.conv = try gpa.alloc([4]u64, n);
        st.pos = try gpa.alloc(usize, slots);
        @memset(st.pos, 0);
        st.ring = .{ s.sliding(c).ring, s.global().ring };
        for (0..n) |i| {
            const l = s.layer(c, i);
            st.kc[i] = try alloc(d, &st.bufs, gpa, slots * l.cacheBytes(s.kv));
            st.vc[i] = try alloc(d, &st.bufs, gpa, slots * l.cacheBytes(s.kv));
            for (0..2) |j| st.conv[i][j] = try alloc(d, &st.bufs, gpa, slots * 3 * s.kv * c.head_dim * 4);
            for (2..4) |j| st.conv[i][j] = try alloc(d, &st.bufs, gpa, slots * 3 * @as(usize, c.hidden) * 4);
        }
        return st;
    }

    pub fn cache(st: *const State, i: usize) kernels.Cache {
        return .{ .k = st.kc[i], .v = st.vc[i] };
    }

    /// Back to empty streams: caches and conv states zeroed.
    pub fn reset(st: *State) !void {
        for (st.bufs.items) |b| try b.fill8(0, null);
        @memset(st.pos, 0);
    }

    /// One slot's conv states zeroed on `s` (a fresh pass; its rings are only read at positions it writes).
    pub fn clearSlot(st: *State, c: *const cfg.Config, d: *const cuda.Driver, slot: usize, s: cuda.Stream) !void {
        for (st.conv) |cv| for (cv, 0..) |base, j| {
            const n = 3 * (if (j < 2) st.kv * c.head_dim else @as(usize, c.hidden)) * 4;
            const view: Buf = .{ .d = d, .ptr = base + slot * n, .len = n, .borrowed = true };
            try view.fill8(0, s.handle);
        };
        st.pos[slot] = 0;
    }

    pub fn deinit(st: *State) void {
        for (st.bufs.items) |*b| b.free();
        st.bufs.deinit(st.gpa);
        st.gpa.free(st.kc);
        st.gpa.free(st.vc);
        st.gpa.free(st.conv);
        st.gpa.free(st.pos);
    }
};
