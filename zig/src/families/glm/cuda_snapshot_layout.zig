//! A compact prefix of a rank's committed GLM state; live sequence addresses and kernel layouts stay fixed.
const std = @import("std");
const State = @import("cuda_state.zig").State;
const Config = @import("config.zig").Config;
const cache = @import("cuda_dsa_block.zig");
const sparse = @import("cuda_sparse.zig");

pub const Part = struct { live: u64, len: usize };
pub const Plan = struct {
    parts: [5 * (@import("config.zig").max_layers + 1) + 2]Part = undefined,
    count: usize = 0,
    len: usize = 0,

    pub fn init(st: *const State, c: *const Config, at: usize, head: ?usize) !Plan {
        if (at > st.capacity or st.caches.len > @import("config.zig").max_layers) return error.SnapshotShape;
        if (st.rings.len != 2 * (st.caches.len + 1)) return error.SnapshotShape;
        var p: Plan = .{};
        const first = if (st.caches.len > 0) st.caches[0].lc else st.mtp_caches.lc;
        if (first < st.arena.buf.ptr) return error.SnapshotShape;
        try p.add(st, st.arena.buf.ptr, @intCast(first - st.arena.buf.ptr));
        for (st.caches, 0..) |x, j| try p.rows(st, c, x, j, at);
        if (head) |n| {
            if (n > st.capacity) return error.SnapshotShape;
            try p.rows(st, c, st.mtp_caches, st.caches.len, n);
        }
        if (st.mtp_pos_dev < st.arena.buf.ptr or st.mtp_pos_dev - st.arena.buf.ptr > st.arena.buf.len)
            return error.SnapshotShape;
        try p.add(st, st.mtp_pos_dev, st.arena.buf.len - @as(usize, @intCast(st.mtp_pos_dev - st.arena.buf.ptr)));
        return p;
    }

    fn rows(p: *Plan, st: *const State, c: *const Config, x: cache.Caches, j: usize, at: usize) !void {
        const latent = try std.math.mul(usize, at, @as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1));
        const ring = try std.math.mul(usize, @min(at, st.ring_rows), st.ring_row);
        const pooled = try std.math.mul(usize, @min(at / sparse.pool + 2, x.pk_rows), @as(usize, c.i_dim) * 2);
        try p.add(st, x.lc, latent);
        if (x.ls != 0) try p.add(st, x.ls, try std.math.mul(usize, at, @sizeOf(f32)));
        try p.add(st, st.rings[2 * j], ring);
        try p.add(st, st.rings[2 * j + 1], ring);
        try p.add(st, x.pk, pooled);
    }

    fn add(p: *Plan, st: *const State, live: u64, len: usize) !void {
        if (len == 0) return;
        if (live < st.arena.buf.ptr) return error.SnapshotShape;
        const off = live - st.arena.buf.ptr;
        if (off > st.arena.buf.len or len > st.arena.buf.len - off or p.count == p.parts.len)
            return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len };
        p.count += 1;
    }
};
