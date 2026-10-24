//! A compact prefix of a rank's committed GLM state; live sequence addresses and kernel layouts stay fixed.
const std = @import("std");
const State = @import("cuda_state.zig").State;
const page_rows = @import("cuda_state.zig").page_rows;
const Config = @import("config.zig").Config;
const cache = @import("cuda_dsa_block.zig");
const sparse = @import("cuda_sparse.zig");

/// `len` bytes from `live`, or on a paged sequence's pages `span` bytes a page, the pages `page` bytes apart.
pub const Part = struct { live: u64, len: usize, span: usize = 0, page: u64 = 0 };
pub const Plan = struct {
    parts: [5 * (@import("config.zig").max_layers + 1) + 2]Part = undefined,
    count: usize = 0,
    len: usize = 0,
    ids: []const u32 = &.{}, // a paged sequence's pages in row order
    sizing: bool = false, // `size`: the rows counted, not read, so no page need be mapped

    pub fn init(st: *const State, c: *const Config, at: usize, head: ?usize) !Plan {
        return make(st, c, at, head, false);
    }

    /// A prefix's bytes at `at`, rows to `capacity` counted held or not (a paged sequence holds its request's alone).
    pub fn size(st: *const State, c: *const Config, capacity: usize, at: usize, head: ?usize) !usize {
        var whole = st.*;
        whole.capacity = capacity;
        return (try make(&whole, c, at, head, true)).len;
    }

    fn make(st: *const State, c: *const Config, at: usize, head: ?usize, sizing: bool) !Plan {
        if (at > st.capacity or st.caches.len > @import("config.zig").max_layers) return error.SnapshotShape;
        if (st.rings.len != 2 * (st.caches.len + 1)) return error.SnapshotShape;
        var p: Plan = .{ .ids = st.pages, .sizing = sizing };
        // a paged sequence's caches are its pages: its fields up to them end at its first ring
        const first = if (st.table != 0) st.rings[0] else if (st.caches.len > 0) st.caches[0].lc else st.mtp_caches.lc;
        if (first < st.arena.buf.ptr) return error.SnapshotShape;
        try p.add(st, st.arena.buf.ptr, @intCast(first - st.arena.buf.ptr));
        for (st.caches, 0..) |x, j| try p.rows(st, c, x, j, at);
        if (head) |n| {
            if (n > st.capacity) return error.SnapshotShape;
            try p.rows(st, c, st.mtp_caches, st.caches.len, n);
        }
        if (st.mtp_pos_dev < st.arena.buf.ptr or st.mtp_pos_dev - st.arena.buf.ptr > st.arena.buf.len)
            return error.SnapshotShape;
        const end = if (st.table != 0) st.table else st.arena.buf.ptr + st.arena.buf.len; // its table stays its own
        if (end < st.mtp_pos_dev) return error.SnapshotShape;
        try p.add(st, st.mtp_pos_dev, @intCast(end - st.mtp_pos_dev));
        return p;
    }

    fn rows(p: *Plan, st: *const State, c: *const Config, x: cache.Caches, j: usize, at: usize) !void {
        const latent = try std.math.mul(usize, at, @as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1));
        const ring = try std.math.mul(usize, @min(at, st.ring_rows), st.ring_row);
        const pooled = try std.math.mul(usize, @min(at / sparse.pool + 2, x.pk_rows), @as(usize, c.i_dim) * 2);
        try p.addCache(st, x, x.lc, latent, page_rows * @as(usize, c.kv_lora) * 2);
        if (x.ls != 0) try p.add(st, x.ls, try std.math.mul(usize, at, @sizeOf(f32)));
        try p.add(st, st.rings[2 * j], ring);
        try p.add(st, st.rings[2 * j + 1], ring);
        try p.addCache(st, x, x.pk, pooled, page_rows / sparse.pool * @as(usize, c.i_dim) * 2);
    }

    /// A cache's rows: in the arena, or on the sequence's pages, `span` bytes a page.
    fn addCache(p: *Plan, st: *const State, x: cache.Caches, live: u64, len: usize, span: usize) !void {
        if (x.pt == 0) return p.add(st, live, len);
        if (len == 0) return;
        if ((!p.sizing and (len - 1) / span >= p.ids.len) or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len, .span = span, .page = x.ps };
        p.count += 1;
    }

    /// Where a part's bytes from `done` on sit, as many as lie in a row there.
    pub fn piece(p: *const Plan, part: Part, done: usize) Part {
        if (part.span == 0) return .{ .live = part.live + done, .len = part.len - done };
        const off = done % part.span;
        return .{ .live = part.live + @as(u64, p.ids[done / part.span]) * part.page + off, .len = @min(part.span - off, part.len - done) };
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
