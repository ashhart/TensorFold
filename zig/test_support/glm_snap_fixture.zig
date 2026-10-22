//! A small actual GLM State and Engine with host-owned CUDA buffers, including both mixer kinds.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const rec = @import("glm_snap_driver.zig").Driver;
const t = std.testing;
pub const Fixture = struct {
    e: glm.engine.Engine,
    seq: @typeInfo(@FieldType(glm.engine.Engine, "cur")).pointer.child,

    pub fn init(f: *Fixture, d: *const cuda.Driver, rank: usize, world: usize, mtp: bool) !void {
        return f.initFormat(d, rank, world, mtp, false);
    }

    pub fn initFormat(f: *Fixture, d: *const cuda.Driver, rank: usize, world: usize, mtp: bool, kv8: bool) !void {
        var w: glm.weights.Weights = undefined;
        w.config = .{ .layers = 2, .kda_heads = 64, .kv_lora = if (kv8) 512 else 16, .i_dim = 8, .hidden = 64 };
        w.share = try glm.shard.Share.of(&w.config, rank, world);
        var layers: [2]glm.weights.Layer = undefined;
        layers[0].mixer = .{ .mla = undefined };
        layers[1].mixer = .{ .kda = undefined };
        w.layers = &layers;
        f.seq = .{ .st = try glm.forward.State.initFormat(t.allocator, d, &w, 16448, kv8) };
        f.e = undefined;
        f.e.cur = &f.seq;
        f.e.gpa = t.allocator;
        f.e.d = d;
        f.e.stream = .{ .d = d, .handle = null };
        f.e.c = w.config;
        f.e.share = w.share;
        f.e.capacity = f.seq.st.capacity;
        f.e.mtp = mtp;
        f.e.snaps = .empty;
        f.e.snap_pages = null;
        f.e.snap_setup = null;
        f.e.dropped = .empty;
        f.e.next_snap = 1;
        f.e.announce = 0;
    }
    pub fn deinit(f: *Fixture) void {
        var it = f.e.snaps.valueIterator();
        while (it.next()) |s| glm.snapshots.free(&f.e, s);
        f.e.snaps.deinit(t.allocator);
        f.e.dropped.deinit(t.allocator);
        if (f.e.snap_pages) |*p| p.deinit();
        f.seq.st.deinit();
    }
    pub fn mark(f: *Fixture, at: usize, head: usize, salt: u8) !void {
        const s = &f.seq.st;
        const buf = rec.bytes(s.arena.buf.ptr, s.arena.buf.len);
        for (buf, 0..) |*b, i| b.* = @as(u8, @truncate(i *% 37 +% i / 251)) ^ salt;
        try s.restoreHost(f.e.stream, at, head, &.{@as(u1, @truncate(at))}, .{ if (at > 4096) at / 4 * 4 else 0, if (head > 4096) (head -| 8) / 4 * 4 else 0 });
        f.seq.in_head = 0;
    }

    /// Independent read-set oracle from State's layout, without using the packing plan.
    pub fn expectPrefix(f: *Fixture, expected: []const u8, at: usize, head: usize) !void {
        const s = &f.seq.st;
        try same(s, expected, s.pos_dev, 4);
        try same(s, expected, s.conv, s.kda_layers * 3 * s.conv_c * 2);
        try same(s, expected, s.rec, 2 * s.kda_layers * s.rec_layer);
        try same(s, expected, s.proj, s.kda_layers * s.proj_layer * 2);
        try same(s, expected, s.save_k, s.kda_layers * 8 * s.heads * 128 * 4);
        try same(s, expected, s.save_v, s.kda_layers * 8 * s.heads * 128 * 2);
        try same(s, expected, s.save_g, s.kda_layers * 8 * s.heads * 128 * 4);
        try same(s, expected, s.save_b, s.kda_layers * 8 * s.heads * 4);
        for (s.caches, 0..) |c, j| try f.expectCache(expected, c, j, at);
        if (f.e.mtp) try f.expectCache(expected, s.mtp_caches, s.caches.len, head);
        try same(s, expected, s.mtp_pos_dev, 4);
        try same(s, expected, s.prompt_hidden, f.e.c.hidden * 2);
    }
    fn expectCache(f: *Fixture, expected: []const u8, c: anytype, j: usize, at: usize) !void {
        const s = &f.seq.st;
        try same(s, expected, c.lc, at * f.e.c.kv_lora * @as(usize, if (c.ls == 0) 2 else 1));
        if (c.ls != 0) try same(s, expected, c.ls, at * @sizeOf(f32));
        for (s.rings[2 * j ..][0..2]) |r| try same(s, expected, r, @min(at, s.ring_rows) * s.ring_row);
        try same(s, expected, c.pk, @min(at / 4 + 2, c.pk_rows) * f.e.c.i_dim * 2);
    }
    fn same(s: *const glm.forward.State, expected: []const u8, ptr: u64, len: usize) !void {
        const off: usize = @intCast(ptr - s.arena.buf.ptr);
        try t.expectEqualSlices(u8, expected[off..][0..len], rec.bytes(ptr, len));
    }
};
