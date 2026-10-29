//! A recorded call's streams in one dispatch a kernel and layer: each layer's stream table, the rows' streams, tiles.
const std = @import("std");
const mtl = @import("metal");
const st = @import("state.zig");
const c = @import("config.zig");

/// lfm2_segments.metal's lfm2_seg: a stream's cache at one layer, its rows from `base`, key count and sub-spans.
pub const Seg = extern struct { keys: u64 = 0, values: u64 = 0, conv: u64 = 0, base: u32, rows: u32, len: u32, cap: u32, subs: u32, pad: u32 = 0 };

/// Bytes of one layer's table in Scratch.segs.
pub const table_bytes = st.batch_rows * @sizeOf(Seg);

/// The attention's 8-row tiles over every stream and the grid depth (chunks times sub-spans) the deepest needs.
pub const Plan = struct { tiles: u32 = 0, used: u32 = 0 };

/// Whether a call takes these kernels: a recorded window of more than one stream.
pub fn batched(record: bool, streams: usize) bool {
    return record and streams > 1;
}

/// The streams' tables, row map and tiles into the scratch, and every cache buffer the kernels reach declared on `e`.
pub fn prepare(e: anytype, s: *st.Scratch, streams: anytype) Plan {
    var plan: Plan = .{};
    const rowseg = s.rowseg.slice(u32, st.batch_rows);
    const tiles = s.tiles.slice(u32, st.batch_rows);
    var base: usize = 0;
    for (streams, 0..) |seg, j| {
        // A stream of more than split_rows rows folds a chunk's sub-spans in its tiles, as its own call would.
        const subs: u32 = if (seg.rows > st.split_rows) 1 else st.subs;
        for (0..(seg.rows + 7) / 8) |t| {
            tiles[plan.tiles] = @intCast((j << 16) | t);
            plan.tiles += 1;
        }
        const used: u32 = @intCast((seg.cache.len + seg.rows + st.chunk_keys - 1) / st.chunk_keys * subs);
        plan.used = @max(plan.used, used);
        @memset(rowseg[base..][0..seg.rows], @intCast(j));
        for (seg.cache.blocks, 0..) |block, layer| {
            var entry: Seg = .{ .base = @intCast(base), .rows = @intCast(seg.rows), .len = @intCast(seg.cache.len), .cap = @intCast(seg.cache.capacity), .subs = subs };
            switch (block) {
                .conv => |b| entry.conv = b.gpuAddress(),
                .attention => |a| {
                    entry.keys = a.keys.gpuAddress();
                    entry.values = a.values.gpuAddress();
                },
            }
            s.segs.slice(Seg, c.layers * st.batch_rows)[layer * st.batch_rows + j] = entry;
        }
        // The kernels reach the caches through the table: each buffer is declared, in a residency set or not.
        for (seg.cache.memory.buffers.items) |b| e.useResource(b, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        base += seg.rows;
    }
    return plan;
}

/// Buffers a call declares to its encoder (useResource), in order.
const Declared = struct {
    ids: [8 * c.layers]?*anyopaque = undefined,
    count: usize = 0,
    pub fn useResource(self: *Declared, b: mtl.Buffer, _: usize) void {
        self.ids[self.count] = @ptrCast(b.id);
        self.count += 1;
    }
    fn has(self: *const Declared, b: mtl.Buffer) bool {
        return std.mem.indexOfScalar(?*anyopaque, self.ids[0..self.count], @ptrCast(b.id)) != null;
    }
};

test "lfm2_moe: a shared round's table reaches only cache buffers it declares, and maps its rows and tiles" {
    const gpa = std.testing.allocator;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var attention: [c.layers]bool = @splat(false);
    for (&attention, 0..) |*a, i| a.* = i % 4 == 2;
    var s = try st.Scratch.init(gpa, device, &attention, st.batch_rows);
    defer s.deinit();
    var caches: [2]st.Cache = undefined;
    for (&caches) |*cache| cache.* = try st.Cache.init(gpa, device, &attention, 64);
    defer for (&caches) |*cache| cache.deinit();
    caches[0].len = 5;
    caches[1].len = 40;
    const streams = [_]struct { cache: *st.Cache, rows: usize }{ .{ .cache = &caches[0], .rows = 9 }, .{ .cache = &caches[1], .rows = 1 } };
    var declared: Declared = .{};
    const plan = prepare(&declared, &s, &streams);
    // Each cache's every buffer: a buffer the kernels reach through the table only is not resident unless declared.
    var want: usize = 0;
    for (&caches) |*cache| for (cache.memory.buffers.items) |b| {
        try std.testing.expect(declared.has(b));
        want += 1;
    };
    try std.testing.expectEqual(want, declared.count);
    try std.testing.expectEqual(@as(u32, 3), plan.tiles);
    try std.testing.expectEqual(@as(u32, st.subs), plan.used);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1 << 16 }, s.tiles.slice(u32, 3));
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, s.rowseg.slice(u32, 10));
    const table = s.segs.slice(Seg, c.layers * st.batch_rows);
    for (attention, 0..) |a, layer| for (&caches, 0..) |*cache, j| {
        const e = table[layer * st.batch_rows + j];
        try std.testing.expectEqual(@as(u32, if (j == 0) 0 else 9), e.base);
        try std.testing.expectEqual(@as(u32, if (j == 0) 1 else st.subs), e.subs);
        try std.testing.expectEqual(@as(u32, @intCast(cache.len)), e.len);
        if (a) try std.testing.expectEqual(cache.blocks[layer].attention.keys.gpuAddress(), e.keys) else try std.testing.expectEqual(cache.blocks[layer].conv.gpuAddress(), e.conv);
    };
}
