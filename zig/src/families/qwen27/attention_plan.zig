//! Absolute key tiles and ancestor paths preserve a row's attention independently of siblings and stream packing.
const std = @import("std");
pub const max_streams = 8;
pub const max_rows = 128;
pub const path_width = 128;
pub const global_words = 8;
pub const stream_words = 12;
pub const chunk = 512;
pub const tile = 64;
pub const Window = struct { parents: []const i32, start: u32, capacity: u32 };
pub const KeepRow = extern struct { stream: u32, source: u32, destination: u32, row: u32 };
pub const Geometry = struct {
    heads: u32 = 24,
    kv_heads: u32 = 4,
    dim: u32 = 256,
    rotary: u32 = 64,
    shared_prefix: bool = true,
    /// Chains of rows take the prefix kernel to their last key; a test turns this off to compare with the tail kernel.
    fuse_chain: bool = true,
    pub fn check(g: Geometry) !void {
        if (g.heads != 24 or g.kv_heads != 4 or g.dim != 256 or g.rotary != 64) return error.TargetAttentionShape;
    }
    pub fn group(g: Geometry) u32 {
        return g.heads / g.kv_heads;
    }
};
pub const Plan = struct {
    gpa: std.mem.Allocator,
    geometry: Geometry,
    streams: u32,
    rows: u32,
    tiles: u32,
    prefix_chunks: u32,
    tail_chunks: u32,
    /// Every stream is a chain of rows: the prefix kernel reads each row's own keys itself, so no tail kernel runs.
    fused: bool,
    meta: [global_words + max_streams * stream_words]i32 = @splat(0),
    tile_stream: []i32,
    q_rows: []i32,
    nodes: []i32,
    paths: []i32,
    positions: []i32,
    pub fn init(gpa: std.mem.Allocator, windows: []const Window, geometry: Geometry) !Plan {
        try geometry.check();
        if (windows.len == 0 or windows.len > max_streams) return error.AttentionStreams;
        var meta: [global_words + max_streams * stream_words]i32 = @splat(0);
        var row_count: u32 = 0;
        var tile_count: u32 = 0;
        var ca_max: u32 = 0;
        var cb_max: u32 = 0;
        var fused = geometry.shared_prefix and geometry.fuse_chain;
        for (windows) |w| for (w.parents, 0..) |p, r| if (r > 0 and p != @as(i32, @intCast(r)) - 1) {
            fused = false;
        };
        for (windows, 0..) |w, st| {
            if (w.parents.len == 0 or w.parents.len > max_rows or w.start > w.capacity or @as(u64, w.start) + w.parents.len > w.capacity or w.capacity > std.math.maxInt(i32) / geometry.dim) return error.AttentionCapacity;
            if (w.parents[0] != -1) return error.AttentionParents;
            var depths: [max_rows]u32 = undefined;
            var deepest: u32 = 0;
            for (w.parents, 0..) |p, r| {
                if (p < -1 or p >= r) return error.AttentionParents;
                depths[r] = if (p < 0) 0 else depths[@intCast(p)] + 1;
                deepest = @max(deepest, depths[r]);
            }
            const pt = if (geometry.shared_prefix) w.start / tile * tile else 0;
            // a chain's prefix kernel runs to its last row's key, any other round stops at the tile before its own keys
            const limit: u32 = if (fused) w.start + @as(u32, @intCast(w.parents.len)) else pt;
            const ca = (limit + chunk - 1) / chunk;
            const cb = if (fused) 0 else (w.start + deepest) / chunk - pt / chunk + 1;
            const count: u32 = @intCast((geometry.group() * w.parents.len + 15) / 16);
            const at = global_words + st * stream_words;
            meta[at] = @intCast(pt);
            meta[at + 1] = @intCast(ca);
            meta[at + 2] = @intCast(w.parents.len);
            meta[at + 3] = @intCast(tile_count);
            meta[at + 4] = @intCast(w.start);
            meta[at + 5] = @intCast(cb);
            meta[at + 6] = @intCast(row_count);
            meta[at + 7] = @intCast(w.capacity * geometry.dim);
            meta[at + 8] = meta[at + 7];
            meta[at + 9] = @intCast(limit);
            meta[at + 10] = @intCast(if (fused) ca else pt / chunk);
            row_count = try std.math.add(u32, row_count, @intCast(w.parents.len));
            tile_count += count;
            ca_max = @max(ca_max, ca);
            cb_max = @max(cb_max, cb);
        }
        if (row_count > max_rows) return error.AttentionRows;
        meta[0] = @intCast(windows.len);
        meta[1] = @intCast(ca_max);
        meta[2] = @intCast(tile_count);
        meta[3] = @intCast(cb_max);
        meta[4] = @intCast(row_count);
        meta[5] = @intCast(tile_count * 16);
        meta[6] = @intCast(ca_max);
        const ts = try gpa.alloc(i32, tile_count);
        errdefer gpa.free(ts);
        const qr = try gpa.alloc(i32, tile_count * 16);
        errdefer gpa.free(qr);
        const nodes = try gpa.alloc(i32, row_count * 2);
        errdefer gpa.free(nodes);
        const paths = try gpa.alloc(i32, row_count * path_width);
        errdefer gpa.free(paths);
        @memset(paths, 0);
        const positions = try gpa.alloc(i32, row_count);
        errdefer gpa.free(positions);
        for (windows, 0..) |w, st| {
            const at = global_words + st * stream_words;
            const first: usize = @intCast(meta[at + 6]);
            const first_tile: usize = @intCast(meta[at + 3]);
            const count = (geometry.group() * w.parents.len + 15) / 16;
            for (0..count) |i| ts[first_tile + i] = @intCast(st);
            for (0..count * 16) |i| qr[first_tile * 16 + i] = @intCast(first * geometry.group() + (if (i < w.parents.len * geometry.group()) i else 0));
            for (w.parents, 0..) |parent, r| {
                const node = first + r;
                const depth: usize = if (parent < 0) 0 else @as(usize, @intCast(nodes[(first + @as(usize, @intCast(parent))) * 2])) + 1;
                nodes[node * 2] = @intCast(depth);
                nodes[node * 2 + 1] = @intCast(st);
                positions[node] = @intCast(w.start + depth);
                if (parent >= 0) @memcpy(paths[node * path_width ..][0..depth], paths[(first + @as(usize, @intCast(parent))) * path_width ..][0..depth]);
                paths[node * path_width + depth] = @intCast(r);
            }
        }
        return .{ .gpa = gpa, .geometry = geometry, .streams = @intCast(windows.len), .rows = row_count, .tiles = tile_count, .prefix_chunks = ca_max, .tail_chunks = cb_max, .fused = fused, .meta = meta, .tile_stream = ts, .q_rows = qr, .nodes = nodes, .paths = paths, .positions = positions };
    }
    pub fn deinit(p: *Plan) void {
        p.gpa.free(p.tile_stream);
        p.gpa.free(p.q_rows);
        p.gpa.free(p.nodes);
        p.gpa.free(p.paths);
        p.gpa.free(p.positions);
        p.* = undefined;
    }
    pub fn physical(p: Plan, node: usize, logical: u32) !?u32 {
        if (node >= p.rows) return error.AttentionRows;
        const stream: usize = @intCast(p.nodes[node * 2 + 1]);
        const start: u32 = @intCast(p.meta[global_words + stream * stream_words + 4]);
        const depth: u32 = @intCast(p.nodes[node * 2]);
        if (logical < start) return logical;
        if (logical > start + depth) return null;
        return start + @as(u32, @intCast(p.paths[node * path_width + logical - start]));
    }
    pub fn workspaceBytes(p: Plan) !usize {
        const rp: usize = p.tiles * 16;
        const hk: usize = p.geometry.kv_heads;
        const d: usize = p.geometry.dim;
        const a = try std.math.mul(usize, try std.math.mul(usize, hk, p.prefix_chunks), rp);
        const b = try std.math.mul(usize, try std.math.mul(usize, hk, p.tail_chunks), @as(usize, p.rows) * 16);
        return std.math.add(usize, try std.math.mul(usize, a + b, (d + 2) * 4), try std.math.mul(usize, hk * (rp + @as(usize, p.rows) * 16) * d, 2));
    }
    pub fn keepRows(p: Plan, gpa: std.mem.Allocator, kept: []const []const u32) ![]KeepRow {
        if (kept.len != p.streams) return error.AttentionStreams;
        var count: usize = 0;
        for (kept, 0..) |path, st| {
            const at = global_words + st * stream_words;
            const first: usize = @intCast(p.meta[at + 6]);
            const width: usize = @intCast(p.meta[at + 2]);
            if (path.len > width) return error.AttentionKeep;
            if (path.len > 0) {
                const last = path[path.len - 1];
                if (last >= width or p.nodes[(first + last) * 2] + 1 != path.len) return error.AttentionKeep;
                for (path, 0..) |r, depth| if (r >= width or p.paths[(first + last) * path_width + depth] != r) return error.AttentionKeep;
            }
            count += path.len;
        }
        const rows = try gpa.alloc(KeepRow, count);
        var i: usize = 0;
        for (kept, 0..) |path, st| for (path, 0..) |r, depth| {
            rows[i] = .{ .stream = @intCast(st), .source = r, .destination = @intCast(depth), .row = @intCast(i) };
            i += 1;
        };
        return rows;
    }
};

test "siblings at one absolute position see only their own ancestors and committed keys" {
    var p = try Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0, 0, 1, 2 }, .start = 600, .capacity = 1024 }}, .{});
    defer p.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 600, 601, 601, 602, 602 }, p.positions);
    try std.testing.expectEqual(@as(?u32, 602), try p.physical(4, 601));
    try std.testing.expectEqual(@as(?u32, 604), try p.physical(4, 602));
    try std.testing.expectEqual(@as(?u32, null), try p.physical(4, 603));
    try std.testing.expectEqual(@as(i32, 576), p.meta[8]);
    try std.testing.expectEqual(@as(i32, 2), p.meta[9]);
    try std.testing.expectEqual(@as(i32, 1), p.meta[13]);
}
test "absolute64tile512chunk boundaries retain prefix continuation without sibling padding" {
    for ([_]u32{ 0, 63, 64, 511, 512, 600 }) |start| {
        var p = try Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0 }, .start = start, .capacity = 2048 }}, .{});
        defer p.deinit();
        try std.testing.expectEqual(start / 64 * 64, @as(u32, @intCast(p.meta[8])));
        try std.testing.expectEqual(start + 1, @as(u32, @intCast(p.positions[1])));
        try std.testing.expect(try p.workspaceBytes() > 0);
    }
}
test "mixed streams keep distinct physical windows and reject capacity and parent defects" {
    var p = try Plan.init(std.testing.allocator, &.{ .{ .parents = &.{ -1, 0, 0 }, .start = 63, .capacity = 512 }, .{ .parents = &.{ -1, 0 }, .start = 511, .capacity = 1024 } }, .{});
    defer p.deinit();
    try std.testing.expectEqual(@as(u32, 5), p.rows);
    try std.testing.expectEqual(@as(?u32, 512), try p.physical(4, 512));
    try std.testing.expectError(error.AttentionParents, Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 2 }, .start = 0, .capacity = 8 }}, .{}));
    try std.testing.expectError(error.AttentionCapacity, Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0 }, .start = 7, .capacity = 8 }}, .{}));
}
test "nonprefix keep uses two-pass gather rows rather than clobbering a later source" {
    var p = try Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0, 0, 1, 2 }, .start = 600, .capacity = 1024 }}, .{});
    defer p.deinit();
    const rows = try p.keepRows(std.testing.allocator, &.{&.{ 0, 2, 4 }});
    defer std.testing.allocator.free(rows);
    try std.testing.expectEqual(@as(u32, 2), rows[1].source);
    try std.testing.expectEqual(@as(u32, 1), rows[1].destination);
    try std.testing.expectEqual(@as(u32, 2), rows[2].destination);
    try std.testing.expectError(error.AttentionKeep, p.keepRows(std.testing.allocator, &.{&.{ 0, 1, 4 }}));
}

test "canonical tail keeps the same causal and ancestor map across multiple key chunks" {
    const windows = [_]Window{.{ .parents = &.{ -1, 0, 0, 2 }, .start = 513, .capacity = 1024 }};
    var ordinary = try Plan.init(std.testing.allocator, &windows, .{});
    defer ordinary.deinit();
    var canonical = try Plan.init(std.testing.allocator, &windows, .{ .shared_prefix = false });
    defer canonical.deinit();
    try std.testing.expectEqual(@as(u32, 0), canonical.prefix_chunks);
    try std.testing.expectEqual(@as(u32, 2), canonical.tail_chunks);
    try std.testing.expectEqual(@as(u32, 1), ordinary.prefix_chunks);
    try std.testing.expectEqualSlices(i32, ordinary.positions, canonical.positions);
    for (0..canonical.rows) |row| for (0..518) |key| {
        try std.testing.expectEqual(try ordinary.physical(row, @intCast(key)), try canonical.physical(row, @intCast(key)));
    };
}

test "a chain's prefix runs to its last key and leaves no tail chunk, any other round keeps the tail kernel" {
    for ([_]u32{ 0, 1, 63, 64, 511, 512, 600 }) |start| {
        const chain = [_]Window{.{ .parents = &.{ -1, 0, 1 }, .start = start, .capacity = 2048 }};
        var fused = try Plan.init(std.testing.allocator, &chain, .{});
        defer fused.deinit();
        var tailed = try Plan.init(std.testing.allocator, &chain, .{ .fuse_chain = false });
        defer tailed.deinit();
        try std.testing.expect(fused.fused and !tailed.fused);
        try std.testing.expectEqual(start + 3, @as(u32, @intCast(fused.meta[global_words + 9])));
        try std.testing.expectEqual((start + 3 + chunk - 1) / chunk, fused.prefix_chunks);
        try std.testing.expectEqual(@as(u32, @intCast(fused.meta[global_words + 10])), fused.prefix_chunks);
        try std.testing.expectEqual(@as(u32, 0), fused.tail_chunks);
        try std.testing.expect(tailed.tail_chunks > 0);
        try std.testing.expectEqual(start / tile * tile, @as(u32, @intCast(tailed.meta[global_words + 9])));
        try std.testing.expectEqualSlices(i32, tailed.positions, fused.positions);
        for (0..3) |row| for (0..start + 3) |key| try std.testing.expectEqual(try tailed.physical(row, @intCast(key)), try fused.physical(row, @intCast(key)));
    }
    const tree = [_]Window{.{ .parents = &.{ -1, 0, 0 }, .start = 600, .capacity = 1024 }};
    var forked = try Plan.init(std.testing.allocator, &tree, .{});
    defer forked.deinit();
    try std.testing.expect(!forked.fused);
    const mixed = [_]Window{ .{ .parents = &.{ -1, 0 }, .start = 5, .capacity = 512 }, tree[0] };
    var both = try Plan.init(std.testing.allocator, &mixed, .{});
    defer both.deinit();
    try std.testing.expect(!both.fused);
    var canonical = try Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0 }, .start = 5, .capacity = 512 }}, .{ .shared_prefix = false });
    defer canonical.deinit();
    try std.testing.expect(!canonical.fused);
}
