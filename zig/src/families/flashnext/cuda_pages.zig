//! Flash Next attention caches as pages: per layer keys, values, indexer keys, pooled rows (bf16); a tier: two planes.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const Pg = @import("cuda_glue.zig").Pg;

const kv_row = wt.kv_heads * wt.head_dim * 2;
const index_row = wt.index_dim * 2;
const ratio = wt.index_ratio;

pub const Part = enum { key, value, index, pooled };

/// Bytes one row of `part` takes: a position's (a block's for pooled).
pub fn rowBytes(part: Part) usize {
    return switch (part) {
        .key, .value => kv_row,
        .index, .pooled => index_row,
    };
}

/// Rows of `part` a page of `tokens` positions holds.
pub fn rows(part: Part, tokens: usize) usize {
    return if (part == .pooled) tokens / ratio else tokens;
}

/// Bytes one attention layer's share of a page takes in the K/V plane.
pub fn kvLayerBytes(tokens: usize) usize {
    return 2 * tokens * kv_row;
}

/// Bytes one attention layer's share of a page takes in the index plane.
pub fn indexLayerBytes(tokens: usize) usize {
    return tokens * index_row + tokens / ratio * index_row;
}

/// Bytes one attention layer's share of a page takes, both planes.
pub fn layerBytes(tokens: usize) usize {
    return kvLayerBytes(tokens) + indexLayerBytes(tokens);
}

/// Bytes a page takes: every attention layer's share.
pub fn pageBytes(tokens: usize) usize {
    return pageBytesOf(tokens, wt.attn_n);
}

/// Bytes a page of `layers` attention layers takes (the model's, and the MTP head's cache when it drafts), both planes.
pub fn pageBytesOf(tokens: usize, layers: usize) usize {
    return layers * layerBytes(tokens);
}

/// Bytes a page of `layers` layers takes in the K/V plane.
pub fn kvPageBytes(tokens: usize, layers: usize) usize {
    return layers * kvLayerBytes(tokens);
}

/// Bytes a page of `layers` layers takes in the index plane.
pub fn indexPageBytes(tokens: usize, layers: usize) usize {
    return layers * indexLayerBytes(tokens);
}

/// Whether `part` is in the index plane.
pub fn inIndexPlane(part: Part) bool {
    return part == .index or part == .pooled;
}

/// Where `part` of layer `layer` starts in a one-plane page: per layer keys, values, indexer keys, pooled rows.
pub fn offset(tokens: usize, layer: usize, part: Part) usize {
    const at = layer * layerBytes(tokens);
    return at + switch (part) {
        .key => 0,
        .value => tokens * kv_row,
        .index => 2 * tokens * kv_row,
        .pooled => 2 * tokens * kv_row + tokens * index_row,
    };
}

/// Where `part` of attention layer `layer` starts inside its plane's page when a host tier keeps two planes.
pub fn planeOffset(tokens: usize, layer: usize, part: Part) usize {
    return switch (part) {
        .key => layer * kvLayerBytes(tokens),
        .value => layer * kvLayerBytes(tokens) + tokens * kv_row,
        .index => layer * indexLayerBytes(tokens),
        .pooled => layer * indexLayerBytes(tokens) + tokens * index_row,
    };
}

/// A page size the kernels take: whole blocks of the indexer's ratio.
pub fn valid(tokens: usize) bool {
    return tokens > 0 and tokens <= wt.window and tokens <= std.math.maxInt(c_int) and tokens % ratio == 0;
}

/// TF_FLASHNEXT_KV_PAGE_TEST=n: each stream's caches read through reversed n-position pages (checks); 0 when unset.
pub fn pageTest() !usize {
    const v = std.c.getenv("TF_FLASHNEXT_KV_PAGE_TEST") orelse return 0;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadFlashNextPageTest;
    if (!valid(n)) return error.BadFlashNextPageTest;
    return n;
}

/// A bounded diagnostic allocation: the padded arena, table, and their combined device charge.
pub const Layout = struct { pages: usize, arena: usize, table: usize, total: usize };

pub fn checkedLayout(cap: usize, tokens: usize) !Layout {
    if (cap == 0 or cap > wt.window or !valid(tokens)) return error.BadFlashNextPageTest;
    const n = cap / tokens + @intFromBool(cap % tokens != 0);
    if (n > std.math.maxInt(c_int)) return error.BadFlashNextPageTest;
    const stride = std.math.mul(usize, tokens, wt.attn_n * (2 * kv_row + index_row + index_row / ratio)) catch return error.BadFlashNextPageTest;
    const arena = std.math.mul(usize, n, stride) catch return error.BadFlashNextPageTest;
    const table = std.math.mul(usize, n, @sizeOf(u32)) catch return error.BadFlashNextPageTest;
    if (arena / 2 > std.math.maxInt(i64)) return error.BadFlashNextPageTest;
    const total = std.math.add(usize, arena, table) catch return error.BadFlashNextPageTest;
    return .{ .pages = n, .arena = arena, .table = table, .total = total };
}

/// One stream's pages: page i of its positions at `base` + ids[i] pages.
pub const Paged = struct {
    /// Where page 0 starts (a host tier: of the K/V plane).
    base: u64,
    /// A host tier: where page 0 of the index plane starts.
    ibase: u64 = 0,
    /// A host tier keeps two planes: the K/V rows in frames, the indexer keys by page id.
    planes: bool = false,
    /// The stream's table on the device, an int a page (`ids` uploaded).
    table: cuda.DeviceBuffer,
    ids: []u32,
    /// The frame of each page's K/V rows when a host tier maps them (empty: the page id is the frame).
    fids: []const u32 = &.{},
    /// A host tier's K/V store as the device addresses it (0: none): snapshots use its rows, not the frames'.
    hbase: u64 = 0,
    /// The index plane's table on the device, an int a page naming the logical page (0: the K/V table's).
    itable: u64 = 0,
    /// Positions a page holds.
    tokens: usize,
    /// The paged state owns `ids` (the page test's); a pool's lane maps its lease's.
    own_ids: bool = true,
    /// The page test's own arena.
    arena: ?cuda.DeviceBuffer = null,
    /// Attention layers a page holds: the model's, and the MTP head's after them in a pool's pages.
    layers: usize = wt.attn_n,
    stream: ?cuda.Stream = null,

    pub fn pg(p: *const Paged) Pg {
        if (!p.planes) return .{ .pt = p.table.ptr, .ps = @intCast(pageBytesOf(p.tokens, p.layers) / 2), .ptok = @intCast(p.tokens) };
        return .{ .pt = p.table.ptr, .ps = @intCast(kvPageBytes(p.tokens, p.layers) / 2), .ptok = @intCast(p.tokens), .ips = @intCast(indexPageBytes(p.tokens, p.layers) / 2), .ipt = p.itable };
    }

    /// `part` of attention layer `layer` in page 0: the cache pointer a kernel takes beside pg().
    pub fn at(p: *const Paged, layer: usize, part: Part) u64 {
        if (!p.planes) return p.base + offset(p.tokens, layer, part);
        return (if (inIndexPlane(part)) p.ibase else p.base) + planeOffset(p.tokens, layer, part);
    }

    /// Byte address of row `row` of `part` of `layer` (a position, or a block for pooled).
    pub fn rowAt(p: *const Paged, layer: usize, part: Part, row: usize) u64 {
        const per = rows(part, p.tokens);
        if (!p.planes) return p.base + @as(u64, p.ids[row / per]) * pageBytesOf(p.tokens, p.layers) + offset(p.tokens, layer, part) + (row % per) * rowBytes(part);
        const kv_id = if (p.fids.len > 0) p.fids[row / per] else p.ids[row / per];
        const plane = if (inIndexPlane(part)) p.ibase + @as(u64, p.ids[row / per]) * indexPageBytes(p.tokens, p.layers) else p.base + @as(u64, kv_id) * kvPageBytes(p.tokens, p.layers);
        return plane + planeOffset(p.tokens, layer, part) + (row % per) * rowBytes(part);
    }

    /// Byte address of row `row` of a K/V `part` in the host store (a host tier's canonical rows).
    pub fn hostRowAt(p: *const Paged, layer: usize, part: Part, row: usize) u64 {
        const per = rows(part, p.tokens);
        return p.hbase + @as(u64, p.ids[row / per]) * kvPageBytes(p.tokens, p.layers) + planeOffset(p.tokens, layer, part) + (row % per) * rowBytes(part);
    }

    /// Borrowed views of each attention layer's `part` in page 0 (what kc, vc, ikc and pooled hold when paged).
    pub fn views(p: *const Paged, gpa: std.mem.Allocator, d: *const cuda.Driver, part: Part) ![]cuda.DeviceBuffer {
        const out = try gpa.alloc(cuda.DeviceBuffer, wt.attn_n);
        for (out, 0..) |*b, layer| b.* = .{ .d = d, .ptr = p.at(layer, part), .len = 0, .borrowed = true };
        return out;
    }

    pub fn deinit(p: *Paged, gpa: std.mem.Allocator) void {
        if (p.stream) |stream| cuda.completion.drain(p.table.d, stream, "diagnostic pages did not drain");
        p.table.free();
        if (p.own_ids) gpa.free(p.ids);
        if (p.arena) |*a| a.free();
    }
};

/// The page test's zeroed pages for `cap` positions in one buffer, in reverse order (no contiguous layout matches).
pub fn testPages(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, cap: usize, tokens: usize) !Paged {
    const layout = try checkedLayout(cap, tokens);
    const n = layout.pages;
    var arena = try cuda.DeviceBuffer.alloc(d, layout.arena);
    errdefer arena.free();
    const ids = try gpa.alloc(u32, n);
    errdefer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(n - 1 - i);
    var table = try cuda.DeviceBuffer.alloc(d, layout.table);
    errdefer table.free();
    errdefer cuda.completion.drain(d, stream, "diagnostic page construction did not drain");
    try arena.fill8(0, stream.handle);
    try d.check(d.api.cuMemcpyHtoDAsync_v2(table.ptr, ids.ptr, layout.table, stream.handle), "cuMemcpyHtoDAsync");
    try stream.synchronize();
    return .{ .base = arena.ptr, .table = table, .ids = ids, .tokens = tokens, .arena = arena, .stream = stream };
}

test "a page's parts tile it in order and pooled rows follow their positions' page" {
    const t = 16;
    try std.testing.expectEqual(@as(usize, 0), offset(t, 0, .key));
    try std.testing.expectEqual(t * kv_row, offset(t, 0, .value));
    try std.testing.expectEqual(offset(t, 0, .pooled) + t / ratio * index_row, offset(t, 1, .key));
    try std.testing.expectEqual(pageBytes(t), offset(t, wt.attn_n - 1, .pooled) + t / ratio * index_row);
    try std.testing.expect(!valid(6) and valid(8) and !valid(0));
    var ids = [_]u32{ 2, 0, 1 };
    const p: Paged = .{ .base = 1 << 20, .table = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true }, .ids = &ids, .tokens = t };
    // position 17 is page 1 (id 0) row 1; block 5 (positions 20..23) is page 1 row 1 of the pooled rows
    try std.testing.expectEqual(p.base + offset(t, 3, .value) + kv_row, p.rowAt(3, .value, 17));
    try std.testing.expectEqual(p.base + offset(t, 3, .pooled) + index_row, p.rowAt(3, .pooled, 5));
    try std.testing.expectEqual(p.base + 2 * pageBytes(t) + offset(t, 0, .index), p.rowAt(0, .index, 0));
    try std.testing.expectEqual(@as(i64, 0), p.pg().ips);
}

test "with a host tier a page's parts tile its two planes; the K/V rows follow their frames, the rest their pages" {
    const t = 16;
    try std.testing.expectEqual(@as(usize, 0), planeOffset(t, 0, .key));
    try std.testing.expectEqual(t * kv_row, planeOffset(t, 0, .value));
    try std.testing.expectEqual(kvLayerBytes(t), planeOffset(t, 1, .key));
    try std.testing.expectEqual(@as(usize, 0), planeOffset(t, 0, .index));
    try std.testing.expectEqual(planeOffset(t, 0, .pooled) + t / ratio * index_row, planeOffset(t, 1, .index));
    try std.testing.expectEqual(kvPageBytes(t, wt.attn_n), planeOffset(t, wt.attn_n - 1, .value) + t * kv_row);
    try std.testing.expectEqual(indexPageBytes(t, wt.attn_n), planeOffset(t, wt.attn_n - 1, .pooled) + t / ratio * index_row);
    try std.testing.expectEqual(pageBytes(t), kvPageBytes(t, wt.attn_n) + indexPageBytes(t, wt.attn_n));
    var ids = [_]u32{ 2, 0, 1 };
    var frames = [_]u32{ 5, 4, 3 };
    const p: Paged = .{ .base = 1 << 20, .ibase = 1 << 30, .planes = true, .table = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true }, .ids = &ids, .fids = &frames, .tokens = t };
    try std.testing.expectEqual(p.base + 4 * kvPageBytes(t, wt.attn_n) + planeOffset(t, 3, .value) + kv_row, p.rowAt(3, .value, 17));
    try std.testing.expectEqual(p.ibase + 0 * indexPageBytes(t, wt.attn_n) + planeOffset(t, 3, .pooled) + index_row, p.rowAt(3, .pooled, 5));
    try std.testing.expectEqual(p.ibase + 2 * indexPageBytes(t, wt.attn_n) + planeOffset(t, 0, .index), p.rowAt(0, .index, 0));
    try std.testing.expect(p.pg().ips != 0);
}

test {
    _ = @import("cuda_page_test_safety.zig");
}
