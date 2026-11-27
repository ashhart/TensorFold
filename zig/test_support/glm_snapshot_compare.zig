//! Independent committed-state byte comparisons between window and paged GLM sequences.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const rec = @import("glm_snap_driver.zig").Driver;
const Fixture = @import("glm_snap_fixture.zig").Fixture;
const t = std.testing;

/// A paged sequence of the fixture's shape (a DSA and a KDA layer), its caches on the pages at `kv`, `ids` mapped.
pub fn pagedSeq(d: *const cuda.Driver, f: *const Fixture, kv: u64, page: u64, ids: []const u32) !glm.forward.State {
    var w: glm.weights.Weights = undefined;
    w.config = f.e.c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    var st = try glm.forward.State.initPaged(t.allocator, d, &w, 16448, .{ .kv = kv, .pages = glm.pages.tableLen(16448), .page_bytes = page });
    st.pages = ids;
    st.capacity = ids.len * 512;
    return st;
}

/// Row `r`'s first byte in a cache of `width`-byte rows, `per` rows a page when the cache is on pages.
fn rowAt(st: *const glm.forward.State, x: glm.dsa_block.Caches, base: u64, r: usize, width: usize, per: usize) u64 {
    return if (x.pt == 0) base + r * width else base + @as(u64, st.pages[r / per]) * x.ps + (r % per) * width;
}

fn same(a: u64, b: u64, len: usize) !void {
    try t.expectEqualSlices(u8, rec.bytes(a, len), rec.bytes(b, len));
}

/// Every field a prefix at `at` reads (the MTP layer's `head` rows), in `a` and in `b`, from either layout.
pub fn expectSamePrefix(c: glm.config.Config, a: *const glm.forward.State, b: *const glm.forward.State, at: usize, head: ?usize) !void {
    const kv = a.kda_layers * 8 * a.heads * 128;
    try same(a.pos_dev, b.pos_dev, 4);
    try same(a.conv, b.conv, a.kda_layers * 3 * a.conv_c * 2);
    try same(a.rec, b.rec, 2 * a.kda_layers * a.rec_layer);
    try same(a.proj, b.proj, a.kda_layers * a.proj_layer * 2);
    try same(a.save_k, b.save_k, kv * 4);
    try same(a.save_v, b.save_v, kv * 2);
    try same(a.save_g, b.save_g, kv * 4);
    try same(a.save_b, b.save_b, a.kda_layers * 8 * a.heads * 4);
    for (0..a.caches.len + 1) |j| {
        const n = if (j < a.caches.len) at else head orelse break;
        const x = if (j < a.caches.len) a.caches[j] else a.mtp_caches;
        const y = if (j < b.caches.len) b.caches[j] else b.mtp_caches;
        const lat = c.kv_lora * 2;
        for (0..n) |r| try same(rowAt(a, x, x.lc, r, lat, 512), rowAt(b, y, y.lc, r, lat, 512), lat);
        for (0..2) |k| try same(a.rings[2 * j + k], b.rings[2 * j + k], @min(n, a.ring_rows) * a.ring_row);
        for (0..@min(n / 4 + 2, x.pk_rows)) |p| try same(rowAt(a, x, x.pk, p, c.i_dim * 2, 128), rowAt(b, y, y.pk, p, c.i_dim * 2, 128), c.i_dim * 2);
    }
    try same(a.mtp_pos_dev, b.mtp_pos_dev, 4);
    try same(a.prompt_hidden, b.prompt_hidden, c.hidden * 2);
}
