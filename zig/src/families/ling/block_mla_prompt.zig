//! Ling's MLA prompt rows expanded: kv_b over cache rows, flash tiles.

// Per head 192 q.k dims and 128 v dims a key instead of the absorbed 576 + 512; decode windows stay absorbed.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("graph.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");
const block_kda = @import("block_kda.zig");

const Ops = graph.Ops;
const kw = state.cache_width;

/// tf_ling_pattn's dynamic shared bytes: two stages of 64 keys, 192 + 128 bf16 dims each.
pub const smem: u32 = 2 * 64 * (24 + 16) * 16;
/// Query rows a tf_ling_pattn block takes.
pub const block_rows = 128;

/// Whether MLA rows take the expanded form: prompt rows.
pub fn takes(o: Ops, b: *const state.Buffers) bool {
    return o.prompt and b.prompt;
}

/// Keys one expansion holds: the scratch's absorbed-query area (rows x heads x 576 bf16) as [2 rows][heads][256] bf16.
pub fn blockKeys(b: *const state.Buffers) usize {
    return 2 * b.rows;
}

/// Segment `sg`'s rows (q roped in b.q) over its cache through its own rows into b.vn.
pub fn attend(o: Ops, b: *const state.Buffers, a: weights.Mla, g: *const graph.Graph, mi: usize, sg: block_kda.Seg, scale: f32) !void {
    const H: usize = g.w.config.mla_heads;
    const st = sg.st;
    const end = st.pos + sg.rows;
    const kb = blockKeys(b);
    const co = b.ol; // fp32 carry [rows][heads][128], then m and l [rows][heads]: inside the unused latent-output area
    const cm = co + b.rows * H * 128 * 4;
    const cl = cm + b.rows * H * 4;
    const cache = st.cacheAt(mi);
    var k0: usize = 0;
    while (k0 < end) : (k0 += kb) {
        const n = @min(kb, end - k0);
        const rows_at = cache + k0 * kw * 2;
        var j: usize = 0; // kv_b on the decode lane, as the absorbed copies were made
        while (j < n) : (j += b.rows) try o.m.linear().decode(o.s, a.kv_b, .{ .x = rows_at + j * kw * 2, .ldx = kw, .part = o.part }, b.qa + j * H * 256 * 2, @min(b.rows, n - j));
        var x: cuda.Args = .{};
        inline for (.{ b.q + sg.row0 * H * 192 * 2, b.qa, rows_at, co, cm, cl, b.vn + sg.row0 * H * 128 * 2 }) |v| x.add(v);
        inline for (.{ sg.rows, H, n, k0, st.pos }) |v| x.add(@as(c_int, @intCast(v)));
        x.add(scale);
        x.add(@as(c_int, @intFromBool(k0 == 0)));
        x.add(@as(c_int, @intFromBool(k0 + kb >= end)));
        const grid: cuda.launch.Dim3 = .{ .x = @intCast((sg.rows + block_rows - 1) / block_rows), .y = @intCast(H) };
        try cuda.launch.launch(o.m.pattn, .{ .grid = grid, .block = .{ .x = 256 }, .shared = smem }, o.s, &x);
    }
}
