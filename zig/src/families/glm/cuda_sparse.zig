//! GLM-5.3-Flash's DSA indexer on CUDA (the Python family's sparse): each row keeps its best 512 4-token pools.

const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const index_mod = @import("cuda_index.zig");

/// sparse.POOL, TOPK_POOLS, SELECT_LIST, SCORE_LOOP_FROM, SCORE_LOOP; geometry.MLA_SELECT_ROWS.
pub const pool = 4;
pub const topk_pools = 512;
pub const select_list = 1024;
pub const score_loop_from = 64;
pub const score_loop = 32;
pub const select_rows = 512;
/// A row's token list: the 512 pools' tokens, then the incomplete pool's up to 3.
pub const width = topk_pools * pool + pool - 1;

/// The indexer's shape: rows a pooled key sums (1: keys as written), pools a row keeps, roped dims (`inv` [rope / 2]).
pub const Geometry = struct {
    pool: usize = pool,
    topk: usize = topk_pools,
    rope: usize = 0,
    inv: u64 = 0,

    /// A row's token list: its pools' tokens, then the incomplete pool's up to pool - 1.
    pub fn width(g: Geometry) usize {
        return g.topk * g.pool + g.pool - 1;
    }

    /// The kernels' cases: pools of 4 (gated) or 1, rope on whole lanes' pairs (a multiple of 4) with its frequencies.
    pub fn check(g: Geometry) !void {
        if ((g.pool != pool and g.pool != 1) or g.topk == 0 or g.rope % 4 != 0 or g.rope > 128 or (g.rope > 0) != (g.inv != 0))
            return error.IndexShape;
    }
};

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// At most wide_rows rows (one wave on 48 SMs) over wide_pools pools or more: 1,024 threads a row, the same bytes.
pub const wide_rows = 48;
const wide_pools = 4096;

fn wideSelect(rows: usize, np: usize) bool {
    return rows <= wide_rows and np >= wide_pools;
}

/// glm_index_decode.cuh's sizes: a scan CTA's pools and threads, a merge CTA's candidates and threads, the floor's aim.
pub const dsel_span = 4096;
pub const dsel_scan_threads = 256;
pub const dsel_cap = 4096;
pub const dsel_merge_threads = 1024;
/// The floor's aim, for speed only: every path picks tf_glm_select_rows' pools.
pub const dsel_target = 2048;

/// The bytes of a decode select's gt and eq bit words for `rows` rows at `np` pools (Scratch.ls, its bits_rows).
pub fn bitsBytes(rows: usize, np: usize) usize {
    return rows * 2 * cdiv(np, 32) * 4;
}

/// tf_glm_dsel_scan's grid for `rows` rows from host_pos: the last row's pools (it sees the most) in spans.
pub fn dselScanCtas(host_pos: usize, rows: usize, np: usize, g: Geometry) usize {
    const last = @min(np, @max((host_pos + rows) / g.pool, g.topk));
    // 0 when every row fits a merge CTA's candidates: the merge takes the rows whole and the scan marks nothing
    return if (last <= dsel_cap) 0 else cdiv(last, dsel_span);
}

/// A decode window's picks (the wide select's rows) on many CTAs: scores [rows][np] at `at` into pools [rows][g.topk].
pub fn decodeSelect(
    set: glue.Set,
    stream: cuda.Stream,
    scores: u64,
    pools: u64,
    np: usize,
    at: u64,
    host_pos: usize,
    rows: usize,
    bits: u64,
    /// 0, or [rows] int32 for each row's path (glm_index_decode.cuh; the checks count them).
    path: u64,
    g: Geometry,
) !void {
    const words = cdiv(np, 32);
    const ctas = dselScanCtas(host_pos, rows, np, g);
    if (ctas > 0) {
        var sa: cuda.Args = .{};
        inline for (.{ scores, @as(i64, @intCast(np)), at, @as(c_int, @intCast(g.topk)), @as(c_int, dsel_target), bits, @as(i64, @intCast(words)), @as(c_int, @intCast(g.pool)) }) |v| sa.add(v);
        try cuda_pdl.launch(set.glue.dsel_scan, .{ .grid = .{ .x = @intCast(ctas), .y = @intCast(rows) }, .block = .{ .x = dsel_scan_threads } }, stream, &sa);
    }
    var ma: cuda.Args = .{};
    inline for (.{ scores, pools, @as(i64, @intCast(np)), at, @as(c_int, @intCast(g.topk)), bits, @as(i64, @intCast(words)), path, @as(c_int, @intCast(g.pool)) }) |v| ma.add(v);
    try cuda_pdl.launch(set.glue.dsel_merge, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = dsel_merge_threads } }, stream, &ma);
}

/// sparse.index_update: the window's LayerNormed (and roped) index keys and gates into ik, ig; completed pools into pk.
pub fn indexUpdate(set: glue.Set, stream: cuda.Stream, k_raw: u64, k_stride: usize, gate: u64, ln_w: u64, ln_b: u64, ape: u64, ik: u64, ig: u64, pk: u64, t: latent.Table, pos: u64, rows: usize, g: Geometry) !void {
    try g.check();
    var a: cuda.Args = .{};
    inline for (.{ k_raw, @as(i64, @intCast(k_stride)), gate, ln_w, ln_b, ik, ig, pos, @as(f32, 1e-6), @as(c_int, @intCast(g.rope)), g.inv }) |v| a.add(v);
    try cuda_pdl.launch(set.glue.index_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
    var b: cuda.Args = .{};
    inline for (.{ ik, ig, ape, pk, pos, @as(i64, @intCast(rows)), t.pt, @as(i64, @intCast(t.ps)), @as(c_int, @intCast(g.pool)) }) |v| b.add(v);
    try cuda_pdl.launch(set.glue.pool_keys, .{ .grid = .{ .x = @intCast(rows / g.pool + 2) }, .block = .{ .x = 32 } }, stream, &b);
}

/// sparse.pool_bucket: visible pools for rows pos .. pos + rows - 1, a power of two (at least 1024), at most `np_max`.
pub fn poolBucket(pos: usize, rows: usize, np_max: usize, g: Geometry) usize {
    const visible = (pos + rows) / g.pool + 1;
    return @min(np_max, @max(1024, std.math.ceilPowerOfTwoAssert(usize, visible)));
}

/// sparse.score_columns: the pool columns selection reads (the last row's complete pools, at least topk).
pub fn scoreColumns(pos: usize, rows: usize, np_max: usize, g: Geometry) usize {
    return @min(np_max, @max(g.topk, (pos + rows) / g.pool));
}

/// select_tokens' 1/sqrt(heads) head-weight scale (32 heads: 1 / 5.656854249492381, as the Python family writes it).
pub fn headScale(heads: usize) f32 {
    if (heads == 32) return @floatCast(1.0 / 5.656854249492381);
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(heads))));
}

/// The scratch of one selectTokens: scores fp32 [min(rows, 512)][np], and with np past select_list the list rows.
pub const Scratch = struct {
    scores: u64,
    ls: u64,
    li: u64,
    pools: u64,
    at: u64,
    /// The rows of decode select bit words (bitsBytes) `ls` also holds; 0: none, and the wide select runs.
    bits_rows: usize = 0,
};

/// A prompt chunk's indexer rows split across the ranks (TF_GLM_INDEX_SPLIT): a block each, the picks all-gathered.
pub const Split = struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize };

/// Whether rank 0 splits a prompt chunk's indexer rows at `start`: from `from` on (0: never), over two ranks or more.
pub fn splitAt(start: usize, from: usize, world: usize) bool {
    return from > 0 and world > 1 and start >= from;
}

/// sparse.select_tokens: rows' tokens (i32 [rows][g.width()], ascending, -1 padded; on pages their rows); 0 if dense.
pub fn selectTokens(set: glue.Set, ops: cuda.Module, stream: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, t: latent.Table, pos: u64, host_pos: usize, rows: usize, heads: usize, d: usize, g: Geometry, np_max_cap: usize, s: Scratch, tokens: u64, counts: u64) !void {
    return selectTokensProfile(set, ops, stream, qi, wts, w_stride, pk, t, pos, host_pos, rows, heads, d, g, np_max_cap, s, tokens, counts, null, null);
}

/// With `split`, this rank picks for its block of rows only (as one pass would, block by block) and gathers the rest.
pub fn selectTokensProfile(set: glue.Set, ops: cuda.Module, stream: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, t: latent.Table, pos: u64, host_pos: usize, rows: usize, heads: usize, d: usize, g: Geometry, np_max_cap: usize, s: Scratch, tokens: u64, counts: u64, split: ?Split, profile: ?*@import("cuda_prof.zig").Prof) !void {
    const prof = @import("cuda_prof.zig");
    try g.check();
    const np_max = poolBucket(host_pos, rows, np_max_cap, g);
    const b = @min(rows, select_rows);
    const loop: usize = if (rows >= score_loop_from) score_loop else 1;
    const add = try ops.function("tf_glm_add_i32");
    const per = if (split) |x| cdiv(rows, x.world) else rows;
    const lo = if (split) |x| @min(rows, x.rank * per) else 0;
    const hi = @min(rows, lo + per);
    var a: usize = lo;
    while (a < hi) : (a += b) {
        const n = @min(b, hi - a);
        var at = pos;
        if (a > 0) {
            var args: cuda.Args = .{};
            args.add(pos);
            args.add(@as(c_int, @intCast(a)));
            args.add(s.at);
            try cuda_pdl.launch(add, .{ .grid = .{}, .block = .{ .x = 32 } }, stream, &args);
            at = s.at;
        }
        const nb = cdiv(scoreColumns(host_pos + a, n, np_max, g), 64);
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(d), -0.5));
        var sa: cuda.Args = .{};
        inline for (.{ qi + a * heads * d * 2, wts + a * w_stride * 2, @as(i64, @intCast(w_stride)), pk, s.scores, at, @as(i64, @intCast(np_max)), @as(i64, @intCast(nb)), scale, headScale(heads), @as(i64, @intCast(heads)), @as(c_int, @intCast(loop)), t.pt, @as(i64, @intCast(t.ps)), @as(c_int, @intCast(g.pool)), @as(c_int, @intCast(g.rope)), g.inv }) |v| sa.add(v);
        if (d != 128 or heads > 32) return error.IndexShape;
        const quick = if (set.index) |k| if (k.takes(rows, np_max, pk, t.ps)) k else null else null;
        // a decode window's rows: the same scores from glm_index_scores (8 rows a CTA, key tiles staged once)
        const dec = if (set.dscore) |k| if (k.takesDecode(rows, np_max, nb, pk, t.ps)) k else null else null;
        var stamp = prof.span(profile, stream, .index_score);
        if (quick) |k| try k.scores(stream, qi + a * heads * d * 2, wts + a * w_stride * 2, w_stride, pk, s.scores, at, n, np_max, nb, scale, headScale(heads), heads, t.pt, t.ps, g) else if (dec) |k| try k.scoresTiles(stream, qi + a * heads * d * 2, wts + a * w_stride * 2, w_stride, pk, s.scores, at, n, np_max, nb, scale, headScale(heads), heads, t.pt, t.ps, index_mod.decodeLoop(n, nb), g) else try cuda_pdl.launch(set.glue.scores, .{ .grid = .{ .x = @intCast(n), .y = @intCast(cdiv(nb, loop)) }, .block = .{ .x = 128 } }, stream, &sa);
        stamp.end();
        stamp = prof.span(profile, stream, .index_select);
        if (quick) |k| if (k.select) {
            try k.pools(stream, s.scores, s.pools + a * g.topk * 8, at, n, np_max, 0, g);
            stamp.end();
            continue;
        };
        const wide = wideSelect(n, np_max);
        if (wide and set.dsel and n <= s.bits_rows) {
            try decodeSelect(set, stream, s.scores, s.pools + a * g.topk * 8, np_max, at, host_pos + a, n, s.ls, 0, g);
            stamp.end();
            continue;
        }
        var sel: cuda.Args = .{};
        inline for (.{ s.scores, s.pools + a * g.topk * 8, @as(i64, @intCast(np_max)), at, @as(c_int, @intCast(g.topk)), @as(c_int, @intCast(g.pool)) }) |v| sel.add(v);
        try cuda_pdl.launch(if (wide) set.glue.select_rows_wide else set.glue.select_rows, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = if (wide) 1024 else 128 } }, stream, &sel);
        stamp.end();
    }
    const stamp = prof.span(profile, stream, .index_select);
    defer stamp.end();
    if (split) |x| { // in place: each rank's equal block (the prefill buffers' pools hold world rows more), rank order
        const bytes = per * g.topk * 8;
        try x.lib.check(x.lib.api.ncclAllGather(s.pools + x.rank * bytes, s.pools, bytes, .u8, x.comm, stream.handle), "ncclAllGather");
    }
    const assemble = try ops.function("tf_glm_dsa_tokens");
    var args: cuda.Args = .{};
    args.add(s.pools);
    args.add(pos);
    args.add(@as(c_int, @intCast(g.topk)));
    args.add(tokens);
    args.add(counts);
    args.add(t.pt);
    args.add(@as(i64, @intCast(t.page_rows)));
    args.add(@as(c_int, @intCast(g.pool)));
    try cuda_pdl.launch(assemble, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
}

test "sparse.py's sizing rules" {
    try std.testing.expectEqual(@as(usize, 1024), poolBucket(100, 1, 1 << 20, .{}));
    try std.testing.expectEqual(@as(usize, 2048), poolBucket(5000, 1, 1 << 20, .{})); // 1251 visible pools
    try std.testing.expectEqual(@as(usize, 512), scoreColumns(100, 1, 1024, .{}));
    try std.testing.expectEqual(@as(usize, 1250), scoreColumns(5000, 1, 2048, .{}));
    try std.testing.expectEqual(@as(usize, 2051), width);
    try std.testing.expectEqual(width, (Geometry{}).width());
    try std.testing.expectEqual(@as(f32, @floatCast(1.0 / 5.656854249492381)), headScale(32));
}

test "GLM index: a decode select's scan covers the last row's pools in spans, and none when the merge takes rows whole" {
    // 864K: the last row sees 216,064 pools of a 216,080 bucket
    try std.testing.expectEqual(@as(usize, 53), dselScanCtas(864256 - 16, 16, 216080, .{}));
    try std.testing.expectEqual(@as(usize, 8), dselScanCtas(131072 - 1, 1, 65536, .{})); // 32,768 visible of 65,536
    try std.testing.expectEqual(@as(usize, 0), dselScanCtas(4 * 4096 - 1, 1, 8192, .{})); // 4,096 pools: one merge CTA
    try std.testing.expectEqual(@as(usize, 2), dselScanCtas(4 * 4096 + 3, 1, 8192, .{}));
    try std.testing.expectEqual(@as(usize, 0), dselScanCtas(100, 1, 4096, .{})); // a few visible: K pools
    try std.testing.expect(dsel_span % (32 * (dsel_scan_threads / 32)) == 0); // whole words for every warp
    try std.testing.expectEqual(@as(usize, 16 * 2 * 6753 * 4), bitsBytes(16, 216080));
}

test "rank 0 splits a prompt chunk's indexer rows from TF_GLM_INDEX_SPLIT's position on, over two ranks or more" {
    try std.testing.expect(!splitAt(65536, 0, 3)); // 0: never
    try std.testing.expect(splitAt(65536, 65536, 3));
    try std.testing.expect(!splitAt(63488, 65536, 3));
    try std.testing.expect(!splitAt(65536, 65536, 1)); // one rank holds every row
}

test "GLM index geometry: full GLM-5.3's pools of 1 score every key, keep 2,048 and list them with no tail" {
    const full: Geometry = .{ .pool = 1, .topk = 2048, .rope = 64, .inv = 0x1000 };
    try full.check();
    try std.testing.expectEqual(@as(usize, 2048), full.width());
    try std.testing.expectEqual(@as(usize, 8192), poolBucket(5000, 1, 1 << 20, full)); // 5,002 visible keys
    try std.testing.expectEqual(@as(usize, 2048), scoreColumns(100, 1, 1 << 20, full));
    try std.testing.expectEqual(@as(usize, 5001), scoreColumns(5000, 1, 8192, full));
    try std.testing.expectEqual(@as(usize, 2), dselScanCtas(4097, 1, 8192, full)); // 4,098 keys past one merge CTA
    try std.testing.expectEqual(@as(usize, 0), dselScanCtas(4095, 1, 8192, full));
    try (Geometry{}).check();
    for ([_]Geometry{ .{ .pool = 2 }, .{ .rope = 64 }, .{ .rope = 62, .inv = 0x1000 }, .{ .inv = 0x1000 }, .{ .topk = 0 } }) |g|
        try std.testing.expectError(error.IndexShape, g.check());
}

test "GLM index geometry: indexUpdate passes GLM-5.3-Flash's rope 0 and pools of 4, full GLM-5.3's rope 64 and pools of 1" {
    const R = struct {
        var n: usize = 0;
        var grids: [2]c_uint = undefined;
        var last: [2]c_int = undefined; // index_write's rope, pool_keys' pool
        fn launch(_: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
            grids[n] = gx;
            last[n] = @as(*const c_int, @ptrCast(@alignCast(args.?[if (n == 0) 9 else 8].?))).*;
            n += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = R.launch;
    d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(R.launch);
    var ks: glue.Kernels = undefined;
    ks.index_write = .{ .d = &d, .handle = @ptrFromInt(1) };
    ks.pool_keys = .{ .d = &d, .handle = @ptrFromInt(2) };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]Geometry{ .{}, .{ .pool = 1, .topk = 2048, .rope = 64, .inv = 0x1000 } }) |g| {
        R.n = 0;
        try indexUpdate(.{ .glue = &ks }, stream, 0x100, 160, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800, .{}, 0x900, 64, g);
        try std.testing.expectEqual(@as(usize, 2), R.n);
        try std.testing.expectEqual([2]c_uint{ 64, @intCast(64 / g.pool + 2) }, R.grids);
        try std.testing.expectEqual([2]c_int{ @intCast(g.rope), @intCast(g.pool) }, R.last);
    }
    try std.testing.expectError(error.IndexShape, indexUpdate(.{ .glue = &ks }, stream, 0, 160, 0, 0, 0, 0, 0, 0, 0, .{}, 0, 64, .{ .pool = 2 }));
}
