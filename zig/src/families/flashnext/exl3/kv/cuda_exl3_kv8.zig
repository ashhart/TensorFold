//! The int8 KV kernels of K10 and the pass's attention block over an int8 cache.

const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const hub = @import("cuda_exl3_kv.zig");

const group: usize = 32;
const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;

const tile_heads: u32 = 16;
const tile_keys: u32 = 32;

/// Dynamic shared bytes of fn_kv8_chunks with `nb` raw buffers (see the .cu: q rows, one bf16 tile pair, p, raw codes).
pub fn chunkSmem(nb: u32) u32 {
    const row: u32 = @intCast((head_dim + 8) * 2);
    return tile_heads * row + 2 * tile_keys * row + tile_heads * (tile_keys + 8) * 2 + nb * 2 * tile_keys * @as(u32, @intCast(head_dim)) + nb * 2 * tile_keys * @as(u32, @intCast(head_dim / group)) * 2;
}

pub const Kv8 = struct {
    module: cuda.Module,
    prep: cuda.Function,
    chunks: cuda.Function,
    chunks_ahead: cuda.Function,
    merge: cuda.Function,
    pack: cuda.Function,
    gather: cuda.Function,
    roundtrip: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kv8 {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kv8 = undefined;
        k.module = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_kv);
        errdefer k.module.unload();
        k.prep = try k.module.function("fn_kv8_prep");
        k.chunks = try k.module.function("fn_kv8_chunks");
        k.chunks_ahead = try k.module.function("fn_kv8_chunks_ahead");
        k.merge = try k.module.function("fn_kv8_merge");
        k.pack = try k.module.function("fn_kv8_pack");
        k.gather = try k.module.function("fn_kv_gather");
        k.roundtrip = try k.module.function("fn_kv_roundtrip");
        try k.chunks.allowDynamicShared(chunkSmem(1));
        try k.chunks_ahead.allowDynamicShared(chunkSmem(2));
        return k;
    }

    pub fn deinit(k: *Kv8) void {
        k.module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: u32, shared: u32, a: *cuda.Args) !void {
        if (grid[0] == 0 or grid[1] == 0 or grid[2] == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared }, s, a);
    }

    fn int(v: usize) c_int {
        return @intCast(v);
    }

    /// A layer's int8 cache: codes [cap][heads][256] and scales [cap][heads][8] for K and for V.
    pub const Cache = struct { k8: u64, ks: u64, v8: u64, vs: u64 };

    pub fn cacheBytes(cap: usize, heads: usize) struct { codes: usize, scales: usize } {
        return .{ .codes = cap * heads * head_dim, .scales = cap * heads * (head_dim / group) * 2 };
    }

    /// fn_attn_prep with the K/V rows written as codes and scales; `nq`/`nkv` the rank's heads (proj row `pw` wide).
    pub fn prepRows(k: *const Kv8, s: cuda.Stream, proj: u64, pos: u64, q_scale: u64, k_scale: u64, i_scale: u64, inv: u64, q: u64, c: Cache, iq: u64, ikc: u64, eps: f32, rows: usize, pw: usize, nq: usize, nkv: usize, rot: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ proj, pos, q_scale, k_scale, i_scale, inv, q, c.k8, c.ks, c.v8, c.vs, iq, ikc }) |v| a.add(v);
        a.add(eps);
        for ([_]usize{ pw, nq, nkv, wt.index_heads, wt.index_dim, wt.half, @intFromBool(rot) }) |v| a.add(int(v));
        try go(k.prep, s, .{ rows, nq + nkv + wt.index_heads + 1, 1 }, @intCast(head_dim), 0, &a);
    }

    /// Upstream's attnChunks over an int8 cache (same grid, same 512-key chunks, `nq` query heads over `nkv` KV heads).
    pub fn attnChunks(k: *const Kv8, s: cuda.Stream, q: u64, c: Cache, pos: u64, po: u64, pm: u64, pl: u64, ids: u64, nk: u64, sparse: u64, rows: usize, nq: usize, nkv: usize, chunks: usize, qsa: bool) !void {
        const g = nq / nkv;
        if (g > tile_heads) return error.RowShape;
        var a: cuda.Args = .{};
        for ([_]u64{ q, c.k8, c.ks, c.v8, c.vs, pos, po, pm, pl, ids, nk, sparse }) |v| a.add(v);
        for ([_]usize{ nq, nkv, g, 512, if (qsa) wt.nch else wt.dense_nch }) |v| a.add(int(v));
        a.add(@as(f32, 0.0625));
        for ([_]usize{ wt.index_idw, @intFromBool(qsa) }) |v| a.add(int(v));
        const ahead = rows <= hub.ahead_rows;
        try go(if (ahead) k.chunks_ahead else k.chunks, s, .{ rows, nkv, chunks }, 128, chunkSmem(if (ahead) 2 else 1), &a);
    }

    pub fn attnMerge(k: *const Kv8, s: cuda.Stream, po: u64, pm: u64, pl: u64, pos: u64, out: u64, nk: u64, sparse: u64, rows: usize, nq: usize, nkv: usize, qsa: bool, rot: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ po, pm, pl, pos, out, nk, sparse }) |v| a.add(v);
        for ([_]usize{ nq, nkv, nq / nkv, 512, if (qsa) wt.nch else wt.dense_nch, @intFromBool(qsa), @intFromBool(rot) }) |v| a.add(int(v));
        try go(k.merge, s, .{ rows, nq, 1 }, @intCast(head_dim), 0, &a);
    }

    /// `rows` rows of 256 bf16 (a (pos, head) row each) into codes and scales.
    pub fn packRows(k: *const Kv8, s: cuda.Stream, x: u64, codes: u64, scales: u64, rows: usize, rot: bool) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(codes);
        a.add(scales);
        a.add(@as(i64, @intCast(rows)));
        a.add(int(@intFromBool(rot)));
        try go(k.pack, s, .{ rows, 1, 1 }, @intCast(head_dim), 0, &a);
    }

    /// Test support: `rows` bf16 rows of 256 replaced in place by an int8 (bits 8) or int4 (bits 4) cache's values.
    pub fn roundTrip(k: *const Kv8, s: cuda.Stream, x: u64, rows: usize, rot: bool, bits: u32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(rows)));
        a.add(int(@intFromBool(rot)));
        a.add(@as(c_int, @intCast(bits)));
        try go(k.roundtrip, s, .{ rows, 1, 1 }, @intCast(head_dim), 0, &a);
    }

    /// `n` rows of `row_bytes` from `src` at `ids` into `dst` (a warp a row, 8 rows a block).
    pub fn gatherRows(k: *const Kv8, s: cuda.Stream, src: u64, dst: u64, ids: u64, n: usize, row_bytes: usize) !void {
        if (row_bytes % 16 != 0) return error.RowShape;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(ids);
        a.add(int(n));
        a.add(int(row_bytes / 16));
        try go(k.gather, s, .{ (n + 7) / 8, 1, 1 }, 256, 0, &a);
    }
};

// -- the pass's attention block over an int8 cache ------------------------------------------------------------------

/// cuda_batch.zig Pass.attnBlock with the layer's K/V in `c` (int8 codes + scales) instead of seq.kc/vc: the same
pub fn attnBlock(p: anytype, k8: *const Kv8, lw: anytype, ai: usize, c: Kv8.Cache, rot: bool) !void {
    const s = p.s;
    const n = p.rows;
    if (p.lanes.len > 0) return error.LanesUnsupported;
    const pa = s.proj;
    const att_rows: usize = 256;
    const ratio = wt.index_ratio;
    const top = wt.index_top;
    const idw = wt.index_idw;
    const proj_n = wt.proj_n;
    _ = try p.lane(s.mixed, s.xsm, lw.proj, pa, s.dummy, n, 1);
    const ikc = p.seq.ikc[ai].ptr;
    const pooled = p.seq.pooled[ai].ptr;
    try k8.prepRows(p.stream, pa, s.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, s.q, c, s.iq, ikc, 1e-6, n, proj_n, wt.q_heads, kv_heads, rot);
    try p.tri.qsaPool(ikc, pooled, s.pos, lw.ik_scale.ptr, s.inv, 1e-6, n, .{});
    const nb = (cdiv(p.seq.cap, ratio) + 15) / 16 * 16;
    var r0: usize = 0;
    var b: usize = 0;
    while (r0 < n) : ({
        r0 += att_rows;
        b += 1;
    }) {
        const m = @min(att_rows, n - r0);
        const ends = p.pos0 + r0 + m;
        if (ends > p.seq.cap) return error.UnexpectedTensor;
        const pos = s.pos + b * 16;
        const iq = s.iq + r0 * wt.index_heads * wt.index_dim * 2;
        if (ends / ratio > top) try p.tri.qsaScores(iq, pooled, pos, p.seq.scores, nb, m, @min(nb, cdiv(ends, ratio)), .{});
        var a: cuda.Args = .{};
        a.add(p.seq.scores);
        a.add(pos);
        a.add(s.qids);
        a.add(s.qnk);
        a.add(s.qspr);
        for ([_]usize{ nb, idw, top, ratio }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(p.k.select, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1024 } }, p.stream, &a);
        const chunks = cdiv(@min(ends, wt.dense_keys), 512);
        const q = s.q + r0 * wt.q_heads * head_dim * 2;
        try k8.attnChunks(p.stream, q, c, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, m, wt.q_heads, kv_heads, chunks, true);
        try k8.attnMerge(p.stream, s.po, s.pm, s.pl, pos, s.ao + r0 * wt.out_k * 2, s.qnk, s.qspr, m, wt.q_heads, kv_heads, true, rot);
    }
    try p.tri.attnGate(s.ao, pa, s.gated, s.gxs_attn, n, proj_n, wt.q_heads, head_dim);
    _ = try p.lane(s.gated, s.gxs_attn, lw.out, s.dummy, s.part, n, wt.out_sk);
}

pub fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}
