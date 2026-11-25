//! CUDA launchers for Hy4-flash compressed/paged sparse indexer: RoPE table, 4:1 pooling, scoring, and top-511 select.
const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Function = cuda.Function;
const Args = cuda.Args;

pub const heads: usize = 16;
pub const dim: usize = 128;
pub const ratio: usize = 4;
pub const groups: usize = 512;
pub const rope: usize = 64;
pub const rope_pairs: usize = rope / 2;
pub const tokens_per_row: usize = groups * ratio;
pub const small_max_jobs: usize = 48;
pub const select_cap: usize = 4096;
pub const select_scan_span: usize = 2048;
pub const pool_1cta_smem: u32 = (16 * 4 * dim + 2 * 64 * 32) * 4;
pub const pool_persist_smem: u32 = (64 * dim + 2 * 128 * 16 + 2 * 32) * 4;

/// Device pointers for 4:1 gated group pooling weights and epsilon.
pub const PoolWeights = struct {
    values: u64,
    gates: u64,
    ape: u64,
    key_norm: u64,
    eps: f32 = 1e-5,
};

pub const Kernels = struct {
    mod: cuda.Module,
    rope_table: Function,
    pool_1cta: Function,
    pool_small: Function,
    pool_persist: Function,
    q_mrope: Function,
    score_decode: Function,
    score_decode_rope: Function,
    score_prefill: Function,
    select_1cta: Function,
    select_scan: Function,
    select_merge: Function,
    pool_1cta_paged: Function,
    pool_small_paged: Function,
    pool_persist_paged: Function,
    score_decode_paged: Function,
    score_decode_rope_paged: Function,
    score_prefill_paged: Function,
    clusters: bool,
    pdl: bool = false,
    sms: usize,
    occ_pre: usize,
    persist_clusters: usize,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const cap = try d.currentCapability();
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_index_paged);
        errdefer mod.unload();
        var k: Kernels = .{
            .mod = mod,
            .rope_table = try mod.function("tf_hy4_idx_rope_table"),
            .pool_1cta = try mod.function("tf_hy4_idx_pool_1cta"),
            .pool_small = try mod.function("tf_hy4_idx_pool_small"),
            .pool_persist = try mod.function("tf_hy4_idx_pool_persist"),
            .q_mrope = try mod.function("tf_hy4_idx_q_mrope"),
            .score_decode = try mod.function("tf_hy4_idx_score_decode"),
            .score_decode_rope = try mod.function("tf_hy4_idx_score_decode_rope"),
            .score_prefill = try mod.function("tf_hy4_idx_score_prefill"),
            .select_1cta = try mod.function("tf_hy4_idx_select_1cta"),
            .select_scan = try mod.function("tf_hy4_idx_select_scan"),
            .select_merge = try mod.function("tf_hy4_idx_select_merge"),
            .pool_1cta_paged = try mod.function("tf_hy4_idx_pool_1cta_paged"),
            .pool_small_paged = try mod.function("tf_hy4_idx_pool_small_paged"),
            .pool_persist_paged = try mod.function("tf_hy4_idx_pool_persist_paged"),
            .score_decode_paged = try mod.function("tf_hy4_idx_score_decode_paged"),
            .score_decode_rope_paged = try mod.function("tf_hy4_idx_score_decode_rope_paged"),
            .score_prefill_paged = try mod.function("tf_hy4_idx_score_prefill_paged"),
            .clusters = cap >= 120,
            .pdl = cap >= 90,
            .sms = 1,
            .occ_pre = 1,
            .persist_clusters = 1,
        };
        try k.pool_1cta.allowDynamicShared(pool_1cta_smem);
        if (k.clusters) try k.pool_persist.allowDynamicShared(pool_persist_smem);
        try k.pool_1cta_paged.allowDynamicShared(pool_1cta_smem);
        if (k.clusters) try k.pool_persist_paged.allowDynamicShared(pool_persist_smem);
        var dev: cuda.abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
        var sm_count: c_int = 1;
        try d.check(d.api.cuDeviceGetAttribute(&sm_count, .multiprocessor_count, dev), "cuDeviceGetAttribute");
        k.sms = @intCast(@max(sm_count, 1));
        k.occ_pre = @max(1, try k.score_prefill.occupancy(256, 0));
        if (k.clusters) {
            const occ_p = try k.pool_persist.occupancy(256, pool_persist_smem);
            k.persist_clusters = @max(1, (occ_p * k.sms) / 4);
        }
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// Newly completed 4-token groups for rows at `[first, first + rows)`: exact partition `[first/4 .. (first+rows)/4)`.
pub fn completedGroups(first: usize, rows: usize) struct { lo: usize, len: usize } {
    const lo = first / ratio;
    const hi = (first + rows) / ratio;
    return .{ .lo = lo, .len = hi - lo };
}

/// Selected row slice `[from, from + count)` starting at `list_from` (default `pos >= recall_from`).
pub fn selectRange(first: usize, rows: usize, list_from: ?usize, recall_from: usize) struct { from: usize, count: usize } {
    const start = if (list_from) |lf| @min(lf, rows) else if (first + rows > recall_from)
        (if (first >= recall_from) 0 else recall_from - first)
    else
        rows;
    return .{ .from = start, .count = rows - start };
}

pub fn scoreSplit(nb: usize, span_own: usize, sms: usize, occ_pre: usize) u32 {
    const ns = @max(@as(usize, 1), (span_own + 15) / 16);
    if (nb == 1) return @intCast(@max(@as(usize, 1), @min(ns, @max(@as(usize, 48), @min(@as(usize, 288), ns / 8)))));
    return @intCast(@max(@as(usize, 1), @min((4 * sms * occ_pre + nb - 1) / nb, @max(@as(usize, 1), ns / 4))));
}

pub fn ropeTable(f: Function, s: Stream, cs: u64, first: usize, npos: usize, freqs: u64, pos3: u64, ld3: usize, sec_h: usize, sec_w: usize, round_bf16: bool) !void {
    if (npos == 0) return;
    var a: Args = .{};
    a.add(cs);
    a.add(cint(first));
    a.add(cint(npos));
    a.add(freqs);
    a.add(pos3);
    inline for (.{ ld3, sec_h, sec_w, @intFromBool(round_bf16) }) |v| a.add(cint(v));
    try cuda.launch.launch(f, .{ .grid = .{ .x = cdiv(npos * rope_pairs, 256) }, .block = .{ .x = 256 } }, s, &a);
}

pub fn poolWith(f: Function, s: Stream, cluster: ?u32, smem: u32, grid_x: u32, block_x: u32, js: u64, jg: u64, g0: usize, njobs: usize, tkc: u64, ctx: usize, w: PoolWeights, rt: u64, rt_stream_stride: usize, rt_group_stride: usize, gkc: u64, pt: u64, pt_stride: usize, g_base: usize) !void {
    if (njobs == 0) return;
    var a: Args = .{};
    inline for (.{ js, jg, cint(g0), cint(njobs), tkc, cint(ctx) }) |v| a.add(v);
    inline for (.{ w.values, w.gates, w.ape, w.key_norm, rt }) |v| a.add(v);
    inline for (.{ cint(rt_stream_stride), cint(rt_group_stride), w.eps, gkc }) |v| a.add(v);
    if (pt != 0) inline for (.{ pt, cint(pt_stride), cint(g_base) }) |v| a.add(v);
    try cuda.launch.launch(f, .{
        .grid = .{ .x = grid_x },
        .block = .{ .x = block_x },
        .shared = smem,
        .cluster = if (cluster) |cl| .{ .x = cl, .y = 1, .z = 1 } else null,
    }, s, &a);
}

pub fn poolEx(k: *const Kernels, s: Stream, w: PoolWeights, tkc: u64, gkc: u64, rt: u64, ctx: usize, g0: usize, njobs: usize, js: u64, jg: u64, rt_stream_stride: usize, rt_group_stride: usize, pt: u64, pt_stride: usize, g_base: usize) !void {
    if (njobs == 0) return;
    const pg = pt != 0;
    if (!k.clusters) {
        return poolWith(if (pg) k.pool_1cta_paged else k.pool_1cta, s, null, pool_1cta_smem, cdiv(njobs, 16), 128, js, jg, g0, njobs, tkc, ctx, w, rt, rt_stream_stride, rt_group_stride, gkc, pt, pt_stride, g_base);
    }
    if (njobs <= small_max_jobs) {
        return poolWith(if (pg) k.pool_small_paged else k.pool_small, s, 4, 0, @intCast(4 * njobs), 128, js, jg, g0, njobs, tkc, ctx, w, rt, rt_stream_stride, rt_group_stride, gkc, pt, pt_stride, g_base);
    }
    const nblk = (njobs + 31) / 32;
    const ncl = @min(nblk, k.persist_clusters);
    return poolWith(if (pg) k.pool_persist_paged else k.pool_persist, s, 4, pool_persist_smem, @intCast(4 * ncl), 256, js, jg, g0, njobs, tkc, ctx, w, rt, rt_stream_stride, rt_group_stride, gkc, pt, pt_stride, g_base);
}

pub fn qMrope(f: Function, s: Stream, sel: u64, n_sel: usize, iq: u64, row_pos: u64, pos3: u64, r_stride: usize, cs: u64, qi: u64) !void {
    if (n_sel == 0) return;
    if (qi == 0 or iq == 0 or cs == 0) return error.ScratchShape;
    var a: Args = .{};
    a.add(sel);
    a.add(cint(n_sel));
    a.add(iq);
    a.add(row_pos);
    a.add(pos3);
    a.add(cint(r_stride));
    a.add(cs);
    a.add(qi);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n_sel) }, .block = .{ .x = 256 } }, s, &a);
}

pub const ScoreCall = struct {
    sel: u64 = 0,
    n_sel: usize,
    iq_or_qi: u64,
    hw: u64,
    row_stream: u64 = 0,
    row_pos: u64,
    gkc: u64,
    gpool: usize,
    sc: u64,
    gcap: usize,
    span_own: usize,
    cs: u64,
    pos3: u64 = 0,
    r_stride: usize,
    qi_scratch: u64,
    page_table: u64 = 0,
    pt_stride: usize = 0,
};

pub fn score(k: *const Kernels, s: Stream, c: ScoreCall) !void {
    if (c.n_sel == 0) return;
    const fused = c.n_sel <= 4;
    const rb: usize = if (fused) 4 else 8;
    const nb = cdiv(c.n_sel, rb);
    const split = scoreSplit(nb, c.span_own, k.sms, k.occ_pre);
    if (fused) {
        var a: Args = .{};
        a.add(c.sel);
        a.add(cint(c.n_sel));
        a.add(c.iq_or_qi);
        a.add(c.hw);
        a.add(c.row_stream);
        a.add(c.row_pos);
        a.add(c.gkc);
        a.add(cint(c.gpool));
        a.add(c.sc);
        a.add(cint(c.gcap));
        a.add(c.cs);
        a.add(c.pos3);
        a.add(cint(c.r_stride));
        if (c.page_table != 0) {
            a.add(c.page_table);
            a.add(cint(c.pt_stride));
        }
        try cuda.launch.launch(if (c.page_table != 0) k.score_decode_rope_paged else k.score_decode_rope, .{ .grid = .{ .x = split, .y = nb }, .block = .{ .x = 128 } }, s, &a);
    } else {
        try qMrope(k.q_mrope, s, c.sel, c.n_sel, c.iq_or_qi, c.row_pos, c.pos3, c.r_stride, c.cs, c.qi_scratch);
        var a: Args = .{};
        a.add(c.sel);
        a.add(cint(c.n_sel));
        a.add(c.qi_scratch);
        a.add(c.hw);
        a.add(c.row_stream);
        a.add(c.row_pos);
        a.add(c.gkc);
        a.add(cint(c.gpool));
        a.add(c.sc);
        a.add(cint(c.gcap));
        if (c.page_table != 0) {
            a.add(c.page_table);
            a.add(cint(c.pt_stride));
        }
        try cuda.launch.launch(if (c.page_table != 0) k.score_prefill_paged else k.score_prefill, .{ .grid = .{ .x = split, .y = nb }, .block = .{ .x = 256 }, .pdl = k.pdl }, s, &a);
    }
}

pub fn select(k: *const Kernels, s: Stream, sel: u64, n_sel: usize, row_pos: u64, sc: u64, gcap: usize, bits: u64, groups_out: u64, tokens_out: u64, max_own: usize) !void {
    if (n_sel == 0) return;
    if (max_own <= select_cap) {
        var a: Args = .{};
        inline for (.{ sel, row_pos, sc, cint(gcap), groups_out, tokens_out }) |v| a.add(v);
        try cuda.launch.launch(k.select_1cta, .{ .grid = .{ .x = @intCast(n_sel) }, .block = .{ .x = 1024 }, .pdl = k.pdl }, s, &a);
    } else {
        const words = (gcap + 31) / 32;
        const scan_blocks = cdiv(max_own, select_scan_span);
        if (scan_blocks > 0) {
            var sa: Args = .{};
            inline for (.{ sel, row_pos, sc, cint(gcap), bits, cint(words) }) |v| sa.add(v);
            try cuda.launch.launch(k.select_scan, .{ .grid = .{ .x = scan_blocks, .y = @intCast(n_sel) }, .block = .{ .x = 256 }, .pdl = k.pdl }, s, &sa);
        }
        var ma: Args = .{};
        inline for (.{ sel, row_pos, sc, cint(gcap), bits, cint(words), groups_out, tokens_out, @as(u64, 0) }) |v| ma.add(v);
        try cuda.launch.launch(k.select_merge, .{ .grid = .{ .x = @intCast(n_sel) }, .block = .{ .x = 1024 }, .pdl = k.pdl }, s, &ma);
    }
}

test "completedGroups partitions any chunk and decode cut across 4-token boundaries" {
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 0, .len = 0 }, completedGroups(0, 3));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 0, .len = 1 }, completedGroups(0, 4));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 0, .len = 512 }, completedGroups(0, 2051));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 512, .len = 127 }, completedGroups(2051, 506));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 639, .len = 1 }, completedGroups(2559, 1));
    try std.testing.expectEqual(@TypeOf(completedGroups(0, 0)){ .lo = 0, .len = 640 }, completedGroups(0, 2560));
}

test "selectRange selects only positions at or past recall_from across cuts" {
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 2048, .count = 0 }, selectRange(0, 2048, null, 2048));
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 2048, .count = 3 }, selectRange(0, 2051, null, 2048));
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 0, .count = 506 }, selectRange(2051, 506, null, 2048));
    try std.testing.expectEqual(@TypeOf(selectRange(0, 0, null, 2048)){ .from = 2048, .count = 512 }, selectRange(0, 2560, 2048, 2048));
}

test "index_paged_cuda launchers compile on host without embedded fatbins" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(PoolWeights);
    std.testing.refAllDecls(ScoreCall);
}
