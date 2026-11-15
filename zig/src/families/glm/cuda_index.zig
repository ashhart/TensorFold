//! A prompt chunk's DSA indexer in CUDA (glm_index.cu): the bytes of tf_glm_scores and tf_glm_select_rows.

const std = @import("std");
const cuda = @import("cuda");

const sparse = @import("cuda_sparse.zig");

pub const topk = sparse.topk_pools;
/// glm_index_scores: rows a CTA (a warp each), 64-pool tiles a CTA, threads a CTA.
pub const score_rows = 8;
pub const score_tiles = 16;
pub const threads = score_rows * 32;
/// glm_index_pools' block: glm_index::T.
pub const pool_threads = 256;
/// The rows a launch must have to take it: where tf_glm_scores' `loop` tiling starts.
pub const rows_from = sparse.score_loop_from;
pub const cap_max = 4096;

/// TF_GLM_INDEX_CUDA: unset, blank or 1 (both kernels), scores (only the scores, to bisect), 0 (neither).
pub const Mode = enum {
    off,
    both,
    scores,

    pub fn parse(text: ?[]const u8) !Mode {
        const t = text orelse return .both;
        if (std.mem.trim(u8, t, " \t\r\n").len == 0 or std.mem.eql(u8, t, "1")) return .both;
        if (std.mem.eql(u8, t, "0")) return .off;
        if (std.mem.eql(u8, t, "scores")) return .scores;
        return error.BadIndexMode;
    }

    pub fn read() !Mode {
        const v = std.c.getenv("TF_GLM_INDEX_CUDA") orelse return .both;
        return parse(std.mem.span(v));
    }
};

/// glm_index_scores' grid: rows in CTAs of 8, the `nb` pool blocks in runs of score_tiles.
pub fn scoresGrid(rows: usize, nb: usize) [2]u32 {
    return .{ @intCast(cdiv(rows, score_rows)), @intCast(@max(1, cdiv(nb, score_tiles))) };
}

/// The candidates a row of the pools kernel keeps and the floor's target (speed only; the pools are the same).
pub const Tile = struct { cap: usize, target: usize };

pub fn tile(np: usize) Tile {
    if (np <= 12288) return .{ .cap = 1024, .target = 724 };
    if (np <= 32768) return .{ .cap = 2048, .target = 1024 };
    return .{ .cap = cap_max, .target = 2048 };
}

pub const Kernels = struct {
    module: cuda.Module,
    scores_fn: cuda.Function,
    pools_fn: cuda.Function,
    /// Whether the pools kernel replaces tf_glm_select_rows (Mode.both).
    select: bool,

    /// `image`: cuda.kernels.glm_index.
    pub fn load(d: *const cuda.Driver, image: []const u8, mode: Mode) !Kernels {
        if (image.len == 0) return error.NoIndexKernels;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .scores_fn = try m.function("tf_glm_index_scores"), .pools_fn = try m.function("tf_glm_index_pools"), .select = mode == .both };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }

    /// Whether `rows` rows over `np` pools with this key cache layout go through the kernels.
    pub fn takes(k: *const Kernels, rows: usize, np: usize, pk: u64, ps: u64) bool {
        _ = k;
        return rows >= rows_from and np >= topk and pk % 16 == 0 and ps % 16 == 0;
    }

    /// tf_glm_scores' launch for `rows` rows at `at` into out [rows][np]; the first `nb` 64-pool blocks are written.
    pub fn scores(k: *const Kernels, s: cuda.Stream, qi: u64, wts: u64, w_stride: usize, pk: u64, out: u64, at: u64, rows: usize, np: usize, nb: usize, scale: f32, wscale: f32, heads: usize, pt: u64, ps: u64) !void {
        const g = scoresGrid(rows, nb);
        var a: cuda.Args = .{};
        inline for (.{ qi, wts, @as(i64, @intCast(w_stride)), pk, out, at, @as(i64, @intCast(rows)), @as(i64, @intCast(np)), @as(i64, @intCast(nb)), scale, wscale, @as(i64, @intCast(heads)), @as(c_int, score_tiles), pt, @as(i64, @intCast(ps)) }) |v| a.add(v);
        try cuda.launch.launch(k.scores_fn, .{ .grid = .{ .x = g[0], .y = g[1] }, .block = .{ .x = threads } }, s, &a);
    }

    /// tf_glm_select_rows' launch: each row's 512 best pools of scores [rows][np] ascending into out [rows][512] int64.
    pub fn pools(k: *const Kernels, s: cuda.Stream, scores_ptr: u64, out: u64, at: u64, rows: usize, np: usize, path: u64) !void {
        const t = tile(np);
        var a: cuda.Args = .{};
        inline for (.{ scores_ptr, out, @as(i64, @intCast(np)), at, @as(c_int, topk), @as(c_int, @intCast(t.cap)), @as(c_int, @intCast(t.target)), path }) |v| a.add(v);
        try cuda.launch.launch(k.pools_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = pool_threads } }, s, &a);
    }
};

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

test "GLM index: TF_GLM_INDEX_CUDA is on unless 0, scores bisects, and a typo is refused" {
    try std.testing.expectEqual(Mode.both, try Mode.parse(null));
    try std.testing.expectEqual(Mode.both, try Mode.parse(""));
    try std.testing.expectEqual(Mode.both, try Mode.parse(" \t"));
    try std.testing.expectEqual(Mode.both, try Mode.parse("1"));
    try std.testing.expectEqual(Mode.off, try Mode.parse("0"));
    try std.testing.expectEqual(Mode.scores, try Mode.parse("scores"));
    try std.testing.expectError(error.BadIndexMode, Mode.parse("on"));
    try std.testing.expectError(error.BadIndexMode, Mode.parse("off"));
}

test "GLM index: the scores grid covers 8 rows a CTA and 16 pool blocks a CTA" {
    // a 512-row chunk over 147,456 pools: 2,304 blocks
    try std.testing.expectEqual([2]u32{ 64, 144 }, scoresGrid(512, 2304));
    try std.testing.expectEqual([2]u32{ 9, 1 }, scoresGrid(65, 16));
    try std.testing.expectEqual([2]u32{ 8, 2 }, scoresGrid(64, 17));
    try std.testing.expectEqual([2]u32{ 1, 1 }, scoresGrid(1, 0));
}

test "GLM index: the candidate cap follows the pool count and stays in shared memory" {
    try std.testing.expectEqual(Tile{ .cap = 1024, .target = 724 }, tile(12288));
    try std.testing.expectEqual(Tile{ .cap = 2048, .target = 1024 }, tile(12289));
    try std.testing.expectEqual(Tile{ .cap = 2048, .target = 1024 }, tile(32768));
    try std.testing.expectEqual(Tile{ .cap = 4096, .target = 2048 }, tile(32769));
    try std.testing.expect(tile(1 << 30).cap <= cap_max);
}

test "GLM index: only prompt-sized launches on aligned key caches take the kernels" {
    const k: Kernels = undefined;
    try std.testing.expect(k.takes(rows_from, 1024, 0x1000, 32768));
    try std.testing.expect(!k.takes(rows_from - 1, 1024, 0x1000, 32768));
    try std.testing.expect(!k.takes(512, 1024, 0x1008, 32768));
    try std.testing.expect(!k.takes(512, 1024, 0x1000, 32760));
    try std.testing.expect(!k.takes(512, topk - 1, 0x1000, 32768));
}

/// A driver that records each launched function's handle, in order (selectTokens below, without a device).
const Recorded = struct {
    var n: usize = 0;
    var ids: [16]usize = undefined;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        if (n < ids.len) ids[n] = @intFromPtr(f);
        n += 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        out.* = @ptrFromInt(if (std.mem.eql(u8, std.mem.span(name), "tf_glm_dsa_tokens")) @as(usize, 41) else 40);
        return 0;
    }
};

test "GLM index: prompt chunks take the kernels; a decode row, an unaligned key cache and a small pool cap keep tf_glm_scores" {
    const glue = @import("cuda_glue.zig");
    const latent = @import("cuda_latent.zig");
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Recorded.launch;
    d.api.cuModuleGetFunction = Recorded.function;
    var ks: glue.Kernels = undefined; // 20 tf_glm_scores, 21 tf_glm_select_rows, 22 its wide form
    ks.scores = .{ .d = &d, .handle = @ptrFromInt(20) };
    ks.select_rows = .{ .d = &d, .handle = @ptrFromInt(21) };
    ks.select_rows_wide = .{ .d = &d, .handle = @ptrFromInt(22) };
    const ops: cuda.Module = .{ .d = &d, .handle = null }; // 40 tf_glm_add_i32, 41 tf_glm_dsa_tokens
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const s: sparse.Scratch = .{ .scores = 0x10000, .ls = 0, .li = 0, .pools = 0x20000, .at = 0x30000 };
    const table: latent.Table = .{ .pt = 0x4000, .ps = 32768, .page_rows = 512 };
    const Case = struct { rows: usize, pk: u64, cap: usize, want: []const usize };
    for ([_]Mode{ .both, .scores }) |mode| {
        // 30 tf_glm_index_scores, 31 tf_glm_index_pools
        const k: Kernels = .{ .module = undefined, .scores_fn = .{ .d = &d, .handle = @ptrFromInt(30) }, .pools_fn = .{ .d = &d, .handle = @ptrFromInt(31) }, .select = mode == .both };
        const set: glue.Set = .{ .glue = &ks, .index = &k };
        const prompt: []const usize = if (mode == .both) &.{ 30, 31, 41 } else &.{ 30, 21, 41 };
        for ([_]Case{
            .{ .rows = 512, .pk = 0x1000, .cap = 1 << 20, .want = prompt }, // a prompt chunk of 512 rows at 65,536
            .{ .rows = 1, .pk = 0x1000, .cap = 1 << 20, .want = &.{ 20, 22, 41 } }, // a decode row: the wide select
            .{ .rows = 512, .pk = 0x1008, .cap = 1 << 20, .want = &.{ 20, 21, 41 } }, // keys off 16 bytes
            .{ .rows = 512, .pk = 0x1000, .cap = 256, .want = &.{ 20, 21, 41 } }, // fewer than 512 pools
        }) |c| {
            Recorded.n = 0;
            try sparse.selectTokens(set, ops, stream, 0x5000, 0x6000, 160, c.pk, table, 0x7000, 65536, c.rows, 32, 128, c.cap, s, 0x8000, 0x9000);
            try std.testing.expectEqualSlices(usize, c.want, Recorded.ids[0..Recorded.n]);
        }
    }
}
