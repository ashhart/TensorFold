//! sparse_index.cu for every CUDA family: indexer scores of query rows over keys, and keys' block summaries.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Key width; keys a scores block; query heads a row at most (their bf16 rows fill 16 KiB of shared memory).
pub const dim = 128;
const keys_a_block = 64;
pub const max_heads = 64;

pub const Phi = enum(i32) { identity = 0, relu = 1 };
pub const Reduce = enum(i32) { sum = 0, max = 1 };
pub const Summary = enum(i32) { mean = 0, max = 1, first = 2, last = 3 };

/// rows of h query heads over n keys of hk key heads, `ks` elements apart; scores `ld` apart (ld >= n).
pub const Scores = struct { rows: usize, n: usize, h: usize, hk: usize, ks: usize, ld: usize, scale: f32, phi: Phi = .relu, reduce: Reduce = .sum };

/// n keys `ks` elements apart into summaries `os` apart; `open`: the incomplete last block too (core's open_block).
pub const Blocks = struct { n: usize, block: usize, ks: usize, os: usize, summary: Summary = .mean, unit: bool = false, open: bool = true };

/// The summaries a launch writes: core sparse_index.blocks' count for n keys.
pub fn summaries(b: Blocks) usize {
    return if (b.open) (b.n + b.block - 1) / b.block else b.n / b.block;
}

/// What these kernels refuse of a core sparse_index.Policy, by name.
pub const Unsupported = error{ IndexHeadDim, IndexHeads, IndexKeyHeads, IndexPhi, IndexReduce, IndexSummary, IndexStreams, IndexPagedKeys, IndexPrecision };

/// The kernels read bf16 keys, queries and summaries: a policy storing any of them in NVFP4 is refused by name.
fn bf16Only(p: anytype, fields: []const []const u8) Unsupported!void {
    inline for (.{ "keys_nvfp4", "queries_nvfp4", "summary_nvfp4" }) |f| if (@hasField(@TypeOf(p), f)) {
        for (fields) |want| if (std.mem.eql(u8, want, f) and @field(p, f) != null) return error.IndexPrecision;
    };
}

/// Where the keys sit: one linear run of key rows is all these kernels read (no stream slots, no page table yet).
pub const KeyLayout = struct { streams: usize = 1, paged: bool = false };

fn linear(l: KeyLayout) Unsupported!void {
    if (l.streams != 1) return error.IndexStreams;
    if (l.paged) return error.IndexPagedKeys;
}

/// The scores launch core policy `p` (a sparse_index.Policy) asks for: its phi, head reduction, key heads and scale.
pub fn scoresFor(p: anytype, l: KeyLayout, head_dim: usize, heads: usize, rows: usize, n: usize, ks: usize, ld: usize) Unsupported!Scores {
    try linear(l);
    try bf16Only(p, &.{ "keys_nvfp4", "queries_nvfp4" });
    if (head_dim != dim) return error.IndexHeadDim;
    if (heads == 0 or heads > max_heads) return error.IndexHeads;
    if (p.key_heads == 0 or heads % p.key_heads != 0) return error.IndexKeyHeads;
    const phi = std.meta.stringToEnum(Phi, @tagName(p.phi)) orelse return error.IndexPhi;
    const reduce = std.meta.stringToEnum(Reduce, @tagName(p.reduce)) orelse return error.IndexReduce;
    return .{ .rows = rows, .n = n, .h = heads, .hk = p.key_heads, .ks = ks, .ld = ld, .scale = @floatCast(p.scale), .phi = phi, .reduce = reduce };
}

/// The block summaries `p` asks for: its block, summary, unit keys and open block; one key head of 128 only.
pub fn blocksFor(p: anytype, l: KeyLayout, head_dim: usize, n: usize, ks: usize, os: usize) Unsupported!Blocks {
    try linear(l);
    try bf16Only(p, &.{ "keys_nvfp4", "summary_nvfp4" });
    if (head_dim != dim) return error.IndexHeadDim;
    if (p.key_heads != 1) return error.IndexKeyHeads;
    const summary = std.meta.stringToEnum(Summary, @tagName(p.summary)) orelse return error.IndexSummary;
    return .{ .n = n, .block = p.block, .ks = ks, .os = os, .summary = summary, .unit = p.norm_first, .open = p.open_block };
}

pub fn checkScores(s: Scores) error{IndexShape}!void {
    if (s.h == 0 or s.h > max_heads or s.hk == 0 or s.h % s.hk != 0 or s.ks < s.hk * dim or s.ks % 8 != 0) return error.IndexShape;
    if (s.ld < s.n or s.n > std.math.maxInt(i32) or s.rows > 65535 or s.ld > std.math.maxInt(i64)) return error.IndexShape;
}

pub fn checkBlocks(b: Blocks) error{IndexShape}!void {
    if (b.block == 0 or b.block > std.math.maxInt(i32) or b.ks < dim or b.os < dim) return error.IndexShape;
    if (b.n > std.math.maxInt(i64) or (b.n + b.block - 1) / b.block > std.math.maxInt(i32)) return error.IndexShape;
}

pub const SparseIndex = struct {
    mod: Module,
    scores: Function,
    blocks: Function,
    open_blocks: Function,

    pub fn load(d: *const Driver) !SparseIndex {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.sparse_index);
        errdefer mod.unload();
        return .{ .mod = mod, .scores = try mod.function("tf_index_scores"), .blocks = try mod.function("tf_index_blocks"), .open_blocks = try mod.function("tf_index_open_blocks") };
    }

    pub fn unload(x: *SparseIndex) void {
        x.mod.unload();
    }

    /// q bf16 [rows, h, 128]; w f32 [rows, h] or 0; counts i32 [rows] or 0; out f32 [rows, ld], -inf from count to ld.
    pub fn runScores(x: *const SparseIndex, s: Stream, q: u64, k: u64, w: u64, counts: u64, out: u64, sh: Scores) !void {
        try checkScores(sh);
        if (sh.rows == 0 or sh.ld == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ q, k }) |v| a.add(v);
        a.add(@as(i64, @intCast(sh.ks)));
        for ([_]u64{ w, counts, out }) |v| a.add(v);
        a.add(@as(i64, @intCast(sh.ld)));
        for ([_]usize{ sh.n, sh.h, sh.hk }) |v| a.add(@as(i32, @intCast(v)));
        a.add(sh.scale);
        a.add(@backingInt(sh.phi));
        a.add(@backingInt(sh.reduce));
        const grid: launch_.Dim3 = .{ .x = @intCast((sh.ld + keys_a_block - 1) / keys_a_block), .y = @intCast(sh.rows), .z = 1 };
        const smem: u32 = @intCast(sh.h * dim * 2 + sh.h * 4);
        try launch_.launch(x.scores, .{ .grid = grid, .block = .{ .x = keys_a_block }, .shared = smem }, s, &a);
    }

    /// k bf16 key rows; out bf16 [summaries(b), os] (an open last block summarises the keys it has).
    pub fn runBlocks(x: *const SparseIndex, s: Stream, k: u64, out: u64, b: Blocks) !void {
        try checkBlocks(b);
        if (summaries(b) == 0) return;
        var a: launch_.Args = .{};
        a.add(k);
        a.add(@as(i64, @intCast(b.ks)));
        a.add(out);
        a.add(@as(i64, @intCast(b.os)));
        a.add(@as(i64, @intCast(b.n)));
        a.add(@as(i32, @intCast(b.block)));
        a.add(@backingInt(b.summary));
        a.add(@as(i32, @intFromBool(b.unit)));
        const grid: launch_.Dim3 = .{ .x = @intCast(summaries(b)), .y = 1, .z = 1 };
        try launch_.launch(x.blocks, .{ .grid = grid, .block = .{ .x = dim } }, s, &a);
    }

    /// Each row's open block (ends i32 [rows] = t + 1, clamped to [0, b.n]) into out bf16 [rows, os], 0 if none.
    pub fn runOpenBlocks(x: *const SparseIndex, s: Stream, k: u64, ends: u64, out: u64, rows: usize, b: Blocks) !void {
        try checkBlocks(b);
        if (rows == 0) return;
        if (rows > std.math.maxInt(i32)) return error.IndexShape;
        var a: launch_.Args = .{};
        a.add(k);
        a.add(@as(i64, @intCast(b.ks)));
        a.add(out);
        a.add(@as(i64, @intCast(b.os)));
        a.add(ends);
        a.add(@as(i64, @intCast(b.n))); // the kernel clamps every end to the keys there are
        a.add(@as(i32, @intCast(b.block)));
        a.add(@backingInt(b.summary));
        a.add(@as(i32, @intFromBool(b.unit)));
        try launch_.launch(x.open_blocks, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = dim } }, s, &a);
    }
};

test "checks take Kolibri-sized indexers and refuse heads past 64, partial groups and overlapping key heads" {
    const ok: Scores = .{ .rows = 4, .n = 16384, .h = 64, .hk = 1, .ks = dim, .ld = 16384, .scale = 1 };
    try checkScores(ok);
    var x = ok;
    x.h = 65;
    try std.testing.expectError(error.IndexShape, checkScores(x));
    x = ok;
    x.h = 48;
    x.hk = 5;
    try std.testing.expectError(error.IndexShape, checkScores(x));
    x = ok;
    x.hk = 2; // two key heads in rows one key head wide
    try std.testing.expectError(error.IndexShape, checkScores(x));
    x = ok;
    x.ld = 100;
    try std.testing.expectError(error.IndexShape, checkScores(x));
    try checkBlocks(.{ .n = 16384, .block = 64, .ks = dim, .os = dim });
    try std.testing.expectError(error.IndexShape, checkBlocks(.{ .n = 16, .block = 0, .ks = dim, .os = dim }));
}

test "a policy maps onto the launches, and what the kernels lack is refused by name" {
    const P = struct { phi: enum { relu, identity, softplus } = .relu, reduce: enum { sum, max, mean } = .sum, summary: enum { mean, max, first, last, median } = .mean, key_heads: usize = 1, scale: f64 = 0.5, block: usize = 16, norm_first: bool = true, open_block: bool = false };
    const s = try scoresFor(P{}, .{}, dim, 48, 2, 100, dim, 100);
    try std.testing.expectEqual(Phi.relu, s.phi);
    try std.testing.expectEqual(@as(f32, 0.5), s.scale);
    try std.testing.expectError(error.IndexPhi, scoresFor(P{ .phi = .softplus }, .{}, dim, 48, 2, 100, dim, 100));
    try std.testing.expectError(error.IndexReduce, scoresFor(P{ .reduce = .mean }, .{}, dim, 48, 2, 100, dim, 100));
    try std.testing.expectError(error.IndexKeyHeads, scoresFor(P{ .key_heads = 5 }, .{}, dim, 48, 2, 100, dim, 100));
    try std.testing.expectError(error.IndexHeadDim, scoresFor(P{}, .{}, 64, 48, 2, 100, dim, 100));
    try std.testing.expectError(error.IndexHeads, scoresFor(P{}, .{}, dim, 65, 2, 100, dim, 100));
    const b = try blocksFor(P{}, .{}, dim, 40, dim, dim);
    try std.testing.expectEqual(@as(usize, 2), summaries(b)); // the open block of 8 left out
    var o = b;
    o.open = true;
    try std.testing.expectEqual(@as(usize, 3), summaries(o));
    try std.testing.expectError(error.IndexSummary, blocksFor(P{ .summary = .median }, .{}, dim, 40, dim, dim));
    try std.testing.expectError(error.IndexKeyHeads, blocksFor(P{ .key_heads = 2 }, .{}, dim, 40, dim, dim));
    // keys of several streams or behind a page table: refused until the kernels address them
    try std.testing.expectError(error.IndexStreams, scoresFor(P{}, .{ .streams = 4 }, dim, 48, 2, 100, dim, 100));
    try std.testing.expectError(error.IndexPagedKeys, scoresFor(P{}, .{ .paged = true }, dim, 48, 2, 100, dim, 100));
    try std.testing.expectError(error.IndexPagedKeys, blocksFor(P{}, .{ .paged = true }, dim, 40, dim, dim));
    // NVFP4 keys, queries or summaries: the kernels read bf16 (planted: each was mapped as if bf16)
    const Q = struct { phi: enum { relu } = .relu, reduce: enum { sum } = .sum, summary: enum { mean } = .mean, key_heads: usize = 1, scale: f64 = 1, block: usize = 16, norm_first: bool = false, open_block: bool = true, keys_nvfp4: ?u8 = null, queries_nvfp4: ?u8 = null, summary_nvfp4: ?u8 = null };
    _ = try scoresFor(Q{}, .{}, dim, 48, 2, 100, dim, 100);
    _ = try blocksFor(Q{ .queries_nvfp4 = 1 }, .{}, dim, 40, dim, dim); // the summaries never read queries
    try std.testing.expectError(error.IndexPrecision, scoresFor(Q{ .keys_nvfp4 = 1 }, .{}, dim, 48, 2, 100, dim, 100));
    try std.testing.expectError(error.IndexPrecision, scoresFor(Q{ .queries_nvfp4 = 1 }, .{}, dim, 48, 2, 100, dim, 100));
    try std.testing.expectError(error.IndexPrecision, blocksFor(Q{ .summary_nvfp4 = 1 }, .{}, dim, 40, dim, dim));
    try std.testing.expectError(error.IndexPrecision, blocksFor(Q{ .keys_nvfp4 = 1 }, .{}, dim, 40, dim, dim));
}
