// CUDA wrappers for GLM-5.3-Flash's DSA sparse indexer (glm5_next/cuda/sparse.py ported to kernels/cuda/glm_sparse.cu).

// The index update writes LayerNorm'd keys and gate rows into the caches and re-pools completed 4-token pools; the

// scores kernel ranks every visible pool per row; the select runs a radix select a row a block; tokens and counts

// assemble each row's attended tokens. Every function takes device pointers, shapes and a stream.

const std = @import("std");

const launch_ = @import("launch.zig");

const stream_ = @import("stream.zig");

const module = @import("module.zig");

const driver = @import("driver.zig");

const kernels = @import("kernels.zig");

const Driver = driver.Driver;

const Module = module.Module;

const Function = module.Function;

const Stream = stream_.Stream;

pub const d = 128; // index key dims

pub const heads = 32; // index heads (index_heads)

pub const pool = 4; // tokens a pool

pub const topk_pools = 512; // pools a row keeps

pub const width = topk_pools * pool + pool - 1; // tokens a row keeps (2051)

pub const score_loop = 32; // pool blocks a scores program takes

pub const block_pools = 64; // pools a scores block

pub const select_block = 1024; // scores a select pass reads a chunk

pub const Sparse = struct {
    d: *const Driver,
    m: Module,
    index_fn: Function,
    pool_fn: Function,
    scores_fn: Function,
    select_fn: Function,
    tokens_fn: Function,
    loaded: bool = false,

    pub fn load(dr: *const Driver) !Sparse {
        var k: Sparse = .{
            .d = dr,
            .m = undefined,
            .index_fn = undefined,
            .pool_fn = undefined,
            .scores_fn = undefined,
            .select_fn = undefined,
            .tokens_fn = undefined,
        };
        errdefer k.unload();
        k.m = try Module.load(dr, kernels.glm_sparse);
        k.index_fn = try k.m.function("tf_glm_index_write");
        k.pool_fn = try k.m.function("tf_glm_pool_keys");
        k.scores_fn = try k.m.function("tf_glm_scores");
        k.select_fn = try k.m.function("tf_glm_select_rows");
        k.tokens_fn = try k.m.function("tf_glm_tokens_counts");
        k.loaded = true;
        return k;
    }

    pub fn unload(self: *Sparse) void {
        if (!self.loaded) return;
        self.m.unload();
        self.loaded = false;
    }

    /// Row writes: kr [R, d] bf16 rows k_stride apart, gr [R, d] fp32 -> ik/ig rows pos.. (bf16).
    pub fn indexWrite(
        self: *Sparse,
        s: *Stream,
        kr: u64,
        k_stride: i64,
        gr: u64,
        lnw: u64,
        lnb: u64,
        ik: u64,
        ig: u64,
        pos: i32,
        posp: u64, // the position on the device instead (0: pos)
        eps: f32,
        rows: u32,
    ) !void {
        var args = launch_.Args{};
        args.add(kr);
        args.add(k_stride);
        args.add(gr);
        args.add(lnw);
        args.add(lnb);
        args.add(ik);
        args.add(ig);
        args.add(pos);
        args.add(eps);
        args.add(posp);
        try launch_.launch(self.index_fn, .{ .grid = .{ .x = rows, .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// Pool keys for the pools the window completes: pk rows (pos / 4).., capped by the window's last full pool.
    pub fn poolKeys(self: *Sparse, s: *Stream, ik: u64, ig: u64, ape: u64, pk: u64, pos: i32, posp: u64, rows: u32) !void {
        var args = launch_.Args{};
        args.add(ik);
        args.add(ig);
        args.add(ape);
        args.add(pk);
        args.add(pos);
        args.add(@as(i32, @intCast(rows)));
        args.add(posp);
        try launch_.launch(self.pool_fn, .{ .grid = .{ .x = rows / 4 + 2, .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// Pool scores: qi rows r.. (qi_stride a row), weights w (w_stride a row), pos first row's position; out f32.
    pub fn scores(
        self: *Sparse,
        s: *Stream,
        qi: u64,
        qi_stride: i64,
        w: u64,
        w_stride: i64,
        pk: u64,
        out: u64,
        pos: i32,
        posp: u64, // the position on the device instead (0: pos)
        rows: u32,
        np: u32,
        scale: f32,
        wscale: f32,
    ) !void {
        var args = launch_.Args{};
        args.add(qi);
        args.add(qi_stride);
        args.add(w);
        args.add(w_stride);
        args.add(pk);
        args.add(out);
        args.add(pos);
        args.add(@as(i32, @intCast(np)));
        args.add(scale);
        args.add(wscale);
        args.add(posp);
        const blocks_y = (np + block_pools * score_loop - 1) / (block_pools * score_loop);
        try launch_.launch(self.scores_fn, .{ .grid = .{ .x = rows, .y = @intCast(blocks_y), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// Radix top-pool select per row: visible bounds the per-row pool limit; out [rows, k] int64 pools ascending.
    pub fn selectRows(self: *Sparse, s: *Stream, sc: u64, np: u32, out: u64, pos: i32, posp: u64, k: u32, rows: u32, visible: bool) !void {
        var args = launch_.Args{};
        args.add(sc);
        args.add(@as(i64, @intCast(np)));
        args.add(@as(i64, @intCast(k)));
        args.add(out);
        args.add(@as(i32, @intCast(np)));
        args.add(pos);
        args.add(@as(i32, @intCast(k)));
        args.add(@as(i32, if (visible) 1 else 0));
        args.add(posp);
        try launch_.launch(self.select_fn, .{ .grid = .{ .x = rows, .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// Tokens/counts from pools: tokens [rows, 2051] i32 (pools ascending then the partial tail), counts [rows].
    pub fn tokensCounts(self: *Sparse, s: *Stream, pools: u64, tokens: u64, counts: u64, pos: i32, posp: u64, rows: u32) !void {
        var args = launch_.Args{};
        args.add(pools);
        args.add(tokens);
        args.add(counts);
        args.add(pos);
        args.add(posp);
        try launch_.launch(self.tokens_fn, .{ .grid = .{ .x = rows, .y = 1, .z = 1 }, .block = .{ .x = 512, .y = 1, .z = 1 } }, s.*, &args);
    }
};
