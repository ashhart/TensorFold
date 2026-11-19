//! CUDA composition of Hy4-flash's compressed sparse indexer, after the reference's CompressedIndexer and
//! `metal_index.zig`: each pass caches its rows' normalized token keys, pools the four-slot groups it closed
//! (FP32 value/gate projections, `intra_group_ape`, `key_norm`), rotates their trailing 64 channels at the
//! group's first token, then ranks a row's closed groups with the 16 rotated query heads and keeps 511 of
//! them plus the causal part of its own group. Slots are cache slots; a group is pooled only once closed,
//! so a rejected draft suffix leaves no stale group behind.
const std = @import("std");
const cuda = @import("cuda");
const mla = @import("mla_cuda.zig");
const rotary = @import("rope.zig");
const recall = @import("index_recall.zig");
const recall_cuda = @import("index_recall_cuda.zig");
const groups = @import("index_groups.zig");

pub const heads = recall.heads;
pub const width = recall.width;
pub const nope = width - 2 * rotary.pairs;
pub const budget_groups: u32 = 512;

/// Device weights: BF16 projections and norm, FP32 pooling parameters.
pub const Weights = struct {
    token_keys: u64, // [128, hidden]
    token_norm: u64, // [128]
    query: u64, // [16 * 128, q_rank]
    head_weights: u64, // [16, hidden]
    values: u64, // f32 [128, 128]
    gates: u64, // f32 [128, 128]
    bias: u64, // f32 [4, 128]
    norm: u64, // f32 [128]
};

pub const Indexer = struct {
    pool: cuda.Module,
    groups_fn: cuda.Function,
    select_kernels: recall_cuda.Kernels,

    pub fn load(driver: *const cuda.Driver) !Indexer {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var pool = try cuda.Module.load(driver, cuda.kernels.hy4_indexpool);
        errdefer pool.unload();
        return .{
            .pool = pool,
            .groups_fn = try pool.function("tf_hy4_index_groups"),
            .select_kernels = try recall_cuda.Kernels.load(driver),
        };
    }

    pub fn deinit(self: *Indexer) void {
        self.select_kernels.deinit();
        self.pool.unload();
        self.* = undefined;
    }

    pub const Keys = struct {
        hidden: u64, // BF16 [gemm_rows, hidden], zero past the active rows
        gemm_rows: usize, // fixed per lane class so cuBLASLt keeps one reduction order
        count: usize,
        projected: u64, // BF16 [gemm_rows, 128]
        norm_tmp: u64,
        staged: u64, // BF16 [gemm_rows, 128]
        first_slot: usize,
    };

    /// Normalized BF16 token keys of the active rows into contiguous cache slots from `first_slot`.
    pub fn cacheKeys(_: Indexer, s: cuda.Stream, gemm: *cuda.vit.Gemms, norms: *const cuda.norms.Norms, w: Weights, b: Keys, cache: cuda.DeviceBuffer, hidden_size: usize, eps: f32) !void {
        if (b.count == 0 or b.count > b.gemm_rows) return error.BadHy4IndexRows;
        try gemm.run(s, b.hidden, w.token_keys, b.projected, b.gemm_rows, width, hidden_size, .bf16);
        try norms.addRmsTail(s, b.projected, null, w.token_norm, b.norm_tmp, b.staged, null, b.gemm_rows, width, eps, .{ .round = true });
        try cache.copyFrom(b.first_slot * width * 2, b.staged, b.count * width * 2, s.handle);
    }

    /// Pool, normalize and rotate `count` closed groups from `first_group` into the group arena.
    /// `positions` holds each group's first-token coordinates, `table` room for `count` rotary rows.
    pub fn poolGroups(self: Indexer, s: cuda.Stream, project: mla.Projections, w: Weights, cache: u64, arena: u64, first_group: u32, count: u32, positions: u64, table: u64, p: rotary.Parameters, eps: f32) !void {
        if (count == 0) return;
        var args: cuda.Args = .{};
        inline for (.{ cache, w.values, w.gates, w.bias, w.norm, arena, first_group, eps }) |x| args.add(x);
        try cuda.launch.launch(self.groups_fn, .{ .grid = .{ .x = count }, .block = .{ .x = width } }, s, &args);
        try project.rotaryTable(s, positions, table, count, p);
        const rope_base = arena + (@as(u64, first_group) * width + nope) * 2;
        try project.rotate(s, rope_base, table, rope_base, .{ .rows = count, .heads = 1, .input_stride = width, .output_stride = width });
    }

    pub const Select = struct {
        q_resid: u64, // BF16 [gemm_rows, q_rank], the q_a_layernorm output
        hidden: u64, // BF16 [gemm_rows, hidden]
        gemm_rows: usize,
        positions: u64, // i32 [rows, 3]
        table: u64, // BF16 [rows, 64]
        query: u64, // BF16 [rows, 16 * 128]
        head_weights: u64, // BF16 [rows, 16]
        arena: u64,
        rows: u64, // recall.Row [rows]
        scores: u64,
        tokens: u64,
        counts: u64,
    };

    /// Rotated query heads, head weights, closed-group scores and the recalled slot list of each row.
    pub fn select(self: Indexer, s: cuda.Stream, gemm: *cuda.vit.Gemms, project: mla.Projections, w: Weights, b: Select, shape: recall.Shape, q_rank: usize, hidden_size: usize, p: rotary.Parameters) !void {
        if (shape.rows > b.gemm_rows) return error.BadHy4IndexRows;
        try gemm.run(s, b.q_resid, w.query, b.query, b.gemm_rows, heads * width, q_rank, .bf16);
        try project.rotaryTable(s, b.positions, b.table, shape.rows, p);
        const rope_base = b.query + nope * 2;
        try project.rotate(s, rope_base, b.table, rope_base, .{ .rows = shape.rows, .heads = heads, .input_stride = width, .output_stride = width });
        try gemm.run(s, b.hidden, w.head_weights, b.head_weights, b.gemm_rows, heads, hidden_size, .bf16);
        try self.select_kernels.run(s, .{
            .query = b.query,
            .keys = b.arena,
            .weights = b.head_weights,
            .rows = b.rows,
            .scores = b.scores,
            .tokens = b.tokens,
            .counts = b.counts,
        }, shape);
    }
};

/// Closed groups a pass of `count` new slots after `previous` cached slots makes poolable.
pub fn closedBy(previous: usize, count: usize) !groups.Append {
    return groups.append(previous, count);
}
