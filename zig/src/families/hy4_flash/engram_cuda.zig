//! CUDA Engram branch for Hy4-flash, after `metal_engram.zig`: gathered BF16 table rows (direct reads or a
//! resident copy) are optionally FP8 round-tripped (the stock checkpoint profile), normalized per hash head,
//! projected once to a value and once per hyper-connection stream to a key, then gated into the four streams.
const std = @import("std");
const cuda = @import("cuda");
const engram = @import("engram.zig");
const ihc = @import("ihc.zig");

pub const head_width = 96;
pub const columns = engram.columns;
pub const input = columns * head_width;

pub const Weights = struct {
    lookup_gamma: u64, // BF16 [16, 96]
    value: u64, // BF16 [hidden, 1536]
    value_bias: u64,
    value_gain: u64, // unified_pre_gamma
    key: [ihc.streams]u64, // BF16 [hidden, 1536] each
    key_bias: u64, // BF16 [4, hidden]
    key_gain: u64,
    query_gain: u64,
    delta_gain: u64, // unified_per_hc_scales
};

pub const Buffers = struct {
    raw: u64, // BF16 [gemm_rows, 1536], gathered rows
    normalized: u64, // BF16 [gemm_rows, 1536]
    keys: u64, // BF16 [4, gemm_rows, hidden]
    value: u64, // BF16 [gemm_rows, hidden]
    media: u64, // u32 [gemm_rows]
    state: u64, // BF16 [gemm_rows, 4, hidden]
    output: u64, // BF16 [gemm_rows, 4, hidden]
};

pub const Kernels = struct {
    module: cuda.Module,
    lookup_fn: cuda.Function,
    apply_fn: cuda.Function,

    pub fn load(driver: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(driver, cuda.kernels.hy4_engram);
        errdefer module.unload();
        return .{ .module = module, .lookup_fn = try module.function("tf_hy4_engram_lookup"), .apply_fn = try module.function("tf_hy4_engram_apply") };
    }

    pub fn deinit(self: *Kernels) void {
        self.module.unload();
        self.* = undefined;
    }

    /// The Engram residual added to `rows` rows of four hidden streams. GEMMs run on the fixed `gemm_rows`.
    pub fn forward(self: Kernels, s: cuda.Stream, gemm: *cuda.vit.Gemms, w: Weights, b: Buffers, rows: u32, gemm_rows: u32, hidden: u32, fp8: bool) !void {
        if (rows == 0 or rows > gemm_rows) return error.BadHy4EngramRows;
        var lookup: cuda.Args = .{};
        inline for (.{ b.raw, w.lookup_gamma, b.normalized, @as(u32, @intFromBool(fp8)) }) |x| lookup.add(x);
        try cuda.launch.launch(self.lookup_fn, .{ .grid = .{ .x = columns, .y = gemm_rows }, .block = .{ .x = 32 } }, s, &lookup);
        try gemm.run(s, b.normalized, w.value, b.value, gemm_rows, hidden, input, .bf16);
        for (w.key, 0..) |key, i| try gemm.run(s, b.normalized, key, b.keys + i * @as(u64, gemm_rows) * hidden * 2, gemm_rows, hidden, input, .bf16);
        var apply: cuda.Args = .{};
        inline for (.{ b.state, b.keys, w.key_bias, w.key_gain, w.query_gain, w.delta_gain, b.value, w.value_bias, w.value_gain, b.media, b.output, gemm_rows, hidden }) |x| apply.add(x);
        try cuda.launch.launch(self.apply_fn, .{ .grid = .{ .x = ihc.streams, .y = rows }, .block = .{ .x = 128 } }, s, &apply);
    }
};
