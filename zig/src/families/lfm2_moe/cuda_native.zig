//! LFM2.5-24B-A2B (MLX affine 4-bit) behind the native CUDA registry: liquid's glue, this family's checks.
const std = @import("std");
const liquid = @import("liquid_cuda");
const cfg = @import("config.zig");
const mdl = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");
const snap = @import("cuda_snap.zig");

const N = liquid.native.Native(struct {
    pub const wts = @import("cuda_weights.zig");
    pub const kern = @import("cuda_kernels.zig");
    pub const mdl = @import("cuda_model.zig");
    pub const be = @import("cuda_backend.zig");
    pub const snap = @import("cuda_snap.zig");
    pub const pages_env = "TENSORFOLD_LFM2_KV_PAGES";
    pub const check = checkLayers;
});

pub const model_type = "lfm2_moe";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const explains_open = true;
pub const prompt_rows = N.prompt_rows;
pub const snap_lookahead = N.snap_lookahead;
pub const window_rows = N.window_rows;
pub const Room = N.Room;
pub const Options = N.Options;
pub const Loaded = N.Loaded;
pub const open = N.open;
pub const attach = N.attach;
pub const fits = N.fits;
pub const snap_bytes = N.snap_bytes;
pub const snap_ready = N.snap_ready;
pub const snap_charged = N.snap_charged;
pub const snap_spare = N.snap_spare;
pub const snap_reuses = N.snap_reuses;
pub const snap_save = N.snap_save;
pub const snap_restore = N.snap_restore;
pub const snap_drop = N.snap_drop;

/// The attention layers the kernels' page arena holds: the admitted 10 of 40 (conv the rest).
fn checkLayers(config: *const cfg.Config) !void {
    var n: usize = 0;
    for (config.attention) |a| n += @intFromBool(a);
    if (n != mdl.attention_layers) return error.UnsupportedLfmGeometry;
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "LFM2.5-24B-A2B on CUDA has no draft head: it drafts from the prompt's own text",
        error.LfmTextOnly => "LFM2.5-24B-A2B is a text model: this request has images",
        error.StructuresUnsupported => "structured output is not built for LFM2.5-24B-A2B on CUDA yet",
        error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
        error.CapturedKernels => "LFM2.5-24B-A2B runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.UnsupportedLfmQuantization => "LFM2.5-24B-A2B on CUDA reads MLX affine 4-bit checkpoints in groups of 64 (the embedding at 4, 6 or 8 bits): the 6- and 8-bit exports are served on Metal",
        error.UnsupportedLfmGeometry, error.UnsupportedLfmConfig, error.UnsupportedLfmRotary => "this lfm2_moe checkpoint's geometry is not LFM2.5-24B-A2B's, which this engine serves",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: LFM2.5-24B-A2B on CUDA holds at most {d} streams (a round's rows): serve --parallel {d} or fewer, or auto", .{ be.round_rows, be.round_rows }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer LFM2.5-24B-A2B streams than asked (each stream's caches are allocated at load): lower --parallel or --context, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.NoStreamFits => "the CUDA memory budget fits no LFM2.5-24B-A2B stream after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        else => null,
    };
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.LfmTextOnly, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.UnsupportedLfmQuantization, error.UnsupportedLfmGeometry, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}

test "the admitted layer pattern has the arena's ten attention layers" {
    var config = try cfg.parse(std.testing.allocator, cfg.fixture);
    try checkLayers(&config);
    config.attention[0] = true;
    try std.testing.expectError(error.UnsupportedLfmGeometry, checkLayers(&config));
}

test "the prompt cache's hooks compile against the store's vtable" {
    const pc: struct { bytes: *const fn (*anyopaque, u32) u64, save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque, restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void, drop: *const fn (*anyopaque, *anyopaque) void } = .{ .bytes = snap_bytes, .save = snap_save, .restore = snap_restore, .drop = snap_drop };
    try std.testing.expect(@intFromPtr(pc.save) != 0);
    try std.testing.expectEqual(snap.bytes(100), snap_bytes(undefined, 100));
}
