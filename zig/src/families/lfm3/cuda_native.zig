//! LFM3-1.7B (MLX 4-bit; NVFP4 and block FP8 weight-only) behind the native CUDA registry: drafts, prompt reuse.
const std = @import("std");
const liquid = @import("liquid_cuda");
const be = @import("cuda_backend.zig");
const snap = @import("cuda_snap.zig");

const N = liquid.native.Native(struct {
    pub const wts = @import("cuda_weights.zig");
    pub const kern = @import("cuda_kernels.zig");
    pub const mdl = @import("cuda_model.zig");
    pub const be = @import("cuda_backend.zig");
    pub const snap = @import("cuda_snap.zig");
    pub const pages_env = "TENSORFOLD_LFM3_KV_PAGES";
    pub fn check(_: anytype) !void {} // the admission already fixed the layer schedule
});

pub const model_type = "lfm3";
pub const formats: []const []const u8 = &.{ "mlx-q4g64", "modelopt-nvfp4", "fp8-block128" };
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

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "LFM3 on CUDA has no draft head: it drafts from the prompt's own text",
        error.Lfm3TextOnly => "LFM3-1.7B is a text model, and LFM3-VL's image tower is not built on CUDA yet: this request has images",
        error.StructuresUnsupported => "structured output is not built for LFM3 on CUDA yet",
        error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
        error.CapturedKernels => "LFM3 runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.UnsupportedLfm3Quantization => "LFM3 on CUDA reads the text exports at MLX 4 bits in groups of 64, ModelOpt NVFP4 and block FP8: its 6- and 8-bit MLX exports and LFM3-VL are not loaded yet",
        error.UnsupportedLfm3Config, error.UnsupportedLfm3Geometry, error.UnsupportedLfm3LayerSchedule, error.UnsupportedLfm3Rotary, error.UnsupportedLfm3Context, error.BadLfm3Config => "this lfm3 checkpoint's config is not LFM3-1.7B's, which this engine serves",
        error.ParallelPastRound => std.fmt.comptimePrint("--parallel: LFM3 on CUDA holds at most {d} streams (a round's rows): serve --parallel {d} or fewer, or auto", .{ be.round_rows, be.round_rows }),
        error.ParallelPastMemory => "--parallel: the CUDA memory budget fits fewer LFM3 streams than asked (each stream's caches are allocated at load): lower --parallel or --context, serve --parallel auto, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.NoStreamFits => "the CUDA memory budget fits no LFM3 stream after the weights: lower --context, free device memory, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        else => null,
    };
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.Lfm3TextOnly, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.UnsupportedLfm3Quantization, error.UnsupportedLfm3Geometry, error.ParallelPastRound, error.ParallelPastMemory, error.NoStreamFits }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}

test "the prompt cache's hooks compile against the store's vtable" {
    const pc: struct { bytes: *const fn (*anyopaque, u32) u64, save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque, restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void, drop: *const fn (*anyopaque, *anyopaque) void } = .{ .bytes = snap_bytes, .save = snap_save, .restore = snap_restore, .drop = snap_drop };
    try std.testing.expect(@intFromPtr(pc.save) != 0);
    try std.testing.expectEqual(snap.bytes(100), snap_bytes(undefined, 100));
}
