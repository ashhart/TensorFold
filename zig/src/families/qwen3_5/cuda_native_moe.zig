//! The qwen3_5_moe model type (Qwen3.6-35B-A3B) on the same CUDA engine: routed experts in place of the dense MLP.
const native = @import("cuda_native.zig");

pub const model_type = "qwen3_5_moe";
/// MLX's 4-bit tables, and ModelOpt's exports: MIXED_PRECISION (nvidia/Qwen3.6-35B-A3B-NVFP4) and single-format NVFP4.
pub const formats: []const []const u8 = &.{ "mlx-q4g64", "modelopt-mixed", "modelopt-nvfp4" };
pub const default_context = native.default_context;
pub const max_segments = native.max_segments;
pub const prompt_rows = native.prompt_rows;
/// --drafter names the checkpoint's own directory: its MTP layer (mtp-4bit.safetensors) drafts chains.
pub const takes_drafter = true;
pub const Options = native.Options;
pub const open = native.open;
pub const attach = native.attach;
pub const pagedLayout = native.pagedLayout; // TF_QWEN_KV_PAGES: the same pool as the dense models'
pub const fits = native.fits;
pub const window_rows = native.window_rows;
pub const openDrafter = native.openMtp;
pub const explain = native.explain;
/// The dense family's kept prompt states, its open refusals' words and its image tower: the MLP is the only change.
pub const snap_bytes = native.snap_bytes;
pub const snap_save = native.snap_save;
pub const snap_restore = native.snap_restore;
pub const snap_drop = native.snap_drop;
pub const snap_host = native.snap_host;
pub const snap_lookahead = native.snap_lookahead;
pub const snap_grid = native.snap_grid;
pub const kept_disk = native.kept_disk;
pub const explains_open = native.explains_open;
pub const vision = native.vision;

test "the MoE family's capabilities name the ModelOpt formats it reads beside MLX's" {
    const std = @import("std");
    for ([_][]const u8{ "mlx-q4g64", "modelopt-mixed", "modelopt-nvfp4" }) |want| {
        var found = false;
        for (formats) |f| found = found or std.mem.eql(u8, f, want);
        try std.testing.expect(found);
    }
}
