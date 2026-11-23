//! The Metal host's family table: each `model_type` it serves and the weight formats it reads.
const std = @import("std");
const api = @import("engine_api");

/// Every checkpoint-wide MLX affine width the per-module readers take (oQ quants): 2-8 bits but 7, groups 32-128.
const affine_widths = blk: {
    var out: [18][]const u8 = undefined;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }, 0..) |bits, i| for ([_]u16{ 32, 64, 128 }, 0..) |group, j| {
        out[3 * i + j] = std.fmt.comptimePrint("mlx-q{d}g{d}", .{ bits, group });
    };
    const done = out;
    break :blk &done;
};
/// Gemma 4's: MLX affine 2-6 or 8 bits in groups of 64, uniform or per module (oQ packs).
const gemma4_widths: []const []const u8 = &.{ "mlx-q2g64", "mlx-q3g64", "mlx-q4g64", "mlx-q5g64", "mlx-q6g64", "mlx-q8g64" };
pub const families: []const api.Family = &.{
    .{ .model_type = "kimi_k3", .formats = &.{"mxfp4"} },
    .{ .model_type = "kimi_k31", .formats = &.{"mxfp4"} },
    .{ .model_type = "qwen4", .formats = @import("qwen4").family.native_host.formats },
    .{ .model_type = "qwen4_text", .formats = @import("qwen4").family.native_host.formats },
    .{ .model_type = "fish_omni", .formats = &.{ "fp8-block128", "nvfp4" } },
    .{ .model_type = "nemotron_h", .formats = affine_widths },
    .{ .model_type = "nemotron_h_puzzle", .formats = affine_widths },
    .{ .model_type = "qwen4_exp", .formats = affine_widths },
    .{ .model_type = "glm5_next", .formats = &.{"mlx-q4g64"} },
    .{ .model_type = "qwen3_5", .formats = affine_widths },
    .{ .model_type = "qwen3_5_moe", .formats = affine_widths },
    .{ .model_type = "lfm3", .formats = &.{ "mlx-q4g32", "mlx-q4g64", "mlx-q6g64", "mlx-q8g64" } },
    .{ .model_type = "lfm3_vl", .formats = &.{ "mlx-q4g32", "mlx-q4g64", "mlx-q6g64", "mlx-q8g64" } },
    .{ .model_type = "lfm2_moe", .formats = &.{ "mlx-q4g64", "mlx-q6g64", "mlx-q8g64" } },
    .{ .model_type = "glm_moe_dsa", .formats = &.{"exlr8"} },
    .{ .model_type = "gemma4", .formats = gemma4_widths },
    .{ .model_type = "gemma4_unified", .formats = gemma4_widths },
    .{ .model_type = "step5_vl", .formats = &.{"mlx-q2g128"} },
    .{ .model_type = "spark2_5", .formats = &.{ "bf16", "mlx-q2g64", "mlx-q3g64", "mlx-q4g64", "mlx-q5g64", "mlx-q6g64", "mlx-q8g64" } },
    .{ .model_type = "minimax_m3_vl", .formats = &.{ "mlx-q3g64", "mlx-q8g64" } },
    .{ .model_type = "hy_v4_flash", .formats = &.{ "mlx-q4g64", "mlx-q8g64" } },
    .{ .model_type = "inkling2_loop", .formats = &.{"modelopt-mixed"} },
    .{ .model_type = "qwen3", .formats = &.{"bf16"} },
    .{ .model_type = "llama", .formats = &.{ "bf16", "compressed-tensors-int4-g32" } },
    .{ .model_type = "flint", .formats = &.{"mlx-q4g64"} },
    .{ .model_type = "unlimited-ocr", .formats = &.{"bf16"} },
    .{ .model_type = "ricursive", .formats = &.{ "bf16", "modelopt-nvfp4-mlp" } },
};
