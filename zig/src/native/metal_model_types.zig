//! Model types that share one Metal family, and the Llama family's checkpoint formats.
const std = @import("std");

/// The Llama family's: the bf16 release, compressed-tensors' int4 g32, or one MLX affine width with bf16 tables.
pub const llama_formats: []const []const u8 = &.{ "bf16", "compressed-tensors-int4-g32", "mlx-q4g32", "mlx-q4g64", "mlx-q8g32", "mlx-q8g64" };

/// Llama (llama) and Mistral (mistral): one native family, llama.zig's host.
pub fn isLlama(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "llama") or std.mem.eql(u8, model_type, "mistral");
}

/// Gemma 4 31B (gemma4) and 12B (gemma4_unified, its text model): one native family, gemma4.zig's host.
pub fn isGemma4(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "gemma4") or std.mem.eql(u8, model_type, "gemma4_unified");
}
