//! The qwen3_5_moe model type (Qwen3.6-35B-A3B) on the Metal engines: the dense 27B's engine and its lane host,
//! routed experts in place of the dense MLP (cuda_native_moe.zig is the CUDA twin).
const std = @import("std");

pub const model_type = "qwen3_5_moe";
pub const name = "Qwen3.6-35B-A3B";
/// What the expert tiles and routers qualify (metal_experts.zig): 4-bit g64 experts, 8-bit g64 routers.
pub const formats: []const []const u8 = &.{"mlx-q4g64"};

pub fn is(t: []const u8) bool {
    return std.mem.eql(u8, t, model_type);
}

/// The flag asking for what later slices of the port bring (the checkpoint's MTP drafts, images), refused before
/// any load; null when the open asks for neither.
pub fn later(drafter: bool, vision: bool) ?[]const u8 {
    return if (drafter) "--drafter" else if (vision) "--vision" else null;
}

/// The refusal's words after the flag.
pub const later_words = "the native engine serves " ++ name ++ " as text without drafts for now";

test "drafts and images are refused by flag, text is not" {
    try std.testing.expectEqualStrings("--drafter", later(true, true).?);
    try std.testing.expectEqualStrings("--vision", later(false, true).?);
    try std.testing.expect(later(false, false) == null);
    try std.testing.expect(is("qwen3_5_moe") and !is("qwen3_5"));
}
