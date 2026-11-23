//! DINOv3 ViT-H+/16's geometry at Idol's 1024 x 1024 input, and the check that a timm snapshot names that model.
const std = @import("std");

pub const width = 1280;
pub const heads = 20;
pub const head_dim = width / heads;
/// SwiGLU's hidden width: gate and up projections of 5,120 each.
pub const ffn = 5120;
pub const layers = 32;
pub const patch = 16;
pub const image = 1024;
pub const grid = image / patch;
pub const patches = grid * grid;
/// The class token and four registers, ahead of the patches: no rotary.
pub const prefix = 5;
pub const rows = prefix + patches;
/// Every nn.LayerNorm of the model and ImageEncoder's last F.layer_norm.
pub const eps: f32 = 1e-5;
/// RoPE's base: periods 100^(4i/64) for i < 16, on each axis.
pub const rope_theta: f64 = 100;
pub const rope_freqs = head_dim / 4;
/// One patch's pixels, channel then row then column: the conv weight [1280, 3, 16, 16] flattened.
pub const patch_k = 3 * patch * patch;
/// timm's tensors: six outside the blocks (patch conv and bias, class token, registers, last norm), fifteen a block.
pub const tensors = 6 + 15 * layers;

/// Accepts timm's vit_huge_plus_patch16_dinov3 snapshot config (1,280 features); anything else is refused.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const arch = doc.value.object.get("architecture") orelse return error.BadConfig;
    if (arch != .string or !std.mem.eql(u8, arch.string, "vit_huge_plus_patch16_dinov3")) return error.NotDinov3;
    const features = doc.value.object.get("num_features") orelse return error.BadConfig;
    if (features != .integer or features.integer != width) return error.UnsupportedGeometry;
}

test "timm's DINOv3 H+/16 config passes; another architecture or width does not" {
    const a = std.testing.allocator;
    try check(a, "{\"architecture\": \"vit_huge_plus_patch16_dinov3\", \"num_classes\": 0, \"num_features\": 1280}");
    try std.testing.expectError(error.NotDinov3, check(a, "{\"architecture\": \"vit_large_patch16_dinov3\", \"num_features\": 1024}"));
    try std.testing.expectError(error.UnsupportedGeometry, check(a, "{\"architecture\": \"vit_huge_plus_patch16_dinov3\", \"num_features\": 1024}"));
    try std.testing.expectError(error.BadConfig, check(a, "[]"));
    try std.testing.expectEqual(@as(usize, 4101), rows);
    try std.testing.expectEqual(@as(usize, 486), tensors);
}
