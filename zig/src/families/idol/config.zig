//! Idol's denoiser geometry, and the check that a snapshot's config.json names that denoiser.
const std = @import("std");

pub const hidden = 1024;
pub const heads = 16;
pub const head_dim = hidden / heads;
pub const ffn = 4 * hidden;
pub const layers = 24;
/// The noise and condition refiners' blocks, each.
pub const refiners = 2;
pub const latent_rows = 8192;
pub const latent_channels = 16;
pub const camera_channels = 5;
/// DINOv3 ViT-H+/16 rows at 1024 x 1024: 64 x 64 patches, the class token and four registers.
pub const cond_rows = 4101;
pub const cond_channels = 1280;
pub const cond2_channels = 128;
/// Latent, condition and camera rows through the joint blocks.
pub const joint_rows = latent_rows + cond_rows + 1;
/// RePo's hidden width (model_channels / 8) and its frequencies per head: 10, 10 and 12 for x, y and z.
pub const repo_hidden = hidden / 8;
pub const repo_pairs = head_dim / 2;
pub const time_dim = 256;
/// Every transformer block's LayerNorm32, fixed in idol-sdk's layers.py, not a config setting.
pub const block_eps: f32 = 1e-6;
/// RePo's LayerNorm32 and the output's F.layer_norm take torch's default.
pub const default_eps: f32 = 1e-5;
/// Modulated blocks: the noise refiner's, then the joint blocks.
pub const modulated = refiners + layers;

/// Accepts the snapshot config whose denoiser this family runs (LatentSeqMMFlowModel at its released size).
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const den = doc.value.object.get("denoiser") orelse return error.BadConfig;
    if (den != .object) return error.BadConfig;
    const name = den.object.get("name") orelse return error.BadConfig;
    if (name != .string or !std.mem.eql(u8, name.string, "LatentSeqMMFlowModel")) return error.NotIdol;
    const args = den.object.get("args") orelse return error.BadConfig;
    if (args != .object) return error.BadConfig;
    const want = .{ .{ "q_token_length", latent_rows }, .{ "in_channels", latent_channels }, .{ "cam_channels", camera_channels }, .{ "model_channels", hidden }, .{ "cond_channels", cond_channels }, .{ "cond2_channels", cond2_channels }, .{ "num_blocks", layers }, .{ "num_refiner_blocks", refiners }, .{ "num_heads", heads }, .{ "mlp_ratio", 4 } };
    inline for (want) |kv| {
        const v = args.object.get(kv[0]) orelse return error.BadConfig;
        if (v != .integer or v.integer != kv[1]) {
            std.log.warn("idol: denoiser {s} is not {d}; this family runs the released geometry only", .{ kv[0], kv[1] });
            return error.UnsupportedGeometry;
        }
    }
}

const sample =
    \\{"denoiser": {"name": "LatentSeqMMFlowModel", "args": {"q_token_length": 8192, "in_channels": 16,
    \\ "cam_channels": 5, "model_channels": 1024, "cond_channels": 1280, "cond2_channels": 128, "num_blocks": 24,
    \\ "num_refiner_blocks": 2, "num_heads": 16, "mlp_ratio": 4}}, "image_size": 1024, "max_voxel_level": 8}
;

test "the released denoiser config passes; another depth, width or class does not" {
    const a = std.testing.allocator;
    try check(a, sample);
    const deeper = try std.mem.replaceOwned(u8, a, sample, "\"num_blocks\": 24", "\"num_blocks\": 12");
    defer a.free(deeper);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, deeper));
    const missing = try std.mem.replaceOwned(u8, a, sample, "\"mlp_ratio\": 4", "\"mlp\": 4");
    defer a.free(missing);
    try std.testing.expectError(error.BadConfig, check(a, missing));
    const other = try std.mem.replaceOwned(u8, a, sample, "LatentSeqMMFlowModel", "LatentFlowModel");
    defer a.free(other);
    try std.testing.expectError(error.NotIdol, check(a, other));
    try std.testing.expectError(error.BadConfig, check(a, "{\"decoder\": {}}"));
}
