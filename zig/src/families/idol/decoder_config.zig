//! Idol's two decoders' geometry, and their config.json check: the octree decoder and the Gaussian decoder.
const std = @import("std");

pub const hidden = 1024;
pub const heads = 16;
pub const ffn = 4 * hidden;
pub const octree_blocks = 4;
pub const gs_blocks = 16;
/// The latent both decoders attend to: the denoiser's final sample [8192, 16].
pub const context_rows = 8192;
pub const context_channels = 16;
/// An octree cell's children: the decoder's 8 logits a parent.
pub const children = 8;
pub const max_level = 8;
pub const per_anchor = 32;
/// The Gaussian decoder's outputs an anchor: offset 3, colour 3, scale 3, rotation 4, opacity 1, offset scale 1.
pub const raw_width = per_anchor * 15;
/// PcdAbsolutePositionEmbedderv2's frequencies an axis: channels // 3 // 2.
pub const pe_freqs = hidden / 3 / 2;
/// SinusoidalEmbedder's frequency vector (the level embedding).
pub const level_freqs = 256;
pub const level_period: f32 = 1024;
pub const block_eps: f32 = 1e-6;
pub const out_eps: f32 = 1e-5;
/// IdolPipeline.decode's Gaussian counts: 32,768 to 1,048,576, a multiple of 32.
pub const min_gaussians = 32768;
pub const max_gaussians = 1 << 20;
pub const max_anchors = max_gaussians / per_anchor;

/// representation_config: the activations' learning-rate multipliers, offsets and biases, as python's doubles.
pub const Representation = struct {
    lr_xyz: f64,
    lr_features_dc: f64,
    lr_scaling: f64,
    lr_rotation: f64,
    lr_opacity: f64,
    perturbe_size: f64,
    filter: f64,
    scaling_bias: f64,
    opacity_bias: f64,
};

/// The released representation_config, for tests.
pub const released: Representation = .{ .lr_xyz = 1, .lr_features_dc = 1, .lr_scaling = 1, .lr_rotation = 0.1, .lr_opacity = 1, .perturbe_size = 1.5, .filter = 0.0009, .scaling_bias = 0.004, .opacity_bias = 0.1 };

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadConfig,
    };
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadConfig;
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadConfig;
}

fn args(root: std.json.ObjectMap, key: []const u8, class: []const u8) !std.json.ObjectMap {
    const d = try object(try field(root, key));
    const name = try field(d, "name");
    if (name != .string or !std.mem.eql(u8, name.string, class)) return error.NotIdol;
    return object(try field(d, "args"));
}

fn expect(o: std.json.ObjectMap, comptime want: anytype) !void {
    inline for (want) |kv| {
        const v = try field(o, kv[0]);
        if (v != .integer or v.integer != kv[1]) {
            std.log.warn("idol: decoder {s} is not {d}; this family runs the released geometry only", .{ kv[0], kv[1] });
            return error.UnsupportedGeometry;
        }
    }
}

/// The snapshot config's decoders checked against the released geometry; returns their representation_config.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !Representation {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    const root = try object(doc.value);
    const oct = try args(root, "decoder", "OctreeProbabilityFixedlenDecoder");
    try expect(oct, .{ .{ "model_channels", hidden }, .{ "cond_channels", context_channels }, .{ "num_blocks", octree_blocks }, .{ "num_heads", heads }, .{ "mlp_ratio", 4 } });
    const gs = try args(root, "decoder_gs", "GaussianFixedlenDecoder");
    try expect(gs, .{ .{ "in_channels", 3 }, .{ "model_channels", hidden }, .{ "cond_channels", context_channels }, .{ "num_blocks", gs_blocks }, .{ "num_heads", heads }, .{ "mlp_ratio", 4 } });
    try expect(root, .{.{ "max_voxel_level", max_level }});
    const rc = try object(try field(gs, "representation_config"));
    try expect(rc, .{.{ "num_gaussians", per_anchor }});
    const lr = try object(try field(rc, "lr"));
    const f = struct {
        fn f(o: std.json.ObjectMap, k: []const u8) !f64 {
            return number(try field(o, k));
        }
    }.f;
    return .{
        .lr_xyz = try f(lr, "_xyz"),
        .lr_features_dc = try f(lr, "_features_dc"),
        .lr_scaling = try f(lr, "_scaling"),
        .lr_rotation = try f(lr, "_rotation"),
        .lr_opacity = try f(lr, "_opacity"),
        .perturbe_size = try f(rc, "perturbe_size"),
        .filter = try f(rc, "3d_filter_kernel_size"),
        .scaling_bias = try f(rc, "scaling_bias"),
        .opacity_bias = try f(rc, "opacity_bias"),
    };
}

/// Anchors for `gaussians` as IdolPipeline.decode counts them: max(1, gaussians // 32), within the card's range.
pub fn anchors(gaussians: usize) !usize {
    if (gaussians < min_gaussians or gaussians > max_gaussians) return error.GaussianCount;
    return gaussians / per_anchor;
}

const sample =
    \\{"decoder": {"name": "OctreeProbabilityFixedlenDecoder", "args": {"model_channels": 1024, "cond_channels": 16,
    \\ "num_blocks": 4, "num_heads": 16, "mlp_ratio": 4}}, "decoder_gs": {"name": "GaussianFixedlenDecoder", "args":
    \\ {"in_channels": 3, "model_channels": 1024, "cond_channels": 16, "num_blocks": 16, "num_heads": 16, "mlp_ratio": 4,
    \\ "representation_config": {"num_gaussians": 32, "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0,
    \\ "_scaling": 1.0, "_rotation": 0.1}, "perturbe_size": 1.5, "3d_filter_kernel_size": 0.0009,
    \\ "scaling_bias": 0.004, "opacity_bias": 0.1}}}, "image_size": 1024, "max_voxel_level": 8}
;

test "the released decoders pass with their representation_config; another depth or class does not" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(released, try check(a, sample));
    const deeper = try std.mem.replaceOwned(u8, a, sample, "\"num_blocks\": 16", "\"num_blocks\": 12");
    defer a.free(deeper);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, deeper));
    const level = try std.mem.replaceOwned(u8, a, sample, "\"max_voxel_level\": 8", "\"max_voxel_level\": 9");
    defer a.free(level);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, level));
    const other = try std.mem.replaceOwned(u8, a, sample, "GaussianFixedlenDecoder", "GaussianDecoder");
    defer a.free(other);
    try std.testing.expectError(error.NotIdol, check(a, other));
    const missing = try std.mem.replaceOwned(u8, a, sample, "\"opacity_bias\"", "\"opacity\"");
    defer a.free(missing);
    try std.testing.expectError(error.BadConfig, check(a, missing));
}

test "anchors: the card's range, rounded down to whole anchors" {
    try std.testing.expectEqual(@as(usize, 8192), try anchors(262144));
    try std.testing.expectEqual(@as(usize, 1024), try anchors(32768 + 31));
    try std.testing.expectError(error.GaussianCount, anchors(1000));
    try std.testing.expectError(error.GaussianCount, anchors((1 << 20) + 32));
}
