//! The HIP loader's configuration: the family's geometry (../config.zig) with the checkpoint's quantization table.

const std = @import("std");
const quant = @import("core").quant;
const shared = @import("../config.zig");

pub const Error = shared.Error || error{UnsupportedQuantization};

/// One affine width: MLX groups of `group` weights share a scale and a bias, `bits` per weight.
pub const Width = quant.mlx.Width;

pub const Spec = shared.Spec;

pub const Config = struct {
    arena: std.heap.ArenaAllocator,
    spec: Spec,
    quant: quant.Config,
    /// `tie_word_embeddings`: the output head is the embedding.
    tied: bool,

    pub fn deinit(c: *Config) void {
        c.arena.deinit();
        c.* = undefined;
    }

    /// The config of the checkpoint directory `dir`.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// `load()`'s reading of config.json: top level or `text_config`, model_type qwen3_5 or qwen3_5_moe.
    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const doc = try shared.declared(a, bytes);
        const tower = try shared.textTower(doc);
        const table = try quant.detect(a, .{ .root = doc, .quantization = try shared.pickObject(doc, tower, "quantization") });
        const width = try table.width("");
        const spec = try shared.geometry(tower, width.bits, width.group);
        return .{ .arena = arena, .spec = spec, .quant = table, .tied = shared.isTied(doc, tower) };
    }
};

const sample =
    \\{"model_type": "qwen3_5", "tie_word_embeddings": false,
    \\ "quantization": {"group_size": 64, "bits": 4, "mode": "affine",
    \\   "language_model.model.layers.0.mlp.gate": {"group_size": 32, "bits": 8},
    \\   "vision_tower.x": {"group_size": 64, "bits": 7}},
    \\ "text_config": {"hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 8,
    \\   "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
    \\   "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
    \\   "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 248320,
    \\   "rms_norm_eps": 1e-06, "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
    \\   "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\     "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    \\   "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}}
;

test "the config carries the geometry, the quantization width and the tying" {
    var c = try Config.parse(std.testing.allocator, sample);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 4096), c.spec.hidden);
    try std.testing.expectEqual(@as(u8, 4), c.spec.bits);
    try std.testing.expectEqual(@as(u16, 64), c.spec.group);
    try std.testing.expect(!c.tied);
}

test "per-tensor widths override the global one, and a bad entry fails only when used" {
    var c = try Config.parse(std.testing.allocator, sample);
    defer c.deinit();
    try std.testing.expectEqual(Width{ .bits = 8, .group = 32 }, try c.quant.width("language_model.model.layers.0.mlp.gate"));
    try std.testing.expectEqual(Width{ .bits = 4, .group = 64 }, try c.quant.width("language_model.model.layers.1.mlp.gate"));
    try std.testing.expectError(error.UnsupportedQuantization, c.quant.width("vision_tower.x"));
}

test "the Python loader's refusals of a quantization" {
    const a = std.testing.allocator;
    const bits = try std.mem.replaceOwned(u8, a, sample, "\"bits\": 4,", "\"bits\": 7,");
    defer a.free(bits);
    try std.testing.expectError(error.UnsupportedQuantization, Config.parse(a, bits));
    const gptq = try std.mem.replaceOwned(u8, a, sample, "\"mode\": \"affine\"", "\"mode\": \"gptq\"");
    defer a.free(gptq);
    try std.testing.expectError(error.UnsupportedQuantization, Config.parse(a, gptq));
}
