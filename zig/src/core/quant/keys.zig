//! Override keys by module: every spelling checkpoints and loaders give one projection reduces to one canonical key.

const std = @import("std");

/// The prefixes a module path may carry, each stripped at most once and in this order.
const prefixes = [_][]const u8{ "model.language_model.", "language_model.", "text_model.", "model." };

/// The module `key` names, without a trailing `.weight` or the known prefixes; a slice of `key`.
pub fn canonical(key: []const u8) []const u8 {
    var k = key;
    if (std.mem.endsWith(u8, k, ".weight")) k = k[0 .. k.len - ".weight".len];
    for (prefixes) |prefix| {
        if (std.mem.startsWith(u8, k, prefix)) k = k[prefix.len..];
    }
    return k;
}

test "every prefix form and a trailing weight name one module" {
    for ([_][]const u8{
        "layers.0.mlp.down_proj",
        "model.layers.0.mlp.down_proj",
        "language_model.model.layers.0.mlp.down_proj",
        "language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.model.layers.0.mlp.down_proj",
        "text_model.layers.0.mlp.down_proj",
        "text_model.model.layers.0.mlp.down_proj",
        "layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
    }) |key| try std.testing.expectEqualStrings("layers.0.mlp.down_proj", canonical(key));
}

test "multi-token prediction and vision modules keep their own names" {
    for ([_][]const u8{ "mtp.layers.0.mlp.gate_proj", "model.mtp.layers.0.mlp.gate_proj", "language_model.mtp.layers.0.mlp.gate_proj", "language_model.model.mtp.layers.0.mlp.gate_proj.weight" }) |key|
        try std.testing.expectEqualStrings("mtp.layers.0.mlp.gate_proj", canonical(key));
    try std.testing.expectEqualStrings("vision_tower.blocks.0.attn.qkv", canonical("vision_tower.blocks.0.attn.qkv.weight"));
    try std.testing.expectEqualStrings("visual.blocks.0.attn.qkv", canonical("model.visual.blocks.0.attn.qkv"));
    try std.testing.expectEqualStrings("lm_head", canonical("language_model.lm_head"));
    // a prefix only counts at the start and a suffix only at the end
    try std.testing.expectEqualStrings("layers.0.weight_proj", canonical("layers.0.weight_proj"));
    try std.testing.expectEqualStrings("vision_tower.model.x", canonical("vision_tower.model.x"));
}
