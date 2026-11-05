//! Validate untrusted index structure and portable shard names before any filesystem operation.
const std = @import("std");

pub const max_bytes = 64 << 20;

pub const plainName = @import("checkpoint_config").index.plainName;
pub const shardName = @import("checkpoint_config").index.shardName;
pub const weightMap = @import("checkpoint_config").index.weightMap;
pub const plainWeightMap = @import("checkpoint_config").index.plainWeightMap;

test "portable shard names exclude path and Windows device aliases" {
    const bad = [_][]const u8{
        "",                 ".",         "..",        ".. ",     "model.",    "model ",       "model\x00suffix", "model\nname",
        "../model",         "sub/model", "..\\model", "C:model", "C:\\model", "model:stream", "CON",             "nul.safetensors",
        "LPT1.safetensors", "com9",      "file?",     "file*",
    };
    for (bad) |name| try std.testing.expect(!plainName(name));
    for ([_][]const u8{ "model.safetensors", "model-00001-of-00002.safetensors", "model.v1", "unicode-λ" }) |name|
        try std.testing.expect(plainName(name));
}

test "a shard name may name a folder inside the model folder, each part plain and none starting with '.'" {
    const bad = [_][]const u8{
        "",                   "/model.safetensors", "sub/",       "sub//model",  "./model",      "sub/./model",    "sub/../model",
        "../sub",             "sub/..",             "sub\\model", "C:/model",    "CON/model",    "sub/CON",        "sub./model",
        "sub /model",         "sub/model ",         "sub/m\x00x", "a/b/../../x", ".cache/model", "sub/.hid/model", "sub/.model",
        "tensors/native-QB.",
    };
    for (bad) |name| try std.testing.expect(!shardName(name));
    for ([_][]const u8{ "model.safetensors", "tensors/native-QB.safetensors", "a/b/c.safetensors" }) |name|
        try std.testing.expect(shardName(name));
}
