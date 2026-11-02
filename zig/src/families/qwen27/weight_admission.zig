//! Scalar roles added for native GGUF images require the image's config marker, never a tensor spelling alone.
const std = @import("std");
const DType = @import("../../core/safetensors.zig").DType;

pub fn gain(dtype: DType, gguf_image: bool) !DType {
    if (dtype == .bf16 or (gguf_image and dtype == .f32)) return dtype;
    return error.BadNativeTensor;
}

pub fn storedDecay(gguf_image: bool, direct: bool, logged: bool) !bool {
    if (direct and !gguf_image) return error.BadNativeTensor;
    if (direct and logged) return error.AmbiguousGdnDecay;
    return direct;
}

test "GGUF Qwen: F32 gains require an image marker while BF16 remains admitted" {
    const t = std.testing;
    try t.expectError(error.BadNativeTensor, gain(.f32, false));
    try t.expectEqual(DType.f32, try gain(.f32, true));
    for ([_]bool{ false, true }) |marked| {
        try t.expectEqual(DType.bf16, try gain(.bf16, marked));
        try t.expectError(error.BadNativeTensor, gain(.f16, marked));
        try t.expectError(error.BadNativeTensor, gain(.u32, marked));
    }
}

test "GGUF Qwen: stored ssm_a requires an image marker even beside legacy A_log" {
    const t = std.testing;
    for ([_]bool{ false, true }) |logged| {
        try t.expectError(error.BadNativeTensor, storedDecay(false, true, logged));
        try t.expect(!try storedDecay(false, false, logged));
    }
    try t.expect(try storedDecay(true, true, false));
    try t.expect(!try storedDecay(true, false, true));
    try t.expectError(error.AmbiguousGdnDecay, storedDecay(true, true, true));
}
