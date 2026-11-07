//! Explicit admission for plain dense BF16 weights; quantized formats keep their existing parser.
const std = @import("std");
pub fn admit(root: std.json.ObjectMap, text: std.json.ObjectMap, moe: bool) !void {
    if (moe) return error.UnsupportedQwenQuantization;
    var found = false;
    for ([_]std.json.ObjectMap{ root, text }) |o| for ([_][]const u8{ "dtype", "torch_dtype" }) |key| {
        if (o.get(key)) |v| {
            if (v != .string or !std.mem.eql(u8, v.string, "bfloat16")) return error.UnsupportedQwenQuantization;
            found = true;
        }
    };
    if (!found) return error.UnsupportedQwenQuantization;
}

/// These modes need their own BF16 qualification; other formats retain their current admission.
pub fn options(bf16: bool, image: bool, int8: bool, learning: bool) !void {
    if (!bf16) return;
    if (image) return error.QwenBf16VisionUnqualified;
    if (int8 or learning) return error.UnsupportedQwenBf16Mode;
}

/// Plain HF GDN scales can be FP32; other formats retain their stored BF16 requirement.
pub fn normIsF32(t: @import("core").checkpoint.Tensor, allow_f32: bool, width: usize) !bool {
    if (allow_f32 and t.is(.f32, &.{width})) return true;
    if (t.is(.bf16, &.{width})) return false;
    return error.UnexpectedTensor;
}
