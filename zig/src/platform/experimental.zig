//! Whole-model release gates remain separate from host compilation and kernel availability.
const std = @import("std");
pub const deepseek_message = "DeepSeek-V4.1 CUDA is experimental in 1.1.1 while whole-model gate #827 is pending; set TF_EXPERIMENTAL=deepseek to opt in";

pub fn refusal(model_type: []const u8, value: ?[]const u8) ?[]const u8 {
    const deepseek = std.mem.eql(u8, model_type, "deepseek_v4") or std.mem.eql(u8, model_type, "deepseek_v41");
    if (!deepseek) return null;
    if (value) |v| if (std.mem.eql(u8, v, "deepseek")) return null;
    return deepseek_message;
}
