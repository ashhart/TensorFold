//! Explicit development opt-in while the shared whole-model release gate remains open.
const std = @import("std");
pub fn refusal(value: ?[]const u8) ?[]const u8 {
    if (value) |v| if (std.mem.eql(u8, v, "deepseek")) return null;
    return "DeepSeek-V4.1 Metal is experimental while whole-model gate #827 is pending; set TF_EXPERIMENTAL=deepseek to opt in";
}
test "Metal DeepSeek admission requires the exact experimental opt-in" {
    for ([_]?[]const u8{ null, "", "1", "deepseek-v41", "DeepSeek" }) |value| try std.testing.expect(refusal(value) != null);
    try std.testing.expect(refusal("deepseek") == null);
}
