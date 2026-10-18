//! Release policy branches use explicit inputs without mutating the process environment.
const std = @import("std");
const policy = @import("sys").experimental;
test "experimental family: DeepSeek defaults to refused until the exact opt-in" {
    for ([_][]const u8{ "deepseek_v4", "deepseek_v41" }) |name| {
        try std.testing.expect(policy.refusal(name, null) != null);
        try std.testing.expectEqualStrings(policy.deepseek_message, policy.refusal(name, null).?);
        for ([_][]const u8{ "", "1", "true", "DeepSeek", "deepseek,other" }) |value|
            try std.testing.expect(policy.refusal(name, value) != null);
        try std.testing.expect(policy.refusal(name, "deepseek") == null);
    }
    try std.testing.expect(policy.refusal("qwen3_5", null) == null);
}

test "experimental family: admission precedes CUDA model or asset work" {
    const common = @embedFile("src/native/cuda_refusal.zig");
    const shared = std.mem.indexOf(u8, common, "experimental.refusal(").?;
    try std.testing.expect(shared < std.mem.indexOf(u8, common, "if (o.mtp_drafts)").?);
    const server = @embedFile("src/families/deepseek_v41/serve_engine.zig");
    const entry = std.mem.indexOf(u8, server, "pub fn open(").?;
    const guard = std.mem.indexOfPos(u8, server, entry, "experimental.refusal(").?;
    try std.testing.expect(guard < std.mem.indexOfPos(u8, server, entry, "TF_DSV41_ASSETS").?);
    const model = @embedFile("src/families/deepseek_v41/cuda_model.zig");
    const open = std.mem.indexOf(u8, model, "pub fn open(").?;
    try std.testing.expect(std.mem.indexOfPos(u8, model, open, "experimental.refusal(").? < std.mem.indexOfPos(u8, model, open, "gpa.create(Model)").?);
}
