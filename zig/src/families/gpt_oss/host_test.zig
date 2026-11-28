//! Host-only family bring-up, without native registration or a model execution claim.
const std = @import("std");
const family = @import("gpt_oss.zig");

test {
    _ = @import("config_test.zig");
    _ = @import("names_test.zig");
    _ = @import("arithmetic_test.zig");
    _ = @import("mxfp4_test.zig");
}

test "gpt-oss uses the existing Harmony splitter for analysis and final channels" {
    const text = "<|start|>assistant<|channel|>analysis<|message|>work<|end|><|start|>assistant<|channel|>final<|message|>42<|return|>";
    const reply = family.parseHarmony(text);
    try std.testing.expectEqualStrings("42", reply.content);
    try std.testing.expectEqualStrings("work", reply.reasoning.?);
    try std.testing.expectEqual(@intFromPtr(&@import("reply_text").parseHarmony), @intFromPtr(&family.parseHarmony));
}
