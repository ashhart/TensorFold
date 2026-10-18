//! The actual release admission refuses unavailable DeepSeek backends independently of the environment.
const std = @import("std");
const gate = @import("release_refusal");
const env = @import("policy_env");
const expected = "DeepSeek-V4.1 is not served by this release's CUDA server; its whole-model gate is #827";
fn check(value: ?[:0]const u8) !void {
    env.value = value;
    defer env.value = null;
    for ([_][]const u8{ "deepseek_v4", "deepseek_v41" }) |name| {
        try std.testing.expectEqualStrings(expected, gate.backend(.{ .dir = "not-opened", .model_type = name }) orelse "");
        try std.testing.expectEqualStrings(expected, gate.backend(.{ .dir = "another-unread-checkpoint", .model_type = name, .policy = "unsupported", .mtp_drafts = 32 }) orelse "");
    }
}
test "DeepSeek release: TF_EXPERIMENTAL unset is a constant refusal before device work" {
    try check(null);
}
test "DeepSeek release: TF_EXPERIMENTAL=1 is the same constant refusal" {
    try check("1");
}
test "DeepSeek release: TF_EXPERIMENTAL=deepseek is the same constant refusal" {
    try check("deepseek");
}
test "DeepSeek release: other CUDA admission stays available" {
    env.value = "deepseek";
    defer env.value = null;
    try std.testing.expect(gate.backend(.{ .dir = "not-opened", .model_type = "nemotron_h" }) == null);
}
