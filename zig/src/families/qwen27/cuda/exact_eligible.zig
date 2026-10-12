//! The exact-cache helpers support only the original BF16 fresh text arithmetic and state layout.
const lanes = @import("lanes");

pub fn text(kv8: bool, s: *const lanes.Stream) bool {
    // The server's assistant-header decide marker is inert in Qwen's nonlearning CUDA backend.
    return !kv8 and s.images.len == 0 and s.prompt_logprobs == null and s.decode_spans.len == 0 and
        !s.untaught and !s.reuse_failed;
}

test "FP8, images and changed prompt arithmetic decline every exact-cache treatment" {
    const std = @import("std");
    var s = try lanes.Stream.init(std.testing.allocator, .{ .id = "eligible", .prompt = &.{1}, .max_new = 4 });
    defer s.deinit(std.testing.allocator);
    try std.testing.expect(text(false, &s));
    try std.testing.expect(!text(true, &s));
    s.images = &.{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 0, .tokens = 1 }};
    try std.testing.expect(!text(false, &s));
    s.images = &.{};
    s.prompt_logprobs = 1;
    try std.testing.expect(!text(false, &s));
    s.prompt_logprobs = null;
    s.decode_spans = &.{.{ 0, 1 }};
    try std.testing.expect(!text(false, &s));
    s.decode_spans = &.{};
    s.untaught = true;
    try std.testing.expect(!text(false, &s));
    s.untaught = false;
    s.reuse_failed = true;
    try std.testing.expect(!text(false, &s));
}

test "ordinary chat assistant-header metadata does not change Qwen text eligibility" {
    const std = @import("std");
    const ids: [31]u32 = @splat(1);
    var s = try lanes.Stream.init(std.testing.allocator, .{ .id = "chat-marker", .prompt = &ids, .max_new = 256, .decide = 29 });
    defer s.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?u32, 29), s.decide);
    try std.testing.expect(text(false, &s));
    try std.testing.expect(!text(true, &s));
}
