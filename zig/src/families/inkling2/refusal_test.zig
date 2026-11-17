//! Image parts and --vision are refused with their own words; text parts, strings and plain content pass.
const std = @import("std");
const refusal = @import("refusal.zig");

const t = std.testing;

fn check(json: []const u8) ?[]const u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, t.allocator, json, .{}) catch unreachable;
    defer parsed.deinit();
    return refusal.parts(parsed.value);
}

test "every image part type the template reads is refused, in any message and in tool results" {
    for ([_][]const u8{ "image", "input_image", "image_url" }) |kind| {
        var buf: [256]u8 = undefined;
        const user = try std.fmt.bufPrint(&buf, "[{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"a\"}},{{\"type\":\"{s}\",\"image_url\":{{\"url\":\"data:,\"}}}}]}}]", .{kind});
        try t.expectEqualStrings(refusal.image_part, check(user).?);
        const tool = try std.fmt.bufPrint(&buf, "[{{\"role\":\"user\",\"content\":\"q\"}},{{\"role\":\"tool\",\"content\":[{{\"type\":\"{s}\"}}]}}]", .{kind});
        try t.expectEqualStrings(refusal.image_part, check(tool).?);
    }
}

test "audio and unknown part types are refused; text, plain strings and string parts pass" {
    try t.expectEqualStrings(refusal.audio_part, check("[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\"}]}]").?);
    try t.expectEqualStrings(refusal.other_part, check("[{\"role\":\"user\",\"content\":[{\"type\":\"video\"}]}]").?);
    try t.expectEqualStrings(refusal.other_part, check("[{\"role\":\"user\",\"content\":[{\"type\":7}]}]").?);
    try t.expect(check("[{\"role\":\"user\",\"content\":\"hello\"}]") == null);
    try t.expect(check("[{\"role\":\"user\",\"content\":[\"a\",{\"text\":\"b\"},{\"type\":\"input_text\",\"text\":\"c\"}]}]") == null);
    try t.expect(check("[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[]}]") == null);
    try t.expect(refusal.parts(.null) == null);
}

test "a part without a type is read as text, as the chat template reads it" {
    try t.expect(check("[{\"role\":\"user\",\"content\":[{\"image_url\":{\"url\":\"data:,\"}}]}]") == null);
    try t.expect(check("[{\"role\":\"user\",\"content\":[{}]}]") == null);
}

test "--vision is refused, its absence is not" {
    try t.expectEqualStrings(refusal.vision_flag, refusal.visionFlag(true).?);
    try t.expect(refusal.visionFlag(false) == null);
}
