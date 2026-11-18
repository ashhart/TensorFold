//! What Hy4-flash refuses before any work: image and video parts and --vision, until its tower runs in the engine.
const std = @import("std");

pub const vision_flag = "--vision: Hy4-flash's vision tower has no native kernels yet; serve it without --vision";
pub const image_part = "image input: Hy4-flash's vision tower has no native kernels yet, so this server takes text parts only";
pub const video_part = "video input: Hy4-flash's vision tower has no native kernels yet, so this server takes text parts only";
pub const other_part = "this content part type is not supported: Hy4-flash takes text parts only for now";

/// The refusal for --vision, or null when it was not asked for.
pub fn visionFlag(requested: bool) ?[]const u8 {
    return if (requested) vision_flag else null;
}

fn oneOf(name: []const u8, comptime list: []const []const u8) bool {
    inline for (list) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

/// One content part's refusal, or null for text.
fn part(p: std.json.Value) ?[]const u8 {
    if (p == .string) return null;
    if (p != .object) return other_part;
    const kind = p.object.get("type") orelse return other_part;
    if (kind != .string) return other_part;
    if (oneOf(kind.string, &.{ "text", "input_text" })) return null;
    if (oneOf(kind.string, &.{ "image", "input_image", "image_url" })) return image_part;
    if (oneOf(kind.string, &.{ "video", "input_video", "video_url" })) return video_part;
    return other_part;
}

/// The first message content part the engine cannot serve, or null when every part is text.
pub fn parts(messages: std.json.Value) ?[]const u8 {
    if (messages != .array) return null;
    for (messages.array.items) |m| {
        if (m != .object) continue;
        const content = m.object.get("content") orelse continue;
        if (content != .array) continue; // a plain string is text
        for (content.array.items) |p| if (part(p)) |why| return why;
    }
    return null;
}

const t = std.testing;

fn first(json: []const u8) !?[]const u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    return parts(parsed.value);
}

test "text messages pass, as a string and as text parts" {
    try t.expect(try first("[{\"role\":\"user\",\"content\":\"hi\"}]") == null);
    try t.expect(try first("[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}]") == null);
    try t.expect(visionFlag(false) == null);
}

test "an image part, a video part, another part and --vision are each refused by name" {
    try t.expectEqualStrings(image_part, (try first("[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"x\"}}]}]")).?);
    try t.expectEqualStrings(video_part, (try first("[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"video_url\"}]}]")).?);
    try t.expectEqualStrings(other_part, (try first("[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\"}]}]")).?);
    try t.expectEqualStrings(other_part, (try first("[{\"role\":\"user\",\"content\":[42]}]")).?);
    try t.expectEqualStrings(vision_flag, visionFlag(true).?);
}
