//! What Inkling 2 Small refuses before any work: image and audio parts and --vision, until the tower has kernels.
const std = @import("std");

pub const vision_flag = "--vision: Inkling 2 Small's vision tower has no native kernels yet; serve it without --vision";
pub const image_part = "image input: Inkling 2 Small's vision tower has no native kernels yet, so this server takes text parts only";
pub const audio_part = "audio input: Inkling 2 Small does not accept audio";
pub const other_part = "this content part type is not supported: Inkling 2 Small takes text parts only";

/// The refusal for --vision, or null when it was not asked for.
pub fn visionFlag(requested: bool) ?[]const u8 {
    return if (requested) vision_flag else null;
}

fn among(name: []const u8, list: []const []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

/// The first message content part the engine cannot serve (image, audio or any non-text type), or null.
pub fn parts(messages: std.json.Value) ?[]const u8 {
    if (messages != .array) return null;
    for (messages.array.items) |m| {
        const content = if (m == .object) m.object.get("content") orelse continue else continue;
        if (content != .array) continue;
        for (content.array.items) |p| {
            if (p == .string) continue;
            const kind = if (p == .object) p.object.get("type") orelse continue else return other_part;
            const name = if (kind == .string) kind.string else return other_part;
            if (among(name, &.{ "text", "input_text" })) continue;
            if (among(name, &.{ "image", "input_image", "image_url" })) return image_part;
            if (among(name, &.{ "audio", "input_audio", "audio_url" })) return audio_part;
            return other_part;
        }
    }
    return null;
}
