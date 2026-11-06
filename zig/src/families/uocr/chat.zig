//! Unlimited-OCR-2's chat template, served when the checkpoint ships none (the release has no template).
const std = @import("std");

pub const model_type = "unlimited-ocr";
pub const source = @embedFile("chat_template.jinja");

/// The template for a checkpoint whose config.json (a parsed JSON value) names this family, else "".
pub fn builtIn(config: anytype) []const u8 {
    if (config != .object) return "";
    const kind = config.object.get("model_type") orelse return "";
    return if (kind == .string and std.mem.eql(u8, kind.string, model_type)) source else "";
}
