//! Nemotron Parse's chat template, which hf_text.zig loads in place of the checkpoint's.
const std = @import("std");

/// The bridge from server media parts to Parse's visual prefix (its OCR example passes images separately), else null.
pub fn template(model_config: std.json.Value) ?[]const u8 {
    if (model_config != .object) return null;
    const kind = model_config.object.get("model_type") orelse return null;
    if (kind != .string or !std.mem.eql(u8, kind.string, "nemotron_parse")) return null;
    return @embedFile("fixtures/nemotron_parse.jinja");
}
