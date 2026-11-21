//! Built-in chat templates: Nemotron Parse's replaces the checkpoint's; Unlimited-OCR-2's fills in when it ships none.
const std = @import("std");

/// The template hf_text.zig serves in place of `checkpoint` (the checkpoint's own, "" when it has none), else null.
pub fn builtIn(model_config: std.json.Value, checkpoint: []const u8) ?[]const u8 {
    if (model_config != .object) return null;
    const kind = model_config.object.get("model_type") orelse return null;
    if (kind != .string) return null;
    // the bridge from server media parts to Parse's visual prefix (its OCR example passes images separately)
    if (std.mem.eql(u8, kind.string, "nemotron_parse")) return @embedFile("fixtures/nemotron_parse.jinja");
    const uocr = @import("uocr_chat");
    if (checkpoint.len == 0 and std.mem.eql(u8, kind.string, uocr.model_type)) return uocr.source;
    return null;
}

test "built-in templates: Parse's replaces the checkpoint's, Unlimited-OCR-2's only fills an empty one, others keep theirs" {
    const a = std.testing.allocator;
    const Case = struct { config: []const u8, checkpoint: []const u8, want: ?[]const u8 };
    const uocr = @import("uocr_chat");
    const parse = @embedFile("fixtures/nemotron_parse.jinja");
    for ([_]Case{
        .{ .config = "{\"model_type\":\"nemotron_parse\"}", .checkpoint = "", .want = parse },
        .{ .config = "{\"model_type\":\"nemotron_parse\"}", .checkpoint = "theirs", .want = parse },
        .{ .config = "{\"model_type\":\"unlimited-ocr\"}", .checkpoint = "", .want = uocr.source },
        .{ .config = "{\"model_type\":\"unlimited-ocr\"}", .checkpoint = "theirs", .want = null },
        .{ .config = "{\"model_type\":\"qwen3\"}", .checkpoint = "", .want = null },
        .{ .config = "{\"model_type\":\"qwen3\"}", .checkpoint = "theirs", .want = null },
        .{ .config = "{\"model_type\":7}", .checkpoint = "", .want = null },
        .{ .config = "[]", .checkpoint = "", .want = null },
    }) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, c.config, .{});
        defer parsed.deinit();
        const got = builtIn(parsed.value, c.checkpoint);
        if (c.want) |w| try std.testing.expectEqualStrings(w, got.?) else try std.testing.expect(got == null);
    }
}
