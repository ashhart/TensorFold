//! The package's ASR prompt: chat_template.json with one audio part.
const std = @import("std");

pub const audio_pad = "<|audio_pad|>";
pub const asr_text = "<asr_text>";

/// The prompt text for a clip of `rows` audio rows: an empty system turn, the audio, the assistant's turn opened.
pub fn text(gpa: std.mem.Allocator, rows: usize, context: []const u8, language: ?[]const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, "<|im_start|>system\n");
    try out.appendSlice(gpa, context);
    try out.appendSlice(gpa, "<|im_end|>\n<|im_start|>user\n<|audio_start|>");
    for (0..rows) |_| try out.appendSlice(gpa, audio_pad);
    try out.appendSlice(gpa, "<|audio_end|><|im_end|>\n<|im_start|>assistant\n");
    if (language) |l| {
        try out.appendSlice(gpa, "language ");
        try out.appendSlice(gpa, l);
        try out.appendSlice(gpa, asr_text);
    }
    return out.toOwnedSlice(gpa);
}

test "text: the template's turns around the audio rows, and a forced language" {
    const a = std.testing.allocator;
    const t = try text(a, 2, "", null);
    defer a.free(t);
    try std.testing.expectEqualStrings("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n", t);
    const f = try text(a, 0, "", "English");
    defer a.free(f);
    try std.testing.expect(std.mem.endsWith(u8, f, "assistant\nlanguage English<asr_text>"));
}
