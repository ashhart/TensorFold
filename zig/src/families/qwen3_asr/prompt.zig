//! The package's ASR prompt (chat_template.json with one audio part) and the reply's "language X<asr_text>" split.
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

/// A reply split as parse_asr_output splits it: "language X<asr_text>body" gives X and body, else no language.
pub fn split(reply: []const u8) struct { language: []const u8, body: []const u8 } {
    const s = std.mem.trim(u8, reply, " \t\r\n");
    const at = std.mem.indexOf(u8, s, asr_text) orelse return .{ .language = "", .body = s };
    const meta = std.mem.trim(u8, s[0..at], " \t\r\n");
    const lang = if (std.mem.startsWith(u8, meta, "language ")) std.mem.trim(u8, meta["language ".len..], " ") else meta;
    if (std.mem.eql(u8, lang, "None")) return .{ .language = "", .body = "" };
    return .{ .language = lang, .body = std.mem.trim(u8, s[at + asr_text.len ..], " \t\r\n") };
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

test "split: language and body, and the silent clip" {
    const s = split("language English<asr_text>Mr. Quilter is the apostle.");
    try std.testing.expectEqualStrings("English", s.language);
    try std.testing.expectEqualStrings("Mr. Quilter is the apostle.", s.body);
    try std.testing.expectEqualStrings("", split("language None<asr_text>").body);
    try std.testing.expectEqualStrings("plain", split(" plain ").body);
}
