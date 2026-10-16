//! Drama 3 prompt rows in the reference-bracket-v1 layout of SGLang-Omni's prompt.py, each part tokenized on its own.
const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");

const Tokenizer = core.tokenizer.Tokenizer;
const W = cfg.frame_width;

pub const tts_instruction = "convert the provided text to speech";
pub const reference_instruction = "convert the provided text to speech reference to the following:";
const message_end = "<|im_end|>\n";
const generation_start = "<|im_start|>assistant\n<|voice|>";

pub const Reference = struct {
    transcript: []const u8,
    codes: []const [cfg.codebooks]i32, // [frames][10]
};

/// prompt.py's UNICODE_WHITE_SPACE (the White_Space property).
fn isWhite(cp: u21) bool {
    return (cp >= 0x9 and cp <= 0xD) or cp == 0x20 or cp == 0x85 or cp == 0xA0 or cp == 0x1680 or
        (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2028 or cp == 0x2029 or cp == 0x202F or cp == 0x205F or cp == 0x3000;
}

fn lstripLen(s: []const u8) usize {
    var i: usize = 0;
    while (i < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return i;
        if (i + n > s.len) return i;
        const cp = std.unicode.utf8Decode(s[i..][0..n]) catch return i;
        if (!isWhite(cp)) return i;
        i += n;
    }
    return i;
}

/// Length of a leading `<|speaker:N|>` tag (N digits), or 0.
fn speakerTag(s: []const u8) usize {
    const head = "<|speaker:";
    if (!std.mem.startsWith(u8, s, head)) return 0;
    var i = head.len;
    while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    if (i == head.len) return 0;
    if (!std.mem.startsWith(u8, s[i..], "|>")) return 0;
    return i + 2;
}

/// with_speaker_tag(text, 0): tag text with speaker 0 unless a tag leads it after white space.
fn withSpeaker(a: std.mem.Allocator, text: []const u8) ![]u8 {
    if (speakerTag(text[lstripLen(text)..]) != 0) return a.dupe(u8, text);
    return std.fmt.allocPrint(a, "<|speaker:0|> {s}", .{text});
}

/// A reference transcript: the tag must follow "Text: " directly, so a tag behind leading white space moves in front.
fn refTranscript(a: std.mem.Allocator, transcript: []const u8) ![]u8 {
    const tagged = try withSpeaker(a, transcript);
    const lead = lstripLen(tagged);
    const body = tagged[lead..];
    const tl = speakerTag(body);
    return std.mem.concat(a, u8, &.{ body[0..tl], tagged[0..lead], body[tl..] });
}

pub fn build(gpa: std.mem.Allocator, tok: *const Tokenizer, text: []const u8, refs: []const Reference) ![][W]i32 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sem_begin = tok.specialTokenId("<|semantic:0|>") orelse return error.NoSemanticToken;
    var rows: std.ArrayList([W]i32) = .empty;
    errdefer rows.deinit(gpa);
    const Seg = union(enum) { text: []const u8, codes: []const [cfg.codebooks]i32 };
    var segs: std.ArrayList(Seg) = .empty;
    try segs.append(a, .{ .text = "<|im_start|>system\n" });
    try segs.append(a, .{ .text = if (refs.len > 0) reference_instruction else tts_instruction });
    for (refs, 0..) |r, i| {
        if (r.codes.len == 0) return error.EmptyReference;
        try segs.append(a, .{ .text = try std.fmt.allocPrint(a, "\n\n[Reference {d}]\nText: {s}\nSpeech: ", .{ i + 1, try refTranscript(a, r.transcript) }) });
        try segs.append(a, .{ .codes = r.codes });
    }
    try segs.append(a, .{ .text = message_end });
    try segs.append(a, .{ .text = "<|im_start|>user\n" });
    try segs.append(a, .{ .text = try withSpeaker(a, text) });
    try segs.append(a, .{ .text = message_end });
    try segs.append(a, .{ .text = generation_start });
    for (segs.items) |seg| switch (seg) {
        .text => |t| {
            const ids = try tok.encode(a, t);
            for (ids) |id| {
                var row: [W]i32 = @splat(0);
                row[0] = @intCast(id);
                try rows.append(gpa, row);
            }
        },
        .codes => |cs| for (cs) |f| {
            var row: [W]i32 = undefined;
            row[0] = @as(i32, @intCast(sem_begin)) + f[0];
            for (0..cfg.codebooks) |c| row[1 + c] = f[c];
            try rows.append(gpa, row);
        },
    };
    return rows.toOwnedSlice(gpa);
}

test "speaker tags as prompt.py places them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("<|speaker:0|> hi", try withSpeaker(a, "hi"));
    try std.testing.expectEqualStrings("  <|speaker:1|> hi", try withSpeaker(a, "  <|speaker:1|> hi"));
    try std.testing.expectEqualStrings("<|speaker:1|>   hi", try refTranscript(a, "  <|speaker:1|> hi"));
    try std.testing.expectEqualStrings("<|speaker:0|> We asked", try refTranscript(a, "We asked"));
}
