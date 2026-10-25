//! Drama 3's default-voice prompt as the reference builds it; each literal is tokenized alone to keep seam ids.
const std = @import("std");
const model_text = @import("model_text.zig");
const pyrepr = @import("pyrepr.zig");
const Allocator = std.mem.Allocator;

/// ``REQUEST_ERROR_PREFIX``: every refusal the model's request lowering writes starts with it.
pub const error_prefix = "Drama 3 request:";

pub const tts_instruction = "convert the provided text to speech";
pub const message_end = "<|im_end|>\n";
pub const generation_start = "<|im_start|>assistant\n<|voice|>";

/// Only the builder writes these; other special-token text in the caller's input is content.
pub const structural_tokens = [_][]const u8{ "<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:" };

/// The tokenizer spells <|speaker:0|> to <|speaker:15|>.
pub const max_speaker_id = 15;

/// The model's sampling defaults (``speech_request.py``), not the generic chat defaults.
pub const temperature: f32 = 1.0;
pub const top_p: f32 = 0.9;
pub const top_k: u32 = 30;

/// The segments one prompt is encoded from, in order.
pub const segment_count = 7;

/// One frame row a prompt token, the token in the text column.
pub fn rows(a: Allocator, prompt: []const u32) error{ OutOfMemory, InvalidToken }![]const [11]i32 {
    const frames = try a.alloc([11]i32, prompt.len);
    for (frames, prompt) |*row, token| {
        row.* = @splat(0);
        row[0] = std.math.cast(i32, token) orelse return error.InvalidToken;
    }
    return frames;
}

/// The Unicode White_Space property (``UNICODE_WHITE_SPACE``); U+001C..U+001F are not in it.
pub fn isWhiteSpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// ``text.lstrip(UNICODE_WHITE_SPACE)``; ``text`` is valid UTF-8.
pub fn trimStart(text: []const u8) []const u8 {
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (true) {
        const at = it.i;
        const cp = it.nextCodepoint() orelse return text[text.len..];
        if (!isWhiteSpace(cp)) return text[at..];
    }
}

const Tag = struct { digits: []const u8, end: usize };

/// ``SPEAKER_TAG_PATTERN.match(text, start)``: ``<|speaker:`` one or more ASCII digits ``|>``.
fn tagAt(text: []const u8, start: usize) ?Tag {
    const open = "<|speaker:";
    if (!std.mem.startsWith(u8, text[start..], open)) return null;
    const first = start + open.len;
    var i = first;
    while (i < text.len and std.ascii.isDigit(text[i])) i += 1;
    if (i == first or !std.mem.startsWith(u8, text[i..], "|>")) return null;
    return .{ .digits = text[first..i], .end = i + 2 };
}

/// ``SPEAKER_IDS``: the decimal spellings 0..15, no leading zero.
fn speakerId(digits: []const u8) ?u8 {
    if (digits.len == 0 or digits.len > 2 or (digits.len == 2 and digits[0] == '0')) return null;
    const n = std.fmt.parseInt(u8, digits, 10) catch return null;
    return if (n <= max_speaker_id) n else null;
}

/// Why input cannot be spoken, in the reference's words and order, or null.
pub fn problem(a: Allocator, input: []const u8) Allocator.Error!?[]const u8 {
    // speech_request.py: well-formed tags naming a speaker other than 0 (malformed ones are the builder's)
    var others: u16 = 0;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, input, at, "<|speaker:")) |start| {
        if (tagAt(input, start)) |tag| {
            if (speakerId(tag.digits)) |n| if (n != 0) {
                others |= @as(u16, 1) << @intCast(n);
            };
            at = tag.end;
        } else at = start + 1;
    }
    if (others != 0) {
        var list: std.ArrayList(u8) = .empty;
        try list.appendSlice(a, error_prefix ++ " only speaker 0 is supported; got speaker tags [");
        var first = true;
        for (1..max_speaker_id + 1) |n| if (others & (@as(u16, 1) << @intCast(n)) != 0) {
            if (!first) try list.appendSlice(a, ", ");
            try list.print(a, "{d}", .{n});
            first = false;
        };
        try list.append(a, ']');
        return list.items;
    }
    // prompt.py's validate_caller_text(text, "input")
    if (trimStart(input).len == 0) return error_prefix ++ " input: text must not be empty";
    for (structural_tokens) |token| if (std.mem.indexOf(u8, input, token) != null)
        return try std.fmt.allocPrint(a, error_prefix ++ " input: {s} is written by the prompt builder and must not appear in caller text", .{token});
    at = 0;
    while (std.mem.indexOfPos(u8, input, at, "<|speaker")) |start| {
        const tag = tagAt(input, start);
        if (tag == null or speakerId(tag.?.digits) == null)
            return try std.fmt.allocPrint(a, error_prefix ++ " input: speaker tags must be written <|speaker:N|> with N in 0..{d} and no leading zero; got {s}", .{ max_speaker_id, try pyrepr.repr(a, .{ .string = input }) });
        at = start + "<|speaker".len;
    }
    return null;
}

/// ``with_speaker_tag(text, 0)``: speaker 0 speaks unless a tag leads the text after white space.
pub fn tagged(a: Allocator, input: []const u8) Allocator.Error![]const u8 {
    if (tagAt(trimStart(input), 0) != null) return input;
    return std.mem.concat(a, u8, &.{ "<|speaker:0|> ", input });
}

/// The prompt's segments for a checked ``input`` (``problem`` returned null), each tokenized on its own.
pub fn segments(a: Allocator, input: []const u8) Allocator.Error![segment_count][]const u8 {
    return .{ "<|im_start|>system\n", tts_instruction, message_end, "<|im_start|>user\n", try tagged(a, input), message_end, generation_start };
}

/// The prompt's ids: each segment encoded with the server's tokenizer as ``tokenizer.encode(segment).ids`` encodes it.
pub fn encode(a: Allocator, text: model_text.Text, input: []const u8) model_text.Error![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    for (try segments(a, input)) |segment| try ids.appendSlice(a, try text.encode(a, segment, true));
    return ids.items;
}

test "white space is the Unicode property, not Python's str.strip set" {
    try std.testing.expect(isWhiteSpace(0x3000) and isWhiteSpace(0x85) and isWhiteSpace(0x0d) and isWhiteSpace(0x200a));
    for ([_]u21{ 0x1c, 0x1f, 0x200b, 0x180e, 0xfeff }) |cp| try std.testing.expect(!isWhiteSpace(cp));
    try std.testing.expectEqualStrings("x \u{3000}", trimStart("\u{3000}\n x \u{3000}"));
    try std.testing.expectEqualStrings("\x1cx", trimStart(" \x1cx"));
}

test "speaker 0 tags untagged text; a leading tag after white space is kept" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "hello", "<|speaker:0|> hello" },
        .{ "<|speaker:0|> hi", "<|speaker:0|> hi" },
        .{ "\n<|speaker:0|> hi", "\n<|speaker:0|> hi" },
        .{ "\u{3000}<|speaker:0|> hi", "\u{3000}<|speaker:0|> hi" },
        .{ "a <|speaker:0|> b", "<|speaker:0|> a <|speaker:0|> b" },
        .{ "\x1c<|speaker:0|> hi", "<|speaker:0|> \x1c<|speaker:0|> hi" },
        .{ "\x1f", "<|speaker:0|> \x1f" },
    };
    for (cases) |c| {
        try std.testing.expect(try problem(a, c[0]) == null);
        try std.testing.expectEqualStrings(c[1], try tagged(a, c[0]));
    }
}

test "input the reference refuses" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const malformed = [_][]const u8{ "<|speaker:99|> x", "<|speaker:16|> x", "a <|speaker:99|> b", "<|speaker:01|> x", "<|speaker:00|> x", "<|speaker:|> x", "<|speaker0|> x", "hi <|speaker", "<|speaker:abc|> x", "<|speaker:0> x", "  <|speaker:abc|> hi" };
    for (malformed) |text| {
        const message = (try problem(a, text)) orelse return error.TestUnexpectedResult;
        try std.testing.expect(std.mem.startsWith(u8, message, "Drama 3 request: input: speaker tags must be written <|speaker:N|> with N in 0..15 and no leading zero; got "));
    }
    try std.testing.expectEqualStrings("Drama 3 request: only speaker 0 is supported; got speaker tags [1, 15]", (try problem(a, "<|speaker:15|> a <|speaker:1|> b <|speaker:1|> c <|speaker:0|> d")).?);
    // other speakers are refused before malformed tags, as speech_request.py checks them first
    try std.testing.expectEqualStrings("Drama 3 request: only speaker 0 is supported; got speaker tags [2]", (try problem(a, "<|speaker:2|> x <|speaker:01|>")).?);
    for ([_][]const u8{ "", " ", "\u{3000}\u{2028}\t" }) |blank|
        try std.testing.expectEqualStrings("Drama 3 request: input: text must not be empty", (try problem(a, blank)).?);
    for (structural_tokens) |token| {
        const message = (try problem(a, try std.mem.concat(a, u8, &.{ "say ", token, " it" }))).?;
        try std.testing.expect(std.mem.indexOf(u8, message, " is written by the prompt builder and must not appear in caller text") != null);
    }
    for ([_][]const u8{ "<|semantic:", "<|semantic:12|>" }) |text| try std.testing.expect(try problem(a, text) != null);
    for ([_][]const u8{ "hi <|audio_separator|> there", "<|fim_prefix|> hi", "<|speaker:0|> current part<|endofturn|>next part" }) |content|
        try std.testing.expect(try problem(a, content) == null);
}
