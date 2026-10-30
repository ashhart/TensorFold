//! Drama 3 prompt rows in the reference-bracket-v1 layout of SGLang-Omni's prompt.py, each part tokenized on its own.
const std = @import("std");
const cfg = @import("config.zig");

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

pub const Built = struct {
    frames: [][W]i32,
    /// End of the invariant reference prompt, before request text (0 without references).
    reference_rows: u32,
};

pub fn build(gpa: std.mem.Allocator, tok: anytype, text: []const u8, refs: []const Reference) !Built {
    if (text.len == 0 or text.len > 16384 or !std.unicode.utf8ValidateSlice(text) or refs.len > 4) return error.BadPrompt;
    for (refs) |r| {
        if (r.transcript.len == 0 or r.transcript.len > 4096 or !std.unicode.utf8ValidateSlice(r.transcript) or
            r.codes.len > (@import("model.zig").max_reference_samples + 2047) / 2048) return error.BadReference;
        for (r.codes) |row| for (row, 0..) |v, i| {
            if (v < 0 or v >= (if (i == 0) @as(i32, 4096) else 1024)) return error.BadReference;
        };
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sem_begin = tok.specialTokenId("<|semantic:0|>") orelse return error.NoSemanticToken;
    var rows: std.ArrayList([W]i32) = .empty;
    errdefer rows.deinit(gpa);
    const Seg = union(enum) { text: []const u8, codes: []const [cfg.codebooks]i32, reference_end };
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
    try segs.append(a, .reference_end);
    try segs.append(a, .{ .text = try withSpeaker(a, text) });
    try segs.append(a, .{ .text = message_end });
    try segs.append(a, .{ .text = generation_start });
    var reference_rows: u32 = 0;
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
        .reference_end => if (refs.len > 0) {
            reference_rows = @intCast(rows.items.len);
        },
    };
    return .{ .frames = try rows.toOwnedSlice(gpa), .reference_rows = reference_rows };
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

const ByteTokenizer = struct {
    pub fn specialTokenId(_: @This(), _: []const u8) ?u32 {
        return 1000;
    }

    pub fn encode(_: @This(), a: std.mem.Allocator, text: []const u8) ![]u32 {
        const ids = try a.alloc(u32, text.len);
        for (text, ids) |byte, *id| id.* = byte;
        return ids;
    }
};

fn expectTextRows(text: []const u8, rows: []const [W]i32) !void {
    try std.testing.expectEqual(text.len, rows.len);
    for (text, rows) |byte, row| {
        var expected: [W]i32 = @splat(0);
        expected[0] = byte;
        try std.testing.expectEqualSlices(i32, &expected, &row);
    }
}

test "reference prompt boundary preserves every row and ends before request text" {
    const t = std.testing;
    const codes = [_][cfg.codebooks]i32{.{ 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 }};
    const refs = [_]Reference{.{ .transcript = "Ref.", .codes = &codes }};
    const built = try build(t.allocator, ByteTokenizer{}, "One.", &refs);
    defer t.allocator.free(built.frames);
    const before = "<|im_start|>system\n" ++ reference_instruction ++
        "\n\n[Reference 1]\nText: <|speaker:0|> Ref.\nSpeech: ";
    const after = message_end ++ "<|im_start|>user\n";
    const suffix = "<|speaker:0|> One." ++ message_end ++ generation_start;
    const boundary = before.len + codes.len + after.len;
    try t.expectEqual(@as(u32, boundary), built.reference_rows);
    try t.expectEqual(boundary + suffix.len, built.frames.len);
    try expectTextRows(before, built.frames[0..before.len]);
    try t.expectEqualSlices(i32, &.{ 1007, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &built.frames[before.len]);
    try expectTextRows(after, built.frames[before.len + 1 .. boundary]);
    try expectTextRows(suffix, built.frames[boundary..]);

    const other = try build(t.allocator, ByteTokenizer{}, "  <|speaker:1|> Different text.", &refs);
    defer t.allocator.free(other.frames);
    try t.expectEqual(built.reference_rows, other.reference_rows);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(built.frames[0..boundary]), std.mem.sliceAsBytes(other.frames[0..boundary]));
    try expectTextRows("  <|speaker:1|> Different text." ++ message_end ++ generation_start, other.frames[boundary..]);

    const plain = try build(t.allocator, ByteTokenizer{}, "One.", &.{});
    defer t.allocator.free(plain.frames);
    try t.expectEqual(@as(u32, 0), plain.reference_rows);
    try expectTextRows("<|im_start|>system\n" ++ tts_instruction ++ message_end ++ "<|im_start|>user\n" ++ suffix, plain.frames);
}

test "reference prompt boundary includes all transcripts, wrappers and multiple references" {
    const t = std.testing;
    const codes = [_][cfg.codebooks]i32{ @splat(1), @splat(2) };
    const refs = [_]Reference{
        .{ .transcript = "  <|speaker:1|> Ref.", .codes = &codes },
        .{ .transcript = "Second.", .codes = codes[0..1] },
    };
    const built = try build(t.allocator, ByteTokenizer{}, "One.", &refs);
    defer t.allocator.free(built.frames);
    const before = "<|im_start|>system\n" ++ reference_instruction ++
        "\n\n[Reference 1]\nText: <|speaker:1|>   Ref.\nSpeech: ";
    const between = "\n\n[Reference 2]\nText: <|speaker:0|> Second.\nSpeech: ";
    const after = message_end ++ "<|im_start|>user\n";
    const boundary = before.len + codes.len + between.len + 1 + after.len;
    try t.expectEqual(@as(u32, boundary), built.reference_rows);
    try expectTextRows(before, built.frames[0..before.len]);
    try expectTextRows(between, built.frames[before.len + 2 ..][0..between.len]);
    try expectTextRows(after, built.frames[boundary - after.len .. boundary]);
    try expectTextRows("<|speaker:0|> One." ++ message_end ++ generation_start, built.frames[boundary..]);
}

test "reference cache: last reference code and equal-length changed tower lookahead both miss" {
    const t = std.testing;
    const cache = @import("reference_cache.zig");
    var prefix: cache.Prefix = .{};
    defer prefix.deinit(t.allocator);
    var codes = [_][cfg.codebooks]i32{ @splat(1), @splat(2) };
    const refs = [_]Reference{.{ .transcript = "Ref.", .codes = &codes }};
    const first = try build(t.allocator, ByteTokenizer{}, "One.", &refs);
    defer t.allocator.free(first.frames);
    const saved_rows = cache.prefillRows(first.frames.len, first.reference_rows, 1);
    try t.expect(!try prefix.begin(t.allocator, first.frames, saved_rows));
    prefix.saved();

    codes[codes.len - 1][cfg.codebooks - 1] += 1;
    const changed = try build(t.allocator, ByteTokenizer{}, "One.", &refs);
    defer t.allocator.free(changed.frames);
    try t.expectEqual(first.reference_rows, changed.reference_rows);
    try t.expectEqual(saved_rows, cache.prefillRows(changed.frames.len, changed.reference_rows, 1));
    try t.expect(!try prefix.begin(t.allocator, changed.frames, saved_rows));
    prefix.saved();
    try t.expect(!try prefix.begin(t.allocator, first.frames, saved_rows));
    prefix.saved();

    // Every processed row is unchanged; only the next row read by the tower differs.
    first.frames[saved_rows][0] += 1;
    try t.expect(!try prefix.begin(t.allocator, first.frames, saved_rows));
}

test "reference cache: a reference speaker change misses while request speaker changes reuse the prefix" {
    const t = std.testing;
    const cache = @import("reference_cache.zig");
    var prefix: cache.Prefix = .{};
    defer prefix.deinit(t.allocator);
    const codes = [_][cfg.codebooks]i32{@splat(1)};
    var refs = [_]Reference{
        .{ .transcript = "<|speaker:1|> First.", .codes = &codes },
        .{ .transcript = "<|speaker:15|> Second.", .codes = &codes },
    };
    const first = try build(t.allocator, ByteTokenizer{}, "<|speaker:1|> One.", &refs);
    defer t.allocator.free(first.frames);
    const saved_rows = cache.prefillRows(first.frames.len, first.reference_rows, 1);
    try t.expect(!try prefix.begin(t.allocator, first.frames, saved_rows));
    prefix.saved();

    const request = try build(t.allocator, ByteTokenizer{}, "<|speaker:15|> Two. <|speaker:1|> Three.", &refs);
    defer t.allocator.free(request.frames);
    try t.expectEqual(first.reference_rows, request.reference_rows);
    try t.expect(try prefix.begin(t.allocator, request.frames, saved_rows));

    refs[0].transcript = "<|speaker:2|> First.";
    const changed = try build(t.allocator, ByteTokenizer{}, "<|speaker:1|> One.", &refs);
    defer t.allocator.free(changed.frames);
    try t.expectEqual(first.frames.len, changed.frames.len);
    try t.expectEqual(first.reference_rows, changed.reference_rows);
    try t.expect(!try prefix.begin(t.allocator, changed.frames, saved_rows));
}
