//! Spans that follow the reading (zonos3.tts.span_follow): each span's characters as the text tokens that spell them.
const std = @import("std");
const errors = @import("errors.zig");
const text_mod = @import("model_text.zig");
const req = @import("zonos_request");
const tokens = @import("zonos_tokens.zig");
const Allocator = std.mem.Allocator;

const Range = [2]usize;

fn codepoints(a: Allocator, s: []const u8) ![]u21 {
    var out: std.ArrayList(u21) = .empty;
    var it = (std.unicode.Utf8View.init(s) catch return error.BadUtf8).iterator();
    while (it.nextCodepoint()) |c| try out.append(a, c);
    return out.toOwnedSlice(a);
}

/// Python's str.isspace.
fn isSpace(c: u21) bool {
    return switch (c) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// map_offsets: an offset of `original` in `final`: exact in unchanged stretches, proportional in rewritten ones.
fn mapOffset(a: Allocator, original: []const u21, final: []const u21, o: usize, end: bool) !usize {
    if (std.mem.eql(u21, original, final)) return o;
    if (o >= original.len) return final.len;
    const x = try a.alloc(u32, original.len);
    for (x, original) |*d, c| d.* = c;
    const y = try a.alloc(u32, final.len);
    for (y, final) |*d, c| d.* = c;
    const ops = try req.difflib.opcodes(a, x, y);
    // the blocks holding original characters partition it; the last that starts at or before `o` holds it
    var hold: ?req.difflib.Op = null;
    for (ops) |op| if (op.i2 > op.i1 and op.i1 <= o) {
        hold = op;
    };
    const op = hold.?;
    if (op.tag == .equal) return op.j1 + o - op.i1;
    const at = @as(f64, @floatFromInt(op.j1)) + @as(f64, @floatFromInt((o - op.i1) * (op.j2 - op.j1))) / @as(f64, @floatFromInt(op.i2 - op.i1));
    return @intFromFloat(if (end) @ceil(at) else @floor(at));
}

/// The characters of `s` (at `base`) each of its tokens covers: a character split between two tokens is both's.
fn stretch(cx: *errors.Cx, text: text_mod.Text, s: []const u8, base: usize, out: *std.ArrayList(Range)) errors.Refused!void {
    if (s.len == 0) return;
    const a = cx.a;
    const ids = text.encode(a, s, false) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.other("the tokenizer cannot encode this prompt"),
    };
    // the character each byte of `s` belongs to
    const char_of = try a.alloc(usize, s.len);
    var chars: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (chars += 1) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return cx.refuse("input must be valid UTF-8 text");
        @memset(char_of[i..][0..@min(n, s.len - i)], chars);
        i += n;
    }
    var at: usize = 0;
    for (ids) |id| {
        const bytes = try text.tokenBytes(a, id) orelse try text.decode(a, &.{id});
        if (bytes.len == 0 or at + bytes.len > s.len or !std.mem.eql(u8, bytes, s[at..][0..bytes.len])) return cx.other("the tokenizer's pieces do not spell the text");
        try out.append(a, .{ base + char_of[at], base + char_of[at + bytes.len - 1] + 1 });
        at += bytes.len;
    }
    if (at != s.len) return cx.other("the tokenizer's pieces do not spell the text");
}

/// token_char_ranges: the characters behind each text position, BOS and EOS empty, a marker or phone its own.
fn tokenChars(cx: *errors.Cx, text: text_mod.Text, s: []const u8) errors.Refused![]Range {
    const a = cx.a;
    var out: std.ArrayList(Range) = .empty;
    try out.append(a, .{ 0, 0 });
    var at: usize = 0; // a byte of `s`
    var from: usize = 0;
    var chars: usize = 0; // the character at `at`
    var from_char: usize = 0;
    while (at < s.len) {
        if (!tokens.isMarker(s, at)) {
            at += std.unicode.utf8ByteSequenceLength(s[at]) catch return cx.refuse("input must be valid UTF-8 text");
            chars += 1;
            continue;
        }
        try stretch(cx, text, s[from..at], from_char, &out);
        if (std.mem.startsWith(u8, s[at..], tokens.group_open)) {
            // the open marker, each phone, the close marker: a character each
            const close = std.mem.indexOfPos(u8, s, at + 3, tokens.group_close) orelse return cx.refuse("unbalanced pronunciation group");
            const n = std.unicode.utf8CountCodepoints(s[at..close]) catch return cx.refuse("input must be valid UTF-8 text");
            for (0..n + 1) |k| try out.append(a, .{ chars + k, chars + k + 1 });
            chars += n + 1;
            at = close + 3;
        } else {
            try out.append(a, .{ chars, chars + 1 });
            chars += 1;
            at += 3;
        }
        from = at;
        from_char = chars;
    }
    try stretch(cx, text, s[from..], from_char, &out);
    try out.append(a, .{ chars, chars });
    return out.toOwnedSlice(a);
}

/// span_token_ranges: each span of `input`'s characters as the text positions [a, b) of `final` that spell it.
pub fn ranges(cx: *errors.Cx, text: text_mod.Text, input: []const u8, final: []const u8, spans: []const [2]usize) errors.Refused![]const [2]u32 {
    const a = cx.a;
    const original = codepoints(a, input) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("input must be valid UTF-8 text");
    const said = codepoints(a, final) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else cx.refuse("input must be valid UTF-8 text");
    const chars = try tokenChars(cx, text, final);
    for (chars) |*r| {
        while (r[0] < r[1] and isSpace(said[r[0]])) r[0] += 1;
        while (r[1] > r[0] and isSpace(said[r[1] - 1])) r[1] -= 1;
    }
    const out = try a.alloc([2]u32, spans.len);
    for (out, spans) |*o, span| {
        var start = span[0];
        var end = span[1];
        while (start < end and isSpace(original[start])) start += 1;
        while (end > start and isSpace(original[end - 1])) end -= 1;
        const s = try mapOffset(a, original, said, start, false);
        const e = try mapOffset(a, original, said, end, true);
        var first: ?usize = null;
        var last: usize = 0;
        for (chars, 0..) |r, i| if (r[0] < r[1] and r[0] < e and s < r[1]) {
            if (first == null) first = i;
            last = i;
        };
        if (first) |f| {
            o.* = .{ @intCast(f), @intCast(last + 1) };
            continue;
        }
        // a span on no token keeps the token at its start, and never steers
        var at = chars.len - 1;
        for (chars, 0..) |r, i| if (r[0] < r[1] and r[1] > s) {
            at = i;
            break;
        };
        o.* = .{ @intCast(at), @intCast(at) };
    }
    return out;
}

test "ranges: a span's tokens through the markers, trimmed of whitespace, through a rewritten stretch" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const bytes = tokens.Bytes.text();
    // a byte a token: "ab cd" is positions 1..5; the span " cd" is the tokens of c and d
    try t.expectEqualSlices([2]u32, &.{.{ 4, 6 }}, try ranges(&cx, bytes, "ab cd", "ab cd", &.{.{ 2, 5 }}));
    // the start marker is position 1, a pause marker its own position
    const marked = tokens.start_marker ++ "ab\xee\x80\x84cd";
    try t.expectEqualSlices([2]u32, &.{ .{ 2, 4 }, .{ 5, 7 } }, try ranges(&cx, bytes, "ab cd", marked, &.{ .{ 0, 2 }, .{ 3, 5 } }));
    // "$5 now" normalized to "five dollars now": the span on "$5" covers the rewritten stretch
    try t.expectEqualSlices([2]u32, &.{ .{ 1, 13 }, .{ 14, 17 } }, try ranges(&cx, bytes, "$5 now", "five dollars now", &.{ .{ 0, 2 }, .{ 3, 6 } }));
    // a span of whitespace only maps onto no token: it keeps the token after it, and never steers
    try t.expectEqualSlices([2]u32, &.{.{ 4, 4 }}, try ranges(&cx, bytes, "ab cd", "ab cd", &.{.{ 2, 3 }}));
}
