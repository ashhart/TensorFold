//! Reply text as it streams: think blocks, tool markup held back, Harmony channels, stop strings.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Markers = struct { open: []const u8, close: []const u8 };
/// Qwen's prompt opens the think block; Gemma 4's reply opens its thought channel.
pub const think_markers: Markers = .{ .open = "", .close = "</think>" };
/// The opener a reply writes itself when its template leaves the think block to it (Kolibri 1's does).
const think_open = "<think>";
pub const channel_markers: Markers = .{ .open = "<|channel>thought", .close = "<channel|>" };
/// (opener, closer) of a tool call's markup: Qwen's, Gemma 4's, DeepSeek-V4's DSML block.
pub const calls = [_][2][]const u8{
    .{ "<tool_call>", "</tool_call>" },
    .{ "<|tool_call>", "<tool_call|>" },
    .{ "<\u{ff5c}DSML\u{ff5c}tool_calls>", "</\u{ff5c}DSML\u{ff5c}tool_calls>" },
};

/// ``text.find(tag, from)`` led by a vector scan for the tag's first byte: markup is rare in prose, so it is fast.
pub fn findTag(text: []const u8, from: usize, tag: []const u8) ?usize {
    if (tag.len == 0) return if (from <= text.len) from else null;
    var at = from;
    while (at < text.len) {
        const i = std.mem.indexOfScalarPos(u8, text, at, tag[0]) orelse return null;
        if (std.mem.startsWith(u8, text[i..], tag)) return i;
        at = i + 1;
    }
    return null;
}

fn find(text: []const u8, tag: []const u8) ?usize {
    return findTag(text, 0, tag);
}

pub fn isChannel(m: Markers) bool {
    return std.mem.eql(u8, m.close, channel_markers.close);
}

/// How many bytes at the end of ``text`` could begin ``tag``.
pub fn partialTag(text: []const u8, tag: []const u8) usize {
    if (tag.len == 0) return 0;
    var k = @min(tag.len - 1, text.len);
    while (k > 0) : (k -= 1) if (std.mem.endsWith(u8, text, tag[0..k])) return k;
    return 0;
}

fn trimNewlines(s: []const u8) []const u8 {
    return std.mem.trimStart(u8, s, "\n");
}

pub const Split = struct { reasoning: []const u8, answer: []const u8 };

/// (reasoning, answer) of a thinking reply; while it streams, a tail that could begin a marker is held back.
pub fn splitThinking(a: Allocator, full: []const u8, finished: bool, m: Markers) Allocator.Error!Split {
    var text = full;
    if (m.open.len > 0 and !std.mem.startsWith(u8, text, m.open)) {
        if (find(text, m.open)) |start| if (start > 0) {
            var prefix = text[0..start];
            prefix = prefix[0 .. prefix.len - partialTag(prefix, m.open)]; // a stray partial opener
            const inner = try splitThinking(a, text[start..], finished, m);
            return .{ .reasoning = inner.reasoning, .answer = try std.mem.concat(a, u8, &.{ prefix, inner.answer }) };
        };
        if (!finished) {
            var held = @max(partialTag(text, m.open), partialTag(text, m.close));
            for (calls) |c| held = @max(held, partialTag(text, c[0]));
            return .{ .reasoning = "", .answer = text[0 .. text.len - held] };
        }
        return .{ .reasoning = "", .answer = text };
    }
    if (m.open.len > 0) {
        text = trimNewlines(text[m.open.len..]);
    } else if (std.mem.startsWith(u8, text, think_open)) {
        text = trimNewlines(text[think_open.len..]);
    } else if (!finished and std.mem.startsWith(u8, think_open, text)) {
        return .{ .reasoning = "", .answer = "" }; // held while the reply may still be writing that opener
    }
    if (find(text, m.close)) |end| return .{ .reasoning = text[0..end], .answer = trimNewlines(text[end + m.close.len ..]) };
    var call: ?usize = null;
    for (calls) |c| if (find(text, c[0])) |at| {
        if (call == null or at < call.?) call = at;
    };
    if (call) |at| {
        // a call written before the block closes: held while the block may still close, the answer if the reply ends
        if (!finished) return .{ .reasoning = text[0..at], .answer = "" };
        for (calls) |c| if (std.mem.startsWith(u8, text[at..], c[0]) and find(text[at..], c[1]) != null)
            return .{ .reasoning = text[0..at], .answer = text[at..] };
        return .{ .reasoning = text, .answer = "" };
    }
    var held: usize = 0;
    if (!finished) {
        held = partialTag(text, m.close);
        for (calls) |c| held = @max(held, partialTag(text, c[0]));
    }
    return .{ .reasoning = text[0 .. text.len - held], .answer = "" };
}

/// Hides tool-call blocks and partial opening tags while streaming, so calls arrive as deltas instead.
pub fn hideToolCalls(a: Allocator, text: []const u8, finished: bool) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    return hideInto(a, &out, text, finished);
}

/// ``hideToolCalls`` into ``out`` (cleared first); text without a call block comes back as a slice of ``text``.
pub fn hideInto(a: Allocator, out: *std.ArrayList(u8), text: []const u8, finished: bool) Allocator.Error![]const u8 {
    out.clearRetainingCapacity();
    var pos: usize = 0;
    while (true) {
        var start: ?usize = null;
        var which: usize = 0;
        for (calls, 0..) |c, i| if (findTag(text, pos, c[0])) |at| {
            if (start == null or at < start.?) {
                start = at;
                which = i;
            }
        };
        if (start == null) {
            const tail = text[pos..];
            var held: usize = 0;
            if (!finished) for (calls) |c| {
                held = @max(held, partialTag(tail, c[0]));
            };
            if (pos == 0) return text[0 .. text.len - held];
            try out.appendSlice(a, tail[0 .. tail.len - held]);
            return out.items;
        }
        try out.appendSlice(a, text[pos..start.?]);
        const opener = calls[which][0];
        const end = findTag(text, start.? + opener.len, calls[which][1]) orelse return out.items;
        pos = end + calls[which][1].len;
    }
}

const harmony_final = "<|channel|>final<|message|>";
const harmony_analysis = "<|channel|>analysis<|message|>";
const harmony_ends = [_][]const u8{ "<|return|>", "<|end|>", "<|call|>", "<|start|>" };

fn cutAt(s: []const u8, marker: []const u8) []const u8 {
    return if (find(s, marker)) |i| s[0..i] else s;
}

pub const Harmony = struct { content: []const u8, reasoning: ?[]const u8 };

/// Harmony's final and analysis channels; other text passes unchanged.
pub fn parseHarmony(text: []const u8) Harmony {
    if (find(text, "<|channel|>") == null) return .{ .content = text, .reasoning = null };
    var reasoning: ?[]const u8 = null;
    if (find(text, harmony_analysis)) |i| {
        var r = text[i + harmony_analysis.len ..];
        r = cutAt(r, harmony_final);
        for (harmony_ends) |e| r = cutAt(r, e);
        reasoning = r;
    }
    const f = find(text, harmony_final) orelse return .{ .content = "", .reasoning = reasoning };
    var content = text[f + harmony_final.len ..];
    for (harmony_ends) |e| content = cutAt(content, e);
    return .{ .content = content, .reasoning = reasoning };
}

/// The part of partly decoded output that should stream.
pub fn streamingVisible(text: []const u8) []const u8 {
    if (find(text, "<|channel|>") == null) return text;
    return parseHarmony(text).content;
}

pub fn stripTrailing(tokens: []const u32, stops: []const u32) []const u32 {
    var end = tokens.len;
    while (end > 0 and std.mem.indexOfScalar(u32, stops, tokens[end - 1]) != null) end -= 1;
    return tokens[0..end];
}

/// A thinking reply's reasoning tokens: through its close, else all of them.
pub fn reasoningCount(tokens: []const u32, think_end: ?u32) usize {
    const end = think_end orelse return 0;
    return if (std.mem.indexOfScalar(u32, tokens, end)) |i| i + 1 else tokens.len;
}

/// Python's ``s[n:]`` with ``n`` counted in characters.
pub fn afterChars(s: []const u8, n: usize) []const u8 {
    var seen: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] & 0xc0 == 0x80) continue;
        if (seen == n) return s[i..];
        seen += 1;
    }
    return "";
}

pub fn charCount(s: []const u8) usize {
    var n: usize = 0;
    for (s) |b| n += @intFromBool(b & 0xc0 != 0x80);
    return n;
}

/// Python's ``str.strip()`` for the whitespace replies hold.
pub fn pyStrip(s: []const u8) []const u8 {
    var start: usize = 0;
    var end = s.len;
    while (start < end) {
        const n = spaceAt(s, start);
        if (n == 0) break;
        start += n;
    }
    while (end > start) {
        var back: usize = 1;
        while (back < 4 and end - back > start and s[end - back] & 0xc0 == 0x80) back += 1;
        if (spaceAt(s[0..end], end - back) != back) break;
        end -= back;
    }
    return s[start..end];
}

/// ``str.lstrip()``: the leading whitespace off.
pub fn pyLstrip(s: []const u8) []const u8 {
    return s[@intFromPtr(pyStrip(s).ptr) - @intFromPtr(s.ptr) ..];
}

/// ``str.rstrip()``: the trailing whitespace off.
pub fn pyRstrip(s: []const u8) []const u8 {
    const stripped = pyStrip(s);
    if (stripped.len == 0) return "";
    return s[0 .. @intFromPtr(stripped.ptr) - @intFromPtr(s.ptr) + stripped.len];
}

/// Python's ``\s`` on one byte (ASCII whitespace and the separators below space).
pub fn isSpaceByte(ch: u8) bool {
    return ch == ' ' or (ch >= 0x09 and ch <= 0x0d) or (ch >= 0x1c and ch <= 0x1f);
}

fn spaceAt(s: []const u8, i: usize) usize {
    const c = s[i];
    if (c == ' ' or (c >= 0x09 and c <= 0x0d) or (c >= 0x1c and c <= 0x1f)) return 1;
    if (c < 0x80) return 0;
    const n = std.unicode.utf8ByteSequenceLength(c) catch return 0;
    if (i + n > s.len) return 0;
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return 0;
    return switch (cp) {
        0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => n,
        else => 0,
    };
}

/// Request stop strings: the visible text ends before the first, and a possible start is held while streaming.
pub const Stops = struct {
    strings: []const []const u8,

    pub fn visible(s: Stops, text: []const u8, partial: bool) []const u8 {
        var first: ?usize = null;
        for (s.strings) |stop| if (std.mem.indexOf(u8, text, stop)) |at| {
            if (first == null or at < first.?) first = at;
        };
        if (first) |at| return text[0..at];
        if (!partial) return text;
        var held: usize = 0;
        for (s.strings) |stop| {
            var size: usize = 1;
            while (size < stop.len and size <= text.len) : (size += 1) {
                if (std.mem.endsWith(u8, text, stop[0..size])) held = @max(held, size);
            }
        }
        return text[0 .. text.len - held];
    }

    /// The first stop the text holds, ties in request order (``matched_stop``).
    pub fn matched(s: Stops, text: []const u8) ?[]const u8 {
        var best: ?usize = null;
        var which: []const u8 = "";
        for (s.strings) |stop| if (std.mem.indexOf(u8, text, stop)) |at| {
            if (best == null or at < best.?) {
                best = at;
                which = stop;
            }
        };
        return if (best != null) which else null;
    }

    /// Tokens the engine's stop check decodes: the longest stop's length plus 8.
    pub fn tail(s: Stops) usize {
        var longest: usize = 0;
        for (s.strings) |stop| longest = @max(longest, charCount(stop));
        return longest + 8;
    }
};

/// Decodes only new tokens with the previous chunk as context, so characters split across tokens stay whole.
pub const Incremental = struct {
    tokens: std.ArrayList(u32) = .empty,
    text: std.ArrayList(u8) = .empty,
    prefix: usize = 0,
    read: usize = 0,

    pub fn extend(inc: *Incremental, a: Allocator, ctx: anytype, decode: anytype, more: []const u32) ![]const u8 {
        try inc.tokens.appendSlice(a, more);
        const before = try decode(ctx, a, inc.tokens.items[inc.prefix..inc.read]);
        const after = try decode(ctx, a, inc.tokens.items[inc.prefix..]);
        const nb = charCount(before);
        if (charCount(after) > nb and !std.mem.endsWith(u8, after, "\u{fffd}")) {
            try inc.text.appendSlice(a, afterChars(after, nb));
            inc.prefix = inc.read;
            inc.read = inc.tokens.items.len;
        }
        return inc.text.items;
    }

    /// A character is still split across tokens, so streaming waits for the rest.
    pub fn pending(inc: *const Incremental) bool {
        return inc.tokens.items.len > 0 and inc.read < inc.tokens.items.len;
    }
};

test "splits and holds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = try splitThinking(a, "plan</think>\n\nHi", true, think_markers);
    try std.testing.expectEqualStrings("plan", s.reasoning);
    try std.testing.expectEqualStrings("Hi", s.answer);
    const held = try splitThinking(a, "plan</thi", false, think_markers);
    try std.testing.expectEqualStrings("plan", held.reasoning);
    try std.testing.expectEqualStrings("a b", try hideToolCalls(a, "a <tool_call>x</tool_call>b<tool", false));
    const stops: Stops = .{ .strings = &.{"END"} };
    try std.testing.expectEqualStrings("ab", stops.visible("abEN", true));
    try std.testing.expectEqualStrings("x", pyStrip(" \u{3000}x\u{a0}\n"));
    try std.testing.expectEqualStrings("lo", afterChars("h\u{e9}lo", 2));
}

test "a reply that writes its own think opener keeps the tag out of its reasoning" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = "<think>\nPlan.\n</think>\n\nDone.";
    var sent: []const u8 = "";
    for (1..reply.len + 1) |n| {
        const s = try splitThinking(a, reply[0..n], false, think_markers);
        try std.testing.expect(std.mem.indexOfScalar(u8, s.reasoning, '<') == null and std.mem.startsWith(u8, s.reasoning, sent));
        sent = s.reasoning;
    }
    for ([_][]const u8{ reply, "Plan.\n</think>\n\nDone." }) |text| {
        const s = try splitThinking(a, text, true, think_markers);
        try std.testing.expectEqualStrings("Plan.\n", s.reasoning);
        try std.testing.expectEqualStrings("Done.", s.answer);
    }
}
