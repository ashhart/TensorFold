//! A long synthetic reply for the stream scan benchmark: word tokens, a close token and each family's tool call.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const KimiText = @import("kimi_text.zig").KimiText;
const Allocator = std.mem.Allocator;

pub const think_end: u32 = 900; // `</think>` as one token
pub const eos: u32 = 901;
const word0: u32 = 1000;

/// Reasoning and prose words, about 4.7 bytes each; two hold a `<` as code in reasoning does.
const words = [_][]const u8{
    " the",     " model", " should", " check", " whether", " value", " is",      " a",      " list", ",",
    ".",        " we",    " need",   " to",    " return",  " it",    " first",   " then",   " call", " write",
    " file",    " so",    " that",   " each",  " line",    " ends",  " in",      " x",      " +",    " 1",
    " Vec<u8>", " a<b",   " if",     " not",   " and",     "\n",     " because", " length", " of",   " output",
};
/// The words a tool value is written in: no newline, quote, backslash or `<`, so its JSON bytes are its own.
const value_words = [_][]const u8{ " alpha", " beta", " gamma", " delta", " notes", " line", " one", " two", " end", "." };
/// Each phase's first word, without the space the others open with.
const leads = [_][]const u8{ "Plan", "Sure", "Begin" };
const vocab = words ++ value_words ++ leads;

pub const Family = enum { qwen, glm, dsml, kimi, none };

/// The reply's token counts: its reasoning, its prose and the words of the tool call's content value.
pub const Shape = struct { reasoning: usize, content: usize, value: usize };

/// The reply's tokens, and where each phase ends in them and in its own decoded bytes.
pub const Reply = struct {
    ids: []u32,
    reasoning_ends: []usize, // bytes of the reasoning text after each reasoning token
    content_ends: []usize,
    value_ends: []usize,
    content_from: usize, // the first content token
    value_from: usize, // the first value token
    reasoning: []const u8,
    content: []const u8,
    value: []const u8,
};

/// Bytes as tokens, `</think>` and an end token, then the words; every prompt renders as "prompt".
pub const Text = struct {
    pub fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn piece(id: u32) []const u8 {
        const all = comptime blk: {
            var b: [256]u8 = undefined;
            for (&b, 0..) |*x, i| x.* = i;
            break :blk b;
        };
        if (id < 256) return all[id..][0..1];
        if (id == think_end) return "</think>";
        if (id >= word0 and id < word0 + vocab.len) return vocab[id - word0];
        return "";
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| try out.appendSlice(a, piece(id));
        if (std.unicode.utf8ValidateSlice(out.items)) return out.items;
        defer out.deinit(a);
        var valid: std.ArrayList(u8) = .empty;
        try @import("tokenizer").unicode.appendLossy(&valid, a, out.items);
        return valid.items;
    }

    fn tokenId(_: *anyopaque, p: []const u8) ?u32 {
        return if (std.mem.eql(u8, p, "</think>")) think_end else null;
    }

    fn tokenString(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return decode(ctx, a, &.{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return word0 + vocab.len;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{eos};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Kimi's synthetic tiktoken (kimi_text_test.zig's) with the words above as tokens, so a token is a word here too.
pub fn kimi(gpa: Allocator) !*KimiText {
    const t = try gpa.create(KimiText);
    const pieces = vocab.len;
    t.* = .{ .gpa = gpa, .arena = .init(gpa), .tok = .{ .a = gpa, .first_special = @intCast(256 + pieces) } };
    errdefer t.deinit();
    t.tok.bytes = try gpa.alloc(u8, 256);
    for (t.tok.bytes, 0..) |*byte, i| {
        byte.* = @intCast(i);
        const p = t.tok.bytes[i .. i + 1];
        try t.tok.tokens.append(gpa, p);
        try t.tok.ranks.put(gpa, p, @intCast(i));
    }
    for (vocab, 0..) |w, i| {
        try t.tok.tokens.append(gpa, w);
        try t.tok.ranks.put(gpa, w, @intCast(256 + i));
    }
    const named = [_][]const u8{ "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>" };
    for (0..256) |i| {
        const name = if (i < named.len) try gpa.dupe(u8, named[i]) else try std.fmt.allocPrint(gpa, "<|reserved_token_{d}|>", .{256 + pieces + i});
        try t.tok.specials.append(gpa, name);
    }
    for (t.tok.tokens.items, 0..) |_, id|
        try t.vocab_lookup.put(t.arena.allocator(), try t.tok.tokenString(t.arena.allocator(), @intCast(id)), @intCast(id));
    t.special_flags[3] = true;
    const end = try t.arena.allocator().alloc(u32, 1);
    end[0] = t.tok.first_special + 3;
    t.eos = end;
    return t;
}

/// The token that writes ``vocab[k]`` in this family's text.
fn wordId(family: Family, k: usize) u32 {
    return @as(u32, @intCast(k)) + if (family == .kimi) @as(u32, 256) else word0;
}

const Builder = struct {
    a: Allocator,
    family: Family,
    kimi_text: ?model_text.Text,
    ids: std.ArrayList(u32) = .empty,

    /// Markup: bytes as tokens, or Kimi's controls as its own tokens.
    fn markup(b: *Builder, s: []const u8) !void {
        if (b.kimi_text) |t| return b.ids.appendSlice(b.a, try t.encode(b.a, s, false));
        for (s) |byte| try b.ids.append(b.a, byte);
    }

    /// ``leads[lead]`` then ``n - 1`` words picked by ``rng``, one token each, their bytes and ends recorded.
    fn run(b: *Builder, rng: std.Random, n: usize, value: bool, lead: usize, ends: *std.ArrayList(usize), bytes: *std.ArrayList(u8)) !void {
        for (0..n) |i| {
            const k = if (i == 0) words.len + value_words.len + lead else if (value) words.len + rng.uintLessThan(usize, value_words.len) else rng.uintLessThan(usize, words.len);
            try b.ids.append(b.a, wordId(b.family, k));
            try bytes.appendSlice(b.a, vocab[k]);
            try ends.append(b.a, bytes.items.len);
        }
    }
};

/// The reply: reasoning, its close, prose, then one write_file call in the family's markup (none: prose only).
pub fn reply(a: Allocator, family: Family, shape: Shape, kimi_text: ?model_text.Text) !Reply {
    var prng: std.Random.DefaultPrng = .init(0x7f3a);
    const rng = prng.random();
    var b: Builder = .{ .a = a, .family = family, .kimi_text = kimi_text };
    var r_ends: std.ArrayList(usize) = .empty;
    var c_ends: std.ArrayList(usize) = .empty;
    var v_ends: std.ArrayList(usize) = .empty;
    var reasoning: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    var value: std.ArrayList(u8) = .empty;
    try b.run(rng, shape.reasoning, false, 0, &r_ends, &reasoning);
    if (family == .kimi) try b.markup("<|close|>think<|sep|><|open|>response<|sep|>") else {
        try b.ids.append(a, think_end);
        try b.markup("\n\n");
    }
    const content_from = b.ids.items.len;
    try b.run(rng, shape.content, false, 1, &c_ends, &content);
    switch (family) {
        .none, .kimi => {},
        .qwen => try b.markup("\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n<parameter=content>\n"),
        .glm => try b.markup("\n\n<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value><arg_key>content</arg_key><arg_value>"),
        .dsml => try b.markup("\n\n<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"write_file\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"path\" string=\"true\">notes.txt</\u{ff5c}DSML\u{ff5c}parameter>\n<\u{ff5c}DSML\u{ff5c}parameter name=\"content\" string=\"true\">"),
    }
    if (family == .kimi) try b.markup("<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"write_file\" index=\"1\"<|sep|><|open|>argument key=\"path\" type=\"string\"<|sep|>notes.txt<|close|>argument<|sep|><|open|>argument key=\"content\" type=\"string\"<|sep|>");
    const value_from = b.ids.items.len;
    if (family != .none) try b.run(rng, shape.value, true, 2, &v_ends, &value);
    switch (family) {
        .none => {},
        .qwen => try b.markup("\n</parameter>\n</function>\n</tool_call>"),
        .glm => try b.markup("</arg_value></tool_call>"),
        .dsml => try b.markup("</\u{ff5c}DSML\u{ff5c}parameter>\n</\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>"),
        .kimi => try b.markup("<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"),
    }
    if (family == .kimi) try b.markup("<|end_of_msg|>") else try b.ids.append(a, eos);
    return .{
        .ids = b.ids.items,
        .reasoning_ends = r_ends.items,
        .content_ends = c_ends.items,
        .value_ends = v_ends.items,
        .content_from = content_from,
        .value_from = value_from,
        .reasoning = reasoning.items,
        .content = content.items,
        .value = value.items,
    };
}

test "stream scan bench: the reply decodes to its phases, one token a word, Kimi's too" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var t: Text = .{};
    const r = try reply(a, .qwen, .{ .reasoning = 40, .content = 10, .value = 10 }, null);
    const whole = try t.text().decode(a, r.ids);
    try std.testing.expect(std.mem.startsWith(u8, whole, r.reasoning));
    try std.testing.expect(std.mem.indexOf(u8, whole, try std.mem.concat(a, u8, &.{ "</think>\n\n", r.content, "\n\n<tool_call>" })) != null);
    try std.testing.expect(std.mem.indexOf(u8, whole, try std.mem.concat(a, u8, &.{ "<parameter=content>\n", r.value, "\n</parameter>" })) != null);
    const k = try kimi(std.testing.allocator);
    defer k.deinit();
    const kr = try reply(a, .kimi, .{ .reasoning = 40, .content = 10, .value = 10 }, k.text());
    try std.testing.expectEqual(@as(usize, 40), kr.reasoning_ends.len);
    try std.testing.expectEqual(r.content_from, 40 + 3); // one token a word: the reasoning, `</think>` and two newlines
    try std.testing.expectEqualStrings(kr.reasoning, try k.text().decode(a, kr.ids[0..40]));
}
