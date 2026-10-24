//! Tokenizer compatibility, memo and decoder regressions.
const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const Tokenizer = tokenizer.Tokenizer;
const parse = tokenizer.parse;
const Memo = tokenizer.Memo;
const eos = tokenizer.eos;
const steps = @import("steps.zig");

test "byte-level BPE with merges, added tokens and decode" {
    const json =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        \\ "normalizer":{"type":"NFC"},
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    const ids = try t.encode(a, "ab ab<|end|>c");
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 5, 7, 6 }, ids);
    const text = try t.decode(a, ids, false);
    defer a.free(text);
    try std.testing.expectEqualStrings("ab ab<|end|>c", text);
    const skipped = try t.decode(a, ids, true);
    defer a.free(skipped);
    try std.testing.expectEqualStrings("ab abc", skipped);
    try std.testing.expectEqual(@as(?u32, 7), t.specialTokenId("<|end|>"));
    try std.testing.expectEqual(@as(usize, 1), t.flagged_specials.len);
    const marker = (try t.tokenBytes(a, 7)).?;
    defer a.free(marker);
    try std.testing.expectEqualStrings("<|end|>", marker);
    const unknown = (try t.tokenBytes(a, 99)).?;
    defer a.free(unknown);
    try std.testing.expectEqualStrings("", unknown);
}

test "with the memo, long segments encode as without it, and the second encode takes their kept ids" {
    const json =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    for (0..300) |_| try text.appendSlice(a, "ab abc ");
    try text.appendSlice(a, "<|end|>");
    for (0..300) |_| try text.appendSlice(a, " ab cab");
    const plain = try t.encode(a, text.items);
    defer a.free(plain);
    t.memo = try Memo.create(a, 1 << 20);
    for (0..2) |_| {
        const kept = try t.encode(a, text.items);
        defer a.free(kept);
        try std.testing.expectEqualSlices(u32, plain, kept);
    }
    try std.testing.expectEqual(@as(u64, 2), t.memo.?.hits); // both segments, the second time
}

test "a kept segment's ids equal a fresh encode's wherever the segment sits" {
    const added =
        \\{"added_tokens":[{"id":20,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        \\ {"id":21,"content":"<|l|>","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":true},
        \\ {"id":22,"content":"<|r|>","single_word":false,"lstrip":false,"rstrip":true,"normalized":false,"special":true}],
    ;
    const kinds = [_][]const u8{
        added ++
            \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
            \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
            \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6,"x":7,"ï":8,"¿":9,"½":10},"merges":[["Ġ","a"],"a b","Ġa b"]}}
        ,
        added ++
            \\ "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true},
            \\ "model":{"type":"BPE","unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,
            \\  "vocab":{"<unk>":0,"a":1,"b":2,"▁":3,"ab":4,"▁a":5,"▁ab":6,"c":7,"x":8,"<0xEF>":9,"<0xBF>":10,"<0xBD>":11},"merges":["a b","▁ a","▁a b"]}}
        ,
    };
    const a = std.testing.allocator;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    for (0..200) |i| try body.appendSlice(a, if (i % 3 == 0) "ab abc " else "cab ab ");
    const b = body.items;
    // first or not, after lstrip and rstrip tokens, with a leading space, around a lossy byte, repeated in one prompt
    const texts = [_][]const []const u8{
        &.{b},               &.{ "<|end|>", b }, &.{ b, "<|end|>", b }, &.{ "x  <|l|>", b },
        &.{ "<|r|>   ", b }, &.{ " ", b },       &.{ b, "\xff", b },    &.{ "<|end|> ", b, "<|r|>", b },
    };
    for (kinds, 0..) |json, kind| {
        var t = try parse(a, json);
        defer t.deinit();
        var fresh: [texts.len][]u32 = undefined;
        for (texts, &fresh) |parts, *ids| {
            const s = try std.mem.concat(a, u8, parts);
            defer a.free(s);
            ids.* = try t.encode(a, s);
        }
        defer for (fresh) |ids| a.free(ids);
        // the Metaspace tokenizer prepends to the first segment alone, so a memo blind to `first` fails below
        if (kind == 1) try std.testing.expect(!std.mem.eql(u32, fresh[0], fresh[1][1..]));
        t.memo = try Memo.create(a, 1 << 20);
        for (0..2) |round| for (0..texts.len) |k| {
            const i = if (round == 0) k else texts.len - 1 - k;
            const s = try std.mem.concat(a, u8, texts[i]);
            defer a.free(s);
            const kept = try t.encode(a, s);
            defer a.free(kept);
            try std.testing.expectEqualSlices(u32, fresh[i], kept);
        };
        try std.testing.expect(t.memo.?.hits >= 10); // every long segment of the second round, at least
    }
}

test "encode, encodeMasked, encodePlain, decode and tokenBytes put only their result in the caller's allocator" {
    const json =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try text.appendSlice(a, "<|end|>");
    while (text.items.len < 1 << 16) try text.appendSlice(a, "ab abc cab ");
    const body = text.items["<|end|>".len..];
    const spans = [_]tokenizer.Span{.{ "<|end|>".len, text.items.len }};
    const encoded = try t.encode(a, text.items);
    defer a.free(encoded);
    // a server's request arena keeps all it is given until the reply ends: this budget holds results, not scratch
    const budget = try a.alloc(u8, 3 * @sizeOf(u32) * encoded.len);
    defer a.free(budget);
    var fba = std.heap.FixedBufferAllocator.init(budget);
    try std.testing.expectEqualSlices(u32, encoded, try t.encode(fba.allocator(), text.items));
    fba.reset();
    try std.testing.expectEqualSlices(u32, encoded, try t.encodeMasked(fba.allocator(), text.items, &spans));
    fba.reset();
    try std.testing.expectEqualSlices(u32, encoded[1..], try t.encodePlain(fba.allocator(), body));
    fba.reset();
    try std.testing.expectEqualStrings(text.items, try t.decode(fba.allocator(), encoded, false));
    var few: [16]u8 = undefined;
    for (encoded[0..16]) |id| {
        const bytes = (try t.tokenBytes(a, id)).?;
        defer a.free(bytes);
        var one = std.heap.FixedBufferAllocator.init(&few);
        try std.testing.expectEqualStrings(bytes, (try t.tokenBytes(one.allocator(), id)).?);
    }
}

test "small calls use stack scratch and a large encode frees its allocator spill" {
    const json =
        \\{"decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","vocab":{"a":0},"merges":[]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    var counting = std.testing.FailingAllocator.init(a, .{});
    t.allocator = counting.allocator();
    defer t.allocator = a;

    const decoded = try t.decode(a, &.{ 0, 0, 0, 0 }, false);
    defer a.free(decoded);
    const bytes = (try t.tokenBytes(a, 0)).?;
    defer a.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), counting.allocations);

    const text = try a.alloc(u8, 64 * 1024);
    defer a.free(text);
    @memset(text, 'a');
    const encoded = try t.encode(a, text);
    defer a.free(encoded);
    try std.testing.expect(counting.allocations > 0);
    try std.testing.expectEqual(counting.allocations, counting.deallocations);
    try std.testing.expectEqual(counting.allocated_bytes, counting.freed_bytes);
}

test "sentencepiece-style BPE with byte fallback" {
    const json =
        \\{"added_tokens":[],"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
        \\ "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
        \\ "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
        \\ "model":{"type":"BPE","unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,
        \\  "vocab":{"<unk>":0,"<0xC3>":1,"<0xA9>":2,"h":3,"i":4,"▁":5,"hi":6,"▁h":7},"merges":["h i","▁ h"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    const ids = try t.encode(a, "hi h\u{E9}");
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 6, 7, 1, 2 }, ids);
    const text = try t.decode(a, ids, false);
    defer a.free(text);
    try std.testing.expectEqualStrings("hi h\u{E9}", text);
    try std.testing.expectEqual(@as(?u32, 0), t.unk_id);
    const first = (try t.tokenBytes(a, 1)).?;
    defer a.free(first);
    const second = (try t.tokenBytes(a, 2)).?;
    defer a.free(second);
    try std.testing.expectEqualSlices(u8, &.{0xc3}, first);
    try std.testing.expectEqualSlices(u8, &.{0xa9}, second);
    const space = (try t.tokenBytes(a, 7)).?;
    defer a.free(space);
    try std.testing.expectEqualStrings(" h", space);
}

test "without a decoder, decode joins known ids' tokens with spaces" {
    const a = std.testing.allocator;
    var t = Tokenizer{ .allocator = a, .vocab = .init(a), .id_to_token = .{ .allocator = a }, .model = .empty(.wordpiece) };
    defer t.deinit();
    try t.vocab.put("t1", 1);
    try t.id_to_token.put(1, "t1");
    try t.id_to_token.put(4, "t4");
    const text = try t.decode(a, &.{ 1, 4, 9 }, false);
    defer a.free(text);
    try std.testing.expectEqualStrings("t1 t4", text);
    try std.testing.expectEqual(@as(?u32, null), t.unk_id);
    try std.testing.expect((try t.tokenBytes(a, 1)) == null);
}

test "raw ByteLevel pieces preserve an incomplete UTF-8 character" {
    const a = std.testing.allocator;
    var t = Tokenizer{
        .allocator = a,
        .vocab = .init(a),
        .id_to_token = .{ .allocator = a },
        .model = .empty(.bpe),
        .decoder = .byte_level,
    };
    defer t.deinit();
    try t.id_to_token.put(0, "\u{c3}");
    try t.id_to_token.put(1, "\u{a9}");
    const first = (try t.tokenBytes(a, 0)).?;
    defer a.free(first);
    const second = (try t.tokenBytes(a, 1)).?;
    defer a.free(second);
    try std.testing.expectEqualSlices(u8, &.{0xc3}, first);
    try std.testing.expectEqualSlices(u8, &.{0xa9}, second);
    const decoded = try t.decode(a, &.{ 0, 1 }, false);
    defer a.free(decoded);
    try std.testing.expectEqualStrings("\u{e9}", decoded);
    const partial = try t.decode(a, &.{0}, false);
    defer a.free(partial);
    try std.testing.expectEqualStrings("\u{fffd}", partial);
}

test "raw token bytes refuse replacements after the decoder joins token boundaries" {
    const a = std.testing.allocator;
    const decoders = [_]steps.Decoder{
        .byte_level, .{ .replace = .{ .pattern = .{ .literal = "ab" }, .content = "x" } },
    };
    var t = Tokenizer{
        .allocator = a,
        .vocab = .init(a),
        .id_to_token = .{ .allocator = a },
        .model = .empty(.bpe),
        .decoder = .{ .sequence = &decoders },
    };
    defer t.deinit();
    try t.id_to_token.put(0, "a");
    try t.id_to_token.put(1, "b");
    const decoded = try t.decode(a, &.{ 0, 1 }, false);
    defer a.free(decoded);
    try std.testing.expectEqualStrings("x", decoded);
    try std.testing.expect((try t.tokenBytes(a, 0)) == null);
}

test "raw token bytes conservatively refuse context-sensitive decoder kinds" {
    const a = std.testing.allocator;
    var scratch = std.heap.ArenaAllocator.init(a);
    defer scratch.deinit();
    const pattern = try @import("regex.zig").Regex.compile(scratch.allocator(), "a+");
    const after_fuse = [_]steps.Decoder{
        .fuse, .{ .replace = .{ .pattern = .{ .literal = "ab" }, .content = "x" } },
    };
    const decoders = [_]steps.Decoder{
        .{ .strip = .{ .content = ' ', .start = 1, .stop = 0 } },
        .{ .metaspace = .{ .replacement = 0x2581, .scheme = .always } },
        .{ .wordpiece = .{ .prefix = "##", .cleanup = false } },
        .{ .replace = .{ .pattern = .{ .regex = pattern }, .content = "x" } },
        .{ .sequence = &after_fuse },
    };
    var t = Tokenizer{
        .allocator = a,
        .vocab = .init(a),
        .id_to_token = .{ .allocator = a },
        .model = .empty(.bpe),
    };
    defer t.deinit();
    try t.id_to_token.put(0, "a");
    for (decoders) |decoder| {
        t.decoder = decoder;
        try std.testing.expect((try t.tokenBytes(a, 0)) == null);
    }
}

test {
    _ = @import("memo.zig");
}

test {
    _ = @import("identity.zig");
    _ = eos;
}
