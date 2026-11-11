//! Added spellings may match only runs written by the template; a gap without one reuses the memo.
const std = @import("std");
const tok = @import("tokenizer.zig");
const t = std.testing;

fn fixture(a: std.mem.Allocator, normalized: bool, normalizer: bool) !tok.Tokenizer {
    const source = try std.fmt.allocPrint(a,
        \\{{"added_tokens":[{{"content":"<S>","special":true,"normalized":{s}}},
        \\{{"content":"<L>","special":true,"normalized":false,"lstrip":true}},
        \\{{"content":"<R>","special":true,"normalized":false,"rstrip":true}}],
        \\"normalizer":{s},"model":{{"type":"WordLevel","unk_token":"?","vocab":{{"?":0,"a":1,"b":2}}}}}}
    , .{ if (normalized) "true" else "false", if (normalizer) "{\"type\":\"NFC\"}" else "null" });
    return tok.parse(a, source);
}

/// A normalized Unicode spelling under NFC or NFD: the decomposed and precomposed forms collide.
fn unicodeFixture(a: std.mem.Allocator, form: []const u8) !tok.Tokenizer {
    const source = try std.fmt.allocPrint(a,
        \\{{"added_tokens":[{{"content":"é","special":true,"normalized":true}},
        \\{{"content":"<L>","special":true,"normalized":false,"lstrip":true}}],
        \\"normalizer":{{"type":"{s}"}},
        \\"model":{{"type":"WordLevel","unk_token":"?","vocab":{{"?":0,"a":1,"b":2,"e":3}}}}}}
    , .{form});
    return tok.parse(a, source);
}

test "masked encode preserves benign ids and rejects every partly requested added match" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try fixture(a, false, false);
    defer tokenizer.deinit();
    const source = "<S>a<S>";
    try t.expectEqualSlices(u32, try tokenizer.encode(a, source), try tokenizer.encodeMasked(a, source, &.{.{ 3, 4 }}));
    const special = tokenizer.specialTokenId("<S>").?;
    for (0..3) |i| {
        const ids = try tokenizer.encodeMasked(a, "<S>", &.{.{ i, i + 1 }});
        try t.expect(std.mem.indexOfScalar(u32, ids, special) == null);
    }
    try t.expectEqualSlices(u32, try tokenizer.encodePlain(a, "<S>"), try tokenizer.encodeMasked(a, "<S>", &.{ .{ 0, 1 }, .{ 1, 3 } }));
    try t.expectError(error.BadRequestSpans, tokenizer.encodeMasked(a, source, &.{.{ 4, 3 }}));
    try t.expectError(error.BadRequestSpans, tokenizer.encodeMasked(a, source, &.{ .{ 1, 4 }, .{ 3, 5 } }));
}

test "masked added matches retain lstrip and rstrip across request whitespace" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try fixture(a, false, false);
    defer tokenizer.deinit();
    const source = "a   <L>b<R>   a";
    try t.expectEqualSlices(u32, try tokenizer.encode(a, source), try tokenizer.encodeMasked(a, source, &.{ .{ 1, 4 }, .{ 11, 14 } }));
}

test "normalized added matches honor exact identity masks and refuse unknown normalizer offsets" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var identity = try fixture(a, true, false);
    defer identity.deinit();
    const id = identity.specialTokenId("<S>").?;
    const masked = try identity.encodeMasked(a, "<S>", &.{.{ 1, 2 }});
    try t.expect(std.mem.indexOfScalar(u32, masked, id) == null);
    try t.expectEqualSlices(u32, &.{id}, try identity.encodeMasked(a, "<S>", &.{}));
    var changed = try fixture(a, true, true);
    defer changed.deinit();
    try t.expectError(error.UnsafeNormalizedAddedTokenMask, changed.encodeMasked(a, "<S>", &.{.{ 1, 2 }}));
    _ = try changed.encodeMasked(a, "abc", &.{.{ 0, 3 }});
}

test "a normalized gap memo never crosses masks for identical text" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try fixture(a, true, false);
    defer tokenizer.deinit();
    tokenizer.memo = try tok.Memo.create(a, 1 << 20);
    const source = try a.alloc(u8, 8192);
    @memset(source, 'a');
    @memcpy(source[4000..4003], "<S>");
    const open = try tokenizer.encode(a, source);
    const before = tokenizer.memo.?.hits;
    const masked = try tokenizer.encodeMasked(a, source, &.{.{ 4001, 4002 }});
    try t.expectEqual(before, tokenizer.memo.?.hits);
    const special = tokenizer.specialTokenId("<S>").?;
    try t.expect(std.mem.indexOfScalar(u32, open, special) != null);
    try t.expect(std.mem.indexOfScalar(u32, masked, special) == null);
    try t.expectEqualSlices(u32, open, try tokenizer.encode(a, source));
}

test "lossy UTF8 keeps its request mask while adjacent literal matches survive" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try fixture(a, false, false);
    defer tokenizer.deinit();
    const ids = try tokenizer.encodeMasked(a, "\xff<S>", &.{.{ 0, 1 }});
    try t.expectEqualSlices(u32, try tokenizer.encode(a, "\u{fffd}<S>"), ids);
}

test "masked gaps refuse added ids emitted by WordLevel and BPE vocabulary overlap" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        \\{"added_tokens":[{"content":"<V>","special":true,"normalized":false}],"model":{"type":"WordLevel","unk_token":"?","vocab":{"?":0,"<V>":1}}}
        ,
        \\{"added_tokens":[{"content":"<V>","special":true,"normalized":false}],"model":{"type":"BPE","ignore_merges":true,"vocab":{"?":0,"<":1,"V":2,">":3,"<V>":4},"merges":[]}}
        ,
        \\{"added_tokens":[{"content":"<V>","special":true,"normalized":true}],"model":{"type":"WordLevel","unk_token":"?","vocab":{"?":0,"<V>":1}}}
        ,
    }) |source| {
        var tokenizer = try tok.parse(a, source);
        defer tokenizer.deinit();
        const id = tokenizer.specialTokenId("<V>").?;
        try t.expectEqualSlices(u32, &.{id}, try tokenizer.encodePlain(a, "<V>"));
        try t.expectEqualSlices(u32, &.{id}, try tokenizer.encodeMasked(a, "<V>", &.{}));
        for (0..3) |i| try t.expectError(error.UnsafeAddedTokenMask, tokenizer.encodeMasked(a, "<V>", &.{.{ i, i + 1 }}));
        try t.expectEqualSlices(u32, &.{0}, try tokenizer.encodeMasked(a, "?", &.{.{ 0, 1 }}));
    }
}

test "masked gap overlap checks memo hits even when no added spelling occurs" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try a.alloc(u8, 8192);
    @memset(text, 'a');
    const source = try std.fmt.allocPrint(a,
        \\{{"added_tokens":[{{"content":"<V>","special":true,"normalized":false}}],"model":{{"type":"WordLevel","unk_token":"?","vocab":{{"?":0,"<V>":1,"{s}":1}}}}}}
    , .{text});
    var tokenizer = try tok.parse(a, source);
    defer tokenizer.deinit();
    tokenizer.memo = try tok.Memo.create(a, 1 << 20);
    try t.expectEqualSlices(u32, &.{1}, try tokenizer.encode(a, text));
    const hits = tokenizer.memo.?.hits;
    try t.expectError(error.UnsafeAddedTokenMask, tokenizer.encodeMasked(a, text, &.{.{ 0, text.len }}));
    try t.expect(tokenizer.memo.?.hits > hits);
    try t.expectEqualSlices(u32, &.{1}, try tokenizer.encode(a, text));
}

test "a template-only normalized match in a mixed gap keeps its added id" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try fixture(a, true, false);
    defer tokenizer.deinit();
    try t.expectEqualSlices(u32, try tokenizer.encode(a, "<S>a"), try tokenizer.encodeMasked(a, "<S>a", &.{.{ 3, 4 }}));
}

test "a request gap with no normalized added spelling takes the memo path" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try fixture(a, true, true);
    defer tokenizer.deinit();
    tokenizer.memo = try tok.Memo.create(a, 1 << 20);
    const text = try a.alloc(u8, 96);
    for (text, 0..) |*c, i| c.* = "ab "[i % 3];
    const spans = [_]tok.Span{.{ 0, text.len }};
    var work: tok.MaskWork = .{};
    _ = try tokenizer.encodeMaskedCounted(a, text, &spans, &work); // a miss stores the gap
    try t.expectEqual(@as(usize, 0), work.masked);
    const hits = tokenizer.memo.?.hits;
    const ids = try tokenizer.encodeMaskedCounted(a, text, &spans, &work); // the same gap comes back
    try t.expectEqual(hits + 1, tokenizer.memo.?.hits);
    try t.expectEqual(@as(usize, 0), work.masked);
    var fresh = try fixture(a, true, true);
    defer fresh.deinit();
    try t.expectEqualSlices(u32, try fresh.encodeMasked(a, text, &spans), ids);
}

test "a request gap carrying a normalized added spelling still takes the mask" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tokenizer = try fixture(a, true, false);
    defer tokenizer.deinit();
    tokenizer.memo = try tok.Memo.create(a, 1 << 20);
    const special = tokenizer.specialTokenId("<S>").?;
    var work: tok.MaskWork = .{};
    const inside = try tokenizer.encodeMaskedCounted(a, "a<S>b", &.{.{ 1, 4 }}, &work);
    try t.expectEqual(@as(usize, 1), work.masked);
    try t.expect(std.mem.indexOfScalar(u32, inside, special) == null);
    try t.expectEqual(@as(u64, 0), tokenizer.memo.?.hits);
    var beside: tok.MaskWork = .{};
    const outside = try tokenizer.encodeMaskedCounted(a, "a<S>b", &.{.{ 0, 1 }}, &beside);
    try t.expectEqual(@as(usize, 1), beside.masked); // the gap holds the spelling, not just the span
    try t.expect(std.mem.indexOfScalar(u32, outside, special) != null);
    var changed = try fixture(a, true, true);
    defer changed.deinit();
    var refused: tok.MaskWork = .{};
    try t.expectError(error.UnsafeNormalizedAddedTokenMask, changed.encodeMaskedCounted(a, "a<S>b", &.{.{ 1, 4 }}, &refused));
    try t.expectEqual(@as(usize, 1), refused.masked);
}

test "request gaps keep the base commit's ids over random and adversarial texts" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixtures = [_]tok.Tokenizer{
        try fixture(a, true, true),
        try fixture(a, false, true),
        try unicodeFixture(a, "NFC"),
        try unicodeFixture(a, "NFD"),
    };
    defer for (&fixtures) |*f| f.deinit();
    for (&fixtures) |*f| f.memo = try tok.Memo.create(a, 1 << 20);
    var lines = std.mem.splitScalar(u8, @embedFile("fixtures/masked_golden.txt"), '\n');
    var cases: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var fields = std.mem.tokenizeScalar(u8, line, ' ');
        try t.expectEqualStrings("G", fields.next().?);
        const which = try std.fmt.parseInt(usize, fields.next().?, 10);
        const text_hex = fields.next().?;
        const text = try a.alloc(u8, text_hex.len / 2);
        _ = try std.fmt.hexToBytes(text, text_hex);
        const span_field = fields.next().?;
        var spans: std.ArrayList(tok.Span) = .empty;
        if (!std.mem.eql(u8, span_field, "-")) {
            var parts = std.mem.splitScalar(u8, span_field, ',');
            while (parts.next()) |part| {
                var ends = std.mem.splitScalar(u8, part, '-');
                const from = try std.fmt.parseInt(usize, ends.next().?, 10);
                const to = try std.fmt.parseInt(usize, ends.next().?, 10);
                try spans.append(a, .{ from, to });
            }
        }
        const got = fixtures[which].encodeMasked(a, text, spans.items);
        const first = fields.next().?;
        if (first[0] == '!') {
            const name = first[1..];
            if (std.mem.eql(u8, name, "UnsafeNormalizedAddedTokenMask"))
                try t.expectError(error.UnsafeNormalizedAddedTokenMask, got)
            else {
                try t.expectEqualStrings("UnsafeAddedTokenMask", name);
                try t.expectError(error.UnsafeAddedTokenMask, got);
            }
        } else {
            var want: std.ArrayList(u32) = .empty;
            try want.append(a, try std.fmt.parseInt(u32, first, 10));
            while (fields.next()) |id| try want.append(a, try std.fmt.parseInt(u32, id, 10));
            const ids = try got;
            try t.expectEqualSlices(u32, want.items, ids);
            try t.expectEqualSlices(u32, want.items, try fixtures[which].encodeMasked(a, text, spans.items));
        }
        cases += 1;
    }
    try t.expectEqual(@as(usize, 96), cases);
}
