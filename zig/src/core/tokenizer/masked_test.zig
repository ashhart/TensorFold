//! Added spellings may match only runs written by the template, with the ordinary gap path unchanged.
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
