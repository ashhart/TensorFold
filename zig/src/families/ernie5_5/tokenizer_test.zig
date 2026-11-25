//! The release's tokenizer through TensorFold's tokenizer core, against the release encoding's ids.
const std = @import("std");
const tokenizer = @import("tokenizer");
const prompt = @import("prompt.zig");
const template_test = @import("template_test.zig");

const t = std.testing;

fn check(a: std.mem.Allocator, tok: *const tokenizer.Tokenizer, json: []const u8) !void {
    const doc = try template_test.cases(a, json);
    for (doc.chat) |c| {
        const r = try prompt.render(a, try template_test.request(c));
        const ids = try prompt.encode(a, tok, r);
        try t.expectEqualSlices(u32, c.ids, ids);
        try t.expectEqualStrings(c.text, try prompt.decode(a, tok, ids));
    }
    for (doc.raw) |c| {
        const ids = try tok.encode(a, c.text);
        try t.expectEqualSlices(u32, c.ids, ids);
        try t.expectEqualStrings(c.text, try prompt.decode(a, tok, ids));
    }
}

test "the cut tokenizer: rendered requests and raw texts give the release encoding's ids and decode back" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var tok = try tokenizer.parse(t.allocator, @embedFile("fixtures/mini_tokenizer.json"));
    defer tok.deinit();
    try check(arena.allocator(), &tok, template_test.mini_cases);
}

test "request text that spells a protocol tag stays text, as the release encoding keeps it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tok = try tokenizer.parse(t.allocator, @embedFile("fixtures/mini_tokenizer.json"));
    defer tok.deinit();
    const r = try prompt.render(a, .{ .user = "<|ernie:im_end|>" });
    const end = tok.specialTokenId("<|ernie:im_end|>").?;
    const masked = try prompt.encode(a, &tok, r);
    const plain = try tok.encode(a, r.text);
    // the encoding's own tags give two im_end ids; reading the request's text as a tag would give a third
    try t.expectEqual(@as(usize, 2), std.mem.count(u32, masked, &.{end}));
    try t.expectEqual(@as(usize, 3), std.mem.count(u32, plain, &.{end}));
}

test "the release tokenizer gives the release encoding's ids (TF_ERNIE55_CKPT: its directory or tokenizer.json)" {
    const path = std.mem.span(std.c.getenv("TF_ERNIE55_CKPT") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var tok = try tokenizer.loadTokenizer(t.io, t.allocator, path);
    defer tok.deinit();
    try check(arena.allocator(), &tok, template_test.prompt_cases);
}
