//! The core tokenizer on a reduced copy of the checkpoint's tokenizer.json, against the reference's ids.
const std = @import("std");
const tokenizer = @import("tokenizer");

const mini = @embedFile("fixtures/mini_tokenizer.json");
const cases = @embedFile("fixtures/tokenizer_cases.json");
const prompts = @embedFile("fixtures/prompt_cases.json");
const t = std.testing;

fn expectIds(tok: *const tokenizer.Tokenizer, text: []const u8, want: std.json.Value) !void {
    const got = try tok.encode(t.allocator, text);
    defer t.allocator.free(got);
    try t.expectEqual(want.array.items.len, got.len);
    for (want.array.items, got) |w, g| try t.expectEqual(@as(u32, @intCast(w.integer)), g);
}

test "plain text, code, numbers, CJK and accents encode to the reference's ids and decode back" {
    var tok = try tokenizer.parse(t.allocator, mini);
    defer tok.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, cases, .{});
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 6), parsed.value.array.items.len);
    for (parsed.value.array.items) |c| {
        const text = c.object.get("text").?.string;
        try expectIds(&tok, text, c.object.get("ids").?);
        const ids = try tok.encode(t.allocator, text);
        defer t.allocator.free(ids);
        const back = try tok.decode(t.allocator, ids, false);
        defer t.allocator.free(back);
        try t.expectEqualStrings(text, back);
    }
}

test "the rendered prompts encode to the reference's ids, special tokens at their own ids" {
    var tok = try tokenizer.parse(t.allocator, mini);
    defer tok.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, prompts, .{});
    defer parsed.deinit();
    for (parsed.value.array.items) |c| try expectIds(&tok, c.object.get("text").?.string, c.object.get("ids").?);
    try t.expectEqual(@as(?u32, 120000), tok.specialTokenId("<｜hy_start:opensource｜>"));
    try t.expectEqual(@as(?u32, 120025), tok.specialTokenId("<｜hy_end:opensource｜>"));
}
