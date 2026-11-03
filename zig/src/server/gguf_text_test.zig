//! Tiny serialized tokenizer metadata exercises the real text provider, without a checkpoint or a device.
const std = @import("std");
const core = @import("core");
const HfText = @import("hf_text.zig").HfText;
const t = std.testing;
fn image(pre: []const u8, with_template: bool) core.gguf_test_fixture.Fixture {
    var f = core.gguf_test_fixture.Fixture.init(0, if (with_template) 10 else 9);
    f.kv("general.architecture", .{ .string = "qwen35" });
    f.kv("tokenizer.ggml.model", .{ .string = "gpt2" });
    f.kv("tokenizer.ggml.pre", .{ .string = pre });
    f.array("tokenizer.ggml.tokens", .string, 5);
    for ([_][]const u8{ "h", "i", "hi", "<end>", "<start>" }) |s| f.string(s);
    f.array("tokenizer.ggml.token_type", .int32, 5);
    for ([_]i32{ 1, 1, 1, 3, 3 }) |kind| f.int(i32, kind);
    f.array("tokenizer.ggml.merges", .string, 1);
    f.string("h i");
    f.kv("tokenizer.ggml.eos_token_id", .{ .uint32 = 3 });
    f.kv("tokenizer.ggml.bos_token_id", .{ .uint32 = 4 });
    f.kv("tokenizer.ggml.add_bos_token", .{ .boolean = false });
    if (with_template) f.kv("tokenizer.chat_template", .{ .string = "{{ messages[0]['content'] }}<end>" });
    f.finish(32, 0);
    return f;
}
test "GGUF text: vocabulary, merges, specials and template come from the serialized source" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var f = image("qwen35", true);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tiny.gguf", .data = f.slice() });
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try tmp.dir.realPathFileAlloc(t.io, "tiny.gguf", a);
    var problem: []const u8 = "";
    const hf = try HfText.load(t.allocator, t.io, path, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const ids = try text.encode(a, "hi<end>", false);
    try t.expectEqualSlices(u32, &.{ 2, 3 }, ids);
    try t.expectEqualSlices(u32, &.{3}, text.eosIds());
    try t.expect(text.isSpecial(3));
    try t.expectEqualStrings("{{ messages[0]['content'] }}<end>", text.templateSource());
    try t.expectEqualStrings("hi<end>", try text.decode(a, ids));
}
test "GGUF text: missing template and unsupported tokenizer are named refusals" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "tiny.gguf" });
    var problem: []const u8 = "";
    var missing = image("qwen35", false);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tiny.gguf", .data = missing.slice() });
    try t.expectError(error.UnsupportedGgufText, HfText.load(t.allocator, t.io, path, null, a, &problem));
    try t.expect(std.mem.startsWith(u8, problem, "tokenizer.chat_template: missing"));
    var unsupported = image("unknown", true);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tiny.gguf", .data = unsupported.slice() });
    try t.expectError(error.UnsupportedGgufText, HfText.load(t.allocator, t.io, path, null, a, &problem));
    try t.expect(std.mem.startsWith(u8, problem, "tokenizer.ggml.pre:"));
}
test "GGUF text: each required tokenizer key is refused by its own name" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "missing.gguf" });
    for ([_][]const u8{ "general.architecture", "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.add_bos_token", "tokenizer.chat_template" }) |key| {
        var missing = image("qwen35", true);
        missing.bytes[std.mem.indexOf(u8, missing.slice(), key).?] = 'X';
        try tmp.dir.writeFile(t.io, .{ .sub_path = "missing.gguf", .data = missing.slice() });
        var problem: []const u8 = "";
        try t.expectError(error.UnsupportedGgufText, HfText.load(t.allocator, t.io, path, null, a, &problem));
        try t.expect(std.mem.startsWith(u8, problem, key));
    }
}
