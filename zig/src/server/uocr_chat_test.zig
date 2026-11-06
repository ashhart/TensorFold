//! Unlimited-OCR-2's built-in chat template: used when the checkpoint ships none; its own and --chat-template win.
const std = @import("std");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;

test "Unlimited-OCR-2 serves its family's template when the checkpoint has none; the checkpoint's and the override win" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"<s>":0,"</s>":1,"H":2},"merges":[]},"added_tokens":[{"id":0,"content":"<s>","special":true},{"id":1,"content":"</s>","special":true}]}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"bos_token\":\"<s>\",\"eos_token\":\"</s>\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"unlimited-ocr\",\"bos_token_id\":0,\"eos_token_id\":1}" });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var problem: []const u8 = "";
    const page = "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,\"}}";
    const messages = (try json.parse(a, "[{\"role\":\"user\",\"content\":[" ++ page ++ "," ++ page ++ ",{\"type\":\"text\",\"text\":\"Free OCR.\"}]}]")).ok;
    const bare = (try json.parse(a, "[{\"role\":\"user\",\"content\":[" ++ page ++ "]}]")).ok;
    const built_in = try HfText.load(gpa, io, dir, null, a, &problem);
    defer built_in.deinit();
    try std.testing.expectEqualStrings("<s><PAGE><image><PAGE><image>Free OCR.", (try built_in.text().render(a, messages, .{}, &problem)).text);
    try std.testing.expectEqualStrings("<s><PAGE><image>Multi page parsing.", (try built_in.text().render(a, bare, .{}, &problem)).text);
    try tmp.dir.writeFile(io, .{ .sub_path = "override.jinja", .data = "override" });
    const override = try std.fmt.allocPrint(a, "{s}/override.jinja", .{dir});
    const overridden = try HfText.load(gpa, io, dir, override, a, &problem);
    defer overridden.deinit();
    try std.testing.expectEqualStrings("override", (try overridden.text().render(a, messages, .{}, &problem)).text);
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = "checkpoint" });
    const shipped = try HfText.load(gpa, io, dir, null, a, &problem);
    defer shipped.deinit();
    try std.testing.expectEqualStrings("checkpoint", (try shipped.text().render(a, messages, .{}, &problem)).text);
    try tmp.dir.deleteFile(io, "chat_template.jinja");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"other\",\"bos_token_id\":0,\"eos_token_id\":1}" });
    const other = try HfText.load(gpa, io, dir, null, a, &problem);
    defer other.deinit();
    try std.testing.expectEqual(@as(usize, 0), other.source.len);
}
