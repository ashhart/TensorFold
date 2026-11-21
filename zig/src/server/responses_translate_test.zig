const std = @import("std");
const json = @import("json");
const responses = @import("responses_translate.zig");
const Cx = @import("errors.zig").Cx;

test "input_file refuses remote file_url" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const items = (try json.parse(cx.a,
        \\[{"role": "user", "content": [{"type": "input_file", "file_url": "https://example.com/doc.pdf"}]}]
    )).ok.array;
    try std.testing.expectError(error.Refused, responses.messages(&cx, items));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "URLs are not accepted") != null);
}

test "reasoning replay retains successive, empty-middle and trailing text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const items = (try json.parse(cx.a,
        \\[
        \\{"type":"reasoning","content":[{"type":"text","text":"one"}]},
        \\{"type":"reasoning","content":[]},
        \\{"type":"reasoning","content":[{"type":"reasoning_text","text":" two"}]},
        \\{"type":"message","role":"assistant","content":"answer"},
        \\{"type":"reasoning","content":[{"type":"text","text":" trailing"}]}
        \\]
    )).ok.array;
    const out = try responses.messages(&cx, items);
    try std.testing.expectEqual(@as(usize, 1), out.len);
    try std.testing.expectEqualStrings("one two trailing", out[0].get("reasoning_content").?.string);
    try std.testing.expectEqualStrings("answer", out[0].get("content").?.string);
}

test "malformed reasoning parts never suppress encrypted refusal" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const invalid = [_][]const u8{
        "[{\"type\":\"reasoning\",\"content\":[{}],\"encrypted_content\":\"cipher\"}]",
        "[{\"type\":\"reasoning\",\"content\":{},\"encrypted_content\":\"cipher\"}]",
        "[{\"type\":\"reasoning\",\"content\":[{\"type\":\"text\",\"text\":2}],\"encrypted_content\":\"cipher\"}]",
        "[{\"type\":\"reasoning\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"},{}],\"encrypted_content\":\"cipher\"}]",
        "[{\"type\":\"reasoning\",\"content\":[],\"encrypted_content\":\"cipher\"}]",
    };
    for (invalid) |body| {
        const items = (try json.parse(cx.a, body)).ok.array;
        try std.testing.expectError(error.Refused, responses.messages(&cx, items));
    }
    const valid = (try json.parse(cx.a, "[{\"type\":\"reasoning\",\"content\":[{\"type\":\"text\",\"text\":\"clear\"}],\"encrypted_content\":\"cipher\"}]")).ok.array;
    const out = try responses.messages(&cx, valid);
    try std.testing.expectEqualStrings("clear", out[0].get("reasoning_content").?.string);
}
