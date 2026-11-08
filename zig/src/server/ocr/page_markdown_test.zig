//! A page-Markdown reader (geometry none, as GLM-OCR) through /v1/ocr against /v1/chat/completions for one reply.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const routes = @import("../routes.zig");
const harness = @import("harness_test.zig");
const Harness = harness.Harness;
const t = std.testing;
const Allocator = std.mem.Allocator;

/// GLM-OCR's reader on Markdown without HTML tables or \( \[ math: the whole reply, trimmed, as one page block.
fn wholePage(a: Allocator, text: []const u8, _: api.Size, _: api.Size) anyerror![]api.ocr.OcrBlock {
    const page = std.mem.trim(u8, text, &std.ascii.whitespace);
    if (page.len == 0) return &.{};
    return a.dupe(api.ocr.OcrBlock, &.{.{ .content = page, .quad = null, .kind = .page }});
}

/// A POST through the dispatcher; the reply must be a 200, and its JSON body is returned.
fn post(a: Allocator, h: *Harness, path: []const u8, body: []const u8) !json.Value {
    var wire: harness.Wire = .{ .a = a };
    routes.dispatch(h.srv, try harness.request(a, &wire, path, "application/json", body), a);
    try t.expectEqual(@as(u16, 200), wire.status());
    return wire.parsed();
}

test "ocr mistral: a page-Markdown reader's page markdown is the chat route's reply for the same image and prompt" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // what GLM-OCR writes for a page: headings, a list, a pipe table, inline math and code, and a trailing newline
    const reply = "# Invoice 0042\n\n- Apples: 3\n- Pears: 5\n\n| Item | Qty |\n| --- | --- |\n| Apples | 3 |\n\nArea $\\pi r^2$, `code`\n";
    var h: Harness = .{ .replies = &.{ .{ .text = reply }, .{ .text = reply } }, .reader = .{ .prompt = "Text Recognition:", .geometry = .none, .parse = wholePage } };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    const png = try harness.png(a, 40, 20);
    const enc = std.base64.standard.Encoder;
    const b64 = try a.alloc(u8, enc.calcSize(png.len));
    const url = try std.fmt.allocPrint(a, "data:image/png;base64,{s}", .{enc.encode(b64, png)});
    const ocr = try post(a, &h, "/v1/ocr", try std.fmt.allocPrint(a, "{{\"model\": \"m\", \"document\": {{\"type\": \"image_url\", \"image_url\": \"{s}\"}}}}", .{url}));
    const chat_body = try std.fmt.allocPrint(a, "{{\"model\": \"m\", \"temperature\": 0, \"chat_template_kwargs\": {{\"enable_thinking\": false}}, " ++
        "\"messages\": [{{\"role\": \"user\", \"content\": [{{\"type\": \"image_url\", \"image_url\": {{\"url\": \"{s}\"}}}}, " ++
        "{{\"type\": \"text\", \"text\": \"Text Recognition:\"}}]}}]}}", .{url});
    const chat = try post(a, &h, "/v1/chat/completions", chat_body);
    const content = chat.get("choices").?.array[0].get("message").?.strField("content").?;
    try t.expectEqualStrings(reply, content);
    const page = ocr.get("pages").?.array[0];
    // the page markdown is the reply byte for byte, only trimmed; a reader without boxes gives no blocks
    try t.expectEqualStrings(std.mem.trim(u8, content, &std.ascii.whitespace), page.strField("markdown").?);
    try t.expectEqual(@as(usize, 0), page.get("blocks").?.array.len);
    try t.expectEqual(@as(usize, 0), page.get("tables").?.array.len);
}
