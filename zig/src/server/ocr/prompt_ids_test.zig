//! An OCR page's prompt ids against a /v1/chat/completions request for the same image and reader prompt.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("../model_text.zig");
const routes = @import("../routes.zig");
const Server = @import("../server.zig").Server;
const TestText = @import("../stream_preflight_test.zig").TestText;
const harness = @import("harness_test.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;

/// dots.mocr's reader prompt (families/dots_mocr/ocr.zig), which a server test cannot import; pinned below.
const dots_prompt =
    \\Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.
    \\
    \\1. Bbox format: [x1, y1, x2, y2]
    \\
    \\2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
    \\
    \\3. Text Extraction & Formatting Rules:
    \\    - Picture: For the 'Picture' category, the text field should be omitted.
    \\    - Formula: Format its text as LaTeX.
    \\    - Table: Format its text as HTML.
    \\    - All Others (Text, Title, etc.): Format their text as Markdown.
    \\
    \\4. Constraints:
    \\    - The output text must be the original text from the image, with no translation.
    \\    - All layout elements must be sorted according to human reading order.
    \\
    \\5. Final Output: The entire output must be a single JSON object.
    \\
;

/// The SHA-256 each family's reader test checks its prompt against: the prompt every oracle run recorded.
const Pinned = struct { prompt: []const u8, sha256: []const u8 };
const readers = [_]Pinned{
    .{ .prompt = dots_prompt, .sha256 = "16ff71ac5d218f35e5b3db41240b6e70741498bc099db3fa922ce1ff972e3b2f" },
};

/// The template's image placeholder: a byte no request text holds.
const image_byte = 1;

/// Bytes as tokens, and a template that writes all it is given: every render option, each message's role and parts.
const EchoText = struct {
    fn text() model_text.Text {
        return .{ .ctx = undefined, .vtable = &.{ .encode = TestText.encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .added = added } };
    }
    fn render(_: *anyopaque, a: Allocator, messages: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        var out: std.ArrayList(u8) = .empty;
        const clear: u8 = if (o.clear_thinking) |c| @intFromBool(c) else 2;
        try out.print(a, "thinking={d} gen={d} images={d} effort={s} tools={d} choice={d} format={d} clear={d}\n", .{
            @intFromBool(o.enable_thinking),         @intFromBool(o.add_generation_prompt), @intFromBool(o.images),
            o.reasoning_effort orelse "-",           @intFromBool(o.tools != null),         @intFromBool(o.tool_choice != null),
            @intFromBool(o.response_format != null), clear,
        });
        if (messages == .array) for (messages.array) |m| {
            try out.print(a, "{s}:", .{m.strField("role") orelse "?"});
            switch (m.get("content") orelse json.Value.null) {
                .string => |s| try out.appendSlice(a, s),
                .array => |parts| for (parts) |p| {
                    const is_text = std.mem.eql(u8, p.strField("type") orelse "", "text");
                    if (is_text) try out.appendSlice(a, p.strField("text") orelse "") else try out.append(a, image_byte);
                },
                else => {},
            }
            try out.append(a, '\n');
        };
        return .{ .text = out.items };
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *b| b.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
    fn added(_: *anyopaque, id: u32) bool {
        return id == image_byte;
    }
};

/// One request as the engine received it: the prompt ids, its images' places and identities, and greedy decoding.
const Seen = struct { prompt: []u32, images: []api.Image, greedy: bool };

/// An engine with images and an OCR reader whose prompt is ``reader``; it records each request and replies "ok".
const Recorder = struct {
    reader: []const u8,
    seen: std.ArrayList(Seen) = .empty,

    fn engine(r: *Recorder) api.Engine {
        return .{ .ctx = r, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn deinit(r: *Recorder) void {
        for (r.seen.items) |s| {
            t.allocator.free(s.prompt);
            t.allocator.free(s.images);
        }
        r.seen.deinit(t.allocator);
    }
    fn info(ctx: *anyopaque) api.Info {
        const r: *Recorder = @ptrCast(@alignCast(ctx));
        const v: api.Vision = .{ .ctx = r, .prepare = prepare, .image_token = image_byte, .image_tokens = 64, .max_images = 4 };
        return .{ .context_window = 1 << 16, .vision = v, .ocr = .{ .prompt = r.reader, .parse = noBlocks } };
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const r: *Recorder = @ptrCast(@alignCast(ctx));
        const prompt = t.allocator.dupe(u32, req.prompt) catch return error.MemoryPressure;
        errdefer t.allocator.free(prompt);
        const imgs = t.allocator.dupe(api.Image, req.images) catch return error.MemoryPressure;
        errdefer t.allocator.free(imgs);
        r.seen.append(t.allocator, .{ .prompt = prompt, .images = imgs, .greedy = req.sampling == null }) catch return error.MemoryPressure;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &[_]u32{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    /// One tower token an image, identified by its bytes, framed as the reader families frame theirs.
    fn prepare(_: *anyopaque, a: Allocator, bytes: []const u8, _: u32) anyerror!api.PreparedImage {
        return .{ .pixels = try a.alloc(f32, 0), .gh = 1, .gw = 1, .tokens = 1, .hash = std.hash.Wyhash.hash(0, bytes), .frame = .{ .width = 32, .height = 16 } };
    }
    fn noBlocks(_: Allocator, _: []const u8, _: api.Size, _: api.Size) anyerror![]api.ocr.OcrBlock {
        return &.{};
    }
};

/// A POST through the dispatcher; the reply must be a 200.
fn post(a: Allocator, srv: *Server, path: []const u8, body: []const u8) !void {
    var wire: harness.Wire = .{ .a = a };
    routes.dispatch(srv, try harness.request(a, &wire, path, "application/json", body), a);
    try t.expectEqual(@as(u16, 200), wire.status());
}

/// A chat client's body: one user turn of the page image and the reader prompt, at temperature 0, plus ``extra``.
fn chatBody(a: Allocator, url: []const u8, prompt: []const u8, thinking_off: bool) ![]const u8 {
    const inner = try json.newObject(a);
    try inner.put(a, "url", .{ .string = url });
    const image = try json.newObject(a);
    try image.put(a, "type", .{ .string = "image_url" });
    try image.put(a, "image_url", .{ .object = inner });
    const text = try json.newObject(a);
    try text.put(a, "type", .{ .string = "text" });
    try text.put(a, "text", .{ .string = prompt });
    const user = try json.newObject(a);
    try user.put(a, "role", .{ .string = "user" });
    try user.put(a, "content", .{ .array = try a.dupe(json.Value, &.{ .{ .object = image }, .{ .object = text } }) });
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "messages", .{ .array = try a.dupe(json.Value, &.{.{ .object = user }}) });
    try body.put(a, "temperature", try json.intValue(a, 0));
    if (thinking_off) {
        // how a chat client turns thinking off (fields.thinkingFields reads chat_template_kwargs)
        const kwargs = try json.newObject(a);
        try kwargs.put(a, "enable_thinking", .{ .bool = false });
        try body.put(a, "chat_template_kwargs", .{ .object = kwargs });
    }
    return json.stringify(a, .{ .object = body }, .{});
}

test "ocr prompt ids: a page's prompt ids equal a chat request's for the same image and reader prompt, thinking off" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const png = try harness.png(a, 32, 16);
    const enc = std.base64.standard.Encoder;
    const b64 = try a.alloc(u8, enc.calcSize(png.len));
    const url = try std.fmt.allocPrint(a, "data:image/png;base64,{s}", .{enc.encode(b64, png)});
    for (readers) |r| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(r.prompt, &digest, .{});
        try t.expectEqualStrings(r.sha256, &std.fmt.bytesToHex(digest, .lower));
        var rec: Recorder = .{ .reader = r.prompt };
        defer rec.deinit();
        // thinking on by default, so only the requests' own switch turns it off
        const srv = try Server.init(t.allocator, t.io, rec.engine(), EchoText.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
        defer srv.deinit();
        try post(a, srv, "/v1/ocr", try std.fmt.allocPrint(a, "{{\"model\": \"m\", \"document\": {{\"type\": \"image_url\", \"image_url\": \"{s}\"}}}}", .{url}));
        try post(a, srv, "/v1/chat/completions", try chatBody(a, url, r.prompt, true));
        try post(a, srv, "/v1/chat/completions", try chatBody(a, url, r.prompt, false));
        try t.expectEqual(@as(usize, 3), rec.seen.items.len);
        const ocr = rec.seen.items[0];
        const chat = rec.seen.items[1];
        try t.expectEqualSlices(u32, chat.prompt, ocr.prompt);
        try t.expectEqual(@as(usize, 1), ocr.images.len);
        try t.expectEqual(chat.images.len, ocr.images.len);
        try t.expectEqual(chat.images[0].hash, ocr.images[0].hash);
        try t.expectEqual(chat.images[0].at, ocr.images[0].at);
        try t.expectEqual(chat.images[0].tokens, ocr.images[0].tokens);
        // temperature 0 on both: greedy decoding (only max_tokens differs, the page filling the room left)
        try t.expect(ocr.greedy and chat.greedy);
        // the template saw the reader prompt after the image, with thinking off
        const rendered = try EchoText.decode(undefined, a, ocr.prompt);
        try t.expect(std.mem.startsWith(u8, rendered, "thinking=0 gen=1 images=1 "));
        try t.expect(std.mem.indexOf(u8, rendered, try std.mem.concat(a, u8, &.{ "user:\x01", r.prompt, "\n" })) != null);
        // the comparison is sensitive: the same chat request without its thinking switch renders other ids
        try t.expect(!std.mem.eql(u32, rec.seen.items[2].prompt, ocr.prompt));
    }
}
