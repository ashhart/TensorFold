//! The OCR routes' chat seam: fill_room, per-request image byte limits and each image part's frame.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const chat = @import("chat.zig");
const images = @import("images.zig");
const model_text = @import("model_text.zig");
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;
const TestText = @import("stream_preflight_test.zig").TestText;
const t = std.testing;

/// An engine with `window` tokens of context and, when set, image input.
const Script = struct {
    window: u32,
    vision: ?api.Vision = null,
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = undefined, .cancel = undefined, .status = undefined, .memory = undefined } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return .{ .context_window = s.window, .vision = s.vision };
    }
};

const Gone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

/// The reply limit chat.prepare gives one user message (TestText renders 6 prompt tokens).
fn replyLimit(window: u32, default_max_tokens: i64, fill_room: bool) !u32 {
    var script: Script = .{ .window = window };
    var text: TestText = .{};
    var srv = try Server.init(t.allocator, t.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .default_max_tokens = default_max_tokens, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const messages = (try json.parse(a, "[{\"role\": \"user\", \"content\": \"read this\"}]")).ok;
    const p = try chat.prepare(srv, &cx, .{ .messages = messages, .fields = (try json.parse(a, "{}")).ok, .fill_room = fill_room }, Gone{});
    defer srv.release(p.preparing, p.request);
    try t.expectEqual(@as(usize, 6), p.prompt_len);
    return p.request.max_tokens;
}

test "ocr chat: fill_room: a reply on a 32-token window may fill the room the prompt leaves" {
    try t.expectEqual(@as(u32, 32 - 6), try replyLimit(32, 4, true));
    // without it the default holds, as today; on an unknown window fill_room keeps the default too
    try t.expectEqual(@as(u32, 4), try replyLimit(32, 4, false));
    try t.expectEqual(@as(u32, 4), try replyLimit(0, 4, true));
    try t.expectEqual(@as(u32, 26), try replyLimit(32, 4096, false));
}

/// The tests' images: a JPEG's first bytes, then any bytes; one token each, framed by their length unless 'n' follows.
fn framed(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, _: u32) anyerror!api.PreparedImage {
    const frame: ?api.Size = if (bytes[3] == 'n') null else .{ .width = @intCast(bytes.len), .height = 7 };
    return .{ .pixels = try a.alloc(f32, 0), .gh = 1, .gw = 1, .tokens = 1, .hash = bytes.len, .frame = frame };
}

fn parse(a: std.mem.Allocator, text: []const u8) !json.Value {
    return switch (try json.parseText(a, text)) {
        .ok => |m| m,
        .err => error.TestUnexpectedResult,
    };
}

test "ocr chat: image_bytes: an 11 MiB limit accepts a 10.5 MiB data URL that chat's default refuses in its own words" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = framed, .image_token = 9, .image_tokens = 64, .max_images = 4 };
    const image = try a.alloc(u8, (21 << 20) / 2);
    @memset(image, 'a');
    @memcpy(image[0..3], "\xff\xd8\xff");
    const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(image.len));
    _ = std.base64.standard.Encoder.encode(encoded, image);
    const body = try std.fmt.allocPrint(a, "[{{\"role\": \"user\", \"content\": [{{\"type\": \"image_url\", \"image_url\": {{\"url\": \"data:image/jpeg;base64,{s}\"}}}}]}}]", .{encoded});
    const msgs = try parse(a, body);
    const ids = [_]u32{ 1, 9, 2 };
    try t.expectError(error.Refused, images.expand(v, null, images.MarkText.text(), &cx, msgs, &ids, 0));
    try t.expectEqualStrings("image data URL exceeds the encoded byte limit", cx.message);
    const x = (try images.expandWith(v, null, images.MarkText.text(), &cx, msgs, &ids, 0, .{ .each = 11 << 20, .total = 11 << 20 })).?;
    try t.expectEqual(@as(u64, image.len), x.images[0].hash);
    // the request's total binds as well: a 10 MiB total refuses the same image under an 11 MiB per-image limit
    try t.expectError(error.Refused, images.expandWith(v, null, images.MarkText.text(), &cx, msgs, &ids, 0, .{ .each = 11 << 20, .total = 10 << 20 }));
    try t.expectEqualStrings("image data URL exceeds the encoded byte limit of 10 MB", cx.message);
}

test "ocr chat: image_bytes: chat's default limits refuse a request past 20 MB with images.py's words" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = framed, .image_token = 9, .image_tokens = 64, .max_images = 4 };
    const image = try a.alloc(u8, 7 << 20); // each under chat's 10 MB; three pass its 20 MB request total
    @memset(image, 'a');
    @memcpy(image[0..3], "\xff\xd8\xff");
    const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(image.len));
    _ = std.base64.standard.Encoder.encode(encoded, image);
    const part = try std.fmt.allocPrint(a, "{{\"type\": \"image_url\", \"image_url\": {{\"url\": \"data:image/jpeg;base64,{s}\"}}}}", .{encoded});
    const msgs = try parse(a, try std.fmt.allocPrint(a, "[{{\"role\": \"user\", \"content\": [{s}, {s}, {s}]}}]", .{ part, part, part }));
    try t.expectError(error.Refused, images.expand(v, null, images.MarkText.text(), &cx, msgs, &[_]u32{ 1, 9, 9, 9, 2 }, 0));
    try t.expectEqualStrings("the request's images exceed 20 MB encoded", cx.message);
}

test "ocr chat: frames: one per image part, in part order, null where the family sets none" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = framed, .image_token = 9, .image_tokens = 64, .max_images = 4 };
    const msgs = try parse(a,
        \\[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,%FF%D8%FFab"}}, {"type": "text", "text": "x"}]},
        \\ {"role": "assistant", "content": "y"},
        \\ {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,%FF%D8%FFnope"}},
        \\  {"type": "image_url", "image_url": {"url": "data:,%FF%D8%FFcdef"}}]}]
    );
    const x = (try images.expand(v, null, images.MarkText.text(), &cx, msgs, &[_]u32{ 1, 9, 2, 9, 9, 3 }, 0)).?;
    try t.expectEqual(@as(usize, 3), x.frames.len);
    try t.expectEqual(@as(?api.Size, .{ .width = 5, .height = 7 }), x.frames[0]);
    try t.expectEqual(@as(?api.Size, null), x.frames[1]);
    try t.expectEqual(@as(?api.Size, .{ .width = 7, .height = 7 }), x.frames[2]);
    // a prompt without image parts has no frames to carry
    try t.expect(try images.expand(v, null, images.MarkText.text(), &cx, try parse(a, "[{\"role\": \"user\", \"content\": \"x\"}]"), &[_]u32{ 1, 2 }, 0) == null);
}

/// Bytes as tokens, and every prompt "a#b": one image placeholder ('#', an added token) between two words.
const PlaceText = struct {
    fn text() model_text.Text {
        return .{ .ctx = undefined, .vtable = &.{ .encode = TestText.encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .added = added } };
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *b| b.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "a#b") };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
    fn added(_: *anyopaque, id: u32) bool {
        return id == '#';
    }
};

test "ocr chat: frames: chat.prepare carries each image part's frame from the prompt to the prepared request" {
    var dummy: u8 = 0;
    var script: Script = .{ .window = 64, .vision = .{ .ctx = &dummy, .prepare = framed, .image_token = '#', .image_tokens = 8, .max_images = 2 } };
    var srv = try Server.init(t.allocator, t.io, script.engine(), PlaceText.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const messages = try parse(a, "[{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:,%FF%D8%FFpage\"}}]}]");
    const p = try chat.prepare(srv, &cx, .{ .messages = messages, .fields = (try json.parse(a, "{}")).ok }, Gone{});
    defer srv.release(p.preparing, p.request);
    try t.expectEqual(@as(usize, 1), p.frames.len);
    try t.expectEqual(@as(?api.Size, .{ .width = 7, .height = 7 }), p.frames[0]);
    try t.expectEqual(@as(usize, 1), p.request.images.len);
}
