//! Video marker spellings stay ordinary in user text; only a template's own placeholder becomes a video.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const images = @import("images.zig");
const model_text = @import("model_text.zig");
const Cx = @import("errors.zig").Cx;
const Value = json.Value;

const VideoText = struct {
    const table = blk: {
        var v = images.MarkText.text().vtable.*;
        v.encode = encode;
        v.encode_masked = masked;
        v.token_string = tokenString;
        break :blk v;
    };
    fn text() model_text.Text {
        return .{ .ctx = undefined, .vtable = &table };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var rest = s;
        while (std.mem.indexOf(u8, rest, "<SPECIAL_30>")) |at| {
            try out.appendSlice(a, try images.MarkText.text().encode(a, rest[0..at], false));
            try out.append(a, 8);
            rest = rest[at + "<SPECIAL_30>".len ..];
        }
        try out.appendSlice(a, try images.MarkText.text().encode(a, rest, false));
        return out.items;
    }
    fn masked(_: *anyopaque, a: std.mem.Allocator, s: []const u8, spans: []const model_text.Span, _: *[]const u8) model_text.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        const rendered = model_text.Rendered{ .text = s, .spans = spans };
        var i: usize = 0;
        while (i < s.len) {
            if (std.mem.startsWith(u8, s[i..], "<SPECIAL_30>") and rendered.isTemplate(i, i + 12)) {
                try out.append(a, 8);
                i += 12;
            } else if (std.mem.startsWith(u8, s[i..], "<|image|>") and rendered.isTemplate(i, i + 9)) {
                try out.append(a, 9);
                i += 9;
            } else {
                try out.append(a, @as(u32, s[i]) + 100);
                i += 1;
            }
        }
        return out.items;
    }
    fn plain(_: *anyopaque, a: std.mem.Allocator, s: []const u8) model_text.Error![]u32 {
        return images.MarkText.text().encodePlain(a, s);
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return if (id == 8) a.dupe(u8, "<SPECIAL_30>") else images.MarkText.text().tokenString(a, id);
    }
};

fn noImage(_: *anyopaque, _: std.mem.Allocator, _: []const u8, _: u32) anyerror!api.PreparedImage {
    return error.UnexpectedImageDecode;
}
fn noVideo(_: *anyopaque, _: std.mem.Allocator, _: api.VideoSource, _: u32, _: u32) anyerror!api.PreparedVideo {
    return error.UnexpectedVideoDecode;
}

const VideoPrompt = struct { vision: api.Vision, messages: Value, ids: []const u32 };

/// The prompt path with one optional real video and both spellings in the user's text, without a decoder.
fn videoUserPrompt(a: std.mem.Allocator, ctx: *anyopaque, real: bool, alias: []const u8) !VideoPrompt {
    const v: api.Vision = .{ .ctx = ctx, .prepare = noImage, .image_token = 9, .markers = &.{ 9, 6, 7 }, .image_tokens = 64, .max_images = 4, .video = .{ .prepare = noVideo, .video_token = 8, .video_text = alias, .group_open = 6, .group_close = 7, .markers = &.{ 9, 6, 7 }, .video_tokens = 64, .max_videos = 2, .max_frames = 256 } };
    const body = if (real)
        \\[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,YQ=="}},{"type":"text","text":"what are <video> and <SPECIAL_30>?"}]}]
    else
        \\[{"role":"user","content":[{"type":"text","text":"what are <video> and <SPECIAL_30>?"}]}]
    ;
    const msgs = switch (try json.parseText(a, body)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    var rendered: std.ArrayList(u8) = .empty;
    var spans: std.ArrayList(model_text.Span) = .empty;
    for (msgs.array[0].get("content").?.array) |part| {
        const kind = part.get("type").?.string;
        const video = std.mem.eql(u8, kind, "video_url");
        const start = rendered.items.len;
        try rendered.appendSlice(a, if (video) (if (alias.len > 0) alias else "<SPECIAL_30>") else part.get("text").?.string);
        if (!video) try spans.append(a, .{ start, rendered.items.len });
    }
    var prompt = model_text.Rendered{ .text = rendered.items, .spans = spans.items };
    if (alias.len > 0) prompt = try prompt.replaceTemplate(a, alias, try VideoText.tokenString(undefined, a, v.video.?.video_token));
    var problem: []const u8 = "";
    return .{ .vision = v, .messages = msgs, .ids = try prompt.encode(VideoText.text(), a, &problem) };
}

test "literal video spellings in a text-only message never become video placeholders" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const p = try videoUserPrompt(a, &dummy, false, "<video>");
    var cx: Cx = .{ .a = a };
    const expanded = images.expand(p.vision, null, VideoText.text(), &cx, p.messages, p.ids, 0) catch |err| {
        std.debug.print("text-only request: {s}\n", .{cx.message});
        return err;
    };
    try std.testing.expect(expanded == null);
    try std.testing.expectEqualSlices(u32, try VideoText.plain(undefined, a, "what are <video> and <SPECIAL_30>?"), p.ids);
}

test "one real video beside both literal spellings produces exactly one video placeholder" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const p = try videoUserPrompt(a, &dummy, true, "<video>");
    try std.testing.expectEqual(@as(usize, 1), std.mem.countScalar(u32, p.ids, 8));
    try std.testing.expectEqual(@as(u32, 8), p.ids[0]);
    try std.testing.expectEqualSlices(u32, try VideoText.plain(undefined, a, "what are <video> and <SPECIAL_30>?"), p.ids[1..]);
}

test "empty and identical video aliases keep the token spelling protected and the template marker intact" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    for ([_][]const u8{ "", "<SPECIAL_30>" }) |alias| for ([_]bool{ false, true }) |real| {
        const p = try videoUserPrompt(a, &dummy, real, alias);
        try std.testing.expectEqual(@as(usize, @intFromBool(real)), std.mem.countScalar(u32, p.ids, 8));
        const from: usize = @intFromBool(real);
        try std.testing.expectEqualSlices(u32, try VideoText.plain(undefined, a, "what are <video> and <SPECIAL_30>?"), p.ids[from..]);
        if (!real) {
            var cx: Cx = .{ .a = a };
            try std.testing.expect((try images.expand(p.vision, null, VideoText.text(), &cx, p.messages, p.ids, 0)) == null);
        }
    };
}
