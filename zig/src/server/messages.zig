//! Chat messages as the template sees them: leading instructions merged, text parts joined, call arguments as objects.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const api = @import("engine_api");
const Value = json.Value;
const Cx = errors.Cx;

const roles = [_][]const u8{ "system", "developer", "user", "assistant", "tool" };

fn isRole(role: ?Value) bool {
    const r = role orelse return false;
    if (r != .string) return false;
    for (roles) |name| if (std.mem.eql(u8, name, r.string)) return true;
    return false;
}

fn withField(cx: *Cx, o: *const json.Object, key: []const u8, value: Value) !*json.Object {
    const copy = try json.copyObject(cx.a, o);
    try copy.put(cx.a, key, value);
    return copy;
}

/// ``normalize_messages`` (text only): leading system and developer text merged, later ones as ``late_system``; a template that needs a user query gains one user turn after a trailing tool run.
pub fn normalize(cx: *Cx, messages: ?Value, late_system: []const u8, needs_user_after_tool: bool) errors.Refused!Value {
    return normalizeWith(cx, messages, late_system, needs_user_after_tool, .{});
}

/// The media parts an engine reads: image_url parts (``Info.vision``), and video_url parts (its ``video``).
pub const Media = struct {
    images: bool = false,
    videos: bool = false,

    pub fn of(v: ?api.Vision) Media {
        return .{ .images = v != null, .videos = v != null and v.?.video != null };
    }
};

/// One video part as the template and the video path read it: {"type": "video_url", "video_url": {"url"}}
/// (videos.py video_source: an object with a url string).
fn videoPart(cx: *Cx, part: Value) errors.Refused!Value {
    const vu = part.get("video_url") orelse return cx.refuse("video_url must contain a non-empty url string");
    const u = if (vu == .object) vu.get("url") else null;
    if (u == null or u.? != .string or u.?.string.len == 0) return cx.refuse("video_url must contain a non-empty url string");
    for ([_][]const u8{ "image", "images", "image_url", "input_image", "audio", "input_audio", "video" }) |k| if (json.truthyField(part, k)) return cx.refuse("video_url parts cannot contain other media");
    const inner = try json.newObject(cx.a);
    try inner.put(cx.a, "url", u.?);
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "type", .{ .string = "video_url" });
    try out.put(cx.a, "video_url", .{ .object = inner });
    return .{ .object = out };
}

/// One image part as the template and the image path read it: {"type": "image_url", "image_url": {"url", "detail"?}}.
fn imagePart(cx: *Cx, part: Value) errors.Refused!Value {
    const iu = part.get("image_url") orelse return cx.refuse("an image_url content part must contain image_url");
    var url: []const u8 = "";
    var detail: ?Value = null;
    if (iu == .string) url = iu.string else if (iu == .object) {
        const u = iu.get("url") orelse return cx.refuse("image_url must contain a url string");
        if (u != .string) return cx.refuse("image_url must contain a url string");
        url = u.string;
        detail = iu.get("detail");
        if (detail) |d| if (d != .string or !(std.mem.eql(u8, d.string, "auto") or std.mem.eql(u8, d.string, "low") or std.mem.eql(u8, d.string, "high")))
            return cx.refuse("image_url detail must be auto, low or high");
    } else return cx.refuse("image_url must be a url string or an object with one");
    if (url.len == 0) return cx.refuse("image_url must contain a url string");
    for ([_][]const u8{ "audio", "input_audio", "video", "video_url" }) |k| if (json.truthyField(part, k)) return cx.refuse("image_url parts cannot contain other media");
    const inner = try json.newObject(cx.a);
    try inner.put(cx.a, "url", .{ .string = url });
    if (detail) |d| try inner.put(cx.a, "detail", d);
    const out = try json.newObject(cx.a);
    try out.put(cx.a, "type", .{ .string = "image_url" });
    try out.put(cx.a, "image_url", .{ .object = inner });
    return .{ .object = out };
}

/// `normalize`, and with `media.images` (an engine that reads them) a user or tool message's image_url parts kept
/// beside its text parts, in order, for the template's image markers, and with `media.videos` a user message's
/// video_url parts the same way; a message without media parts is joined text as before.
pub fn normalizeWith(cx: *Cx, messages: ?Value, late_system: []const u8, needs_user_after_tool: bool, media: Media) errors.Refused!Value {
    const images = media.images;
    const list = messages orelse return cx.refuse("messages must be a non-empty list");
    if (list != .array or list.array.len == 0) return cx.refuse("messages must be a non-empty list");
    var out: std.ArrayList(Value) = .empty;
    var instructions: std.ArrayList(*json.Object) = .empty;
    for (list.array) |message| {
        if (message != .object) return cx.refuse("each message must be an object");
        const role = message.get("role");
        if (!isRole(role)) return cx.refuse("message role must be system, developer, user, assistant or tool");
        if (fields.hasMedia(message)) return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
        const content = message.get("content");
        var item: *json.Object = message.object;
        if (content != null and content.? == .array) {
            var text: std.ArrayList(u8) = .empty;
            var parts: std.ArrayList(Value) = .empty; // with images: text and image parts in order
            var has_image = false;
            for (content.?.array) |part| {
                const kind = if (part == .object and part.get("type") != null and part.get("type").? == .string) part.get("type").?.string else "";
                if (images and std.mem.eql(u8, kind, "image_url")) {
                    // a tool's result carries images as agents send screenshots; the template renders them in its response
                    if (!std.mem.eql(u8, role.?.string, "user") and !std.mem.eql(u8, role.?.string, "tool")) return cx.refuse("image parts are accepted in user and tool messages only");
                    try parts.append(cx.a, try imagePart(cx, part));
                    has_image = true;
                    continue;
                }
                if (media.videos and std.mem.eql(u8, kind, "video_url")) {
                    if (!std.mem.eql(u8, role.?.string, "user")) return cx.refuse("video_url parts are supported only in user messages");
                    try parts.append(cx.a, try videoPart(cx, part));
                    has_image = true;
                    continue;
                }
                if (!std.mem.eql(u8, kind, "text") or fields.hasMedia(part)) return cx.refuse(if (media.videos) "content parts must be text, image_url or video_url; audio is unsupported" else if (images) "content parts must be text or image_url; audio and video inputs are unsupported" else "this server accepts text parts only; image, audio and video inputs are unsupported");
                const t = part.get("text") orelse return cx.refuse("a text content part must contain a text string");
                if (t != .string) return cx.refuse("a text content part must contain a text string");
                try text.appendSlice(cx.a, t.string);
                if (images) {
                    const tp = try json.newObject(cx.a);
                    try tp.put(cx.a, "type", .{ .string = "text" });
                    try tp.put(cx.a, "text", t);
                    try parts.append(cx.a, .{ .object = tp });
                }
            }
            item = try withField(cx, message.object, "content", if (has_image) Value{ .array = parts.items } else Value{ .string = text.items });
        } else if (content == null or content.? == .null) {
            item = try withField(cx, message.object, "content", .{ .string = "" });
        } else if (content.? != .string) {
            return cx.refuse("message content must be text or an array of text parts");
        }
        const r = role.?.string;
        if (std.mem.eql(u8, r, "system") or std.mem.eql(u8, r, "developer")) {
            if (out.items.len == 0) {
                try instructions.append(cx.a, item);
                continue;
            }
            if (!std.mem.eql(u8, r, late_system)) item = try withField(cx, item, "role", .{ .string = late_system });
        }
        try out.append(cx.a, .{ .object = item });
    }
    if (instructions.items.len > 0) {
        var joined: std.ArrayList(u8) = .empty;
        for (instructions.items, 0..) |m, i| {
            if (i > 0) try joined.appendSlice(cx.a, "\n\n");
            try joined.appendSlice(cx.a, m.get("content").?.string);
        }
        const first = try withField(cx, instructions.items[0], "role", .{ .string = "system" });
        try first.put(cx.a, "content", .{ .string = joined.items });
        try out.insert(cx.a, 0, .{ .object = first });
    }
    // A template that demands a user query (it raises "No user query found") refuses a conversation whose user turn became tool results; by the template's test a user turn wholly inside a <|im_start|> block is no query, so only a conversation without one gains a placeholder user turn after its last tool run.
    if (needs_user_after_tool) {
        var has_query = false;
        var last_tool: ?usize = null;
        for (out.items, 0..) |m, i| {
            const role = m.get("role");
            if (role == null or role.? != .string) continue;
            if (std.mem.eql(u8, role.?.string, "user")) {
                const content = m.get("content");
                const text = if (content != null and content.? == .string) content.?.string else "";
                const trimmed = std.mem.trim(u8, text, " \t\n\r\x0b\x0c");
                if (!(std.mem.startsWith(u8, trimmed, "<tool_response>") and std.mem.endsWith(u8, trimmed, "</tool_response>"))) has_query = true;
            } else if (std.mem.eql(u8, role.?.string, "tool")) {
                last_tool = i;
            }
        }
        if (!has_query) {
            const at = if (last_tool) |i| i + 1 else out.items.len;
            const turn = try json.newObject(cx.a);
            try turn.put(cx.a, "role", .{ .string = "user" });
            try turn.put(cx.a, "content", .{ .string = "(tool results above)" });
            try out.insert(cx.a, at, .{ .object = turn });
        }
    }
    return .{ .array = out.items };
}

/// ``_normalize_tool_call_arguments``: call arguments as objects for templates; bad ones under ``_invalid_arguments``.
pub fn toolArguments(cx: *Cx, messages: Value) errors.Refused!Value {
    if (messages != .array) return messages;
    const out = try cx.a.alloc(Value, messages.array.len);
    for (messages.array, out) |message, *slot| {
        slot.* = message;
        const calls = message.get("tool_calls") orelse continue;
        if (calls != .array or calls.array.len == 0) continue;
        var touched = false;
        const new_calls = try cx.a.alloc(Value, calls.array.len);
        for (calls.array, new_calls) |call, *dst| {
            dst.* = call;
            const function = call.get("function") orelse continue;
            if (function != .object or !function.has("arguments")) continue;
            const args = function.get("arguments").?;
            if (args == .object) continue;
            var parsed: ?Value = args;
            if (args == .string) {
                parsed = switch (try json.parseText(cx.a, args.string)) {
                    .ok => |v| v,
                    .err => null,
                };
            }
            if (parsed == null or parsed.? != .object) {
                const wrapped = try json.newObject(cx.a);
                try wrapped.put(cx.a, "_invalid_arguments", args);
                parsed = .{ .object = wrapped };
            }
            const fn_copy = try withField(cx, function.object, "arguments", parsed.?);
            dst.* = .{ .object = try withField(cx, call.object, "function", .{ .object = fn_copy }) };
            touched = true;
        }
        if (touched) slot.* = .{ .object = try withField(cx, message.object, "tool_calls", .{ .array = new_calls }) };
    }
    return .{ .array = out };
}

/// ``{"type": "text", "text": text}``: a chat content part.
pub fn textPart(a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error!Value {
    const p = try json.newObject(a);
    try p.put(a, "type", .{ .string = "text" });
    try p.put(a, "text", .{ .string = text });
    return .{ .object = p };
}

/// A short title request without tools (``is_title_request``): it yields to foreground turns.
pub fn isTitleRequest(messages: Value, has_tools: bool) bool {
    if (has_tools or messages != .array or messages.array.len == 0) return false;
    const first = messages.array[0];
    const role = first.get("role") orelse return false;
    if (role != .string or !std.mem.eql(u8, role.string, "system")) return false;
    var size: usize = 0;
    for (messages.array) |m| {
        const c = m.get("content");
        size += if (c != null and c.? == .string) std.unicode.utf8CountCodepoints(c.?.string) catch c.?.string.len else 4096;
    }
    const text = first.get("content") orelse return false;
    if (size >= 4096 or text != .string) return false;
    return std.ascii.findIgnoreCase(text.string, "title") != null;
}

test "a conversation with a user query stays byte-identical under the rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "user", "content": "run it"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "g", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t2", "content": "ok"}, {"role": "assistant", "content": "done"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", true);
    // byte-identical: the same messages, in order, nothing gained
    try std.testing.expectEqualStrings(try json.stringify(cx.a, messages, .{ .ascii = false }), try json.stringify(cx.a, out, .{ .ascii = false }));
}

test "a conversation with no user query gains one placeholder turn after its last tool run" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}, {"role": "tool", "tool_call_id": "t2", "content": "ok"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", true);
    try std.testing.expectEqual(@as(usize, 4), out.array.len);
    try std.testing.expectEqualStrings("tool", out.array[1].get("role").?.string);
    try std.testing.expectEqualStrings("42 rows", out.array[1].get("content").?.string); // the tool messages unchanged
    try std.testing.expectEqualStrings("tool", out.array[2].get("role").?.string);
    try std.testing.expectEqualStrings("ok", out.array[2].get("content").?.string);
    try std.testing.expectEqualStrings("user", out.array[3].get("role").?.string);
    try std.testing.expectEqualStrings("(tool results above)", out.array[3].get("content").?.string);
}

test "a user message whose content is wholly a tool block does not count as a query" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const user = try json.newObject(cx.a);
    try user.put(cx.a, "role", .{ .string = "user" });
    try user.put(cx.a, "content", .{ .string = "<tool_response>\ngot 42 rows\n</tool_response>" });
    const assistant = try json.newObject(cx.a);
    try assistant.put(cx.a, "role", .{ .string = "assistant" });
    try assistant.put(cx.a, "content", .{ .string = "" });
    const tool = try json.newObject(cx.a);
    try tool.put(cx.a, "role", .{ .string = "tool" });
    try tool.put(cx.a, "tool_call_id", .{ .string = "t1" });
    try tool.put(cx.a, "content", .{ .string = "done" });
    const messages: Value = .{ .array = try cx.a.dupe(Value, &.{ .{ .object = user }, .{ .object = assistant }, .{ .object = tool } }) };
    const out = try normalize(&cx, messages, "system", true);
    try std.testing.expectEqual(@as(usize, 4), out.array.len);
    try std.testing.expectEqualStrings("user", out.array[3].get("role").?.string);
    try std.testing.expectEqualStrings("(tool results above)", out.array[3].get("content").?.string);
}

test "a conversation keeps its shape when the template does not need the rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", false);
    try std.testing.expectEqual(@as(usize, 2), out.array.len);
    try std.testing.expectEqualStrings("assistant", out.array[0].get("role").?.string);
    try std.testing.expectEqualStrings("tool", out.array[1].get("role").?.string);
}
