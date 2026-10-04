//! Chat messages as the template sees them: leading instructions merged, text parts joined, call arguments as objects.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
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

/// ``normalize_messages`` (text only): leading system and developer text merged, later ones as ``late_system``.
pub fn normalize(cx: *Cx, messages: ?Value, late_system: []const u8) errors.Refused!Value {
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
            for (content.?.array) |part| {
                const typed = part == .object and part.get("type") != null and part.get("type").? == .string and std.mem.eql(u8, part.get("type").?.string, "text");
                if (!typed or fields.hasMedia(part)) return cx.refuse("this server accepts text parts only; image, audio and video inputs are unsupported");
                const t = part.get("text") orelse return cx.refuse("a text content part must contain a text string");
                if (t != .string) return cx.refuse("a text content part must contain a text string");
                try text.appendSlice(cx.a, t.string);
            }
            item = try withField(cx, message.object, "content", .{ .string = text.items });
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
