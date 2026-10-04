//! Anthropic Messages requests as the chat requests that run them (``anthropic_translate.translate``).
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const pyrepr = @import("pyrepr.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

fn string(cx: *Cx, value: ?Value, name: []const u8, empty: bool) errors.Refused![]const u8 {
    const v = value orelse Value.null;
    if (v != .string or (!empty and v.string.len == 0)) return cx.fail(.request, "{s} must be a {s}string", .{ name, if (empty) "" else "nonempty " });
    return v.string;
}

fn parts(cx: *Cx, value: ?Value) errors.Refused![]const Value {
    const v = value orelse Value.null;
    if (v == .string) {
        const one = try cx.a.alloc(Value, 1);
        one[0] = try textPart(cx.a, v.string);
        return one;
    }
    if (v != .array) return cx.refuse("content must be a string or an array of content blocks");
    for (v.array) |p| if (p != .object) return cx.refuse("content must be a string or an array of content blocks");
    return v.array;
}

const textPart = @import("messages.zig").textPart;

fn image(cx: *Cx, block: Value) errors.Refused!Value {
    const a = cx.a;
    const source = block.get("source") orelse Value.null;
    if (source != .object) return cx.refuse("image.source must be an object");
    var url: []const u8 = undefined;
    if (source.typeIs("base64")) {
        const media = try string(cx, source.get("media_type"), "image.source.media_type", false);
        const ok = for ([_][]const u8{ "image/jpeg", "image/png", "image/gif", "image/webp" }) |m| {
            if (std.mem.eql(u8, m, media)) break true;
        } else false;
        if (!ok) return cx.refuse("unsupported image media_type");
        url = try std.fmt.allocPrint(a, "data:{s};base64,{s}", .{ media, try string(cx, source.get("data"), "image.source.data", false) });
    } else if (source.typeIs("url")) {
        url = try string(cx, source.get("url"), "image.source.url", false);
    } else return cx.refuse("image.source.type must be base64 or url; file IDs are unsupported");
    const inner = try json.newObject(a);
    try inner.put(a, "url", .{ .string = url });
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "image_url" });
    try o.put(a, "image_url", .{ .object = inner });
    return .{ .object = o };
}

fn roleIs(v: Value, names: []const []const u8) bool {
    const r = v.get("role") orelse return false;
    if (r != .string) return false;
    for (names) |n| if (std.mem.eql(u8, n, r.string)) return true;
    return false;
}

fn messages(cx: *Cx, value: ?Value, system: ?Value) errors.Refused!Value {
    const a = cx.a;
    const list = value orelse Value.null;
    if (list != .array or list.array.len == 0) return cx.refuse("messages must be a nonempty array");
    var out: std.ArrayList(Value) = .empty;
    if (system) |sys| if (sys != .null) {
        const ps = try parts(cx, sys);
        for (ps) |p| if (!p.typeIs("text")) return cx.refuse("system accepts text blocks only");
        var joined: std.ArrayList(u8) = .empty;
        for (ps, 0..) |p, i| {
            if (i > 0) try joined.appendSlice(a, "\n\n");
            try joined.appendSlice(a, try string(cx, p.get("text"), "system.text", true));
        }
        const m = try json.newObject(a);
        try m.put(a, "role", .{ .string = "system" });
        try m.put(a, "content", .{ .string = joined.items });
        try out.append(a, .{ .object = m });
    };
    for (list.array, 0..) |message, index| {
        if (message != .object or !roleIs(message, &.{ "user", "assistant", "system" })) return cx.refuse("message role must be user, assistant or system");
        const role = message.get("role").?.string;
        if (std.mem.eql(u8, role, "system")) {
            const previous: Value = if (index > 0) list.array[index - 1] else .{ .object = try json.newObject(a) };
            const following: Value = if (index + 1 < list.array.len) list.array[index + 1] else blk: {
                const f = try json.newObject(a);
                try f.put(a, "role", .{ .string = "assistant" });
                break :blk .{ .object = f };
            };
            if (previous != .object or !roleIs(previous, &.{ "user", "system" }) or following != .object or !roleIs(following, &.{ "assistant", "system" }))
                return cx.refuse("mid-conversation system messages must follow a user turn and precede an assistant or end");
            const clear = message.get("clear_at");
            const clear_ok = clear == null or clear.? == .null or (clear.? == .string and std.mem.eql(u8, clear.?.string, "never"));
            if (!clear_ok or json.truthyField(message, "output_config")) return cx.refuse("turn-scoped system messages and per-message output_config are unsupported");
        }
        var content: std.ArrayList(Value) = .empty;
        var calls: std.ArrayList(Value) = .empty;
        var thoughts: std.ArrayList(u8) = .empty;
        var any_thought = false;
        const blocks = try parts(cx, message.get("content"));
        for (blocks) |block| {
            const kind = block.get("type");
            const k = if (kind != null and kind.? == .string) kind.?.string else "";
            const user = std.mem.eql(u8, role, "user");
            const assistant = std.mem.eql(u8, role, "assistant");
            if (std.mem.eql(u8, k, "text")) {
                try content.append(a, try textPart(a, try string(cx, block.get("text"), "text", true)));
            } else if (std.mem.eql(u8, k, "image") and user) {
                try content.append(a, try image(cx, block));
            } else if (std.mem.eql(u8, k, "thinking") and assistant) {
                try thoughts.appendSlice(a, try string(cx, block.get("thinking"), "thinking", true));
                any_thought = true;
            } else if (std.mem.eql(u8, k, "redacted_thinking") and assistant) {
                return cx.refuse("redacted thinking cannot be decoded by this local model; send plaintext thinking");
            } else if (std.mem.eql(u8, k, "tool_use") and assistant) {
                const arguments = block.get("input") orelse Value.null;
                if (arguments != .object) return cx.refuse("tool_use.input must be an object");
                const id = try string(cx, block.get("id"), "tool_use.id", false);
                const name = try string(cx, block.get("name"), "tool_use.name", false);
                const function = try json.newObject(a);
                try function.put(a, "name", .{ .string = name });
                try function.put(a, "arguments", .{ .string = try json.stringify(a, arguments, .{ .ascii = false }) });
                const call = try json.newObject(a);
                try call.put(a, "id", .{ .string = id });
                try call.put(a, "type", .{ .string = "function" });
                try call.put(a, "function", .{ .object = function });
                try calls.append(a, .{ .object = call });
            } else if (std.mem.eql(u8, k, "tool_result") and user) {
                var texts: std.ArrayList([]const u8) = .empty;
                var result_parts: std.ArrayList(Value) = .empty;
                var has_image = false;
                for (try parts(cx, block.get("content") orelse Value{ .string = "" })) |part| {
                    if (part.typeIs("text")) {
                        const t = try string(cx, part.get("text"), "tool_result.text", true);
                        try texts.append(a, t);
                        try result_parts.append(a, try textPart(a, t));
                    } else if (part.typeIs("image")) {
                        try result_parts.append(a, try image(cx, part));
                        has_image = true;
                    } else return cx.fail(.request, "unsupported tool_result content type {s}", .{try pyrepr.repr(a, part.get("type"))});
                }
                var text = try std.mem.join(a, "\n", texts.items);
                if (json.truthyField(block, "is_error")) {
                    text = try std.mem.concat(a, u8, &.{ "Tool error: ", text });
                    try result_parts.insert(a, 0, try textPart(a, "Tool error: "));
                }
                const m = try json.newObject(a);
                try m.put(a, "role", .{ .string = "tool" });
                try m.put(a, "tool_call_id", .{ .string = try string(cx, block.get("tool_use_id"), "tool_use_id", false) });
                try m.put(a, "content", if (has_image) .{ .array = result_parts.items } else .{ .string = text });
                try out.append(a, .{ .object = m });
            } else return cx.fail(.request, "unsupported {s} content block type {s}", .{ role, try pyrepr.repr(a, kind) });
        }
        if (content.items.len > 0 or calls.items.len > 0 or any_thought) {
            const item = try json.newObject(a);
            try item.put(a, "role", .{ .string = role });
            try item.put(a, "content", if (content.items.len > 0) .{ .array = content.items } else .{ .string = "" });
            if (calls.items.len > 0) try item.put(a, "tool_calls", .{ .array = calls.items });
            if (any_thought) try item.put(a, "reasoning_content", .{ .string = thoughts.items });
            try out.append(a, .{ .object = item });
        } else if (blocks.len == 0) {
            const item = try json.newObject(a);
            try item.put(a, "role", .{ .string = role });
            try item.put(a, "content", .{ .string = "" });
            try out.append(a, .{ .object = item });
        }
    }
    return .{ .array = out.items };
}

fn positiveInt(v: ?Value) ?i64 {
    const x = v orelse return null;
    if (x != .int) return null;
    const n = x.int64() orelse return std.math.maxInt(i64);
    return if (n > 0) n else null;
}

/// The chat request a Messages request runs as; ``count`` for count_tokens (max_tokens is not required).
pub fn translate(cx: *Cx, body: Value, count: bool) errors.Refused!Value {
    const a = cx.a;
    if (body != .object) return cx.refuse("request body must be an object");
    const model = try string(cx, body.get("model"), "model", false);
    const chat = try json.newObject(a);
    try chat.put(a, "model", .{ .string = model });
    try chat.put(a, "messages", try messages(cx, body.get("messages"), body.get("system")));
    if (!count) {
        _ = positiveInt(body.get("max_tokens")) orelse return cx.refuse("max_tokens must be a positive integer");
        try chat.put(a, "max_tokens", body.get("max_tokens").?);
    } else try chat.put(a, "max_tokens", .{ .int = "1" });
    if (body.get("stream")) |s| {
        if (s != .bool) return cx.refuse("stream must be a boolean");
        try chat.put(a, "stream", s);
    }
    for ([_][]const u8{ "temperature", "top_p", "top_k" }) |k| if (body.get(k)) |v| try chat.put(a, k, v);
    if (body.get("stop_sequences")) |stops| {
        const ok = stops == .array and for (stops.array) |s| {
            if (s != .string or s.string.len == 0) break false;
        } else true;
        if (!ok) return cx.refuse("stop_sequences must be an array of nonempty strings");
        try chat.put(a, "stop", stops);
    }
    if (body.field("context_management")) |context| {
        const allowed = (try json.parse(a, "{\"type\": \"clear_thinking_20251015\", \"keep\": \"all\"}")).ok;
        const edits = context.get("edits") orelse Value{ .array = &.{} };
        const ok = context == .object and edits == .array and for (edits.array) |e| {
            if (!json.equal(e, allowed)) break false;
        } else true;
        if (!ok) return cx.refuse("context_management supports clear_thinking with keep: all only");
    }
    for ([_][]const u8{ "container", "mcp_servers", "service_tier" }) |name| if (body.field(name)) |v| {
        const fine = (v == .array and v.array.len == 0) or (v == .string and std.mem.eql(u8, v.string, "auto"));
        if (!fine) return cx.fail(.request, "{s} is not supported by this server", .{name});
    };
    const tools = body.get("tools") orelse Value{ .array = &.{} };
    if (tools != .array) return cx.refuse("tools must be an array");
    var translated: std.ArrayList(Value) = .empty;
    for (tools.array) |tool| {
        const kind = if (tool == .object) tool.get("type") orelse Value{ .string = "custom" } else Value.null;
        if (tool != .object or kind != .string or !std.mem.eql(u8, kind.string, "custom")) return cx.refuse("only client-defined function tools are supported");
        const name = try string(cx, tool.get("name"), "tool.name", false);
        const schema = tool.get("input_schema") orelse Value.null;
        if (schema != .object) return cx.refuse("tool.input_schema must be an object");
        const function = try json.newObject(a);
        try function.put(a, "name", .{ .string = name });
        try function.put(a, "parameters", schema);
        if (tool.has("description")) try function.put(a, "description", .{ .string = try string(cx, tool.get("description"), "tool.description", true) });
        const t = try json.newObject(a);
        try t.put(a, "type", .{ .string = "function" });
        try t.put(a, "function", .{ .object = function });
        try translated.append(a, .{ .object = t });
    }
    if (translated.items.len > 0) try chat.put(a, "tools", .{ .array = translated.items });
    if (body.field("tool_choice")) |choice| {
        if (choice != .object) return cx.refuse("tool_choice must be an object");
        const kind = choice.get("type");
        const k = if (kind != null and kind.? == .string) kind.?.string else "";
        if (std.mem.eql(u8, k, "auto") or std.mem.eql(u8, k, "none") or std.mem.eql(u8, k, "any")) {
            try chat.put(a, "tool_choice", .{ .string = if (std.mem.eql(u8, k, "any")) "required" else k });
        } else if (std.mem.eql(u8, k, "tool")) {
            const function = try json.newObject(a);
            try function.put(a, "name", .{ .string = try string(cx, choice.get("name"), "tool_choice.name", false) });
            const c = try json.newObject(a);
            try c.put(a, "type", .{ .string = "function" });
            try c.put(a, "function", .{ .object = function });
            try chat.put(a, "tool_choice", .{ .object = c });
        } else return cx.refuse("tool_choice.type must be auto, none, any or tool");
        if (choice.get("disable_parallel_tool_use")) |d| {
            if (d != .bool) return cx.refuse("disable_parallel_tool_use must be a boolean");
            try chat.put(a, "parallel_tool_calls", .{ .bool = !d.bool });
        }
    }
    const config = body.field("output_config") orelse Value{ .object = try json.newObject(a) };
    if (config != .object) return cx.refuse("output_config must be an object");
    if (config.get("effort")) |e| {
        const known = e == .string and for (fields.efforts) |name| {
            if (std.mem.eql(u8, name, e.string)) break true;
        } else false;
        if (!known) return cx.refuse("unsupported output_config.effort");
        try chat.put(a, "reasoning_effort", e);
    }
    if (config.field("format")) |fmt| {
        const schema = fmt.get("schema");
        if (fmt != .object or !fmt.typeIs("json_schema") or schema == null or schema.? != .object) return cx.refuse("output_config.format must be json_schema with a schema object");
        const js = try json.newObject(a);
        try js.put(a, "name", .{ .string = "response" });
        try js.put(a, "schema", schema.?);
        try js.put(a, "strict", .{ .bool = true });
        const rf = try json.newObject(a);
        try rf.put(a, "type", .{ .string = "json_schema" });
        try rf.put(a, "json_schema", .{ .object = js });
        try chat.put(a, "response_format", .{ .object = rf });
    }
    const thinking = body.field("thinking") orelse blk: {
        const d = try json.newObject(a);
        try d.put(a, "type", .{ .string = "disabled" });
        break :blk Value{ .object = d };
    };
    const tk = if (thinking == .object) thinking.get("type") else null;
    const t = if (tk != null and tk.? == .string) tk.?.string else "";
    if (thinking != .object or !(std.mem.eql(u8, t, "disabled") or std.mem.eql(u8, t, "enabled") or std.mem.eql(u8, t, "adaptive")))
        return cx.refuse("thinking.type must be disabled, enabled or adaptive");
    const enabled = !std.mem.eql(u8, t, "disabled");
    const kwargs = try json.newObject(a);
    try kwargs.put(a, "enable_thinking", .{ .bool = enabled });
    try chat.put(a, "chat_template_kwargs", .{ .object = kwargs });
    if (!enabled) try chat.put(a, "reasoning_effort", .{ .string = "none" });
    if (std.mem.eql(u8, t, "enabled")) {
        const budget = positiveInt(thinking.get("budget_tokens"));
        const limit = if (body.get("max_tokens")) |m| m.int64() orelse std.math.maxInt(i64) else 0;
        if (budget == null or (!count and budget.? >= limit)) return cx.refuse("thinking.budget_tokens must be positive and less than max_tokens");
        try chat.put(a, "thinking_budget", thinking.get("budget_tokens").?);
    }
    return .{ .object = chat };
}
