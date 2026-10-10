//! A Responses request as the chat completion that runs it (``responses_translate.translate``).
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const pyrepr = @import("pyrepr.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Fields a chat completion reads as they are: OpenAI's and this server's own.
const passed = [_][]const u8{ "model", "temperature", "top_p", "top_k", "min_p", "seed", "stream", "parallel_tool_calls", "stop", "draft", "thinking_budget", "ignore_eos", "priority", "return_token_ids", "chat_template_kwargs" };
const refused = [_][2][]const u8{
    .{ "background", "background responses are not supported: send the request and wait for it" },
    .{ "conversation", "conversations are not supported: send previous_response_id or the items" },
    .{ "prompt", "prompt templates are not supported: send input and instructions" },
    .{ "context_management", "context management is not supported" },
};

pub const Request = struct {
    chat: Value,
    echo: *json.Object,
    added: []Value,
    store: bool,
    stream: bool,
};

fn kindOf(v: Value) ?Value {
    return if (v == .object) v.get("type") else null;
}

/// A message's content as chat content: text parts as text, images as image_url parts.
fn content(cx: *Cx, c: ?Value) errors.Refused!Value {
    const a = cx.a;
    const value = c orelse Value.null;
    if (value == .string) return value;
    if (value != .array) return cx.refuse("message content must be a string or a list of content parts");
    var parts: std.ArrayList(Value) = .empty;
    for (value.array) |part| {
        const kind = kindOf(part);
        const k = if (kind != null and kind.? == .string) kind.?.string else "";
        const is = struct {
            fn any(s: []const u8, names: []const []const u8) bool {
                for (names) |n| if (std.mem.eql(u8, s, n)) return true;
                return false;
            }
        };
        if (kind != null and kind.? == .string and is.any(k, &.{ "input_text", "output_text", "text" }) and part.strField("text") != null) {
            try parts.append(a, try textPart(a, part.strField("text").?));
        } else if (kind != null and kind.? == .string and std.mem.eql(u8, k, "refusal") and part.strField("refusal") != null) {
            try parts.append(a, try textPart(a, part.strField("refusal").?));
        } else if (kind != null and kind.? == .string and std.mem.eql(u8, k, "input_image")) {
            const url_text = part.strField("image_url") orelse return cx.refuse("input_image needs an image_url (a URL or data URL); file ids are not supported");
            const url = try json.newObject(a);
            try url.put(a, "url", .{ .string = url_text });
            if (json.truthyField(part, "detail")) try url.put(a, "detail", part.get("detail").?);
            const p = try json.newObject(a);
            try p.put(a, "type", .{ .string = "image_url" });
            try p.put(a, "image_url", .{ .object = url });
            try parts.append(a, .{ .object = p });
        } else return cx.fail(.request, "content parts of type {s} are not supported: send input_text or input_image", .{try pyrepr.repr(a, kind)});
    }
    return .{ .array = parts.items };
}

const textPart = @import("messages.zig").textPart;

/// A function_call_output's output as a tool message's content.
fn output(cx: *Cx, o: ?Value) errors.Refused!Value {
    const value = o orelse Value.null;
    if (value == .string) return value;
    if (value == .array) {
        var texts_only = true;
        var parts_only = true;
        for (value.array) |p| {
            const k = kindOf(p);
            const name = if (k != null and k.? == .string) k.?.string else "";
            if (!std.mem.eql(u8, name, "input_text")) texts_only = false;
            if (!std.mem.eql(u8, name, "input_text") and !std.mem.eql(u8, name, "input_image")) parts_only = false;
        }
        if (texts_only) {
            var joined: std.ArrayList(u8) = .empty;
            for (value.array) |p| if (p.get("text")) |t| if (t.truthy()) try joined.appendSlice(cx.a, try tool_specs.pyStr(cx.a, t));
            return .{ .string = joined.items };
        }
        if (parts_only) return content(cx, value);
    }
    return cx.refuse("a function_call_output's output must be a string or input_text and input_image parts");
}

/// Input items (output items too) as chat messages: a turn's reasoning, text and calls are one assistant message.
pub fn messages(cx: *Cx, items: []const Value) errors.Refused![]Value {
    const a = cx.a;
    var out: std.ArrayList(*json.Object) = .empty;
    var thought: ?[]const u8 = null;
    for (items) |item| {
        if (item != .object) return cx.refuse("each input item must be an object");
        var kind: ?Value = item.get("type");
        if (kind == null or !kind.?.truthy()) kind = if (item.has("role")) Value{ .string = "message" } else null;
        const k = if (kind != null and kind.? == .string) kind.?.string else "";
        if (std.mem.eql(u8, k, "message")) {
            const role = item.strField("role") orelse "";
            const ok = for ([_][]const u8{ "user", "assistant", "system", "developer" }) |r| {
                if (std.mem.eql(u8, r, role)) break true;
            } else false;
            if (!ok or item.get("role").? != .string) return cx.refuse("a message's role must be user, assistant, system or developer");
            const m = try json.newObject(a);
            try m.put(a, "role", .{ .string = role });
            try m.put(a, "content", try content(cx, item.get("content")));
            if (std.mem.eql(u8, role, "assistant")) if (thought) |t| {
                try m.put(a, "reasoning_content", .{ .string = t });
                thought = null;
            };
            try out.append(a, m);
        } else if (std.mem.eql(u8, k, "function_call")) {
            const call_id = item.strField("call_id");
            const name = item.strField("name");
            if (call_id == null or name == null) return cx.refuse("a function_call item needs a call_id and a name");
            const function = try json.newObject(a);
            try function.put(a, "name", .{ .string = name.? });
            const args = item.get("arguments");
            try function.put(a, "arguments", if (args != null and args.?.truthy()) args.? else .{ .string = "{}" });
            const call = try json.newObject(a);
            try call.put(a, "id", .{ .string = call_id.? });
            try call.put(a, "type", .{ .string = "function" });
            try call.put(a, "function", .{ .object = function });
            const last_role = if (out.items.len > 0) (Value{ .object = out.items[out.items.len - 1] }).strField("role") orelse "" else "";
            if (out.items.len == 0 or !std.mem.eql(u8, last_role, "assistant")) {
                const m = try json.newObject(a);
                try m.put(a, "role", .{ .string = "assistant" });
                try m.put(a, "content", .{ .string = "" });
                try out.append(a, m);
            }
            const last = out.items[out.items.len - 1];
            if (thought) |t| {
                try last.put(a, "reasoning_content", .{ .string = t });
                thought = null;
            }
            const existing = last.get("tool_calls");
            const before: []const Value = if (existing != null and existing.? == .array) existing.?.array else &.{};
            try last.put(a, "tool_calls", .{ .array = try std.mem.concat(a, Value, &.{ before, &.{.{ .object = call }} }) });
        } else if (std.mem.eql(u8, k, "function_call_output")) {
            const call_id = item.strField("call_id") orelse return cx.refuse("a function_call_output item needs the call_id of its call");
            const m = try json.newObject(a);
            try m.put(a, "role", .{ .string = "tool" });
            try m.put(a, "tool_call_id", .{ .string = call_id });
            try m.put(a, "content", try output(cx, item.get("output")));
            try out.append(a, m);
        } else if (std.mem.eql(u8, k, "reasoning")) {
            var texts: std.ArrayList(u8) = .empty;
            var any = false;
            if (item.get("content")) |c| if (c == .array) for (c.array) |p| if (p == .object) {
                any = true;
                if (p.strField("text")) |t| try texts.appendSlice(a, t);
            };
            if (json.truthyField(item, "encrypted_content") and !any) return cx.refuse("encrypted reasoning is not supported: send the reasoning item's content");
            thought = if (texts.items.len > 0) texts.items else null;
        } else return cx.fail(.request, "input items of type {s} are not supported: send messages, function_call, function_call_output and reasoning items", .{try pyrepr.repr(a, kind)});
    }
    const result = try a.alloc(Value, out.items.len);
    for (out.items, result) |m, *slot| slot.* = .{ .object = m };
    return result;
}

fn tools(cx: *Cx, raw: ?Value) errors.Refused!?[]Value {
    const value = raw orelse return null;
    if (value == .null) return null;
    if (value != .array) return cx.refuse("tools must be a list");
    var out: std.ArrayList(Value) = .empty;
    for (value.array) |tool| {
        const kind = kindOf(tool);
        if (kind == null or kind.? != .string or !std.mem.eql(u8, kind.?.string, "function")) return cx.fail(.request, "tools of type {s} are not supported: this server runs function tools only", .{try pyrepr.repr(cx.a, kind)});
        const name = tool.strField("name") orelse "";
        if (name.len == 0) return cx.refuse("a function tool needs a name");
        const function = try json.newObject(cx.a);
        for ([_][]const u8{ "name", "description", "parameters", "strict" }) |k| if (tool.field(k)) |v| try function.put(cx.a, k, v);
        const t = try json.newObject(cx.a);
        try t.put(cx.a, "type", .{ .string = "function" });
        try t.put(cx.a, "function", .{ .object = function });
        try out.append(cx.a, .{ .object = t });
    }
    return out.items;
}

const Choice = struct { choice: ?Value, offered: ?[]Value };

fn toolChoice(cx: *Cx, raw: ?Value, offered: ?[]Value) errors.Refused!Choice {
    const choice = raw orelse return .{ .choice = null, .offered = offered };
    if (choice == .null) return .{ .choice = null, .offered = offered };
    if (choice == .string) for ([_][]const u8{ "none", "auto", "required" }) |s| if (std.mem.eql(u8, s, choice.string)) return .{ .choice = choice, .offered = offered };
    const kind = kindOf(choice);
    const k = if (kind != null and kind.? == .string) kind.?.string else "";
    if (std.mem.eql(u8, k, "function") and choice.strField("name") != null) {
        const function = try json.newObject(cx.a);
        try function.put(cx.a, "name", .{ .string = choice.strField("name").? });
        const c = try json.newObject(cx.a);
        try c.put(cx.a, "type", .{ .string = "function" });
        try c.put(cx.a, "function", .{ .object = function });
        return .{ .choice = .{ .object = c }, .offered = offered };
    }
    const mode: Value = choice.get("mode") orelse .{ .string = "auto" };
    const mode_ok = mode == .string and (std.mem.eql(u8, mode.string, "auto") or std.mem.eql(u8, mode.string, "required"));
    if (std.mem.eql(u8, k, "allowed_tools") and mode_ok) {
        const allowed = choice.get("tools");
        const list: []const Value = if (allowed != null and allowed.? == .array) allowed.?.array else &.{};
        if (allowed != null and allowed.?.truthy() and allowed.? != .array) return cx.refuse("allowed_tools may name function tools only");
        for (list) |t| {
            const tk = kindOf(t);
            if (tk == null or tk.? != .string or !std.mem.eql(u8, tk.?.string, "function")) return cx.refuse("allowed_tools may name function tools only");
        }
        var kept: std.ArrayList(Value) = .empty;
        for (offered orelse &.{}) |t| {
            const name = t.get("function").?.get("name").?;
            for (list) |allow| if (allow.get("name")) |n| if (json.equal(n, name)) {
                try kept.append(cx.a, t);
                break;
            };
        }
        return .{ .choice = mode, .offered = kept.items };
    }
    return cx.refuse("tool_choice must be none, auto, required, a function or allowed_tools of functions");
}

/// ``text.format`` as a chat ``response_format`` (null for plain text).
fn format(cx: *Cx, text: ?Value) errors.Refused!?Value {
    const t = text orelse return null;
    if (t == .null) return null;
    if (t != .object) return cx.refuse("text must be an object");
    const fmt = t.get("format") orelse return null;
    if (fmt == .null) return null;
    if (fmt != .object) return cx.other("'str' object has no attribute 'get'");
    const kind = fmt.get("type");
    const k = if (kind != null and kind.? == .string) kind.?.string else "";
    if (std.mem.eql(u8, k, "text")) return null;
    const out = try json.newObject(cx.a);
    if (std.mem.eql(u8, k, "json_object")) {
        try out.put(cx.a, "type", .{ .string = "json_object" });
        return .{ .object = out };
    }
    if (std.mem.eql(u8, k, "json_schema")) {
        const js = try json.newObject(cx.a);
        for ([_][]const u8{ "name", "schema", "strict", "description" }) |key| if (fmt.field(key)) |v| try js.put(cx.a, key, v);
        try out.put(cx.a, "type", .{ .string = "json_schema" });
        try out.put(cx.a, "json_schema", .{ .object = js });
        return .{ .object = out };
    }
    return cx.refuse("text.format must be text, json_object or json_schema");
}

/// A Responses request as the chat completion that runs it; ``history`` reads a stored conversation.
pub fn translate(cx: *Cx, body: Value, history: anytype) errors.Refused!Request {
    const a = cx.a;
    if (body != .object) return cx.refuse("the request body must be a JSON object");
    _ = try fields.probabilityOptions(cx, body, false); // logprobs come on /v1/chat/completions only
    for (refused) |r| if (json.truthyField(body, r[0])) return cx.refuse(r[1]);
    if (body.get("include")) |inc| if (inc.truthy()) {
        var parts: std.ArrayList([]const u8) = .empty;
        switch (inc) {
            .array => |items| for (items) |i| try parts.append(a, try tool_specs.pyStr(a, i)),
            .object => |o| for (o.keys()) |k| try parts.append(a, k),
            .string => |s| for (0..s.len) |i| try parts.append(a, s[i .. i + 1]),
            else => try parts.append(a, try tool_specs.pyStr(a, inc)),
        }
        return cx.fail(.request, "include is not supported ({s}): reasoning comes as text in its reasoning item, and nothing is encrypted", .{try std.mem.join(a, ", ", parts.items)});
    };
    if (body.field("truncation")) |t| if (!(t == .string and std.mem.eql(u8, t.string, "disabled"))) return cx.refuse("truncation must be \"disabled\": a prompt too long for the context is refused");
    if (json.truthyField(body, "top_logprobs")) return cx.refuse("top_logprobs is not supported");
    const meta_raw = body.get("metadata");
    const metadata: Value = if (meta_raw != null and meta_raw.?.truthy()) meta_raw.? else .{ .object = try json.newObject(a) };
    const meta_ok = metadata == .object and metadata.object.count() <= 16 and for (metadata.object.keys(), metadata.object.values()) |k, v| {
        if (@import("reply_text.zig").charCount(k) > 64 or v != .string or @import("reply_text.zig").charCount(v.string) > 512) break false;
    } else true;
    if (!meta_ok) return cx.refuse("metadata must be at most 16 string pairs (keys up to 64 characters, values up to 512)");
    const instructions = body.field("instructions");
    if (instructions != null and instructions.? != .string) return cx.refuse("instructions must be a string");
    var given = body.get("input") orelse Value.null;
    if (given == .string) {
        const m = try json.newObject(a);
        try m.put(a, "role", .{ .string = "user" });
        try m.put(a, "content", given);
        const one = try a.alloc(Value, 1);
        one[0] = .{ .object = m };
        given = .{ .array = one };
    }
    if (given != .array or given.array.len == 0) return cx.refuse("input must be a string or a non-empty list of items");
    const added = try messages(cx, given.array);
    const parent = body.field("previous_response_id");
    const earlier: []Value = if (parent) |p| try history.conversation(cx, p) else &.{};
    const offered = try tools(cx, body.get("tools"));
    const choice = try toolChoice(cx, body.get("tool_choice"), offered);
    const chat = try json.newObject(a);
    for (passed) |k| if (body.get(k)) |v| try chat.put(a, k, v);
    var msgs: std.ArrayList(Value) = .empty;
    if (instructions != null and instructions.?.string.len > 0) {
        const sys = try json.newObject(a);
        try sys.put(a, "role", .{ .string = "system" });
        try sys.put(a, "content", instructions.?);
        try msgs.append(a, .{ .object = sys });
    }
    try msgs.appendSlice(a, earlier);
    try msgs.appendSlice(a, added);
    try chat.put(a, "messages", .{ .array = msgs.items });
    if (choice.offered) |o| try chat.put(a, "tools", .{ .array = o });
    if (choice.choice) |c| try chat.put(a, "tool_choice", c);
    if (body.field("max_output_tokens")) |m| try chat.put(a, "max_tokens", m);
    const reasoning_raw = body.get("reasoning");
    const reasoning: Value = if (reasoning_raw != null and reasoning_raw.?.truthy()) reasoning_raw.? else .{ .object = try json.newObject(a) };
    if (reasoning != .object) return cx.refuse("reasoning must be an object");
    if (reasoning.field("effort")) |e| try chat.put(a, "reasoning_effort", e);
    if (try format(cx, body.get("text"))) |f| try chat.put(a, "response_format", f);
    const echo = try json.newObject(a);
    try echo.put(a, "instructions", instructions orelse .null);
    try echo.put(a, "max_output_tokens", body.get("max_output_tokens") orelse .null);
    try echo.put(a, "metadata", metadata);
    try echo.put(a, "parallel_tool_calls", body.get("parallel_tool_calls") orelse .{ .bool = true });
    try echo.put(a, "previous_response_id", parent orelse .null);
    const r = try json.newObject(a);
    try r.put(a, "effort", reasoning.get("effort") orelse .null);
    try r.put(a, "summary", reasoning.get("summary") orelse .null);
    try echo.put(a, "reasoning", .{ .object = r });
    const store_field = body.get("store");
    const store = !(store_field != null and store_field.? == .bool and !store_field.?.bool);
    try echo.put(a, "store", .{ .bool = store });
    try echo.put(a, "temperature", body.get("temperature") orelse .null);
    const text_obj = try json.newObject(a);
    const text_raw = body.get("text");
    const fmt_raw: ?Value = if (text_raw != null and text_raw.?.truthy()) text_raw.?.get("format") else null;
    if (fmt_raw != null and fmt_raw.?.truthy()) try text_obj.put(a, "format", fmt_raw.?) else {
        const plain = try json.newObject(a);
        try plain.put(a, "type", .{ .string = "text" });
        try text_obj.put(a, "format", .{ .object = plain });
    }
    try echo.put(a, "text", .{ .object = text_obj });
    const tc = body.get("tool_choice");
    try echo.put(a, "tool_choice", if (tc != null and tc.?.truthy()) tc.? else .{ .string = "auto" });
    const ts = body.get("tools");
    try echo.put(a, "tools", if (ts != null and ts.?.truthy()) ts.? else .{ .array = &.{} });
    try echo.put(a, "top_p", body.get("top_p") orelse .null);
    try echo.put(a, "truncation", .{ .string = "disabled" });
    try echo.put(a, "user", body.get("user") orelse .null);
    try echo.put(a, "background", .{ .bool = false });
    return .{ .chat = .{ .object = chat }, .echo = echo, .added = added, .store = store, .stream = json.truthyField(body, "stream") };
}
