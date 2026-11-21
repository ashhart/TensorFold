//! A Responses request as the chat completion that runs it (``responses_translate.translate``).
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const pyrepr = @import("pyrepr.zig");
const tool_specs = @import("tool_specs.zig");
const response_tools = @import("responses_tools.zig");
const namespace_tools = @import("responses_namespace.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Fields a chat completion reads as they are: OpenAI's and this server's own.
const passed = [_][]const u8{ "model", "temperature", "top_p", "top_k", "min_p", "seed", "stream", "parallel_tool_calls", "stop", "draft", "thinking_budget", "ignore_eos", "priority", "return_token_ids", "chat_template_kwargs", "grammar_budget" };
const refused = [_][2][]const u8{
    .{ "conversation", "conversations are not supported: send previous_response_id or the items" },
    .{ "prompt", "prompt templates are not supported: send input and instructions" },
    .{ "context_management", "context management is not supported" },
    .{ "execution_witness", "execution_witness is supported on /v1/chat/completions and /v1/completions only" },
};

pub const Request = struct {
    chat: Value,
    echo: *json.Object,
    added: []Value,
    store: bool,
    stream: bool,
    /// ``background: true``: the turn runs on after its POST returns, and GET polls it (responses_background.zig).
    background: bool = false,
    /// The client's own name for the grammar field when ``text.format`` asked for a structure; null for plain text.
    format: ?[]const u8 = null,
    /// Only ``include`` asked for logprobs: an engine without rows answers without them, as before it had any.
    rows_if_offered: bool = false,
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
        } else if (kind != null and kind.? == .string and (std.mem.eql(u8, k, "input_file") or std.mem.eql(u8, k, "file"))) {
            const inner = if (part.get("file")) |f| (if (f == .object) f.object else return cx.refuse("file must be an object")) else blk: {
                const f = try json.newObject(a);
                if (part.get("filename")) |v| try f.put(a, "filename", v);
                if (part.get("file_data")) |v| try f.put(a, "file_data", v);
                if (part.get("file_url")) |v| try f.put(a, "url", v);
                if (part.get("file_id")) |v| try f.put(a, "file_id", v);
                break :blk f;
            };
            const iv: Value = .{ .object = inner };
            if (iv.get("file_id") != null and iv.get("file_data") == null and iv.get("data") == null and iv.get("url") == null and iv.get("file_url") == null)
                return cx.refuse("file parts require inline file_data (base64 or data:application/pdf;base64,...); file_id is not supported");
            if (iv.strField("url") orelse iv.strField("file_url")) |u| {
                const trimmed = std.mem.trim(u8, u, " \t\r\n");
                if (trimmed.len > 0 and !std.ascii.startsWithIgnoreCase(trimmed, "data:"))
                    return cx.refuse("URLs are not accepted for PDF inputs; send inline base64 or a data:application/pdf;base64,... URL");
            }
            if ((iv.strField("file_data") orelse iv.strField("data") orelse iv.strField("url") orelse iv.strField("file_url")) == null)
                return cx.fail(.request, "content parts of type {s} are not supported: send input_text or input_image", .{try pyrepr.repr(a, kind)});
            const p = try json.newObject(a);
            try p.put(a, "type", .{ .string = "file" });
            try p.put(a, "file", .{ .object = inner });
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
            if (!std.mem.eql(u8, name, "input_text") and !std.mem.eql(u8, name, "input_image") and !std.mem.eql(u8, name, "input_file")) parts_only = false;
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

/// A function_call item's chat name: its namespace prefixes the child as <ns>__<child>, the spelling the engine got.
fn flatName(cx: *Cx, namespace: ?Value, name: []const u8) errors.Refused![]const u8 {
    const ns = namespace orelse return name;
    if (ns == .null) return name;
    if (ns != .string) return cx.refuse("a function_call item's namespace must be a string");
    if (ns.string.len == 0) return cx.refuse("a function_call item's namespace must be a nonempty string");
    const prefix = try std.fmt.allocPrint(cx.a, "{s}__", .{ns.string});
    if (std.mem.startsWith(u8, name, prefix)) return name;
    return try std.fmt.allocPrint(cx.a, "{s}{s}", .{ prefix, name });
}

/// A namespace child's two names, as the output side writes them.
pub const Child = struct { name: []const u8, namespace: []const u8 };

/// Whether ``name`` is ``group`` followed by the separator and ``own``: the one engine spelling a child can carry.
pub fn spells(name: []const u8, group: []const u8, own: []const u8) bool {
    if (!std.mem.startsWith(u8, name, group)) return false;
    const tail = name[group.len..];
    return std.mem.startsWith(u8, tail, "__") and std.mem.eql(u8, tail[2..], own);
}

/// ``flat`` as the request declares it: a composed <group>__<child> engine name, or null when no group spells it.
pub fn childName(declared: ?Value, flat: []const u8) ?Child {
    if (declared == null or declared.? != .array) return null;
    for (declared.?.array) |tool| {
        if (!tool.typeIs("namespace")) continue;
        const group = tool.strField("name") orelse continue;
        const children = tool.get("tools") orelse continue;
        if (children != .array) continue;
        for (children.array) |child| {
            if (!child.typeIs("function")) continue;
            const own = child.strField("name") orelse continue;
            if (spells(flat, group, own)) return .{ .name = own, .namespace = group };
        }
    }
    return null;
}

/// Write a call's own name and namespace from the request's declared tools.
pub fn writeCallName(a: Allocator, item: *json.Object, declared: ?Value, name: []const u8) Allocator.Error!void {
    if (childName(declared, name)) |child| {
        try item.put(a, "name", .{ .string = child.name });
        try item.put(a, "namespace", .{ .string = child.namespace });
    } else try item.put(a, "name", .{ .string = name });
}

fn flushThought(a: Allocator, out: *std.ArrayList(*json.Object), thought: *?[]const u8) Allocator.Error!void {
    const text = thought.* orelse return;
    const m = try json.newObject(a);
    try m.put(a, "role", .{ .string = "assistant" });
    try m.put(a, "content", .{ .string = "" });
    try m.put(a, "reasoning_content", .{ .string = text });
    try out.append(a, m);
    thought.* = null;
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
            if (!std.mem.eql(u8, role, "assistant")) try flushThought(a, &out, &thought);
            const m = try json.newObject(a);
            try m.put(a, "role", .{ .string = role });
            try m.put(a, "content", try content(cx, item.get("content")));
            if (std.mem.eql(u8, role, "assistant")) if (thought) |t| {
                try m.put(a, "reasoning_content", .{ .string = t });
                thought = null;
            };
            try out.append(a, m);
        } else if (std.mem.eql(u8, k, "function_call") or std.mem.eql(u8, k, "custom_tool_call")) {
            const custom = std.mem.eql(u8, k, "custom_tool_call");
            const call_id = item.strField("call_id") orelse return cx.refuse("a tool call item needs a call_id");
            const call_name = item.strField("name") orelse return cx.refuse("a tool call item needs a name");
            const supplied = if (custom) item.get("input") else item.get("arguments");
            if (custom and (supplied == null or supplied.? != .string)) return cx.refuse("a custom_tool_call item needs string input, including an explicitly empty input");
            const arguments = if (custom) supplied.? else if (supplied != null and supplied.?.truthy()) supplied.? else Value{ .string = "{}" };
            if (!custom and arguments != .string) return cx.refuse("a function_call item's arguments must be a string; incomplete arguments remain incomplete");
            const function = try json.newObject(a);
            try function.put(a, "name", .{ .string = if (custom) call_name else try flatName(cx, item.get("namespace"), call_name) });
            try function.put(a, "arguments", arguments);
            const call = try json.newObject(a);
            try call.put(a, "id", .{ .string = call_id });
            try call.put(a, "type", .{ .string = if (custom) "custom" else "function" });
            try call.put(a, "function", .{ .object = function });
            if (custom) {
                const payload = try json.newObject(a);
                try payload.put(a, "name", .{ .string = call_name });
                try payload.put(a, "input", arguments);
                try call.put(a, "custom", .{ .object = payload });
            }
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
        } else if (std.mem.eql(u8, k, "function_call_output") or std.mem.eql(u8, k, "custom_tool_call_output")) {
            const call_id = item.strField("call_id") orelse return cx.refuse("a tool call output item needs the call_id of its call");
            const m = try json.newObject(a);
            try m.put(a, "role", .{ .string = "tool" });
            try m.put(a, "tool_call_id", .{ .string = call_id });
            try m.put(a, "content", try output(cx, item.get("output")));
            try flushThought(a, &out, &thought);
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
        } else if (std.mem.eql(u8, k, "compaction") and json.truthyField(item, "encrypted_content")) {
            continue; // a foreign opaque compaction has no model-visible content; retain the supplied remaining history
        } else if (std.mem.eql(u8, k, "item_reference")) {
            return cx.refuse("an item_reference requires its original response history; missing history cannot be reconstructed");
        } else return cx.fail(.request, "input items of type {s} are not supported: send messages, function_call, custom_tool_call, tool outputs and reasoning items", .{try pyrepr.repr(a, kind)});
    }
    try flushThought(a, &out, &thought);
    const result = try a.alloc(Value, out.items.len);
    for (out.items, result) |m, *slot| slot.* = .{ .object = m };
    return result;
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

/// OpenAI's include values, sorted: only ``message.output_text.logprobs`` adds output, where the engine has rows.
const include_values = [_][]const u8{ "code_interpreter_call.outputs", "computer_call_output.output.image_url", "file_search_call.results", "message.input_image.image_url", "message.output_text.logprobs", "reasoning.encrypted_content", "web_search_call.action.sources", "web_search_call.results" };

/// ``include`` of OpenAI's values accepted (clients send some by habit), others refused by name.
pub fn include(cx: *Cx, given: Value) errors.Refused!void {
    if (given == .null) return;
    if (given != .array) return cx.refuse("include must be a list of strings");
    var unknown: std.ArrayList([]const u8) = .empty;
    for (given.array) |v| {
        if (v != .string) return cx.refuse("include must be a list of strings");
        for (include_values) |known| {
            if (std.mem.eql(u8, v.string, known)) break;
        } else try unknown.append(cx.a, try pyrepr.repr(cx.a, v));
    }
    if (unknown.items.len > 0) return cx.fail(.request, "include has no value {s}: the values are {s}", .{ try std.mem.join(cx.a, ", ", unknown.items), try std.mem.join(cx.a, ", ", &include_values) });
}

/// A Responses request as the chat completion that runs it; ``history`` reads a stored conversation.
pub fn translate(cx: *Cx, body: Value, history: anytype) errors.Refused!Request {
    const a = cx.a;
    if (body != .object) return cx.refuse("the request body must be a JSON object");
    for (refused) |r| if (json.truthyField(body, r[0])) return cx.refuse(r[1]);
    if (body.get("include")) |inc| try include(cx, inc);
    if (body.field("truncation")) |t| if (!(t == .string and std.mem.eql(u8, t.string, "disabled"))) return cx.refuse("truncation must be \"disabled\": a prompt too long for the context is refused");
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
    try namespace_tools.inputNamespaces(cx, given.array, body.get("tools"));
    const added = try messages(cx, given.array);
    const parent = body.field("previous_response_id");
    const earlier: []Value = if (parent) |p| try history.conversation(cx, p) else &.{};
    const namespaced = namespace_tools.hasNamespace(body.get("tools"));
    const offered = if (namespaced) try namespace_tools.tools(cx, body.get("tools")) else try response_tools.tools(cx, body.get("tools"));
    const choice = if (namespaced) try namespace_tools.toolChoice(cx, body.get("tool_choice"), offered, body.get("tools")) else try response_tools.toolChoice(cx, body.get("tool_choice"), offered);
    const chat = try json.newObject(a);
    // rows come with top_logprobs or logprobs; the include name alone asks for them only where the engine has them
    const listed = if (body.get("include")) |inc| inc == .array and for (inc.array) |v| {
        if (std.mem.eql(u8, v.string, "message.output_text.logprobs")) break true;
    } else false else false;
    const rows = try json.newObject(a);
    for ([_][]const u8{ "n", "logprobs", "top_logprobs" }) |k| if (body.get(k)) |v| try rows.put(a, k, v);
    const asked = body.field("top_logprobs") != null or json.truthyField(body, "logprobs");
    if (listed or asked) try rows.put(a, "logprobs", .{ .bool = true });
    _ = try fields.probabilityOptions(cx, .{ .object = rows }, true);
    for ([_][]const u8{ "logprobs", "top_logprobs" }) |k| if (rows.get(k)) |v| try chat.put(a, k, v);
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
    const rf = try format(cx, body.get("text"));
    if (rf) |f| try chat.put(a, "response_format", f);
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
    const background = json.truthyField(body, "background");
    if (background) try backgroundAsked(cx, body, store);
    try echo.put(a, "background", .{ .bool = background });
    return .{ .chat = .{ .object = chat }, .echo = echo, .added = added, .store = store, .stream = json.truthyField(body, "stream"), .background = background, .format = if (rf != null) "text.format" else null, .rows_if_offered = listed and !asked };
}

/// What a background request needs: a boolean, a stored response to poll, and no stream on the POST.
fn backgroundAsked(cx: *Cx, body: Value, store: bool) errors.Refused!void {
    if (body.get("background").? != .bool) return cx.refuse("background must be a boolean");
    if (!store) return cx.refuse("background: true needs store: true (the default): GET /v1/responses/{id} polls the stored response");
    if (json.truthyField(body, "stream")) return cx.refuse("background: true with stream: true is not supported: send stream: false and poll GET /v1/responses/{id}");
}

test "input_file refuses remote file_url" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const items = (try json.parse(cx.a,
        \\[{"role": "user", "content": [{"type": "input_file", "file_url": "https://example.com/doc.pdf"}]}]
    )).ok.array;
    try std.testing.expectError(error.Refused, messages(&cx, items));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "URLs are not accepted") != null);
}

test "background Responses: pending reasoning stays ahead of following user and tool items" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const items = (try json.parse(cx.a,
        \\[{"type":"reasoning","content":[{"text":"first"}]},{"role":"user","content":"next"},{"type":"reasoning","content":[{"text":"second"}]},{"type":"function_call_output","call_id":"call_1","output":"done"}]
    )).ok.array;
    const got = try messages(&cx, items);
    try std.testing.expectEqual(@as(usize, 4), got.len);
    for ([_][]const u8{ "assistant", "user", "assistant", "tool" }, got) |role, m| try std.testing.expectEqualStrings(role, m.strField("role").?);
    try std.testing.expectEqualStrings("first", got[0].strField("reasoning_content").?);
    try std.testing.expectEqualStrings("second", got[2].strField("reasoning_content").?);
}

test {
    _ = @import("responses_tools_test.zig");
}
