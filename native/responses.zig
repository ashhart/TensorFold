const std = @import("std");
const V = std.json.Value;
const A = std.mem.Allocator;

pub fn route(target: []const u8) ?[]const u8 {
    const path = std.mem.trimEnd(u8, target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len], "/");
    for ([_][]const u8{ "/v1/responses", "/responses" }) |prefix| {
        if (std.mem.eql(u8, path, prefix)) return "";
        if (path.len > prefix.len + 1 and std.mem.startsWith(u8, path, prefix) and path[prefix.len] == '/') {
            const id = path[prefix.len + 1 ..];
            if (std.mem.indexOfScalar(u8, id, '/') == null) return id;
        }
    }
    return null;
}

fn get(value: V, key: []const u8) V {
    return if (value == .object) value.object.get(key) orelse .null else .null;
}
fn is(value: V, text: []const u8) bool {
    return value == .string and std.mem.eql(u8, value.string, text);
}
fn truthy(value: V) bool {
    return switch (value) {
        .null => false,
        .bool => value.bool,
        .integer => value.integer != 0,
        .float => value.float != 0,
        .string, .number_string => |s| s.len > 0,
        .array => value.array.items.len > 0,
        .object => value.object.count() > 0,
    };
}
fn array(a: A) V {
    return .{ .array = std.json.Array.init(a) };
}
fn object(a: A, value: anytype) !V {
    const bytes = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(bytes);
    return (try std.json.parseFromSlice(V, a, bytes, .{ .allocate = .alloc_always })).value;
}
fn clone(a: A, value: V) !V {
    return object(a, value);
}
fn select(a: A, value: V, keys: []const []const u8) !V {
    var out = V{ .object = .empty };
    for (keys) |key| {
        const field = get(value, key);
        if (field != .null) try out.object.put(a, key, field);
    }
    return out;
}

fn content(a: A, value: V) !V {
    if (value == .string) return value;
    if (value != .array) return error.InvalidResponseContent;
    var parts = array(a);
    for (value.array.items) |part| {
        const kind = get(part, "type");
        const text = get(part, if (is(kind, "refusal")) "refusal" else "text");
        if ((is(kind, "input_text") or is(kind, "output_text") or is(kind, "text") or is(kind, "refusal")) and text == .string) {
            try parts.array.append(try object(a, .{ .type = "text", .text = text }));
        } else if (is(kind, "input_image")) {
            const url = get(part, "image_url");
            if (url != .string) return error.ResponseImageNeedsUrl;
            var image = try object(a, .{ .url = url });
            const detail = get(part, "detail");
            if (truthy(detail)) try image.object.put(a, "detail", detail);
            try parts.array.append(try object(a, .{ .type = "image_url", .image_url = image }));
        } else return error.UnsupportedResponseContent;
    }
    return parts;
}

fn toolOutput(a: A, value: V) !V {
    if (value == .string) return value;
    if (value != .array) return error.InvalidFunctionOutput;
    var text: std.ArrayList(u8) = .empty;
    for (value.array.items) |part| {
        if (!is(get(part, "type"), "input_text")) return error.InvalidFunctionOutput;
        const piece = get(part, "text");
        if (truthy(piece)) {
            if (piece != .string) return error.InvalidFunctionOutput;
            try text.appendSlice(a, piece.string);
        }
    }
    return .{ .string = text.items };
}

pub fn messages(a: A, items: V) !V {
    if (items != .array) return error.InvalidResponseInput;
    var out = array(a);
    var thought: ?[]const u8 = null;
    for (items.array.items) |item| {
        if (item != .object) return error.InvalidResponseItem;
        var kind = get(item, "type");
        if (!truthy(kind) and item.object.contains("role")) kind = .{ .string = "message" };
        if (is(kind, "message")) {
            const role = get(item, "role");
            if (!is(role, "user") and !is(role, "assistant") and !is(role, "system") and !is(role, "developer")) return error.InvalidResponseRole;
            var message = try object(a, .{ .role = role, .content = try content(a, get(item, "content")) });
            if (is(role, "assistant")) if (thought) |text| {
                try message.object.put(a, "reasoning_content", .{ .string = text });
                thought = null;
            };
            try out.array.append(message);
        } else if (is(kind, "function_call")) {
            const id = get(item, "call_id");
            const name = get(item, "name");
            if (id != .string or name != .string) return error.InvalidResponseCall;
            const args = get(item, "arguments");
            const call = try object(a, .{ .id = id, .type = "function", .function = .{ .name = name, .arguments = if (truthy(args)) args else V{ .string = "{}" } } });
            if (out.array.items.len == 0 or !is(get(out.array.items[out.array.items.len - 1], "role"), "assistant")) try out.array.append(try object(a, .{ .role = "assistant", .content = "" }));
            const message = &out.array.items[out.array.items.len - 1];
            if (thought) |text| {
                try message.object.put(a, "reasoning_content", .{ .string = text });
                thought = null;
            }
            if (!message.object.contains("tool_calls")) try message.object.put(a, "tool_calls", array(a));
            try message.object.getPtr("tool_calls").?.array.append(call);
        } else if (is(kind, "function_call_output")) {
            const id = get(item, "call_id");
            if (id != .string) return error.InvalidResponseCall;
            try out.array.append(try object(a, .{ .role = "tool", .tool_call_id = id, .content = try toolOutput(a, get(item, "output")) }));
        } else if (is(kind, "reasoning")) {
            const parts = get(item, "content");
            if (truthy(parts) and parts != .array) return error.InvalidResponseReasoning;
            var text: std.ArrayList(u8) = .empty;
            var dictionaries: usize = 0;
            if (parts == .array) for (parts.array.items) |part| {
                if (part == .object) dictionaries += 1;
                const piece = get(part, "text");
                if (piece == .string) try text.appendSlice(a, piece.string);
            };
            if (truthy(get(item, "encrypted_content")) and dictionaries == 0) return error.EncryptedReasoningUnsupported;
            thought = if (text.items.len > 0) text.items else null;
        } else return error.UnsupportedResponseItem;
    }
    return out;
}

fn tools(a: A, value: V) !V {
    if (value == .null) return .null;
    if (value != .array) return error.InvalidResponseTools;
    var out = array(a);
    for (value.array.items) |tool| {
        if (!is(get(tool, "type"), "function")) return error.UnsupportedResponseTool;
        const name = get(tool, "name");
        if (name != .string or name.string.len == 0) return error.InvalidResponseToolName;
        try out.array.append(try object(a, .{ .type = "function", .function = try select(a, tool, &.{ "name", "description", "parameters", "strict" }) }));
    }
    return out;
}

fn toolChoice(a: A, value: V, offered: *V) !V {
    if (value == .null or is(value, "none") or is(value, "auto") or is(value, "required")) return value;
    const kind = get(value, "type");
    if (is(kind, "function") and get(value, "name") == .string) return object(a, .{ .type = "function", .function = .{ .name = get(value, "name") } });
    const mode = if (value == .object) value.object.get("mode") orelse V{ .string = "auto" } else V.null;
    if (is(kind, "allowed_tools") and (is(mode, "auto") or is(mode, "required"))) {
        const allowed = get(value, "tools");
        if (truthy(allowed) and allowed != .array) return error.InvalidAllowedTools;
        if (allowed == .array) for (allowed.array.items) |tool| {
            if (!is(get(tool, "type"), "function")) return error.InvalidAllowedTools;
        };
        var filtered = array(a);
        if (offered.* == .array and allowed == .array) for (offered.array.items) |tool| {
            const name = get(get(tool, "function"), "name").string;
            for (allowed.array.items) |permit| if (is(get(permit, "name"), name)) {
                try filtered.array.append(tool);
                break;
            };
        };
        offered.* = filtered;
        return mode;
    }
    return error.InvalidResponseToolChoice;
}

fn format(a: A, text: V) !V {
    if (text != .null and text != .object) return error.InvalidResponseText;
    const fmt = get(text, "format");
    if (fmt == .null or is(get(fmt, "type"), "text")) return .null;
    if (is(get(fmt, "type"), "json_object")) return object(a, .{ .type = "json_object" });
    if (is(get(fmt, "type"), "json_schema")) return object(a, .{ .type = "json_schema", .json_schema = try select(a, fmt, &.{ "name", "schema", "strict", "description" }) });
    return error.InvalidResponseFormat;
}

pub const Request = struct {
    chat: V,
    echo: V,
    added: V,
    store: bool,
    stream: bool,
};

pub fn base(a: A, request: Request, id: []const u8, model: []const u8, created: i64) !V {
    var out = try clone(a, request.echo);
    const fields = try object(a, .{ .id = id, .object = "response", .created_at = created, .status = "in_progress", .@"error" = @as(?u8, null), .incomplete_details = @as(?u8, null), .model = model, .output = array(a), .usage = @as(?u8, null) });
    var iter = fields.object.iterator();
    while (iter.next()) |pair| try out.object.put(a, pair.key_ptr.*, pair.value_ptr.*);
    return out;
}

pub const Reply = struct {
    a: A,
    text_allocator: A = std.heap.page_allocator,
    io: std.Io,
    initial: V,
    items: std.ArrayList(V) = .empty,
    buffers: std.ArrayList(std.ArrayList(u8)) = .empty,
    current: ?usize = null,
    calls: std.AutoHashMapUnmanaged(i64, usize) = .empty,
    sequence: usize = 0,
    id_sequence: usize = 0,
    final: ?V = null,
    context: ?*anyopaque = null,
    emit: ?*const fn (?*anyopaque, V) anyerror!void = null,
    store: ?*Store = null,
    added: V = .null,

    pub fn deinit(r: *Reply) void {
        for (r.buffers.items) |*buffer| buffer.deinit(r.text_allocator);
    }

    pub fn writeDelta(r: *Reply, value: anytype) !void {
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        try r.delta(try object(scratch.allocator(), value));
    }

    pub fn finishUsage(r: *Reply, reason: []const u8, value: anytype, completed_at: i64) !V {
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        return r.finish(.{ .string = reason }, try object(scratch.allocator(), value), .null, completed_at);
    }

    pub fn complete(r: *Reply, value: anytype, completed_at: i64) !V {
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        return r.completion(try object(scratch.allocator(), value), completed_at);
    }

    fn event(r: *Reply, kind: []const u8, fields: anytype) !void {
        const send = r.emit orelse return;
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const a = scratch.allocator();
        var value = try object(a, fields);
        try value.object.put(a, "type", .{ .string = kind });
        try value.object.put(a, "sequence_number", .{ .integer = @intCast(r.sequence) });
        try send(r.context, value);
        r.sequence += 1;
    }

    pub fn start(r: *Reply) !void {
        try r.event("response.created", .{ .response = r.initial });
        try r.event("response.in_progress", .{ .response = r.initial });
    }

    fn freshId(r: *Reply, prefix: []const u8) ![]const u8 {
        const id = try std.fmt.allocPrint(r.a, "{s}_{s}_{d}", .{ prefix, get(r.initial, "id").string, r.id_sequence });
        r.id_sequence += 1;
        return id;
    }

    fn open(r: *Reply, kind: []const u8, fields: V) !usize {
        try r.close("completed");
        var item: V = undefined;
        var part: ?V = null;
        if (std.mem.eql(u8, kind, "message")) {
            item = try object(r.a, .{ .id = try r.freshId("msg"), .type = "message", .role = "assistant", .status = "in_progress", .content = array(r.a) });
            part = try object(r.a, .{ .type = "output_text", .text = "", .annotations = array(r.a), .logprobs = array(r.a) });
        } else if (std.mem.eql(u8, kind, "reasoning")) {
            item = try object(r.a, .{ .id = try r.freshId("rs"), .type = "reasoning", .summary = array(r.a), .content = array(r.a), .status = "in_progress" });
            part = try object(r.a, .{ .type = "reasoning_text", .text = "" });
        } else {
            item = try clone(r.a, fields);
            try item.object.put(r.a, "id", .{ .string = try r.freshId("fc") });
            try item.object.put(r.a, "type", .{ .string = "function_call" });
            try item.object.put(r.a, "status", .{ .string = "in_progress" });
        }
        const index = r.items.items.len;
        try r.buffers.append(r.a, .empty);
        try r.items.append(r.a, item);
        r.current = index;
        try r.event("response.output_item.added", .{ .output_index = index, .item = item });
        if (part) |value| {
            try r.items.items[index].object.getPtr("content").?.array.append(value);
            try r.event("response.content_part.added", .{ .item_id = get(item, "id"), .output_index = index, .content_index = @as(usize, 0), .part = value });
        }
        return index;
    }

    fn close(r: *Reply, status: []const u8) !void {
        const index = r.current orelse return;
        r.current = null;
        const item = &r.items.items[index];
        try item.object.put(r.a, "status", .{ .string = status });
        const id = get(item.*, "id");
        if (is(get(item.*, "type"), "function_call")) {
            try r.event("response.function_call_arguments.done", .{ .item_id = id, .output_index = index, .name = get(item.*, "name"), .arguments = get(item.*, "arguments") });
        } else {
            const part = get(item.*, "content").array.items[0];
            if (is(get(item.*, "type"), "message")) {
                try r.event("response.output_text.done", .{ .item_id = id, .output_index = index, .content_index = @as(usize, 0), .text = get(part, "text"), .logprobs = array(r.a) });
            } else try r.event("response.reasoning_text.done", .{ .item_id = id, .output_index = index, .content_index = @as(usize, 0), .text = get(part, "text") });
            try r.event("response.content_part.done", .{ .item_id = id, .output_index = index, .content_index = @as(usize, 0), .part = part });
        }
        try r.event("response.output_item.done", .{ .output_index = index, .item = item.* });
    }

    fn write(r: *Reply, kind: []const u8, text: []const u8) !void {
        const index = if (r.current != null and is(get(r.items.items[r.current.?], "type"), kind)) r.current.? else try r.open(kind, .null);
        const item = r.items.items[index];
        const part = &item.object.getPtr("content").?.array.items[0];
        try r.buffers.items[index].appendSlice(r.text_allocator, text);
        part.object.getPtr("text").?.* = .{ .string = r.buffers.items[index].items };
        if (std.mem.eql(u8, kind, "message")) {
            try r.event("response.output_text.delta", .{ .item_id = get(item, "id"), .output_index = index, .content_index = @as(usize, 0), .delta = text, .logprobs = array(r.a) });
        } else try r.event("response.reasoning_text.delta", .{ .item_id = get(item, "id"), .output_index = index, .content_index = @as(usize, 0), .delta = text });
    }

    pub fn delta(r: *Reply, value: V) !void {
        if (r.final != null) return;
        const thought = if (truthy(get(value, "reasoning_content"))) get(value, "reasoning_content") else get(value, "reasoning");
        if (thought == .string and thought.string.len > 0) try r.write("reasoning", thought.string);
        const text = get(value, "content");
        if (text == .string and text.string.len > 0) try r.write("message", text.string);
        const tool_calls = get(value, "tool_calls");
        if (tool_calls == .array) for (tool_calls.array.items) |call| {
            const f = get(call, "function");
            const call_index = get(call, "index");
            const key = if (call_index == .integer) call_index.integer else 0;
            const index = r.calls.get(key) orelse blk: {
                const id = get(call, "id");
                const name = get(f, "name");
                const fields = try object(r.a, .{ .call_id = if (truthy(id)) id else V{ .string = try r.freshId("call") }, .name = if (truthy(name)) name else V{ .string = "" }, .arguments = "" });
                const opened = try r.open("function_call", fields);
                try r.calls.put(r.a, key, opened);
                break :blk opened;
            };
            const args = get(f, "arguments");
            if (args == .string and args.string.len > 0) {
                const item = &r.items.items[index];
                try r.buffers.items[index].appendSlice(r.text_allocator, args.string);
                item.object.getPtr("arguments").?.* = .{ .string = r.buffers.items[index].items };
                try r.event("response.function_call_arguments.delta", .{ .item_id = get(item.*, "id"), .output_index = index, .delta = args });
            }
        };
    }

    pub fn finish(r: *Reply, reason: V, chat_usage: V, stats: V, completed_at: i64) !V {
        if (r.final) |value| return value;
        const incomplete = is(reason, "length");
        const status = if (incomplete) "incomplete" else "completed";
        var only_reasoning = true;
        for (r.items.items) |item| only_reasoning = only_reasoning and is(get(item, "type"), "reasoning");
        if (!incomplete and only_reasoning) _ = try r.open("message", .null);
        try r.close(status);
        var final = try clone(r.a, r.initial);
        try final.object.put(r.a, "status", .{ .string = status });
        try final.object.put(r.a, "output", try object(r.a, r.items.items));
        try final.object.put(r.a, "usage", try usage(r.a, chat_usage));
        try final.object.put(r.a, "incomplete_details", if (incomplete) try object(r.a, .{ .reason = "max_output_tokens" }) else .null);
        if (!incomplete) try final.object.put(r.a, "completed_at", .{ .integer = completed_at });
        if (stats != .null) try final.object.put(r.a, "tensorfold", try clone(r.a, stats));
        r.final = final;
        if (r.store) |store| try store.put(r.io, final, r.added);
        try r.event(if (incomplete) "response.incomplete" else "response.completed", .{ .response = final });
        return final;
    }

    pub fn fail(r: *Reply, err: V) !V {
        if (r.final) |value| return value;
        try r.close("incomplete");
        const code = if (is(get(err, "type"), "invalid_request_error")) "invalid_prompt" else "server_error";
        const message = if (err == .string) err else if (truthy(get(err, "message"))) get(err, "message") else V{ .string = "the reply failed" };
        var final = try clone(r.a, r.initial);
        try final.object.put(r.a, "status", .{ .string = "failed" });
        try final.object.put(r.a, "output", try object(r.a, r.items.items));
        try final.object.put(r.a, "error", try object(r.a, .{ .code = code, .message = message }));
        r.final = final;
        try r.event("response.failed", .{ .response = final });
        return final;
    }

    pub fn chunk(r: *Reply, payload: V, completed_at: i64) !void {
        if (r.final != null) return;
        if (payload == .null) {
            _ = try r.fail(.{ .string = "the reply ended early" });
        } else if (payload == .object and payload.object.contains("error")) {
            _ = try r.fail(get(payload, "error"));
        } else {
            const choices = get(payload, "choices");
            const choice = if (choices == .array and choices.array.items.len > 0) choices.array.items[0] else V.null;
            try r.delta(get(choice, "delta"));
            const reason = get(choice, "finish_reason");
            if (truthy(reason)) _ = try r.finish(reason, get(payload, "usage"), get(payload, "tensorfold"), completed_at);
        }
    }

    pub fn completion(r: *Reply, data: V, completed_at: i64) !V {
        const choice = get(data, "choices").array.items[0];
        const message = get(choice, "message");
        var calls = array(r.a);
        const input_calls = get(message, "tool_calls");
        if (input_calls == .array) for (input_calls.array.items, 0..) |call, i| {
            var copy = try clone(r.a, call);
            if (!copy.object.contains("index")) try copy.object.put(r.a, "index", .{ .integer = @intCast(i) });
            try calls.array.append(copy);
        };
        try r.delta(try object(r.a, .{ .reasoning_content = get(message, "reasoning_content"), .content = get(message, "content"), .tool_calls = calls }));
        return r.finish(get(choice, "finish_reason"), get(data, "usage"), get(data, "tensorfold"), completed_at);
    }
};

fn usage(a: A, chat: V) !V {
    if (!truthy(chat)) return .null;
    const prompt = get(chat, "prompt_tokens");
    const completion = get(chat, "completion_tokens");
    const cached = get(get(chat, "prompt_tokens_details"), "cached_tokens");
    const thought = get(get(chat, "completion_tokens_details"), "reasoning_tokens");
    return object(a, .{ .input_tokens = prompt, .input_tokens_details = .{ .cached_tokens = if (cached == .null) V{ .integer = 0 } else cached }, .output_tokens = completion, .output_tokens_details = .{ .reasoning_tokens = if (thought == .null) V{ .integer = 0 } else thought }, .total_tokens = prompt.integer + completion.integer });
}

pub fn translate(a: A, io: std.Io, body: V, store: *Store) !Request {
    if (body != .object) return error.InvalidResponseRequest;
    for ([_][]const u8{ "background", "conversation", "prompt", "context_management", "include", "top_logprobs" }) |key| if (truthy(get(body, key))) return error.UnsupportedResponseOption;
    const truncation = get(body, "truncation");
    if (truncation != .null and !is(truncation, "disabled")) return error.UnsupportedResponseTruncation;
    const metadata = if (truthy(get(body, "metadata"))) get(body, "metadata") else V{ .object = .empty };
    if (metadata != .object or metadata.object.count() > 16) return error.InvalidResponseMetadata;
    var pairs = metadata.object.iterator();
    while (pairs.next()) |pair| {
        if (try std.unicode.utf8CountCodepoints(pair.key_ptr.*) > 64 or pair.value_ptr.* != .string or try std.unicode.utf8CountCodepoints(pair.value_ptr.string) > 512) return error.InvalidResponseMetadata;
    }
    const instructions = get(body, "instructions");
    if (instructions != .null and instructions != .string) return error.InvalidResponseInstructions;
    var given = get(body, "input");
    if (given == .string) given = try object(a, &.{.{ .role = "user", .content = given }});
    if (given != .array or given.array.items.len == 0) return error.InvalidResponseInput;
    const added = try messages(a, given);
    const parent = get(body, "previous_response_id");
    var history = if (parent != .null) try store.conversation(a, io, parent) else array(a);
    var offered = try tools(a, get(body, "tools"));
    const choice = try toolChoice(a, get(body, "tool_choice"), &offered);
    var chat = V{ .object = .empty };
    for ([_][]const u8{ "model", "temperature", "top_p", "top_k", "min_p", "seed", "stream", "parallel_tool_calls", "stop", "draft", "thinking_budget", "ignore_eos", "priority", "return_token_ids", "chat_template_kwargs" }) |key| if (body.object.get(key)) |value| try chat.object.put(a, key, value);
    if (truthy(instructions)) try history.array.insert(0, try object(a, .{ .role = "system", .content = instructions }));
    try history.array.appendSlice(added.array.items);
    try chat.object.put(a, "messages", history);
    if (offered != .null) try chat.object.put(a, "tools", offered);
    if (choice != .null) try chat.object.put(a, "tool_choice", choice);
    if (get(body, "max_output_tokens") != .null) try chat.object.put(a, "max_tokens", get(body, "max_output_tokens"));
    const reasoning = if (truthy(get(body, "reasoning"))) get(body, "reasoning") else V{ .object = .empty };
    if (reasoning != .object) return error.InvalidResponseReasoning;
    if (get(reasoning, "effort") != .null) try chat.object.put(a, "reasoning_effort", get(reasoning, "effort"));
    const fmt = try format(a, get(body, "text"));
    if (fmt != .null) try chat.object.put(a, "response_format", fmt);
    const keep = get(body, "store") != .bool or get(body, "store").bool;
    const text_format = get(get(body, "text"), "format");
    const raw_choice = get(body, "tool_choice");
    const raw_tools = get(body, "tools");
    const echo = try object(a, .{
        .instructions = instructions,
        .max_output_tokens = get(body, "max_output_tokens"),
        .metadata = metadata,
        .parallel_tool_calls = body.object.get("parallel_tool_calls") orelse V{ .bool = true },
        .previous_response_id = parent,
        .reasoning = .{ .effort = get(reasoning, "effort"), .summary = get(reasoning, "summary") },
        .store = keep,
        .temperature = get(body, "temperature"),
        .text = .{ .format = if (truthy(text_format)) text_format else try object(a, .{ .type = "text" }) },
        .tool_choice = if (truthy(raw_choice)) raw_choice else V{ .string = "auto" },
        .tools = if (truthy(raw_tools)) raw_tools else array(a),
        .top_p = get(body, "top_p"),
        .truncation = "disabled",
        .user = get(body, "user"),
        .background = false,
    });
    return .{ .chat = chat, .echo = echo, .added = added, .store = keep, .stream = truthy(get(body, "stream")) };
}

pub const Store = struct {
    a: A,
    limit: usize = 1024,
    max_bytes: usize = 256 * 1024 * 1024,
    bytes: usize = 0,
    entries: std.ArrayList(Entry) = .empty,
    mutex: std.Io.Mutex = .init,

    const Entry = struct {
        arena: std.heap.ArenaAllocator,
        response: V,
        added: V,
        size: usize,
    };

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |*entry| entry.arena.deinit();
        s.entries.deinit(s.a);
    }

    fn index(s: *Store, id: []const u8) ?usize {
        for (s.entries.items, 0..) |entry, i| if (is(get(entry.response, "id"), id)) return i;
        return null;
    }

    fn remove(s: *Store, i: usize) void {
        var entry = s.entries.orderedRemove(i);
        s.bytes -= entry.size;
        entry.arena.deinit();
    }

    pub fn put(s: *Store, io: std.Io, response: V, added: V) !void {
        var arena = std.heap.ArenaAllocator.init(s.a);
        errdefer arena.deinit();
        const a = arena.allocator();
        const owned = try clone(a, response);
        if (get(owned, "id") != .string or added != .array) return error.InvalidStoredResponse;
        var all = try clone(a, added);
        const output = try messages(a, get(owned, "output"));
        try all.array.appendSlice(output.array.items);
        const size = jsonSize(owned) + jsonSize(all);
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        try s.entries.ensureUnusedCapacity(s.a, 1);
        if (s.index(get(owned, "id").string)) |i| s.remove(i);
        s.entries.appendAssumeCapacity(.{ .arena = arena, .response = owned, .added = all, .size = size });
        s.bytes += size;
        while (s.entries.items.len > 0 and (s.entries.items.len > s.limit or s.bytes > s.max_bytes)) s.remove(0);
    }

    pub fn getResponse(s: *Store, a: A, io: std.Io, id: []const u8) !?V {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const i = s.index(id) orelse return null;
        return try clone(a, s.entries.items[i].response);
    }

    pub fn delete(s: *Store, io: std.Io, id: []const u8) bool {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const i = s.index(id) orelse return false;
        s.remove(i);
        return true;
    }

    pub fn conversation(s: *Store, a: A, io: std.Io, id: V) !V {
        if (id != .string) return error.InvalidPreviousResponse;
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        var chain: std.ArrayList(V) = .empty;
        var at = id;
        while (at != .null) {
            if (at != .string) return error.InvalidPreviousResponse;
            if (chain.items.len >= s.entries.items.len) return error.CyclicResponseHistory;
            const i = s.index(at.string) orelse return error.PreviousResponseNotStored;
            const entry = s.entries.orderedRemove(i);
            s.entries.appendAssumeCapacity(entry);
            try chain.append(a, entry.added);
            at = get(entry.response, "previous_response_id");
        }
        var out = array(a);
        var remaining = chain.items.len;
        while (remaining > 0) {
            remaining -= 1;
            try out.array.appendSlice((try clone(a, chain.items[remaining])).array.items);
        }
        return out;
    }
};

// Upstream budgets json.dumps' ASCII representation, including separator spaces.
fn stringSize(text: []const u8) usize {
    var size: usize = 2;
    var iter = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (iter.nextCodepoint()) |c| size += switch (c) {
        '"', '\\', '\n', '\r', '\t', 8, 12 => 2,
        0...7, 11, 14...31, 127...0xffff => 6,
        0x10000...0x10ffff => 12,
        else => 1,
    };
    return size;
}

fn jsonSize(value: V) usize {
    return switch (value) {
        .null => 4,
        .bool => if (value.bool) 4 else 5,
        .string => stringSize(value.string),
        .integer => blk: {
            var buffer: [32]u8 = undefined;
            break :blk (std.fmt.bufPrint(&buffer, "{d}", .{value.integer}) catch unreachable).len;
        },
        .float => blk: {
            var buffer: [64]u8 = undefined;
            const scientific = std.fmt.bufPrint(&buffer, "{e}", .{value.float}) catch unreachable;
            const at = std.mem.indexOfScalar(u8, scientific, 'e') orelse return scientific.len;
            const exponent = std.fmt.parseInt(i32, scientific[at + 1 ..], 10) catch unreachable;
            if (exponent < -4 or exponent >= 16) {
                const digits: usize = if (@abs(exponent) >= 100) 3 else 2;
                break :blk at + 2 + digits;
            }
            const decimal = std.fmt.bufPrint(&buffer, "{d}", .{value.float}) catch unreachable;
            break :blk decimal.len + @as(usize, if (std.mem.indexOfScalar(u8, decimal, '.') == null) 2 else 0);
        },
        .number_string => value.number_string.len,
        .array => blk: {
            var size: usize = 2;
            for (value.array.items, 0..) |item, i| size += jsonSize(item) + @as(usize, if (i > 0) 2 else 0);
            break :blk size;
        },
        .object => blk: {
            var size: usize = 2;
            var iter = value.object.iterator();
            var first = true;
            while (iter.next()) |item| {
                size += stringSize(item.key_ptr.*) + 2 + jsonSize(item.value_ptr.*) + @as(usize, if (first) 0 else 2);
                first = false;
            }
            break :blk size;
        },
    };
}

fn expectJson(expected: V, actual: V) !void {
    if (expected == .object and actual == .object) {
        try std.testing.expectEqual(expected.object.count(), actual.object.count());
        var iter = expected.object.iterator();
        while (iter.next()) |pair| try expectJson(pair.value_ptr.*, actual.object.get(pair.key_ptr.*) orelse return error.MissingResponseField);
    } else if (expected == .array and actual == .array) {
        try std.testing.expectEqual(expected.array.items.len, actual.array.items.len);
        for (expected.array.items, actual.array.items) |lhs, rhs| try expectJson(lhs, rhs);
    } else if (expected == .string and actual == .string) {
        try std.testing.expectEqualStrings(expected.string, actual.string);
    } else try std.testing.expectEqualDeep(expected, actual);
}

pub fn check(io: std.Io, path: []const u8) !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(debug_allocator.deinit() == .ok);
    const checked = debug_allocator.allocator();
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 * 1024 * 1024));
    const fixture = try std.json.parseFromSlice(V, a, bytes, .{});
    {
        var store = Store{ .a = checked };
        defer store.deinit();
        {
            var input_arena = std.heap.ArenaAllocator.init(checked);
            defer input_arena.deinit();
            const input_a = input_arena.allocator();
            try store.put(io, try object(input_a, .{ .id = "owned", .previous_response_id = @as(?u8, null), .output = &.{.{ .role = "assistant", .content = "retained" }} }), try object(input_a, &.{.{ .role = "user", .content = "question" }}));
        }
        var copy = (try store.getResponse(a, io, "owned")).?;
        try copy.object.put(a, "id", .{ .string = "changed" });
        const retained = (try store.getResponse(a, io, "owned")).?;
        try std.testing.expectEqualStrings("owned", get(retained, "id").string);
        try std.testing.expect(store.delete(io, "owned"));
        try std.testing.expectEqualStrings("retained", get(get(retained, "output").array.items[0], "content").string);
        try std.testing.expectEqual(@as(usize, 0), store.bytes);
    }
    for (get(fixture.value, "sizes").array.items) |case| {
        const input = get(case, "value");
        const expected: usize = @intCast(get(case, "expected").integer);
        if (jsonSize(input) != expected) {
            std.debug.print("Responses store size mismatch for {s}\n", .{try std.json.Stringify.valueAlloc(a, input, .{})});
            return error.ResponseSizeMismatch;
        }
    }
    for (get(fixture.value, "routes").array.items) |case| {
        const expected = get(case, "expected");
        const found = route(get(case, "path").string);
        if (expected == .null) try std.testing.expectEqual(null, found) else try std.testing.expectEqualStrings(expected.string, found orelse return error.MissingResponseRoute);
    }
    const requests = get(fixture.value, "requests").array.items;
    for (requests, 0..) |case, i| {
        var scratch = std.heap.ArenaAllocator.init(checked);
        defer scratch.deinit();
        var store = Store{ .a = checked };
        defer store.deinit();
        const result = translate(scratch.allocator(), io, get(case, "body"), &store) catch |err| {
            if (truthy(get(case, "error"))) continue;
            std.debug.print("Responses request {d}: {s}\n", .{ i, @errorName(err) });
            return err;
        };
        if (truthy(get(case, "error"))) return error.ExpectedResponseRejection;
        try expectJson(get(case, "expected"), try object(scratch.allocator(), result));
    }
    var operations: usize = 0;
    for (get(fixture.value, "stores").array.items) |scenario| {
        var store = Store{ .a = checked, .limit = @intCast(get(scenario, "limit").integer), .max_bytes = @intCast(get(scenario, "max_bytes").integer) };
        defer store.deinit();
        for (get(scenario, "actions").array.items, 0..) |action, i| {
            var scratch = std.heap.ArenaAllocator.init(checked);
            defer scratch.deinit();
            const result = storeAction(scratch.allocator(), io, &store, action) catch |err| blk: {
                if (!truthy(get(action, "error"))) {
                    std.debug.print("Responses store action {d}: {s}\n", .{ i, @errorName(err) });
                    return err;
                }
                break :blk V.null;
            };
            try expectJson(get(action, "expected"), result);
            const ids = get(action, "ids").array.items;
            try std.testing.expectEqual(ids.len, store.entries.items.len);
            for (ids, store.entries.items) |id, entry| try expectJson(id, get(entry.response, "id"));
            try std.testing.expectEqual(@as(usize, @intCast(get(action, "bytes").integer)), store.bytes);
            operations += 1;
        }
    }
    const replies = get(fixture.value, "replies").array.items;
    for (replies, 0..) |case, i| {
        var scratch = std.heap.ArenaAllocator.init(checked);
        defer scratch.deinit();
        const alloc = scratch.allocator();
        var store = Store{ .a = checked };
        defer store.deinit();
        var probe = Probe{ .a = alloc, .io = io, .store = &store };
        var reply = Reply{ .a = alloc, .text_allocator = checked, .io = io, .initial = get(case, "base"), .context = &probe, .emit = Probe.emit, .store = &store, .added = get(case, "added") };
        defer reply.deinit();
        try reply.start();
        if (get(case, "completion") != .null) {
            _ = try reply.completion(get(case, "completion"), 1234);
        } else for (get(case, "chunks").array.items) |chunk| try reply.chunk(chunk, 1234);
        expectJson(get(case, "events"), try object(alloc, probe.events.items)) catch |err| {
            std.debug.print("Responses reply {d} event mismatch\n", .{i});
            return err;
        };
        try expectJson(get(case, "kept"), try object(alloc, probe.kept.items));
        try expectJson(get(case, "expected"), reply.final orelse .null);
        try expectJson(get(case, "stored"), (try store.getResponse(alloc, io, "resp_fixture")) orelse .null);
    }
    std.debug.print("PASS: {d} upstream Responses translations, {d} bounded-store/chain operations, {d} reply/event sequences and route checks\n", .{ requests.len, operations, replies.len });
}

const Probe = struct {
    a: A,
    io: std.Io,
    store: *Store,
    events: std.ArrayList(V) = .empty,
    kept: std.ArrayList(bool) = .empty,
    fn emit(context: ?*anyopaque, event: V) !void {
        const p: *Probe = @ptrCast(@alignCast(context.?));
        try p.events.append(p.a, try clone(p.a, event));
        try p.kept.append(p.a, (try p.store.getResponse(p.a, p.io, "resp_fixture")) != null);
    }
};

fn storeAction(a: A, io: std.Io, store: *Store, action: V) !V {
    const op = get(action, "op");
    if (is(op, "put")) {
        try store.put(io, get(action, "response"), get(action, "added"));
        return .null;
    }
    const id = get(action, "id");
    if (is(op, "get")) return (try store.getResponse(a, io, id.string)) orelse .null;
    if (is(op, "delete")) return .{ .bool = store.delete(io, id.string) };
    if (is(op, "conversation")) return store.conversation(a, io, id);
    if (is(op, "translate")) return object(a, try translate(a, io, get(action, "body"), store));
    return error.InvalidStoreAction;
}
