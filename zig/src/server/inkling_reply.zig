//! Inkling 2's reply blocks split into reasoning, visible text and tool calls, as vLLM's Inkling parser reads them.
const std = @import("std");

/// The release's block markers (special tokens in tokenizer_config.json).
pub const markers = struct {
    pub const model = "<|message_model|>";
    pub const text = "<|content_text|>";
    pub const thinking = "<|content_thinking|>";
    pub const end = "<|end_message|>";
    pub const end_sampling = "<|content_model_end_sampling|>";
    pub const tool_json = "<|content_invoke_tool_json|>";
    pub const tool_text = "<|content_invoke_tool_text|>";
    pub const tool_error = "<|content_tool_error|>";
};

const Marker = enum { model, text, thinking, end, end_sampling, tool_json, tool_text, tool_error };
const State = enum { header, content, reasoning, tool };

pub const Call = struct { name: []const u8, arguments: []const u8 };

pub const Reply = struct {
    reasoning: []const u8,
    content: []const u8,
    calls: []const Call,
    /// A tool block the reply ended inside (max_tokens): its name when the payload got that far, never raw markup.
    cut: ?[]const u8 = null,
};

fn markerAt(s: []const u8) ?struct { m: Marker, len: usize } {
    inline for (comptime std.enums.values(Marker)) |m| {
        const text = @field(markers, @tagName(m));
        if (std.mem.startsWith(u8, s, text)) return .{ .m = m, .len = text.len };
    }
    return null;
}

/// The next state for a marker in a state, or null where vLLM's table has no entry (the marker is then plain text).
fn next(s: State, m: Marker) ?State {
    return switch (m) {
        .end, .end_sampling => .content,
        .model => if (s == .content or s == .header) .header else null,
        .text, .tool_text, .tool_error => if (s == .content or s == .header) .content else null,
        .thinking => if (s == .tool) null else .reasoning,
        .tool_json => if (s == .tool) null else .tool,
    };
}

/// Split one reply (specials spelled out) into `a`; unfinished, it holds back an open header that may name a tool.
pub fn split(a: std.mem.Allocator, reply: []const u8, finished: bool) !Reply {
    var reasoning: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    var header: std.ArrayList(u8) = .empty;
    var payload: std.ArrayList(u8) = .empty;
    var calls: std.ArrayList(Call) = .empty;
    var state: State = .header;
    var i: usize = 0;
    while (i < reply.len) {
        if (reply[i] == '<') if (markerAt(reply[i..])) |hit| if (next(state, hit.m)) |to| {
            i += hit.len;
            if (state == .tool and to == .content) try calls.append(a, try call(a, payload.items));
            if (state == .header and (hit.m == .end or hit.m == .end_sampling)) try content.appendSlice(a, header.items);
            if (to == .header or state == .header) header.clearRetainingCapacity();
            if (to == .tool) payload.clearRetainingCapacity();
            state = to;
            continue;
        };
        const stop = std.mem.indexOfScalarPos(u8, reply, i + 1, '<') orelse reply.len;
        const piece = reply[i..stop];
        switch (state) {
            .header => try header.appendSlice(a, piece),
            .content => try content.appendSlice(a, piece),
            .reasoning => try reasoning.appendSlice(a, piece),
            .tool => try payload.appendSlice(a, piece),
        }
        i = stop;
    }
    var out: Reply = .{
        .reasoning = std.mem.trimEnd(u8, reasoning.items, " \t\r\n"),
        .content = content.items,
        .calls = calls.items,
    };
    if (state == .tool) out.cut = name(payload.items);
    if (finished and state == .header and header.items.len > 0) out.content = try std.mem.concat(a, u8, &.{ content.items, header.items });
    const blank = std.mem.trim(u8, out.content, " \t\r\n").len == 0;
    if (blank and (out.calls.len > 0 or out.cut != null)) out.content = "";
    return out;
}

/// A finished tool block: `{"name": ..., "args": {...}}`; the arguments are the args member as written, or {}.
fn call(a: std.mem.Allocator, payload: []const u8) !Call {
    var doc = std.json.parseFromSlice(std.json.Value, a, payload, .{}) catch return .{ .name = "", .arguments = "{}" };
    defer doc.deinit();
    if (doc.value != .object) return .{ .name = "", .arguments = "{}" };
    const n = doc.value.object.get("name") orelse std.json.Value{ .null = {} };
    const own = if (n == .string) try a.dupe(u8, n.string) else "";
    return .{ .name = own, .arguments = try a.dupe(u8, member(payload, "args") orelse "{}") };
}

/// The verbatim value of a top-level member of a JSON object known to parse.
fn member(raw: []const u8, key: []const u8) ?[]const u8 {
    var depth: usize = 0;
    var i: usize = 0;
    while (i < raw.len) : (i += 1) {
        switch (raw[i]) {
            '{', '[' => depth += 1,
            '}', ']' => depth -|= 1,
            '"' => {
                const close = stringEnd(raw, i) orelse return null;
                const s = raw[i + 1 .. close];
                i = close;
                if (depth != 1 or !std.mem.eql(u8, s, key)) continue;
                var j = close + 1;
                while (j < raw.len and (raw[j] == ' ' or raw[j] == ':' or raw[j] == '\n' or raw[j] == '\t')) j += 1;
                return valueSpan(raw, j);
            },
            else => {},
        }
    }
    return null;
}

fn stringEnd(raw: []const u8, open: usize) ?usize {
    var j = open + 1;
    while (j < raw.len) : (j += 1) {
        if (raw[j] == '\\') j += 1 else if (raw[j] == '"') return j;
    }
    return null;
}

fn valueSpan(raw: []const u8, start: usize) ?[]const u8 {
    if (start >= raw.len) return null;
    if (raw[start] != '{' and raw[start] != '[') {
        var j = start;
        if (raw[j] == '"') return raw[start .. (stringEnd(raw, j) orelse return null) + 1];
        while (j < raw.len and raw[j] != ',' and raw[j] != '}') j += 1;
        return std.mem.trimEnd(u8, raw[start..j], " \t\r\n");
    }
    var depth: usize = 0;
    var j = start;
    while (j < raw.len) : (j += 1) {
        switch (raw[j]) {
            '{', '[' => depth += 1,
            '}', ']' => {
                depth -= 1;
                if (depth == 0) return raw[start .. j + 1];
            },
            '"' => j = stringEnd(raw, j) orelse return null,
            else => {},
        }
    }
    return null;
}

/// The name of a cut tool block, when its payload already holds a whole "name" string.
fn name(payload: []const u8) ?[]const u8 {
    const at = std.mem.indexOf(u8, payload, "\"name\"") orelse return null;
    var j = at + 6;
    while (j < payload.len and (payload[j] == ' ' or payload[j] == ':')) j += 1;
    if (j >= payload.len or payload[j] != '"') return null;
    const close = stringEnd(payload, j) orelse return null;
    return payload[j + 1 .. close];
}

const t = std.testing;

fn splitT(arena: *std.heap.ArenaAllocator, text: []const u8) !Reply {
    return split(arena.allocator(), text, true);
}

test "inkling2 reply: a chat turn gives its reasoning and its answer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const r = try splitT(&arena, "<|message_model|><|content_thinking|>The user says hello. \n<|end_message|>" ++
        "<|message_model|><|content_text|>Hello! How can I help?<|end_message|><|content_model_end_sampling|>");
    try t.expectEqualStrings("The user says hello.", r.reasoning);
    try t.expectEqualStrings("Hello! How can I help?", r.content);
    try t.expectEqual(@as(usize, 0), r.calls.len);
    try t.expect(r.cut == null);
}

test "inkling2 reply: a tool call takes its name from the payload and its arguments verbatim" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const r = try splitT(&arena, "<|message_model|><|content_thinking|>Need the weather.<|end_message|>" ++
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"S<F>\",\"days\":2}}" ++
        "<|end_message|><|content_model_end_sampling|>");
    try t.expectEqualStrings("Need the weather.", r.reasoning);
    try t.expectEqualStrings("", r.content);
    try t.expectEqual(@as(usize, 1), r.calls.len);
    try t.expectEqualStrings("get_weather", r.calls[0].name);
    try t.expectEqualStrings("{\"city\":\"S<F>\",\"days\":2}", r.calls[0].arguments);
}

test "inkling2 reply: two calls, a call without args, and text between blocks" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const r = try splitT(&arena, "<|message_model|><|content_text|>Checking both.<|end_message|>" ++
        "<|message_model|>a<|content_invoke_tool_json|>{\"name\":\"a\",\"args\":{\"x\":[1,{\"y\":\"}\"}]}}<|end_message|>" ++
        "<|message_model|>b<|content_invoke_tool_json|>{\"name\":\"b\"}<|end_message|>");
    try t.expectEqualStrings("Checking both.", r.content);
    try t.expectEqual(@as(usize, 2), r.calls.len);
    try t.expectEqualStrings("{\"x\":[1,{\"y\":\"}\"}]}", r.calls[0].arguments);
    try t.expectEqualStrings("b", r.calls[1].name);
    try t.expectEqualStrings("{}", r.calls[1].arguments);
}

test "inkling2 reply: thinking off, a header with no block, and a cut call" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const plain = try splitT(&arena, "<|message_model|><|content_text|>4<|end_message|><|content_model_end_sampling|>");
    try t.expectEqualStrings("", plain.reasoning);
    try t.expectEqualStrings("4", plain.content);
    const bare = try splitT(&arena, "Hi there<|end_message|>");
    try t.expectEqualStrings("Hi there", bare.content);
    const cut = try splitT(&arena, "<|message_model|><|content_thinking|>x<|end_message|><|message_model|>get_weather" ++
        "<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"ci");
    try t.expectEqualStrings("get_weather", cut.cut.?);
    try t.expectEqualStrings("", cut.content);
    try t.expectEqual(@as(usize, 0), cut.calls.len);
}

test "inkling2 reply: a marker vLLM's table does not take in a state stays text" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const r = try splitT(&arena, "<|message_model|><|content_thinking|>a<|content_text|>b<|end_message|>");
    try t.expectEqualStrings("a<|content_text|>b", r.reasoning);
    try t.expectEqualStrings("", r.content);
}

test "inkling2 reply: a stream holds back a header until its block says what it is" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const so_far = "<|message_model|><|content_thinking|>List /tmp.<|end_message|><|message_model|>run_shell";
    try t.expectEqualStrings("", (try split(a, so_far, false)).content);
    const named = try split(a, so_far ++ "_command<|content_invoke_tool_json|>{\"name\":\"run_shell_command\"}<|end_message|>", false);
    try t.expectEqualStrings("", named.content);
    try t.expectEqualStrings("run_shell_command", named.calls[0].name);
    try t.expectEqualStrings("Hi", (try split(a, "<|message_model|>Hi", true)).content);
}
