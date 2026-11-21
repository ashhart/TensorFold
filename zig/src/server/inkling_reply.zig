//! Inkling 2's reply blocks split into reasoning, visible text and tool calls, as vLLM's Inkling parser reads them.
const std = @import("std");
const json = @import("json");
const json_cut = @import("tool_json_cut.zig");
const tool_parse = @import("tool_parse.zig");

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

/// Inkling 2's replies end their sampling with this marker; a tokenizer that holds it speaks the block format.
pub fn speaks(text: anytype) bool {
    return text.vtable.token_id(text.ctx, markers.end_sampling) != null and text.vtable.token_id(text.ctx, markers.model) != null;
}

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
pub fn split(a: std.mem.Allocator, reply: []const u8, finished: bool, offered: []const json.Value) !Reply {
    var s: Stream = .{ .a = a, .offered = offered };
    try s.feed(reply, false);
    return s.result(reply, finished);
}

/// The split kept across a stream's steps: each step reads only the text past `fed`.
pub const Stream = struct {
    a: std.mem.Allocator,
    offered: []const json.Value = &.{}, // the request's tools: a block that calls none of them is content
    state: State = .header,
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    header: std.ArrayList(u8) = .empty,
    payload: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Call) = .empty,
    fed: usize = 0,
    cut: ?[]const u8 = null, // the open tool block's name, once its payload holds it whole

    /// Read `reply`, which only grows, past `fed`; `hold` keeps back an end that may still become a marker.
    pub fn feed(s: *Stream, reply: []const u8, hold: bool) !void {
        const a = s.a;
        var i = s.fed;
        while (i < reply.len) {
            if (reply[i] == '<') {
                if (markerAt(reply[i..])) |hit| if (next(s.state, hit.m)) |to| {
                    i += hit.len;
                    if (s.state == .tool and to == .content) try s.close();
                    if (s.state == .header and (hit.m == .end or hit.m == .end_sampling)) try s.content.appendSlice(a, s.header.items);
                    if (to == .header or s.state == .header) s.header.clearRetainingCapacity();
                    if (to == .tool) s.payload.clearRetainingCapacity();
                    if (to == .tool) s.cut = null;
                    s.state = to;
                    continue;
                };
                if (hold and opens(reply[i..])) break;
            }
            const stop = std.mem.indexOfScalarPos(u8, reply, i + 1, '<') orelse reply.len;
            try s.open().appendSlice(a, reply[i..stop]);
            i = stop;
        }
        s.fed = i;
    }

    /// A finished tool block: a call to an offered tool, else its payload as written, kept as content.
    fn close(s: *Stream) !void {
        if (try call(s.a, s.payload.items, s.offered)) |c| return s.calls.append(s.a, c);
        try s.content.appendSlice(s.a, s.payload.items);
    }

    fn open(s: *Stream) *std.ArrayList(u8) {
        return switch (s.state) {
            .header => &s.header,
            .content => &s.content,
            .reasoning => &s.reasoning,
            .tool => &s.payload,
        };
    }

    /// The split of `reply` so far, a held end read as text in the open block, as `split` reads it.
    pub fn result(s: *Stream, reply: []const u8, finished: bool) !Reply {
        const a = s.a;
        const tail = reply[s.fed..];
        var reasoning = s.reasoning.items;
        var content = s.content.items;
        var header = s.header.items;
        var payload = s.payload.items;
        if (tail.len > 0) {
            const held = switch (s.state) {
                .header => &header,
                .content => &content,
                .reasoning => &reasoning,
                .tool => &payload,
            };
            held.* = try std.mem.concat(a, u8, &.{ held.*, tail });
        }
        if (s.state == .tool and s.cut == null) s.cut = try name(a, payload);
        var out: Reply = .{
            .reasoning = std.mem.trimEnd(u8, reasoning, " \t\r\n"),
            .content = content,
            .calls = s.calls.items,
        };
        if (s.state == .tool) out.cut = s.cut;
        if (finished and s.state == .header and header.len > 0) out.content = try std.mem.concat(a, u8, &.{ content, header });
        const blank = std.mem.trim(u8, out.content, " \t\r\n").len == 0;
        if (blank and (out.calls.len > 0 or out.cut != null)) out.content = "";
        return out;
    }
};

/// Whether `s` (from a `<`) is a marker's start cut short, so later text may complete it.
fn opens(s: []const u8) bool {
    inline for (comptime std.enums.values(Marker)) |m| {
        const text = @field(markers, @tagName(m));
        if (s.len < text.len and std.mem.startsWith(u8, text, s)) return true;
    }
    return false;
}

/// How many of `ids` lie inside thinking blocks, reading the markers' token ids as `split` reads their text.
pub fn thinkingTokens(text: anytype, ids: []const u32) usize {
    const all = comptime std.enums.values(Marker);
    var marker_ids: [all.len]u32 = undefined;
    inline for (all, 0..) |m, k| marker_ids[k] = text.tokenId(@field(markers, @tagName(m))) orelse std.math.maxInt(u32);
    var state: State = .header;
    var n: usize = 0;
    for (ids) |id| {
        if (std.mem.indexOfScalar(u32, &marker_ids, id)) |k| if (next(state, all[k])) |to| {
            state = to;
            continue;
        };
        if (state == .reasoning) n += 1;
    }
    return n;
}

/// A tool block's `{"name": ..., "args": {...}}` naming an offered tool, under its spelling; null for anything else.
fn call(a: std.mem.Allocator, payload: []const u8, offered: []const json.Value) !?Call {
    var doc = std.json.parseFromSlice(std.json.Value, a, payload, .{}) catch return null;
    defer doc.deinit();
    if (doc.value != .object) return null;
    const n = doc.value.object.get("name") orelse return null;
    if (n != .string) return null;
    var known = try tool_parse.knownNames(a, offered);
    const own = try tool_parse.callName(a, n.string, &known) orelse return null;
    const args = json_cut.find(try json_cut.members(a, std.mem.trimStart(u8, payload, " \t\r\n")), "args");
    return .{ .name = own, .arguments = try a.dupe(u8, if (args) |m| m.value else "{}") };
}

/// The name of a cut tool block, when its payload already holds a whole top-level "name" string.
fn name(a: std.mem.Allocator, payload: []const u8) !?[]const u8 {
    const body = std.mem.trimStart(u8, payload, " \t\r\n");
    if (body.len == 0 or body[0] != '{') return null;
    const m = json_cut.find(try json_cut.members(a, body), "name") orelse return null;
    if (m.cut or m.value.len < 2 or m.value[0] != '"') return null;
    return m.value[1 .. m.value.len - 1];
}

const t = std.testing;

/// The tools a request offers, from their JSON list.
fn toolsT(a: std.mem.Allocator, list: []const u8) ![]const json.Value {
    return (try json.parse(a, list)).ok.items().?;
}

const offered_t = "[{\"name\":\"get_weather\"},{\"name\":\"a\"},{\"name\":\"b\"},{\"name\":\"f\"},{\"name\":\"g\"}," ++
    "{\"type\":\"function\",\"function\":{\"name\":\"run_shell_command\"}}]";

fn splitT(arena: *std.heap.ArenaAllocator, text: []const u8) !Reply {
    const a = arena.allocator();
    return split(a, text, true, try toolsT(a, offered_t));
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
    const tools = try toolsT(a, offered_t);
    const so_far = "<|message_model|><|content_thinking|>List /tmp.<|end_message|><|message_model|>run_shell";
    try t.expectEqualStrings("", (try split(a, so_far, false, tools)).content);
    const named = try split(a, so_far ++ "_command<|content_invoke_tool_json|>{\"name\":\"run_shell_command\"}<|end_message|>", false, tools);
    try t.expectEqualStrings("", named.content);
    try t.expectEqualStrings("run_shell_command", named.calls[0].name);
    try t.expectEqualStrings("Hi", (try split(a, "<|message_model|>Hi", true, tools)).content);
}

test "inkling2 reply: thinking tokens are the ids inside thinking blocks" {
    const Ids = struct {
        fn tokenId(_: @This(), piece: []const u8) ?u32 {
            for ([_][]const u8{ markers.model, markers.thinking, markers.end }, 1..) |m, id| if (std.mem.eql(u8, m, piece)) return @intCast(id);
            return null;
        }
    };
    try t.expectEqual(@as(usize, 3), thinkingTokens(Ids{}, &.{ 1, 2, 70, 71, 72, 3, 1, 80 }));
    try t.expectEqual(@as(usize, 0), thinkingTokens(Ids{}, &.{ 1, 80, 81 }));
}

test "inkling2 reply: a stream fed a byte at a time splits each prefix as the whole split does" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsT(a, offered_t);
    const reply = "<|message_model|><|content_thinking|>a \u{e9}<|content_text|>b<|end_message|><|message_model|><|content_text|>" ++
        "x <|content_te y\u{1f600}<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"k\":1}}" ++
        "<|end_message|><|message_model|>h<|content_invoke_tool_json|>{\"name\":\"h\"}<|end_message|><|message_model|>f" ++
        "<|content_invoke_tool_json|>{\"name\":<|end_message|><|message_model|>g<|content_invoke_tool_json|>{\"name\":\"g\",\"args\":{\"q\":\"<|\"";
    var s: Stream = .{ .a = a, .offered = tools };
    for (1..reply.len + 1) |n| {
        try s.feed(reply[0..n], true);
        const got = try s.result(reply[0..n], false);
        const want = try split(a, reply[0..n], false, tools);
        try t.expectEqualStrings(want.reasoning, got.reasoning);
        try t.expectEqualStrings(want.content, got.content);
        try t.expectEqual(want.calls.len, got.calls.len);
        try t.expectEqualStrings(want.cut orelse "-", got.cut orelse "-");
    }
}

test "inkling2 reply: a call takes the offered spelling, and a call to a tool not offered stays content" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const block = "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\" get_weather\",\"args\":{}}<|end_message|>";
    const spelled = try split(a, block, true, try toolsT(a, "[{\"name\":\"Get_Weather\"}]"));
    try t.expectEqual(@as(usize, 1), spelled.calls.len);
    try t.expectEqualStrings("Get_Weather", spelled.calls[0].name);
    // a named tool_choice leaves only that tool offered (tool_specs.active), so a call to another one is content
    for ([_][]const u8{ "[{\"name\":\"send_mail\"}]", "[]" }) |list| {
        const r = try split(a, block, true, try toolsT(a, list));
        try t.expectEqual(@as(usize, 0), r.calls.len);
        try t.expectEqualStrings("{\"name\":\" get_weather\",\"args\":{}}", r.content);
    }
}

test "inkling2 reply: a tool block that is not a call stays content" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for ([_][]const u8{ "{\"name\":\"f\",\"args\":{", "{\"args\":{}}", "{\"name\":7}", "[\"f\"]", "" }) |payload| {
        const r = try splitT(&arena, try std.mem.concat(arena.allocator(), u8, &.{ "<|message_model|>f<|content_invoke_tool_json|>", payload, "<|end_message|>" }));
        try t.expectEqual(@as(usize, 0), r.calls.len);
        try t.expectEqualStrings(payload, r.content);
    }
}

test "inkling2 reply: args are the member keyed args, whatever the values, key escapes and whitespace" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const cases = [_][2][]const u8{
        .{ "{\"name\":\"f\",\"why\":\"args\",\"args\":{\"k\":\"name\"}}", "{\"k\":\"name\"}" },
        .{ "{\"why\":\"args\",\"name\":\"f\",\"args\":[\"args\"]}", "[\"args\"]" },
        .{ "{\"name\":\"f\",\"\\u0061rgs\":{\"k\":2}}", "{\"k\":2}" },
        .{ "{\n\t\"name\"\t:\t\"f\" ,\r\n\t\"args\"\n:\t{\"k\":\t3}\n}", "{\"k\":\t3}" },
    };
    for (cases) |c| {
        const r = try splitT(&arena, try std.mem.concat(arena.allocator(), u8, &.{ "<|message_model|>f<|content_invoke_tool_json|>", c[0], "<|end_message|>" }));
        try t.expectEqual(@as(usize, 1), r.calls.len);
        try t.expectEqualStrings(c[1], r.calls[0].arguments);
    }
}
