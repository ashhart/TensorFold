//! GLM's streamed arguments join to the whole reply's call, byte for byte.
const std = @import("std");
const json = @import("json");
const pre = @import("stream_preflight_test.zig");
const tool_parse = @import("tool_parse.zig");
const stream = @import("tool_stream.zig");
const TestText = pre.TestText;
const CallEngine = pre.CallEngine;
const sendWith = pre.sendWith;
const Streamer = stream.Streamer;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The tools these cases offer: string, integer, array and object parameters.
const tools_text = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"q\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"},\"tags\":{\"type\":\"array\"},\"obj\":{\"type\":\"object\"}}}}}," ++
    "{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}}]";

/// A chat request's tools, before its body closes: the whole and streamed replies share it.
const chat_tools = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"q\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"},\"tags\":{\"type\":\"array\"},\"obj\":{\"type\":\"object\"}}}}}," ++
    "{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}}]";

fn toolsOf(a: Allocator) ![]const Value {
    return (try json.parse(a, tools_text)).ok.array;
}

/// The deltas a streamer sends while the reply grows through ``pieces``.
fn streamed(a: Allocator, tools: []const Value, pieces: []const []const u8) ![]Value {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    for (pieces) |p| try s.feed(p, &out);
    return out.items;
}

/// The arguments a client joins from streamed deltas.
fn joined(a: Allocator, deltas: []const Value) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        try out.appendSlice(a, one.get("function").?.get("arguments").?.string);
    };
    return out.items;
}

/// How many deltas open a call.
fn headers(deltas: []const Value) usize {
    var n: usize = 0;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        n += @intFromBool(one.get("id") != null);
    };
    return n;
}

/// The whole reply's calls' arguments joined, or "" when it makes none.
fn whole(a: Allocator, tools: []const Value, text: []const u8) ![]const u8 {
    const calls = (try tool_parse.parse(a, text, tools, null)).calls orelse return "";
    var out: std.ArrayList(u8) = .empty;
    for (calls) |c| try out.appendSlice(a, c.get("function").?.get("arguments").?.string);
    return out.items;
}

/// ``<tool_call>\n<function=NAME>\n<parameter=q>\nVALUE\n</parameter>\n</function>\n</tool_call>``.
fn xmlCall(a: Allocator, name: []const u8, value: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ "<tool_call>\n<function=", name, ">\n<parameter=q>\n", value, "\n</parameter>\n</function>\n</tool_call>" });
}

/// ``<tool_call>NAME<arg_key>q</arg_key><arg_value>VALUE</arg_value></tool_call>``.
fn glmCall(a: Allocator, name: []const u8, value: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ "<tool_call>", name, "<arg_key>q</arg_key><arg_value>", value, "</arg_value></tool_call>" });
}

test "a GLM call's two thousand byte value streams before it closes and joins as the whole reply's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const value = try a.alloc(u8, 2000);
    @memset(value, 'x');
    const text = try std.mem.concat(a, u8, &.{
        "<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value>",
        "<arg_key>content</arg_key><arg_value>",
        value,
        "</arg_value></tool_call>",
    });
    const closer = std.mem.lastIndexOf(u8, text, "</arg_value>").?;
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    var early: usize = 0;
    var at: usize = 0;
    while (at < text.len) : (at += 3) {
        const end = @min(at + 3, text.len);
        try s.feed(text[0..end], &out);
        if (end < closer) early = out.items.len;
    }
    try std.testing.expectEqual(@as(usize, 1), headers(out.items));
    try std.testing.expect(out.items.len > 100);
    try std.testing.expect(early > 2); // the header, the key and a value piece went out before the value closed
    try std.testing.expectEqualStrings(try whole(a, tools, text), try joined(a, out.items));
    try std.testing.expect(!s.open());
}

test "a GLM call's stream survives every byte boundary" {
    const texts = [_][]const u8{
        "<tool_call>\nlookup<arg_key>q</arg_key><arg_value>h\u{e9}llo < 2 \"fast\"</arg_value>" ++
            "<arg_key>n</arg_key><arg_value>3</arg_value></tool_call>",
        "<tool_call>lookup<arg_key>tags</arg_key><arg_value>[\"a\", \"b\"]</arg_value>" ++
            "<arg_key>obj</arg_key><arg_value>{\"k\": 1, \"j\": [1, 2]}</arg_value> key</arg_key><arg_value>3</arg_value></tool_call>",
        // a lone closer repairGlmKeys leaves unrepaired (no closers to spare): the stream must not make it a key
        "<tool_call>lookup<arg_key>a<arg_key>b</arg_key><arg_value>1</arg_value> w</arg_key><arg_value>2</arg_value></tool_call>",
    };
    for (texts) |text| for (0..text.len + 1) |boundary| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = try toolsOf(a);
        const expected = try whole(a, tools, text);
        const one = try streamed(a, tools, &.{text});
        const pieces = try streamed(a, tools, &.{ text[0..boundary], text });
        try std.testing.expectEqual(@as(usize, 1), headers(pieces));
        try std.testing.expectEqual(headers(one), headers(pieces));
        try std.testing.expectEqualStrings(expected, try joined(a, pieces));
    };
}

test "GLM cases stream the arguments the whole reply parses" {
    const cases = [_][]const u8{
        // two arguments: a string and an integer
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>fast</arg_value><arg_key>n</arg_key><arg_value>3</arg_value></tool_call>",
        // typed values keep the spaces they were written with
        "<tool_call>lookup<arg_key>tags</arg_key><arg_value>[\"a\", \"b\"]</arg_value>" ++
            "<arg_key>obj</arg_key><arg_value>{\"k\": 1, \"j\": [1, 2]}</arg_value></tool_call>",
        // newlines after the name and between every tag
        "<tool_call>\nlookup\n<arg_key>\nq\n</arg_key>\n<arg_value>\nfast\n</arg_value>\n</tool_call>",
        // junk between pairs, a literal less-than included
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>fast</arg_value> and 1 < 2\n<arg_key>n</arg_key><arg_value>3</arg_value></tool_call>",
        // a key the JSON needs to escape
        "<tool_call>lookup<arg_key>a\"b</arg_key><arg_value>x</arg_value></tool_call>",
        // a value outside ASCII
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>h\u{e9}llo \u{1f600}</arg_value></tool_call>",
        // a lone </arg_key> repairGlmKeys reads as the key
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>fast</arg_value> key</arg_key><arg_value>3</arg_value></tool_call>",
        // a whole call, then prose
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>a</arg_value></tool_call> then some prose",
        // a key whose value never came: the envelope's closer ends the call with no pair
        "<tool_call>lookup<arg_key>q</arg_key></tool_call>",
        // the same after a pair that did come
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>a</arg_value><arg_key>n</arg_key></tool_call>",
    };
    for (cases) |text| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = try toolsOf(a);
        const expected = try whole(a, tools, text);
        const one = try streamed(a, tools, &.{text});
        try std.testing.expectEqual(@as(usize, 1), headers(one));
        try std.testing.expectEqualStrings(expected, try joined(a, one));
        var boundary: usize = 0;
        while (boundary < text.len + 1) : (boundary += 7) {
            const pieces = try streamed(a, tools, &.{ text[0..boundary], text });
            try std.testing.expectEqualStrings(expected, try joined(a, pieces));
        }
    }
}

test "a repeated key streams both pairs, as XML does; the whole reply keeps the last" {
    const cases = [_][]const u8{
        // GLM: the same key twice
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>a</arg_value>" ++
            "<arg_key>q</arg_key><arg_value>b</arg_value></tool_call>",
        // XML: the same parameter twice, the gap both formats share
        "<tool_call>\n<function=lookup>\n<parameter=q>\na\n</parameter>\n" ++
            "<parameter=q>\nb\n</parameter>\n</function>\n</tool_call>",
    };
    for (cases) |text| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = try toolsOf(a);
        const deltas = try streamed(a, tools, &.{text});
        try std.testing.expectEqual(@as(usize, 1), headers(deltas));
        // the stream sends every pair; the whole reply's object keeps the last one
        try std.testing.expectEqualStrings("{\"q\":\"a\",\"q\":\"b\"}", try joined(a, deltas));
        try std.testing.expectEqualStrings("{\"q\":\"b\"}", try whole(a, tools, text));
    }
}

test "a GLM envelope the stream cannot stream sends nothing and leaves the finish its calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const cases = [_][]const u8{
        "<tool_call>launch<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>", // not offered
        "<tool_call>lookup</tool_call>", // no argument
        "<tool_call>call:lookup{q}</tool_call>", // Gemma's call
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>", // JSON
        "<tool_call>look<up<arg_key>q</arg_key>x<arg_value>y</arg_value></tool_call>", // a name with <
    };
    for (cases) |text| {
        var s = try Streamer.init(a, tools);
        var out: std.ArrayList(Value) = .empty;
        try s.feed(text, &out);
        try std.testing.expect(!s.streamed);
        try std.testing.expectEqual(@as(usize, 0), out.items.len);
    }
    // the finish then sends what the whole reply parses: nothing for an unoffered name, {} for no arguments
    try std.testing.expectEqualStrings("", try whole(a, tools, cases[0]));
    try std.testing.expectEqualStrings("{}", try whole(a, tools, cases[1]));
}

test "a GLM call after a DSML block streams nothing, as today, and the finish still gives both calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const text = "<\u{ff5c}DSML\u{ff5c}tool_calls><\u{ff5c}DSML\u{ff5c}invoke name=\"lookup\">" ++
        "<\u{ff5c}DSML\u{ff5c}parameter name=\"q\" string=\"true\">a</\u{ff5c}DSML\u{ff5c}parameter>" ++
        "</\u{ff5c}DSML\u{ff5c}invoke></\u{ff5c}DSML\u{ff5c}tool_calls>" ++
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>fast</arg_value></tool_call>";
    const deltas = try streamed(a, tools, &.{text});
    const expected = "{\"q\":\"a\"}{\"q\":\"fast\"}";
    try std.testing.expectEqual(@as(usize, 2), headers(deltas));
    try std.testing.expectEqualStrings(expected, try joined(a, deltas));
    try std.testing.expectEqualStrings(expected, try whole(a, tools, text));
}

test "a GLM call the end token left open completes as its closing markup completes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const open = "<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value>" ++
        "<arg_key>content</arg_key><arg_value>The history";
    const tail = try tool_parse.closeCall(a, open, tools);
    try std.testing.expectEqualStrings("</arg_value></tool_call>", tail);
    const closed = try std.mem.concat(a, u8, &.{ open, tail });
    const deltas = try streamed(a, tools, &.{ open, closed });
    try std.testing.expectEqual(@as(usize, 1), headers(deltas));
    try std.testing.expectEqualStrings("{\"path\":\"notes.txt\",\"content\":\"The history\"}", try joined(a, deltas));
    try std.testing.expectEqualStrings(try whole(a, tools, closed), try joined(a, deltas));
}

test "a cut GLM value replays from the stream as tool_glm_cut cuts it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a);
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value><arg_key>content</arg_key><arg_value>The history of", "{\"path\":\"notes.txt\",\"content\":\"The history of" },
        .{ "<tool_call>write_file<arg_key>limit</arg_key><arg_value>12", "{\"limit\":12" },
    };
    for (cases) |case| {
        var s = try Streamer.init(a, tools);
        var out: std.ArrayList(Value) = .empty;
        try s.finishCut(case[0], &out);
        try std.testing.expect(s.open());
        try std.testing.expectEqualStrings(case[1], try joined(a, out.items));
    }
}

/// ``reply``'s whole and streamed chat replies: calls, joined arguments, content and finish agree.
fn expectAgree(a: Allocator, reply: []const u8, max_tokens: usize) !void {
    const tokens = try TestText.encode(undefined, a, reply, false);
    const body = try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}", .{ chat_tools, max_tokens });
    for ([_]bool{ false, true }) |batched| {
        var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
        const whole_reply = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
        const stream_reply = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
        try std.testing.expectEqual(@as(?u16, 200), whole_reply.status);
        try std.testing.expect(stream_reply.opened and stream_reply.status == null);
        try std.testing.expectEqual(whole_reply.calls, stream_reply.calls);
        try std.testing.expectEqualStrings(whole_reply.arguments[0..whole_reply.arguments_len], stream_reply.arguments[0..stream_reply.arguments_len]);
        try std.testing.expectEqualStrings(whole_reply.content(), stream_reply.content());
        try std.testing.expectEqual(whole_reply.finish, stream_reply.finish);
        try std.testing.expectEqual(@as(i64, 0), stream_reply.preparing);
    }
}

test "whole and streamed chat replies agree on GLM calls and on the envelopes around them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // a GLM call with prose before it
    try expectAgree(a, try std.mem.concat(a, u8, &.{ "Writing it.\n", try glmCall(a, "write_file", "notes.txt") }), 2000);
    // two GLM calls
    try expectAgree(a, try std.mem.concat(a, u8, &.{ try glmCall(a, "lookup", "a"), "\n", try glmCall(a, "lookup", "b") }), 2000);
    // E1: an unoffered XML call between two the reply makes
    try expectAgree(a, try std.mem.concat(a, u8, &.{ try xmlCall(a, "lookup", "a"), try xmlCall(a, "other", "b"), try xmlCall(a, "lookup", "c") }), 2000);
    // E2: an XML call, a GLM call, an XML call
    try expectAgree(a, try std.mem.concat(a, u8, &.{ try xmlCall(a, "lookup", "a"), try glmCall(a, "lookup", "b"), try xmlCall(a, "lookup", "c") }), 2000);
    // D1/D2: typed XML values as the whole reply's compact JSON
    try expectAgree(a, "<tool_call>\n<function=lookup>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n" ++
        "<parameter=obj>\n{\"k\": 1, \"j\": [1, 2]}\n</parameter>\n</function>\n</tool_call>", 2000);
    // a GLM call max_tokens cuts inside a value
    const cut_reply = try std.mem.concat(a, u8, &.{ "Writing it.\n", try glmCall(a, "write_file", "The history of the bicycle") });
    try expectAgree(a, cut_reply, std.mem.indexOf(u8, cut_reply, " the bicycle").?);
    // a GLM call with no argument, and one whose key never got a value
    try expectAgree(a, "<tool_call>lookup</tool_call>", 2000);
    try expectAgree(a, "<tool_call>lookup<arg_key>q</arg_key></tool_call>", 2000);
}

test "a GLM value that runs into the next pair or never closes is text, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // #2801: a value that runs into the next pair, and a last value the envelope ends before its closer
    const replies = [_][]const u8{
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>a<arg_key>n</arg_key><arg_value>1</arg_value></tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>a</tool_call>",
    };
    for (replies) |reply| {
        const tokens = try TestText.encode(undefined, a, reply, false);
        const body = try std.fmt.allocPrint(a, "{s},\"max_tokens\":2000", .{chat_tools});
        for ([_]bool{ false, true }) |batched| {
            var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
            const whole_reply = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
            const stream_reply = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
            try std.testing.expectEqual(@as(?u16, 200), whole_reply.status);
            try std.testing.expect(stream_reply.opened and stream_reply.status == null);
            // not the deltas: a stream may send the call's first pieces before the slip is visible
            try std.testing.expectEqual(@as(usize, 0), whole_reply.calls);
            try std.testing.expectEqual(.stop, whole_reply.finish.?);
            const window = whole_reply.text.len; // Seen keeps this many of the reply's first bytes
            try std.testing.expectEqualStrings(reply[0..@min(window, reply.len)], whole_reply.content());
            try std.testing.expectEqual(.stop, stream_reply.finish.?);
            try std.testing.expectEqualStrings(whole_reply.content(), stream_reply.content());
        }
    }
}
