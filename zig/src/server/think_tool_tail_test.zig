//! Closed thinking blocks lift only complete trailing call envelopes with no later answer, whole and streamed.
const std = @import("std");
const json = @import("json");
const reply = @import("reply_text.zig");
const pre = @import("think_split_test.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

const Case = struct { call: []const u8, markers: reply.Markers = reply.think_markers, own_open: bool = true };
const formats = [_]Case{
    .{ .call = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"example\"}}</tool_call>" },
    .{ .call = "<tool_call>\n<function=lookup>\n<parameter=q>\nexample\n</parameter>\n</function>\n</tool_call>", .own_open = false },
    .{ .call = "<tool_call>lookup<arg_key>q</arg_key><arg_value>example</arg_value></tool_call>" },
    .{ .call = "<|tool_call>call:lookup{q:<|\"|>example<|\"|>}<tool_call|>", .markers = reply.channel_markers },
    .{ .call = "<|tool_call>:lookup{q:<|\"|>example<|\"|>}<tool_call|>", .markers = reply.channel_markers },
    .{ .call = dsml("tool_calls", "", "") },
    .{ .call = dsml("calls", " ", " ") },
    .{ .call = dsml("function_calls", "", " ") },
    .{ .call = "<|tool_call_start|>[lookup(q='example')]<|tool_call_end|>" },
    .{ .call = "<|tool_call_start|>[{\"name\":\"lookup\",\"arguments\":{\"q\":\"example\"}}]<|tool_call_end|>" },
    .{ .call = "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"lookup\">]<]minimax[>[<q>example]<]minimax[>[</q>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>", .markers = reply.mm_markers },
};

fn dsml(comptime block: []const u8, comptime sep: []const u8, comptime param_sep: []const u8) []const u8 {
    return "<｜DSML｜" ++ sep ++ block ++ "><｜DSML｜" ++ sep ++ "invoke name=\"lookup\">" ++
        "<｜DSML｜" ++ param_sep ++ "parameter name=\"q\" string=\"true\">example</｜DSML｜" ++ param_sep ++ "parameter>" ++
        "</｜DSML｜" ++ sep ++ "invoke></｜DSML｜" ++ sep ++ block ++ ">";
}

const request = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":4096," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}}]";

const Seen = struct {
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    names: [2]std.ArrayList(u8) = .{ .empty, .empty },
    args: [2]std.ArrayList(u8) = .{ .empty, .empty },
    calls: usize = 0,
    finish: []const u8 = "",
};

fn add(a: Allocator, out: *std.ArrayList(u8), value: ?Value) !void {
    if (value) |v| if (v == .string) try out.appendSlice(a, v.string);
}

fn observe(a: Allocator, got: *const pre.Got, streamed: bool) !Seen {
    var seen: Seen = .{};
    const chunks = if (streamed) got.events.items else &[_][]u8{got.body};
    if (!streamed) try std.testing.expectEqual(@as(u16, 200), got.status);
    for (chunks) |raw| {
        const doc = switch (try json.parse(a, raw)) {
            .ok => |value| value,
            .err => {
                std.debug.print("invalid reply JSON: {s}\n", .{raw});
                return error.InvalidReplyJson;
            },
        };
        const choices = doc.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        const choice = choices.array[0];
        const m = choice.get(if (streamed) "delta" else "message").?;
        try add(a, &seen.content, m.get("content"));
        try add(a, &seen.reasoning, m.get("reasoning_content"));
        if (choice.strField("finish_reason")) |f| seen.finish = f;
        if (m.get("tool_calls")) |calls| for (calls.array, 0..) |call, position| {
            const index: usize = if (streamed) @intCast(call.get("index").?.int64().?) else position;
            try std.testing.expect(index < 2);
            seen.calls = @max(seen.calls, index + 1);
            const function = call.get("function").?;
            try add(a, &seen.names[index], function.get("name"));
            try add(a, &seen.args[index], function.get("arguments"));
        };
    }
    return seen;
}

fn verify(seen: Seen, count: usize, reasoning: []const u8, answer: []const u8) !void {
    try std.testing.expectEqual(count, seen.calls);
    try std.testing.expectEqualStrings(reasoning, seen.reasoning.items);
    try std.testing.expectEqualStrings(answer, seen.content.items);
    try std.testing.expectEqualStrings(if (count > 0) "tool_calls" else "stop", seen.finish);
    for (0..count) |i| {
        try std.testing.expectEqualStrings("lookup", seen.names[i].items);
        try std.testing.expectEqualStrings("{\"q\":\"example\"}", seen.args[i].items);
    }
}

fn check(a: Allocator, c: Case, body: []const u8, after: []const u8, count: usize, reasoning: []const u8, answer: []const u8) !void {
    const open = if (!c.own_open) "" else if (c.markers.open.len > 0) c.markers.open else if (std.mem.eql(u8, c.markers.close, reply.mm_markers.close)) "<mm:think>" else "<think>";
    const before = try std.mem.concat(a, u8, &.{ open, "\n", body });
    for ([_]bool{ true, false }) |single_close| {
        var tokens: std.ArrayList(u32) = .empty;
        for (before) |byte| try tokens.append(a, byte);
        if (single_close) try tokens.append(a, pre.think_end) else for (c.markers.close) |byte| try tokens.append(a, byte);
        for (after) |byte| try tokens.append(a, byte);
        try tokens.append(a, pre.eos);
        const settings: pre.Text = .{ .markers = c.markers, .close_token = single_close };
        var whole = try pre.sendText(tokens.items, true, request ++ "}", settings, null);
        defer whole.deinit();
        try verify(try observe(a, &whole, false), count, reasoning, answer);
        for ([_]?usize{ 1, 2, 3, null, 4096 }) |stride| {
            var stream = try pre.sendText(tokens.items, true, request ++ ",\"stream\":true}", settings, stride);
            defer stream.deinit();
            try verify(try observe(a, &stream, true), count, reasoning, answer);
        }
    }
}

fn exercise(index: usize) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = formats[index];
    const one = try std.mem.concat(a, u8, &.{ "Plan.\n", c.call });
    try check(a, c, one, "", 1, "Plan.", "");
    try check(a, c, one, "\n \t\u{3000}", 1, "Plan.", "");
    const two = try std.mem.concat(a, u8, &.{ "Plan.\n", c.call, " \n", c.call, "\n\t" });
    try check(a, c, two, "", 2, "Plan.", "");
}

fn kept(index: usize) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = formats[index];
    const one = try std.mem.concat(a, u8, &.{ "Plan.\n", c.call });
    const tools = (try json.parse(a, request ++ "}")).ok.get("tools").?.array;
    const before = try @import("tool_minimax.zig").dsml(a, one, tools);
    try check(a, c, one, "\n\nDone.", 0, reply.pyStrip(before), "Done.");
    try check(a, c, one, "\u{200b}", 0, reply.pyStrip(before), "\u{200b}");
    const middle = try std.mem.concat(a, u8, &.{ one, "\nMore reasoning." });
    const expected = try @import("tool_minimax.zig").dsml(a, middle, tools);
    try check(a, c, middle, "", 0, expected, "");
}

test "thinking tail: Qwen JSON calls before the close" {
    try exercise(0);
}
test "thinking tail: reporter Qwen XML calls before the close" {
    try exercise(1);
}
test "thinking tail: GLM XML calls before the close" {
    try exercise(2);
}
test "thinking tail: Gemma call prefix before the channel close" {
    try exercise(3);
}
test "thinking tail: Gemma colon prefix before the channel close" {
    try exercise(4);
}
test "thinking tail: DSML tool_calls before the close" {
    try exercise(5);
}
test "thinking tail: spaced DSML calls before the close" {
    try exercise(6);
}
test "thinking tail: DSML function_calls before the close" {
    try exercise(7);
}
test "thinking tail: LFM2 Pythonic calls before the close" {
    try exercise(8);
}
test "thinking tail: LFM2 JSON calls before the close" {
    try exercise(9);
}
test "thinking tail: MiniMax calls before the close" {
    try exercise(10);
}
test "thinking tail: later answer or reasoning keeps every format in reasoning" {
    for (0..formats.len) |index| try kept(index);
}

test "thinking tail: the reported call-only turn has no reasoning or stray content" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try check(arena.allocator(), formats[1], formats[1].call, "", 1, "", "");
}

test "thinking tail: an earlier mentioned call stays reasoning before a final real call" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, request ++ "}")).ok.get("tools").?.array;
    for (formats) |c| {
        const prefix = try std.mem.concat(a, u8, &.{ "Example: ", c.call, "\nStill reasoning.\n" });
        const reasoning = try @import("tool_minimax.zig").dsml(a, prefix, tools);
        try check(a, c, try std.mem.concat(a, u8, &.{ prefix, c.call }), "", 1, reply.pyStrip(reasoning), "");
    }
}

test "thinking tail: multiple invokes inside one DSML, LFM2 or MiniMax envelope" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 5, 6, 7, 8, 9, 10 }) |index| {
        var c = formats[index];
        if (@import("tool_dsml.zig").whole(c.call)) |block| {
            c.call = try @import("tool_dsml_stream_test.zig").block(a, block.spelling, &.{ block.inner, block.inner });
        } else if (index == 10) {
            const mm = @import("tool_minimax.zig");
            const inner = c.call[mm.open.len .. c.call.len - mm.close.len];
            c.call = try std.mem.concat(a, u8, &.{ mm.open, inner, inner, mm.close });
        } else {
            const open = reply.calls[3][0];
            const close = reply.calls[3][1];
            const inner = c.call[open.len + 1 .. c.call.len - close.len - 1];
            c.call = try std.mem.concat(a, u8, &.{ open, "[", inner, ",", inner, "]", close });
        }
        try check(a, c, try std.mem.concat(a, u8, &.{ "Plan.\n", c.call }), "", 2, "Plan.", "");
    }
}

test "thinking tail: partial call envelopes and prose payloads stay reasoning" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (formats) |c| {
        const open = if (c.markers.open.len > 0) c.markers.open else "";
        const body = try std.mem.concat(a, u8, &.{ "Plan.", c.call[0 .. c.call.len - 3] });
        const full = try std.mem.concat(a, u8, &.{ open, body, c.markers.close });
        const split = try reply.splitThinking(a, full, true, c.markers);
        try std.testing.expectEqualStrings(body, split.reasoning);
        try std.testing.expectEqualStrings("", split.answer);
    }
    const prose = "Plan. <|tool_call> is just an example <tool_call|>";
    const split = try reply.splitThinking(a, prose ++ "</think>", true, reply.think_markers);
    try std.testing.expectEqualStrings(prose, split.reasoning);
    try std.testing.expectEqualStrings("", split.answer);
}
