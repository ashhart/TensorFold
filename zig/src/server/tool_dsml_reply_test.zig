//! DSML calls through the chat route, whole and streamed: same calls, arguments, content and finish (#1396's bar).
const std = @import("std");
const json = @import("json");
const pre = @import("stream_preflight_test.zig");
const t = @import("tool_dsml_stream_test.zig");
const tool_dsml = @import("tool_dsml.zig");
const Allocator = std.mem.Allocator;
const spellings = &tool_dsml.spellings;

/// A chat request offering ``tools`` (a JSON array), its body left open for the fields a case adds.
fn chatBody(a: Allocator, tools: []const u8, extra: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":", tools, extra });
}

/// What a client saw of one reply: its calls, their joined arguments, its content (64 bytes at most) and its finish.
const Seen = struct { calls: usize, args: []const u8, content: []const u8, finish: []const u8 };

fn seenOf(a: Allocator, seen: anytype) !Seen {
    const finish = if (seen.finish) |f| @tagName(f) else "";
    return .{ .calls = seen.calls, .args = try a.dupe(u8, seen.arguments[0..seen.arguments_len]), .content = try a.dupe(u8, seen.content()), .finish = finish };
}

/// ``body`` against ``reply`` whole and streamed: what a client sees of each.
fn both(a: Allocator, reply: []const u8, body: []const u8, batched: bool) ![2]Seen {
    var backend: pre.CallEngine = .{ .tokens = try pre.TestText.encode(undefined, a, reply, false), .batched = batched };
    const whole = try pre.sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
    const streamed = try pre.sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
    try std.testing.expectEqual(@as(?u16, 200), whole.status);
    try std.testing.expect(streamed.opened and streamed.status == null);
    try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
    return .{ try seenOf(a, whole), try seenOf(a, streamed) };
}

/// ``reply``'s whole and streamed chat replies, one byte a token and batched, agree; what the whole one saw.
fn agree(a: Allocator, reply: []const u8, body: []const u8) !Seen {
    var seen: [2]Seen = undefined;
    for ([_]bool{ false, true }) |batched| {
        seen = try both(a, reply, body, batched);
        try std.testing.expectEqual(seen[0].calls, seen[1].calls);
        try std.testing.expectEqualStrings(seen[0].args, seen[1].args);
        try std.testing.expectEqualStrings(seen[0].content, seen[1].content);
        try std.testing.expectEqualStrings(seen[0].finish, seen[1].finish);
    }
    return seen[0];
}

test "DSML calls agree whole and streamed in every spelling, with prose, typed values, two invokes and one call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (spellings) |*sp| for ([_][]const u8{ ",\"max_tokens\":2000", ",\"max_tokens\":2000,\"parallel_tool_calls\":false" }) |extra| {
        const body = try chatBody(a, t.tools_text, extra);
        const lookup = try t.invoke(a, sp, "lookup", &.{ .{ "q", true, "h\u{e9}llo \"x\"" }, .{ "tags", false, "[\"a\", 2]" } });
        const write = try t.invoke(a, sp, "write_file", &.{ .{ "path", true, "notes.txt" }, .{ "content", true, "one\ntwo" } });
        const replies = [_]struct { []const u8, usize, []const u8 }{
            .{ try std.mem.concat(a, u8, &.{ "I'll look.\n\n", try t.block(a, sp, &.{lookup}) }), 1, "{\"q\":\"h\u{e9}llo \\\"x\\\"\",\"tags\":[\"a\",2]}" },
            .{ try t.block(a, sp, &.{ lookup, write }), 2, "{\"q\":\"h\u{e9}llo \\\"x\\\"\",\"tags\":[\"a\",2]}{\"path\":\"notes.txt\",\"content\":\"one\\ntwo\"}" },
            .{ try t.block(a, sp, &.{try t.invoke(a, sp, "lookup", &.{})}), 1, "{}" },
        };
        const single = std.mem.indexOf(u8, extra, "parallel") != null;
        for (replies) |r| {
            const seen = try agree(a, r[0], body);
            try std.testing.expectEqual(if (single) 1 else r[1], seen.calls);
            if (!single) try std.testing.expectEqualStrings(r[2], seen.args);
            try std.testing.expectEqualStrings("tool_calls", seen.finish);
        }
    };
}

test "a DSML block that breaks before its first header fails as it did: no call, its text kept, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = try chatBody(a, t.tools_text, ",\"max_tokens\":2000");
    for (spellings) |*sp| for (try t.earlyBreaks(a, sp)) |text| {
        const seen = try agree(a, try std.mem.concat(a, u8, &.{ "Checking.\n", text }), body);
        try std.testing.expectEqual(@as(usize, 0), seen.calls);
        try std.testing.expectEqualStrings("", seen.args);
        try std.testing.expectEqualStrings("stop", seen.finish);
        try std.testing.expect(std.mem.startsWith(u8, seen.content, "Checking.\n"));
        try std.testing.expect(seen.content.len > "Checking.\n".len); // the broken block stays the reply's text
    };
}

test "a DSML call max_tokens cuts after its header returns cut whole and streamed; before it, no call as before" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (spellings) |*sp| {
        const reply = try std.mem.concat(a, u8, &.{ "Writing it.\n", try t.block(a, sp, &.{try t.invoke(a, sp, "write_file", &.{ .{ "path", true, "notes.txt" }, .{ "content", true, "The history of the bicycle" } })}) });
        const cases = [_]struct { usize, usize, []const u8 }{
            .{ std.mem.indexOf(u8, reply, " the bicycle").?, 1, "{\"path\":\"notes.txt\",\"content\":\"The history of" },
            .{ std.mem.indexOf(u8, reply, "tes.txt").?, 0, "" },
        };
        for (cases) |case| {
            const body = try chatBody(a, t.tools_text, try std.fmt.allocPrint(a, ",\"max_tokens\":{d}", .{case[0]}));
            const seen = try agree(a, reply, body);
            try std.testing.expectEqual(case[1], seen.calls);
            try std.testing.expectEqualStrings(case[2], seen.args);
            try std.testing.expectEqualStrings("length", seen.finish);
            // the cut call is not text; before its header the open block stays the reply's text, as it did
            if (case[1] == 1) try std.testing.expectEqualStrings("Writing it.", seen.content) else try std.testing.expect(std.mem.startsWith(u8, seen.content, "Writing it.\n<"));
        }
    }
}

/// The grammar a strict tool gets writes ``function_calls``: unspaced invokes, spaced parameters.
const strict_tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"strict\":true,\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"q\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"}},\"required\":[\"q\",\"n\"],\"additionalProperties\":false}}}," ++
    "{\"type\":\"function\",\"function\":{\"name\":\"now\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";

test "a strict tool's grammar-shaped call agrees whole and streamed, alone, beside another and as one call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sp = &spellings[2];
    const lookup = try t.invoke(a, sp, "lookup", &.{ .{ "q", true, "hello" }, .{ "n", false, "3" } });
    const now = try t.invoke(a, sp, "now", &.{});
    for ([_][]const u8{ ",\"max_tokens\":2000", ",\"max_tokens\":2000,\"parallel_tool_calls\":false" }) |extra| {
        const single = std.mem.indexOf(u8, extra, "parallel") != null;
        for ([_]struct { []const []const u8, []const u8 }{
            .{ &.{lookup}, "{\"q\":\"hello\",\"n\":3}" },
            .{ &.{ lookup, now }, "{\"q\":\"hello\",\"n\":3}{}" },
        }) |case| {
            const seen = try agree(a, try std.mem.concat(a, u8, &.{ "\n", try t.block(a, sp, case[0]), "\n" }), try chatBody(a, strict_tools, extra));
            try std.testing.expectEqual(if (single) 1 else case[0].len, seen.calls);
            try std.testing.expectEqualStrings(if (single) "{\"q\":\"hello\",\"n\":3}" else case[1], seen.args);
            try std.testing.expectEqualStrings("", seen.content);
        }
    }
}

test "a DSML block that breaks after its header: the whole reply keeps it as text, the stream has sent the call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sp = &spellings[1];
    const reply = try std.mem.concat(a, u8, &.{ sp.open, sp.invoke_open, "lookup\">", sp.param_open, "q\" string=\"true\">a", sp.param_close, sp.param_open, "n\" string=\"yes\">3", sp.param_close, sp.invoke_close, sp.close });
    const seen = try both(a, reply, try chatBody(a, t.tools_text, ",\"max_tokens\":2000"), false);
    try std.testing.expectEqual(@as(usize, 0), seen[0].calls);
    try std.testing.expectEqualStrings("stop", seen[0].finish);
    try std.testing.expectEqualStrings(reply[0..seen[0].content.len], seen[0].content);
    try std.testing.expectEqual(@as(usize, 1), seen[1].calls);
    try std.testing.expectEqualStrings("{\"q\":\"a\"", seen[1].args);
    try std.testing.expectEqualStrings("stop", seen[1].finish);
    try std.testing.expectEqualStrings(seen[0].content, seen[1].content);
}
