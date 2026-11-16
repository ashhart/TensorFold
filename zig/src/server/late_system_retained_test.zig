const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const api = @import("engine_api");
const fixtures = @import("late_system_test.zig");
const DropLate = fixtures.DropLate;
const Capture = fixtures.Capture;
const exchange = fixtures.exchange;
const textOf = fixtures.textOf;
const prose_chat = fixtures.prose_chat;

test "a stream ends with a call to a tool the request did not offer as text, as the whole reply keeps it" {
    var text: DropLate = .{};
    // prose, then a call to a tool the request does not offer: the parser leaves the block as text
    var backend: Capture = .{ .reply = "Checking the vault.\n<tool_call>\n<function=vault_list>\n<parameter=path>\nimported/\n</parameter>\n</function>\n</tool_call>" };
    defer std.testing.allocator.free(backend.prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = try exchange(srv, "/v1/chat/completions", prose_chat ++ "}");
    defer std.testing.allocator.free(whole.body);
    const message = (try json.parse(a, whole.body)).ok.get("choices").?.array[0].get("message").?;
    try std.testing.expect(message.get("tool_calls") == null);
    try std.testing.expectEqualStrings(backend.reply, message.get("content").?.string);
    const got = try exchange(srv, "/v1/chat/completions", prose_chat ++ ",\"stream\":true}");
    defer std.testing.allocator.free(got.body);
    var prose: std.ArrayList(u8) = .empty;
    var finish: []const u8 = "";
    var lines = std.mem.splitScalar(u8, got.body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = std.mem.trimEnd(u8, line["data: ".len..], "\r");
        if (std.mem.eql(u8, data, "[DONE]")) continue;
        const choices = (try json.parse(a, data)).ok.get("choices").?.array;
        if (choices.len == 0) continue;
        const delta = choices[0].get("delta").?;
        try std.testing.expect(delta.get("tool_calls") == null);
        if (delta.get("content")) |content| try prose.appendSlice(a, content.string);
        if (choices[0].get("finish_reason")) |f| if (f == .string) {
            finish = f.string;
        };
    }
    try std.testing.expectEqualStrings("stop", finish);
    // the streamed prose joins to the whole reply's content: the block is not silently lost
    try std.testing.expectEqualStrings(message.get("content").?.string, std.mem.trim(u8, prose.items, " \n"));
}

const lookup_reply = "<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>";
const vault_reply = "<tool_call>{\"name\":\"vault_list\",\"arguments\":{}}</tool_call>";

fn expectStreamMatchesWhole(reply: []const u8) !void {
    try expectStreamMatchesWholeCalls(reply, null);
}

fn expectStreamMatchesWholeCalls(reply: []const u8, expected_call_count: ?usize) !void {
    try expectStreamMatchesWholeOptions(reply, expected_call_count, false, null);
}

fn expectStreamMatchesWholeOptions(reply: []const u8, expected_call_count: ?usize, thinking: bool, expected_content: ?[]const u8) !void {
    var text: DropLate = .{};
    var backend: Capture = .{ .reply = reply };
    defer std.testing.allocator.free(backend.prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = thinking,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = try exchange(srv, "/v1/chat/completions", prose_chat ++ "}");
    defer std.testing.allocator.free(whole.body);
    try std.testing.expectEqual(@as(u16, 200), whole.status);
    const message = (try json.parse(a, whole.body)).ok.get("choices").?.array[0].get("message").?;
    const expected_calls = if (message.get("tool_calls")) |calls| calls.array else &.{};
    if (expected_call_count) |count| try std.testing.expectEqual(count, expected_calls.len);
    if (expected_content) |content| try std.testing.expectEqualStrings(content, textOf(message, "content"));
    const got = try exchange(srv, "/v1/chat/completions", prose_chat ++ ",\"stream\":true}");
    defer std.testing.allocator.free(got.body);
    try std.testing.expectEqual(@as(u16, 200), got.status);
    var prose: std.ArrayList(u8) = .empty;
    var arguments: std.ArrayList(u8) = .empty;
    var names: std.ArrayList(u8) = .empty;
    var calls: usize = 0;
    var role_seen = false;
    var reasoning: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, got.body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = std.mem.trimEnd(u8, line["data: ".len..], "\r");
        if (std.mem.eql(u8, data, "[DONE]")) continue;
        const choices = (try json.parse(a, data)).ok.get("choices").?.array;
        if (choices.len == 0) continue;
        const delta = choices[0].get("delta").?;
        if (delta.get("content")) |content| {
            if (expected_content != null and content.string.len > 0) try std.testing.expect(role_seen);
            try prose.appendSlice(a, content.string);
        }
        if (delta.get("role")) |role| {
            try std.testing.expectEqualStrings("assistant", role.string);
            role_seen = true;
        }
        if (delta.get("reasoning_content")) |content| try reasoning.appendSlice(a, content.string);
        if (delta.get("tool_calls")) |tc| for (tc.array) |call| {
            try std.testing.expectEqual(@as(i64, 0), call.get("index").?.int64().?);
            if (call.get("id")) |id| {
                try std.testing.expect(id.string.len > 0);
                try std.testing.expectEqualStrings("function", call.get("type").?.string);
                calls += 1;
            }
            const function = call.get("function").?;
            if (function.get("name")) |name| try names.appendSlice(a, name.string);
            if (function.get("arguments")) |args| try arguments.appendSlice(a, args.string);
        };
    }
    try std.testing.expectEqual(expected_calls.len, calls);
    try std.testing.expectEqualStrings(if (expected_calls.len > 0) expected_calls[0].get("function").?.get("name").?.string else "", names.items);
    try std.testing.expectEqualStrings(if (expected_calls.len > 0) expected_calls[0].get("function").?.get("arguments").?.string else "", arguments.items);
    try std.testing.expectEqualStrings(textOf(message, "content"), prose.items);
    if (thinking) try std.testing.expectEqualStrings("Plan.", reasoning.items);
}

test "stream replay retains the unoffered envelope after an offered call" {
    try expectStreamMatchesWhole("Checking." ++ lookup_reply ++ vault_reply);
}

test "stream retained envelope preserves the leading newline" {
    try expectStreamMatchesWhole("\nChecking." ++ vault_reply);
}

test "stream retained envelope precedes later prose with the leading newline" {
    try expectStreamMatchesWhole("\nChecking." ++ vault_reply ++ "Tail.");
}

test "stream replays an offered call once before retained envelopes and later prose" {
    try expectStreamMatchesWhole("Checking." ++ lookup_reply ++ vault_reply ++ "Tail.");
    try expectStreamMatchesWhole("Checking." ++ lookup_reply ++ "<tool_call>{invalid}</tool_call>Tail.");
}

test "stream replay with leading whitespace retains malformed envelope and later prose" {
    try expectStreamMatchesWholeCalls("\nChecking." ++ lookup_reply ++ "<tool_call>{invalid}</tool_call>Tail.", 1);
}

test "stream retains a malformed closed envelope with whitespace and later prose" {
    try expectStreamMatchesWhole(" \n\t<tool_call>{invalid}</tool_call>Tail.");
}

test "stream without accepted calls preserves a space after streamed reasoning" {
    try expectStreamMatchesWholeOptions("Plan.</think> ", 0, true, " ");
}

test "stream retains an unoffered closed envelope promoted from unclosed reasoning" {
    const content = "<tool_call>{\"name\": \"other\", \"arguments\": {}}</tool_call> done";
    try expectStreamMatchesWholeOptions("Plan." ++ content, 0, true, content);
}

test "stream without accepted calls sends the role before deferred leading-whitespace prose" {
    try expectStreamMatchesWholeOptions("\nPlain prose", 0, false, "\nPlain prose");
}

test "stream content and tool calls occur exactly once across all chunks" {
    try expectStreamMatchesWhole(lookup_reply);
    try expectStreamMatchesWhole("Checking." ++ lookup_reply ++ vault_reply);
    try expectStreamMatchesWhole("\nChecking." ++ vault_reply);
}

const malformed_reply = "<tool_call>{\"arguments\":{}}</tool_call>";

/// Runs one reply as a whole reply and a stream, then checks the kept-call block on both.
fn expectKept(reply: []const u8, unoffered: usize, malformed: usize, names: []const []const u8) !void {
    return expectKeptWith(prose_chat ++ "}", reply, unoffered, malformed, names);
}

fn expectKeptWith(body: []const u8, reply: []const u8, unoffered: usize, malformed: usize, names: []const []const u8) !void {
    var text: DropLate = .{};
    var backend: Capture = .{ .reply = reply };
    defer std.testing.allocator.free(backend.prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = try exchange(srv, "/v1/chat/completions", body);
    defer std.testing.allocator.free(whole.body);
    try expectCounts((try json.parse(a, whole.body)).ok.get("tensorfold").?.get("kept_tool_calls").?, unoffered, malformed, names);
    const streamed = try exchange(srv, "/v1/chat/completions", try std.mem.concat(a, u8, &.{ body[0 .. body.len - 1], ",\"stream\":true}" }));
    defer std.testing.allocator.free(streamed.body);
    var found: ?json.Value = null;
    var finishes: usize = 0;
    var lines = std.mem.splitScalar(u8, streamed.body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = std.mem.trimEnd(u8, line["data: ".len..], "\r");
        if (std.mem.eql(u8, data, "[DONE]")) continue;
        const chunk = (try json.parse(a, data)).ok;
        const choices = chunk.get("choices").?.array;
        if (choices.len == 0) continue;
        if (choices[0].get("finish_reason").? != .null) finishes += 1;
        if (chunk.get("tensorfold")) |tf| if (tf.get("kept_tool_calls")) |kept| {
            try std.testing.expect(found == null);
            try std.testing.expect(choices[0].get("finish_reason").? != .null);
            found = kept;
        };
    }
    try std.testing.expectEqual(@as(usize, 1), finishes);
    try expectCounts(found orelse return error.NoKeptToolCalls, unoffered, malformed, names);
}

fn expectCounts(kept: json.Value, unoffered: usize, malformed: usize, names: []const []const u8) !void {
    try std.testing.expectEqual(@as(i64, @intCast(unoffered)), kept.get("unoffered").?.int64());
    try std.testing.expectEqual(@as(i64, @intCast(malformed)), kept.get("malformed").?.int64());
    if (names.len == 0) {
        try std.testing.expect(kept.get("unoffered_names") == null);
        return;
    }
    const got = kept.get("unoffered_names").?.array;
    try std.testing.expectEqual(names.len, got.len);
    for (names, got) |want, v| try std.testing.expectEqualStrings(want, v.string);
}

test "tool visibility HTTP: kept calls split into unoffered and malformed on whole replies and streams" {
    try expectKept(vault_reply, 1, 0, &.{"vault_list"});
    try expectKept(malformed_reply, 0, 1, &.{});
    try expectKept(vault_reply ++ "\n" ++ malformed_reply, 1, 1, &.{"vault_list"});
}

test "tool visibility HTTP: a clean reply has no kept-call fields" {
    var text: DropLate = .{};
    var backend: Capture = .{ .reply = "hi" };
    defer std.testing.allocator.free(backend.prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = try exchange(srv, "/v1/chat/completions", prose_chat ++ "}");
    defer std.testing.allocator.free(whole.body);
    try std.testing.expect((try json.parse(a, whole.body)).ok.get("tensorfold").?.get("kept_tool_calls") == null);
    const streamed = try exchange(srv, "/v1/chat/completions", prose_chat ++ ",\"stream\":true}");
    defer std.testing.allocator.free(streamed.body);
    var lines = std.mem.splitScalar(u8, streamed.body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = std.mem.trimEnd(u8, line["data: ".len..], "\r");
        if (std.mem.eql(u8, data, "[DONE]")) continue;
        const chunk = (try json.parse(a, data)).ok;
        if (chunk.get("tensorfold")) |tf| try std.testing.expect(tf.get("kept_tool_calls") == null);
    }
}

test "tool visibility HTTP: kept-call names cap at eight and 64 bytes" {
    var long_name: [108]u8 = undefined;
    @memcpy(long_name[0..8], "unknown_");
    @memset(long_name[8..], 'a');
    var cut: [64]u8 = undefined;
    @memcpy(cut[0..8], "unknown_");
    @memset(cut[8..], 'a');
    var reply: std.ArrayList(u8) = .empty;
    var line: [256]u8 = undefined;
    try reply.appendSlice(std.testing.allocator, try std.fmt.bufPrint(&line, "<tool_call>{{\"name\":\"{s}\",\"arguments\":{{}}}}</tool_call>\n", .{long_name[0..]}));
    for (0..9) |i| try reply.appendSlice(std.testing.allocator, try std.fmt.bufPrint(&line, "<tool_call>{{\"name\":\"tool_{d}\",\"arguments\":{{}}}}</tool_call>\n", .{i}));
    defer reply.deinit(std.testing.allocator);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const roomy = try std.mem.replaceOwned(u8, a, prose_chat, "\"max_tokens\":256", "\"max_tokens\":2048");
    var expected: [8][]const u8 = .{ cut[0..], "tool_0", "tool_1", "tool_2", "tool_3", "tool_4", "tool_5", "tool_6" };
    try expectKeptWith(try std.mem.concat(a, u8, &.{ roomy, "}" }), reply.items, 10, 0, &expected);
}

test "tool visibility HTTP: metrics count kept unoffered and malformed calls" {
    var text: DropLate = .{};
    var backend: Capture = .{ .reply = vault_reply };
    defer std.testing.allocator.free(backend.prompt);
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const first = try exchange(srv, "/v1/chat/completions", prose_chat ++ "}");
    defer std.testing.allocator.free(first.body);
    backend.reply = malformed_reply;
    const second = try exchange(srv, "/v1/chat/completions", prose_chat ++ "}");
    defer std.testing.allocator.free(second.body);
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try srv.metrics.render(std.testing.io, &out.writer, .{}, &.{}, .{}, null, 0);
    try std.testing.expectEqual(@as(?u64, 1), metricValue(out.written(), "kept_tool_calls_unoffered_total"));
    try std.testing.expectEqual(@as(?u64, 1), metricValue(out.written(), "kept_tool_calls_malformed_total"));
}

fn metricValue(body: []const u8, comptime name: []const u8) ?u64 {
    const full = "tensorfold:" ++ name;
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, full ++ " ")) continue;
        return std.fmt.parseUnsigned(u64, line[full.len + 1 ..], 10) catch null;
    }
    return null;
}
