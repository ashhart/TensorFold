const std = @import("std");
const server = @import("root.zig");
const json = server.json;
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
