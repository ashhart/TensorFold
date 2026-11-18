//! ``echo`` on /v1/completions: the prompt's text and rows ahead of an unchanged reply, prompt-only replies, refusals.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const openai = @import("openai.zig");
const fixture = @import("logprobs_test.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const T = std.testing;

/// The route's reply to `body` over `backend`'s rows; the bytes-as-tokens text, so "abc" is the prompt 97, 98, 99.
fn sendTo(backend: *fixture.Script, chat: bool, body: []const u8) !fixture.Got {
    var text: fixture.Text = .{};
    var srv = try server.Server.init(T.allocator, T.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: fixture.Got = .{};
    openai.run(srv, a, got.out(), .{ .conn = &conn }, chat, (try json.parse(a, body)).ok);
    return got;
}

/// The whole 200 reply to a completion body, as values in `a`.
fn whole(a: Allocator, reply: []const u32, body: []const u8) !json.Value {
    var backend: fixture.Script = .{ .reply = reply };
    var got = try sendTo(&backend, false, body);
    defer got.deinit();
    try T.expectEqual(@as(u16, 200), got.status);
    return (try json.parse(a, try a.dupe(u8, got.body))).ok;
}

fn choiceOf(v: json.Value) json.Value {
    return v.get("choices").?.array[0];
}

fn compact(a: Allocator, v: json.Value) ![]const u8 {
    return json.stringify(a, v, .{ .compact = true });
}

fn expectList(a: Allocator, choice: json.Value, key: []const u8, want: []const u8) !void {
    try T.expectEqualStrings(want, try compact(a, choice.get("logprobs").?.get(key).?));
}

test "completions echo: the prompt's tokens and rows come first, then the reply's entries as without echo" {
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = [_]u32{ 'H', 'i' };
    const head = "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":2,\"logprobs\":1";
    const plain = choiceOf(try whole(a, &reply, head ++ "}"));
    const echoed_reply = try whole(a, &reply, head ++ ",\"echo\":true}");
    const echoed = choiceOf(echoed_reply);
    try T.expectEqualStrings("Hi", plain.get("text").?.string);
    try T.expectEqualStrings("abcHi", echoed.get("text").?.string);
    // the first prompt token has no row; "b" is not its row's best token, so the best ('z') is its one alternative
    try expectList(a, echoed, "tokens", "[\"a\",\"b\",\"c\",\"H\",\"i\"]");
    try expectList(a, echoed, "token_logprobs", "[null,-0.125,-0.25,-0.25,-0.5]");
    try expectList(a, echoed, "top_logprobs", "[null,{\"z\":-0.0},{\"c\":-0.25},{\"H\":-0.25},{\"i\":-0.5}]");
    try expectList(a, echoed, "text_offset", "[0,1,2,3,4]");
    // the reply's own entries are the ones the request gets without echo; only their offsets move past the prompt
    for ([_][]const u8{ "tokens", "token_logprobs", "top_logprobs" }) |key| {
        const tail: json.Value = .{ .array = echoed.get("logprobs").?.get(key).?.array[3..] };
        try T.expectEqualStrings(try compact(a, plain.get("logprobs").?.get(key).?), try compact(a, tail));
    }
    try expectList(a, plain, "text_offset", "[0,1]");
    try T.expect(echoed.get("prompt_logprobs") == null);
    try T.expectEqualStrings("{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":0}}", try compact(a, echoed_reply.get("usage").?));
    // a prompt of token ids echoes their text; logprobs 0 gives each row no alternatives
    const ids = choiceOf(try whole(a, &reply, "{\"model\":\"test-model\",\"prompt\":[97,98,99],\"max_tokens\":2,\"logprobs\":0,\"echo\":true}"));
    try T.expectEqualStrings("abcHi", ids.get("text").?.string);
    try expectList(a, ids, "top_logprobs", "[null,{},{},{},{}]");
    // without logprobs, echo is the prompt's text alone
    const bare = choiceOf(try whole(a, &reply, "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":2,\"echo\":true}"));
    try T.expectEqualStrings("abcHi", bare.get("text").?.string);
    try T.expect(bare.get("logprobs").? == .null);
}

test "completions echo: max_tokens 0 returns the prompt alone, and without echo keeps meaning the default length" {
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = [_]u32{ 'H', 'i' }; // the engine sends its whole reply whatever it is asked: the server keeps none
    for ([_][]const u8{ "\"max_tokens\":0", "\"max_completion_tokens\":0", "\"max_tokens\":0,\"max_completion_tokens\":null" }) |limit| {
        const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"prompt\":\"abc\",\"logprobs\":2,\"echo\":true,", limit, "}" });
        const got = try whole(a, &reply, body);
        const c = choiceOf(got);
        try T.expectEqualStrings("abc", c.get("text").?.string);
        try T.expectEqualStrings("length", c.get("finish_reason").?.string);
        try expectList(a, c, "tokens", "[\"a\",\"b\",\"c\"]");
        try expectList(a, c, "token_logprobs", "[null,-0.125,-0.25]");
        try expectList(a, c, "text_offset", "[0,1,2]");
        try T.expectEqualStrings("0", got.get("usage").?.get("completion_tokens").?.int);
    }
    const bare = choiceOf(try whole(a, &reply, "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":0,\"echo\":true}"));
    try T.expectEqualStrings("abc", bare.get("text").?.string);
    try T.expect(bare.get("logprobs").? == .null);
    // the engine is asked for one token, the fewest a submitted reply takes, not for the server's default length
    var backend: fixture.Script = .{ .reply = &reply };
    var asked = try sendTo(&backend, false, "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":0,\"logprobs\":1,\"echo\":true}");
    defer asked.deinit();
    try T.expectEqual(@as(u32, 1), backend.max_tokens);
    const plain = choiceOf(try whole(a, &reply, "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":0}"));
    try T.expectEqualStrings("Hi", plain.get("text").?.string);
}

test "completions echo: echo false, null or absent leaves the reply as it was" {
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = [_]u32{ 'H', 'i' };
    const head = "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":2,\"logprobs\":2";
    const want = try whole(a, &reply, head ++ "}");
    for ([_][]const u8{ ",\"echo\":false}", ",\"echo\":null}" }) |tail| {
        const got = try whole(a, &reply, try std.mem.concat(a, u8, &.{ head, tail }));
        try T.expectEqualStrings(try compact(a, want.get("choices").?), try compact(a, got.get("choices").?));
        try T.expectEqualStrings(try compact(a, want.get("usage").?), try compact(a, got.get("usage").?));
    }
}

test "completions echo: beside prompt_logprobs each form keeps its own count of alternatives" {
    var arena = std.heap.ArenaAllocator.init(T.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = [_]u32{ 'H', 'i' };
    // the engine reads two alternatives a row for echo: prompt_logprobs still lists one
    const wide = choiceOf(try whole(a, &reply, "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":2,\"logprobs\":2,\"prompt_logprobs\":1,\"echo\":true}"));
    try T.expectEqualStrings("[null,{\"98\":{\"logprob\":-0.125,\"rank\":2,\"decoded_token\":\"b\"},\"122\":{\"logprob\":-0.0,\"rank\":1,\"decoded_token\":\"z\"}},{\"99\":{\"logprob\":-0.25,\"rank\":1,\"decoded_token\":\"c\"}}]", try compact(a, wide.get("prompt_logprobs").?));
    try T.expectEqualStrings("{\"z\":-0.0,\"B\":-0.0625}", try compact(a, wide.get("logprobs").?.get("top_logprobs").?.array[1]));
    // and the other way round: echo lists one, prompt_logprobs two
    const narrow = choiceOf(try whole(a, &reply, "{\"model\":\"test-model\",\"prompt\":\"abc\",\"max_tokens\":2,\"logprobs\":1,\"prompt_logprobs\":2,\"echo\":true}"));
    try T.expectEqualStrings("{\"z\":-0.0}", try compact(a, narrow.get("logprobs").?.get("top_logprobs").?.array[1]));
    try T.expectEqual(@as(usize, 3), narrow.get("prompt_logprobs").?.array[1].object.count());
    try T.expectEqual(@as(usize, 2), narrow.get("prompt_logprobs").?.array[2].object.count());
}

test "completions echo: refused by name before anything is submitted, while chat routes still ignore the field" {
    const reply = [_]u32{ 'H', 'i' };
    const cases = [_]struct { body: []const u8, message: []const u8, rows: bool = true }{
        .{ .body = "{\"prompt\":\"abc\",\"echo\":true,\"stream\":true}", .message = "echo is supported on /v1/completions without streaming only" },
        .{ .body = "{\"prompt\":\"abc\",\"echo\":\"yes\"}", .message = "echo must be a boolean or null" },
        .{ .body = "{\"prompt\":\"abc\",\"echo\":1}", .message = "echo must be a boolean or null" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"echo\":true}", .message = "echo is supported with a prompt, not messages" },
        .{ .body = "{\"prompt\":\"abc\",\"echo\":true}", .message = "echo is not supported by this model or backend", .rows = false },
        .{ .body = "{\"prompt\":\"abc\",\"echo\":true,\"max_tokens\":0,\"logprobs\":1}", .message = "echo is not supported by this model or backend", .rows = false },
    };
    for (cases) |c| {
        var backend: fixture.Script = .{ .reply = &reply, .prompt_rows = c.rows };
        var got = try sendTo(&backend, false, c.body);
        defer got.deinit();
        try T.expectEqual(@as(u16, 400), got.status);
        try T.expect(std.mem.indexOf(u8, got.body, c.message) != null);
        try T.expectEqual(@as(usize, 0), backend.submits);
    }
    var backend: fixture.Script = .{ .reply = &reply, .prompt_rows = false };
    var got = try sendTo(&backend, true, "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2,\"echo\":true}");
    defer got.deinit();
    try T.expectEqual(@as(u16, 200), got.status);
    try T.expect(std.mem.indexOf(u8, got.body, "\"content\": \"Hi\"") != null);
}
