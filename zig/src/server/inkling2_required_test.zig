//! Inkling 2 grammar replies: an end of message after a bare call ends the turn; finish_reason keeps the cap.
const std = @import("std");
const h = @import("inkling2_chat_test.zig");
const t = std.testing;

const call_x = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
const call_y = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"y\"}}</tool_call>";
/// The tools with tool_choice "required": the call grammar whose end of message may end the turn.
const required = h.lookup ++ ",\"tool_choice\":\"required\"";

test "Inkling 2 reply: a bare call then its end of message ends the turn, not three repeats to the cap" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the served failure's shape: the call, the model's own end of the call's message, then what it wrote past it
    const reply = try h.encodeReply(a, call_x ++ "<|end_message|>" ++ call_x ++ call_x);
    const head = (try h.encodeReply(a, call_x ++ "<|end_message|>")).len;
    const got = try h.probeIds(a, "release", reply, required);
    try t.expectEqualStrings("lookup", got.names);
    try t.expectEqualStrings("{\"q\":\"x\"}", got.arguments);
    try t.expectEqualStrings("", got.content);
    try t.expectEqualStrings("tool_calls", got.finish);
    try t.expectEqual(@as(i64, @intCast(head)), got.usage.completion);
}

test "Inkling 2 reply: a different second call still returns with parallel calls; parallel_tool_calls false keeps one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try h.encodeReply(a, call_x ++ "\n" ++ call_y ++ "<|end_message|>" ++ call_x);
    const two = try h.probeIds(a, "release", reply, required);
    try t.expectEqualStrings("lookuplookup", two.names);
    try t.expectEqualStrings("{\"q\":\"x\"}{\"q\":\"y\"}", two.arguments);
    try t.expectEqualStrings("tool_calls", two.finish);
    const one = try h.probeIds(a, "release", reply, required ++ ",\"parallel_tool_calls\":false");
    try t.expectEqualStrings("lookup", one.names);
    try t.expectEqualStrings("{\"q\":\"x\"}", one.arguments);
    try t.expectEqualStrings("", one.content);
    try t.expectEqualStrings("tool_calls", one.finish);
}

test "Inkling 2 reply: finished calls in a reply the token cap ended keep finish_reason length, bare or native" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bare = try h.encodeReply(a, call_x ++ call_x);
    const capped = try h.probeIds(a, "release", bare, try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}", .{ required, bare.len }));
    try t.expectEqualStrings("lookuplookup", capped.names);
    try t.expectEqualStrings("length", capped.finish);
    const native = try h.encodeReply(a, "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"q\":1}}<|end_message|>");
    const cut = try h.probeIds(a, "release", native, try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}", .{ required, native.len }));
    try t.expectEqualStrings("lookup", cut.names);
    try t.expectEqualStrings("length", cut.finish);
    // the same calls ending before the cap, on the end of sampling, still report tool_calls
    const ended = try h.probeIds(a, "release", try h.encodeReply(a, call_x ++ "<|end_message|>"), try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}", .{ required, bare.len + 8 }));
    try t.expectEqualStrings("tool_calls", ended.finish);
}

test "Inkling 2 reply: an end of message inside a native reply does not end the turn" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try h.encodeReply(a, "<|message_model|><|content_thinking|>Plan.<|end_message|><|message_model|><|content_text|>391<|end_message|>");
    const reply = try std.mem.concat(a, u32, &.{ text, &.{2} }); // the fixture's EOS
    const got = try h.probeIds(a, "release", reply, h.lookup);
    try t.expectEqualStrings("Plan.", got.reasoning);
    try t.expectEqualStrings("391", got.content);
    try t.expectEqualStrings("stop", got.finish);
}

/// The end ids a plain reply's request carries (the release fixture's eos ids).
fn plainEnds(a: std.mem.Allocator) !usize {
    return (try h.probeIds(a, "release", try endsOwn(a, "x"), "")).eos_len;
}

/// `text` then the release fixture's EOS (2), as a reply that ends on its own.
fn endsOwn(a: std.mem.Allocator, text: []const u8) ![]const u32 {
    return std.mem.concat(a, u32, &.{ try h.encodeReply(a, text), &.{2} });
}

test "Inkling 2 reply: KS62 1, a bare end of message in a reply with no tools or auto tools does not end it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try endsOwn(a, "left<|end_message|>right");
    for ([_][]const u8{ "", h.lookup }) |extra| {
        const got = try h.probeIds(a, "release", reply, extra);
        try t.expectEqualStrings("leftright", got.content);
        try t.expectEqualStrings("stop", got.finish);
        try t.expectEqual(try plainEnds(a), got.eos_len);
    }
}

test "Inkling 2 reply: KS62 2, a JSON grammar keeps its own ends and is not cut at an end of message" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try endsOwn(a, "{\"a\":1}<|end_message|>x");
    // the request boundary refuses a format beside a required call (grammar.refusal), so a format here stands alone
    for ([_][]const u8{ "", h.lookup }) |tools| {
        const extra = try std.mem.concat(a, u8, &.{ tools, ",\"response_format\":{\"type\":\"json_object\"}" });
        const got = try h.probeIds(a, "release", reply, extra);
        try t.expectEqualStrings("{\"a\":1}x", got.content);
        try t.expectEqual(try plainEnds(a), got.eos_len);
    }
}

test "Inkling 2 reply: a required call grammar takes the end of message, and only after a complete call" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try h.encodeReply(a, "<|end_message|>" ++ call_x ++ "<|end_message|>" ++ call_x);
    const got = try h.probeIds(a, "release", reply, required);
    try t.expectEqual(try plainEnds(a), got.eos_len);
    try t.expectEqualStrings("lookup", got.names);
    try t.expectEqualStrings("{\"q\":\"x\"}", got.arguments);
    try t.expectEqualStrings("tool_calls", got.finish);
}

test "Inkling 2 reply: KS64, a complete call then an open one: the end of message inside it does not end the turn" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const open_y = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"y\"}}";
    const reply = try endsOwn(a, call_x ++ open_y ++ "<|end_message|></tool_call>");
    const got = try h.probeIds(a, "release", reply, required);
    try t.expectEqualStrings("lookuplookup", got.names);
    try t.expectEqualStrings("{\"q\":\"x\"}{\"q\":\"y\"}", got.arguments);
    try t.expectEqualStrings("tool_calls", got.finish);
}

test "Inkling 2 reply: a required reply ends at an end of message only when it is complete calls and nothing else" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { reply: []const u8, names: []const u8, arguments: []const u8 };
    const cases = [_]Case{
        // a closed but malformed envelope after the call: refused, so the end inside the reply is not its end
        .{ .reply = call_x ++ "<tool_call>{not json}</tool_call><|end_message|>" ++ call_y, .names = "lookuplookup", .arguments = "{\"q\":\"x\"}{\"q\":\"y\"}" },
        // a complete call naming a tool not offered: refused the same way
        .{ .reply = "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call><|end_message|>" ++ call_x, .names = "lookup", .arguments = "{\"q\":\"x\"}" },
        // text beside the call
        .{ .reply = call_x ++ "note<|end_message|>" ++ call_y, .names = "lookuplookup", .arguments = "{\"q\":\"x\"}{\"q\":\"y\"}" },
        // only blanks after the call: the reply is the call, so the end of message ends it
        .{ .reply = call_x ++ "\n <|end_message|>" ++ call_y, .names = "lookup", .arguments = "{\"q\":\"x\"}" },
        // two complete calls: both, then the end
        .{ .reply = call_x ++ call_y ++ "<|end_message|>" ++ call_x, .names = "lookuplookup", .arguments = "{\"q\":\"x\"}{\"q\":\"y\"}" },
    };
    for (cases) |case| {
        const got = try h.probeIds(a, "release", try endsOwn(a, case.reply), required);
        try t.expectEqualStrings(case.names, got.names);
        try t.expectEqualStrings(case.arguments, got.arguments);
        try t.expectEqualStrings("tool_calls", got.finish);
    }
}

test "Inkling 2 reply: a json_schema reply ends on the same token ids as before the call stop, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the model's own end of message after the JSON, then the end of sampling (the fixture's EOS, 2)
    const reply = try endsOwn(a, "{\"a\":1}<|end_message|>");
    const want = try std.json.Stringify.valueAlloc(a, reply, .{});
    const extra = ",\"return_token_ids\":true,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"r\",\"schema\":{\"type\":\"object\"}}}";
    for ([_]usize{ 0, 1, 2, 3, 256 }) |stride| {
        const got = try h.ask(a, "release", reply, extra, stride > 0, @max(stride, 1));
        // every generated id, the end of message and the end of sampling last, as 267544184f returns them
        try t.expectEqualStrings(want, got.token_ids);
        try t.expectEqualStrings("{\"a\":1}", got.content);
        try t.expectEqualStrings("stop", got.finish);
        // compiled by the shared compiler, whose ends are the eos ids alone, and sent with the plain ends
        try t.expectEqual(@as(usize, 1), got.grammars);
        try t.expectEqual(@as(usize, 0), got.tool_grammars);
        try t.expectEqual(try plainEnds(a), got.eos_len);
    }
}

test {
    _ = @import("inkling2_stop_test.zig");
    _ = @import("inkling2_thinking_test.zig");
}
