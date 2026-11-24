//! Inkling 2 grammar replies: an end of message after a bare call ends the turn; finish_reason keeps the cap.
const std = @import("std");
const h = @import("inkling2_chat_test.zig");
const t = std.testing;

const call_x = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
const call_y = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"y\"}}</tool_call>";

test "Inkling 2 reply: a bare call then its end of message ends the turn, not three repeats to the cap" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the served failure's shape: the call, the model's own end of the call's message, then what it wrote past it
    const reply = try h.encodeReply(a, call_x ++ "<|end_message|>" ++ call_x ++ call_x);
    const head = (try h.encodeReply(a, call_x ++ "<|end_message|>")).len;
    const got = try h.probeIds(a, "release", reply, h.lookup);
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
    const two = try h.probeIds(a, "release", reply, h.lookup);
    try t.expectEqualStrings("lookuplookup", two.names);
    try t.expectEqualStrings("{\"q\":\"x\"}{\"q\":\"y\"}", two.arguments);
    try t.expectEqualStrings("tool_calls", two.finish);
    const one = try h.probeIds(a, "release", reply, h.lookup ++ ",\"parallel_tool_calls\":false");
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
    const capped = try h.probeIds(a, "release", bare, try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}", .{ h.lookup, bare.len }));
    try t.expectEqualStrings("lookuplookup", capped.names);
    try t.expectEqualStrings("length", capped.finish);
    const native = try h.encodeReply(a, "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"q\":1}}<|end_message|>");
    const cut = try h.probeIds(a, "release", native, try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}", .{ h.lookup, native.len }));
    try t.expectEqualStrings("lookup", cut.names);
    try t.expectEqualStrings("length", cut.finish);
    // the same calls ending before the cap, on the end of sampling, still report tool_calls
    const ended = try h.probeIds(a, "release", try h.encodeReply(a, call_x ++ "<|end_message|>"), try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}", .{ h.lookup, bare.len + 8 }));
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
